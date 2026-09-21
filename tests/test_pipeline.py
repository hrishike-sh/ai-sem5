import cv2
import numpy as np
import pytest
from PIL import Image

from ivp.media import MediaError, load_image
from ivp.pipeline import Settings, process_frame, quality, scene_distance, scene_signature


def test_original_preserves_pixels_and_no_alias():
    rgb = np.random.default_rng(8).integers(0, 256, (73, 101, 3), dtype=np.uint8)
    result = process_frame(rgb)
    assert np.array_equal(rgb, result.color_rgb)
    assert not np.shares_memory(rgb, result.color_rgb)
    assert np.array_equal(result.transform_to_source, np.eye(3))
    assert result.measurements['original'] == quality(rgb) == result.measurements['processed']
    result.measurements['original']['flags'].append('Caller annotation')
    assert 'Caller annotation' not in result.measurements['processed']['flags']


def test_pixel_centre_mapping_and_aspect_ratio():
    rgb = np.zeros((101, 201, 3), np.uint8)
    result = process_frame(rgb, Settings(max_width=100), timestamp=1.25)
    assert result.color_rgb.shape == (50, 100, 3)
    centre = result.transform_to_source @ np.array([49.5, 24.5, 1])
    assert centre == pytest.approx([100, 50, 1])
    assert result.timestamp == 1.25


def test_gamma_brightens_without_changing_original_colour_mask():
    rgb = np.full((48, 48, 3), (110, 8, 8), np.uint8)
    baseline = process_frame(rgb)
    enhanced = process_frame(rgb, Settings(gamma=2, clahe=True))
    assert enhanced.color_rgb.mean() > baseline.color_rgb.mean()
    assert np.array_equal(enhanced.features["Colour mask"], baseline.features["Colour mask"])
    assert np.array_equal(rgb[0, 0], [110, 8, 8])


def test_red_hue_wrap_excludes_green():
    hsv = np.zeros((40, 60, 3), np.uint8)
    hsv[:, :20] = [179, 255, 255]
    hsv[:, 20:40] = [1, 255, 255]
    hsv[:, 40:] = [60, 255, 255]
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    mask = process_frame(rgb, Settings(morphology="Off")).features["Colour mask"]
    assert mask[:, :40].min() == 255
    assert mask[:, 40:].max() == 0


@pytest.mark.parametrize("method", ["Gaussian", "Median", "Bilateral"])
def test_filters_reduce_noise(method):
    noise = np.clip(np.random.default_rng(3).normal(100, 15, (64, 64, 3)), 0, 255).astype(np.uint8)
    out = process_frame(noise, Settings(denoise=method, kernel=5))
    assert out.color_rgb.std() < noise.std()


@pytest.mark.parametrize("mode", ["Otsu", "Adaptive", "Fixed"])
def test_binary_features_and_candidate_shapes(mode):
    rgb = np.zeros((240, 320, 3), np.uint8)
    for y in (120, 150, 180):
        cv2.rectangle(rgb, (70, y), (250, y + 12), (255, 255, 255), -1)
    result = process_frame(rgb, Settings(threshold_mode=mode))
    assert set(np.unique(result.features["Threshold"])) <= {0, 255}
    assert len(result.candidates["stripe_boxes"]) == 3
    assert result.candidates["horizontal_segments"]


def test_grayscale_alpha_and_exif(tmp_path):
    gray = tmp_path / "gray.png"
    Image.new("L", (30, 20), 80).save(gray)
    assert load_image(gray).shape == (20, 30, 3)
    alpha = tmp_path / "alpha.png"
    Image.new("RGBA", (20, 20), (255, 0, 0, 0)).save(alpha)
    assert load_image(alpha).min() == 255
    exif = Image.Exif()
    exif[274] = 6
    photo = tmp_path / "rotated.jpg"
    Image.new("RGB", (30, 20), "red").save(photo, exif=exif)
    assert load_image(photo).shape == (30, 20, 3)


@pytest.mark.parametrize("gray", [
    [[0, 1, 32, 64], [96, 128, 192, 255]],
    [[0, 1, 2], [3, 4, 5]],
    [[126, 127], [128, 129]],
    [[0, 0], [0, 0]],
    [[255, 255], [255, 255]],
], ids=["gradient", "dark", "narrow-range", "black", "white"])
def test_16bit_grayscale_matches_equivalent_8bit_image(tmp_path, gray):
    gray = np.array(gray, dtype=np.uint8)
    eight_bit = tmp_path / "eight.png"
    sixteen_bit = tmp_path / "sixteen.png"
    Image.fromarray(gray).save(eight_bit)
    Image.fromarray(gray.astype(np.uint16) * 257).save(sixteen_bit)
    original = sixteen_bit.read_bytes()
    expected = load_image(eight_bit)
    actual = load_image(sixteen_bit)
    assert actual.dtype == np.uint8
    assert np.array_equal(actual, expected)
    processed = process_frame(actual)
    assert np.array_equal(processed.color_rgb, expected)
    assert processed.measurements == process_frame(expected).measurements
    assert sixteen_bit.read_bytes() == original


def test_16bit_grayscale_rounding_and_full_range(tmp_path):
    path = tmp_path / "rounding.png"
    gray = np.array([[0, 128, 129, 256], [257, 32768, 65534, 65535]], dtype=np.uint16)
    Image.fromarray(gray).save(path)
    expected = np.array([[0, 0, 1, 1], [1, 128, 255, 255]], dtype=np.uint8)
    assert np.array_equal(load_image(path), np.repeat(expected[:, :, None], 3, axis=2))


def test_16bit_grayscale_retains_exif_orientation(tmp_path):
    path = tmp_path / "oriented.png"
    gray = np.array([[0, 1, 32, 64], [96, 128, 192, 255]], dtype=np.uint16)
    exif = Image.Exif()
    exif[274] = 6
    Image.fromarray(gray * 257).save(path, exif=exif)
    expected = np.array([[96, 0], [128, 1], [192, 32], [255, 64]], dtype=np.uint8)
    assert np.array_equal(load_image(path), np.repeat(expected[:, :, None], 3, axis=2))


@pytest.mark.parametrize("transparent,expected", [
    (0, [[255, 0, 128], [128, 128, 255]]),
    (32768, [[0, 0, 255], [128, 128, 255]]),
])
def test_16bit_grayscale_transparency_uses_original_samples(tmp_path, transparent, expected):
    path = tmp_path / "transparent.png"
    gray = np.array([[0, 1, 32768], [32896, 32769, 65535]], dtype=np.uint16)
    Image.fromarray(gray).save(path, transparency=transparent)
    expected = np.array(expected, dtype=np.uint8)
    assert np.array_equal(load_image(path), np.repeat(expected[:, :, None], 3, axis=2))


def test_dark_quality_and_scene_change():
    dark = np.full((40, 60, 3), 20, np.uint8)
    assert "Dark frame" in quality(dark)["flags"]
    red = np.full((40, 60, 3), (240, 0, 0), np.uint8)
    green = np.full((40, 60, 3), (0, 240, 0), np.uint8)
    assert scene_distance(scene_signature(red), scene_signature(red)) == pytest.approx(0)
    assert scene_distance(scene_signature(red), scene_signature(green)) > 0.9


def test_invalid_images_settings_and_input(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_text("not an image")
    with pytest.raises(MediaError):
        load_image(bad)
    with pytest.raises(ValueError):
        Settings(canny_low=200, canny_high=100)
    with pytest.raises(ValueError):
        Settings(gamma=0)
    with pytest.raises(ValueError):
        process_frame(np.zeros((10, 10), np.uint8))
