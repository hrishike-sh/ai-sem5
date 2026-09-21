import numpy as np
import pytest

from ivp.inspection import inspect_region
from ivp.pipeline import Settings, process_frame, quality
from web_server import parse_settings


def test_exposure_and_saturation_preserve_original_masks():
    rgb = np.full((40, 60, 3), (100, 20, 10), np.uint8)
    baseline = process_frame(rgb)
    exposed = process_frame(rgb, Settings(exposure=1))
    assert np.array_equal(exposed.color_rgb[0, 0], [200, 40, 20])
    assert np.array_equal(exposed.features['Colour mask'], baseline.features['Colour mask'])
    gray = process_frame(rgb, Settings(saturation=0)).color_rgb
    assert np.array_equal(gray[:, :, 0], gray[:, :, 1])
    assert np.array_equal(gray[:, :, 1], gray[:, :, 2])
    assert np.array_equal(rgb[0, 0], [100, 20, 10])


def test_white_balance_neutrality_and_uniform_stretch():
    neutral = np.full((20, 30, 3), 70, np.uint8)
    assert np.array_equal(process_frame(neutral, Settings(white_balance=1, stretch=5)).color_rgb, neutral)
    cast = np.full_like(neutral, (120, 80, 60))
    corrected = process_frame(cast, Settings(white_balance=1)).color_rgb
    assert np.ptp(corrected[0, 0]) <= 1
    ramp = np.tile(np.linspace(60, 140, 100, dtype=np.uint8), (30, 1))
    rgb = np.repeat(ramp[:, :, None], 3, axis=2)
    stretched = process_frame(rgb, Settings(stretch=1)).color_rgb
    assert stretched.min() == 0 and stretched.max() == 255


def test_gradient_and_morphology_on_known_structure():
    rgb = np.zeros((40, 60, 3), np.uint8)
    rgb[10:30, 20:40] = 255
    features = process_frame(rgb).features
    assert features['Sobel magnitude'].max() == 255
    assert features['Sobel magnitude'][20, 30] == 0
    assert features['Morphological gradient'][10, 20] == 255
    assert features['Morphological gradient'][20, 30] == 0
    assert quality(np.zeros_like(rgb))['entropy_bits'] == 0


def test_region_measurements_match_pixels_and_source_mapping():
    rgb = np.zeros((80, 160, 3), np.uint8)
    rgb[:, 80:] = 100
    result = process_frame(rgb, Settings(max_width=80, exposure=1))
    region = inspect_region(result, [.5, 0, 1, 1])
    assert region['bounds_processed_xyxy_exclusive'] == [40, 0, 80, 40]
    assert region['original']['brightness'] == 100
    assert region['processed']['brightness'] == 200
    assert region['source_corner_pixel_centres'] == [[80.5, .5], [158.5, 78.5]]
    assert sum(region['histogram_processed']) == 1600
    for invalid in ([0, 0, 0, 1], [-.1, 0, 1, 1], [0, 0, float('nan'), 1], [False, 0, 1, 1]):
        with pytest.raises(ValueError):
            inspect_region(result, invalid)


@pytest.mark.parametrize('value', [{'gamma':float('nan')}, {'kernel':3.5}, {'gamma':True}, {'white_balance':2}, {'max_width':1}, {'unknown':4}, {'exposure':4}])
def test_api_settings_reject_invalid_values(value):
    with pytest.raises(ValueError):
        parse_settings(value)
