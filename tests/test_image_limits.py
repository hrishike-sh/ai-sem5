import warnings

import numpy as np
import pytest
from PIL import Image, ImageFile

from ivp.media import MediaError, load_image


def source_image(tmp_path, image_format, size):
    path = tmp_path / ("source." + image_format.lower())
    Image.new("RGB", size, (25, 90, 140)).save(path, format=image_format)
    return path


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_oversized_image_is_rejected_before_pixel_decoding(tmp_path, monkeypatch, image_format):
    path = source_image(tmp_path, image_format, (11, 10))
    original = path.read_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    def unexpected_decode(*args, **kwargs):
        pytest.fail("An image above the safe pixel limit must not be decoded")

    monkeypatch.setattr(ImageFile.ImageFile, "load", unexpected_decode)
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(MediaError, match="(?i)resiz"):
            load_image(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_oversized_image_warning_as_error_becomes_an_actionable_media_error(tmp_path, monkeypatch, image_format):
    path = source_image(tmp_path, image_format, (11, 10))
    original = path.read_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with pytest.raises(MediaError, match="(?i)resiz"):
            load_image(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_image_at_the_safe_pixel_limit_still_decodes(tmp_path, monkeypatch, image_format):
    path = source_image(tmp_path, image_format, (10, 10))
    original = path.read_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", Image.DecompressionBombWarning)
        rgb = load_image(path)
    assert not caught
    assert rgb.shape == (10, 10, 3)
    assert rgb.dtype == np.uint8
    assert np.allclose(rgb, (25, 90, 140), atol=2)
    assert path.read_bytes() == original


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_image_above_pillows_hard_limit_keeps_resize_guidance(tmp_path, monkeypatch, image_format):
    path = source_image(tmp_path, image_format, (21, 10))
    original = path.read_bytes()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    with pytest.raises(MediaError, match="(?i)resiz"):
        load_image(path)
    assert path.read_bytes() == original
