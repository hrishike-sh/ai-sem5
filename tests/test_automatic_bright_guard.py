"""Bright automatic enhancement must not erase detail with new clipping."""
import cv2
import numpy as np
import pytest

from ivp.automatic import choose_settings
from ivp.pipeline import Settings, process_frame
from ivp.stages import build_stages


@pytest.mark.parametrize('shape', [(72, 96), (120, 180), (480, 640)])
@pytest.mark.parametrize('value', [206, 218, 230, 244])
def test_automatic_bright_flat_does_not_create_highlights(shape, value):
    rgb = np.full((*shape, 3), value, np.uint8)
    original = rgb.copy()
    settings, _ = choose_settings(rgb)
    result = process_frame(rgb, settings, include_features=False)
    assert result.measurements['processed']['highlight_fraction'] == 0
    assert result.measurements['processed']['brightness'] < value
    assert np.array_equal(rgb, original)
    assert result.timestamp == 0
    assert np.array_equal(result.transform_to_source, np.eye(3))


@pytest.mark.parametrize('shape', [(72, 96), (120, 180), (480, 640)])
def test_bright_gradient_retains_order_and_unclipped_levels(shape):
    h, w = shape
    gray = np.broadcast_to(np.linspace(206, 244, w).astype(np.uint8), (h, w))
    rgb = np.repeat(gray[..., None], 3, axis=2)
    settings, _ = choose_settings(rgb)
    result = process_frame(rgb, settings, include_features=False)
    processed = cv2.cvtColor(result.color_rgb, cv2.COLOR_RGB2GRAY)
    assert len(np.unique(processed)) >= 30
    assert np.all(np.diff(processed[h//2].astype(int)) >= 0)
    assert result.measurements['processed']['highlight_fraction'] == 0
    assert result.measurements['processed']['brightness'] < gray.mean()


def test_bright_policy_explains_withheld_colour_clahe():
    settings, policy = choose_settings(np.full((72, 96, 3), 230, np.uint8))
    assert settings.gamma < 1 and not settings.clahe
    assert policy['method'] == 'deterministic_ivp_v2'
    assert any('bright' in reason.lower() and 'clahe' in reason.lower() for reason in policy['reasons'])


def test_manual_clahe_and_grayscale_demonstration_remain_available():
    rgb = np.full((72, 96, 3), 230, np.uint8)
    settings, _ = choose_settings(rgb)
    result = process_frame(rgb, settings)
    stages = {step['id']: step for step in build_stages(result, settings)}
    assert len(stages) == 28 and 'CLAHE' in stages['contrast']['title']
    assert np.array_equal(stages['gray']['image'], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    manual = process_frame(rgb, Settings(gamma=.85, clahe=True, clip_limit=1.5), include_features=False)
    assert manual.measurements['processed']['highlight_fraction'] == 1
    assert result.measurements['processed']['highlight_fraction'] == 0


def test_video_profile_remains_fixed_to_first_frame_without_http_server():
    from web_server import Lab
    lab = Lab.__new__(Lab)
    lab.media, lab.auto_profiles = {'clip': (None, {'kind': 'video'})}, {}
    calls = []

    def frame(data):
        calls.append(data['timestamp'])
        return None, {'kind': 'video'}, np.full((72, 96, 3), 230, np.uint8), data['timestamp']

    lab.frame = frame
    first = lab.resolve_settings({'media': 'clip', 'mode': 'automatic', 'timestamp': 7})
    later = lab.resolve_settings({'media': 'clip', 'mode': 'automatic', 'timestamp': 15})
    assert calls == [0] and first == later
    assert first[0].gamma < 1 and not first[0].clahe
    assert first[1]['reference_timestamp'] == 0
