import io
import json
import zipfile

import cv2
import numpy as np
import pytest

from ivp.automatic import choose_settings
from ivp.pipeline import process_frame
from ivp.stages import build_stages, manifest
from tests.test_web_server import server as server, request


@pytest.mark.parametrize('kind', ['black', 'white', 'dark', 'noisy', 'stripes'])
def test_automatic_graph_is_finite_and_deterministic(kind):
    rgb = np.full((120, 180, 3), {'black':0, 'white':255, 'dark':25}.get(kind, 90), np.uint8)
    if kind == 'noisy':
        rgb = np.random.default_rng(19).integers(0, 256, rgb.shape, dtype=np.uint8)
    if kind == 'stripes':
        for y in (65, 85, 105):
            rgb[y:y+7, 40:140] = 255
    original = rgb.copy()
    settings, reason = choose_settings(rgb)
    assert settings == choose_settings(rgb)[0]
    result = process_frame(rgb, settings)
    steps = build_stages(result, settings)
    assert len(steps) == 28
    assert {'paint_background', 'paint_contrast', 'paint_mask'} <= {s['id'] for s in steps}
    assert np.array_equal(original, rgb)
    seen = set()
    for step in steps:
        assert step['parent'] is None or step['parent'] in seen
        seen.add(step['id'])
        assert step['image'].dtype == np.uint8
        assert step['image'].shape[:2] == result.color_rgb.shape[:2]
        assert np.isfinite(step['image']).all()
    by_id = {s['id']:s['image'] for s in steps}
    assert np.array_equal(by_id['gray'], cv2.cvtColor(result.stages['Prepared'], cv2.COLOR_RGB2GRAY))
    assert set(np.unique(by_id['canny'])) <= {0, 255}
    assert set(np.unique(by_id['otsu'])) <= {0, 255}
    assert 'image' not in manifest(steps)[0]
    if kind == 'dark':
        assert settings.gamma > 1
    if kind == 'white':
        assert settings.gamma < 1 and not settings.clahe


def test_automatic_preview_export_and_manual_override(server):
    from PIL import Image
    buf = io.BytesIO()
    Image.new('RGB', (80, 60), (20, 20, 20)).save(buf, 'PNG')
    media = request(server, '/api/upload?name=dark.png', raw=buf.getvalue())
    data = {'media':media['id'], 'mode':'automatic', 'settings':{'gamma':4}}
    preview = request(server, '/api/preview', data)
    assert preview['settings']['gamma'] == 1.5
    assert len(preview['ivp_steps']) == 28
    assert preview['automatic']['mode'] == 'automatic'
    exported = request(server, '/api/export', data)
    assert exported['report']['settings'] == preview['settings']
    import urllib.request
    link = next(d['url'] for d in exported['downloads'] if d['name'] == 'ivp-stages.zip')
    with urllib.request.urlopen(link) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            assert len(archive.namelist()) == 30
            assert len(json.loads(archive.read('stages.json'))) == 28
            assert json.loads(archive.read('processing.json'))['settings'] == preview['settings']
            assert Image.open(archive.open('02-gray.png')).mode == 'L'
    manual = request(server, '/api/preview', {**data, 'mode':'manual','settings':{}})
    assert manual['settings']['gamma'] == 1
    assert manual['automatic']['mode'] == 'manual'


def test_video_auto_profile_uses_first_frame_once(server, monkeypatch):
    lab = server.lab
    lab.media['clip'] = (None, {'kind': 'video'})
    calls = []
    def frame(data):
        calls.append(data['timestamp'])
        rgb = np.full((60, 80, 3), 20 if data['timestamp'] == 0 else 240, np.uint8)
        return None, {'kind': 'video'}, rgb, data['timestamp']
    monkeypatch.setattr(lab, 'frame', frame)
    first = lab.resolve_settings({'media': 'clip', 'mode': 'automatic', 'timestamp': 7})
    later = lab.resolve_settings({'media': 'clip', 'mode': 'automatic', 'timestamp': 15})
    assert calls == [0]
    assert first == later and first[0].gamma == 1.5
    assert first[1]['reference_timestamp'] == 0
