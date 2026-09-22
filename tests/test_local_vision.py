from pathlib import Path
import os
from unittest.mock import patch

import pytest

from traffic_ai.local_vision import validate_observation
from traffic_ai.ollama_runtime import environment
from traffic_ai import ollama_runtime


def test_runtime_paths_are_scoped_to_configured_runtime():
    original = os.environ.get('USERPROFILE')
    with patch('pathlib.Path.mkdir'):
        configured = environment()
    for name in ['USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'TEMP', 'TMP', 'OLLAMA_MODELS', 'CUDA_CACHE_PATH']:
        assert Path(configured[name]).is_relative_to(ollama_runtime.ROOT)
    assert configured['OLLAMA_HOST'] == '127.0.0.1:11445'
    assert configured['OLLAMA_NO_CLOUD'] == '1'
    assert os.environ.get('USERPROFILE') == original


def test_vision_response_has_observations_not_rule_decisions():
    record = {'summary': 'Traffic scene', 'visible_features': ['Motorcycle'],
              'sign_text_or_symbols': [], 'uncertainties': ['Helmet is obscured']}
    assert validate_observation(record) == record
    with pytest.raises(ValueError):
        validate_observation({**record, 'violation': True})
    with pytest.raises(ValueError):
        validate_observation({**record, 'visible_features': 'Motorcycle'})
    with pytest.raises(ValueError):
        validate_observation({**record, 'uncertainties': [True]})


def test_review_request_is_local_bounded_and_saved_with_provenance(tmp_path, monkeypatch):
    import json
    import numpy as np
    from traffic_ai import local_vision

    observation = {'summary': 'A road sign', 'visible_features': [], 'sign_text_or_symbols': ['No left turn'],
                   'uncertainties': ['Applicability unresolved']}
    calls = []
    monkeypatch.setattr(local_vision, 'ROOT', tmp_path)
    monkeypatch.setattr(local_vision, 'ensure_running', lambda: {'version': 'test'})

    def fake_request(route, payload=None, timeout=5):
        calls.append((route, payload, timeout))
        if route == '/api/tags':
            return {'models': [{'name': local_vision.MODEL, 'digest': 'sha256:test'}]}
        return {'done': True, 'done_reason': 'stop', 'message': {'content': observation['summary']},
                'total_duration': 2000000000}

    monkeypatch.setattr(local_vision, 'request', fake_request)
    rgb = np.zeros((1600, 800, 3), np.uint8)
    report = local_vision.review_frame_ollama(rgb, 1.25)
    assert report['source_size'] == [800, 1600]
    assert report['review_size'] == [640, 1280]
    assert report['model_digest'] == 'sha256:test'
    assert report['timestamp'] == 1.25
    assert report['observation']['summary'] == 'A road sign'
    assert calls[1][1]['keep_alive'] == 0
    assert calls[1][1]['options']['num_ctx'] == 4096
    assert calls[1][1]['stream'] is False
    assert len(list((tmp_path / 'reviews').glob('*.json'))) == 1
    assert not rgb.any()
