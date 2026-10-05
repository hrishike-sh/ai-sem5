"""Synthetic pixels/model selections only; no detector, credentials or network."""
import base64
import json
import os
import socket

import numpy as np
from PIL import Image
import pytest

from ivp.media import source_revision
from traffic_ai import direct_review as module
from traffic_ai.direct_vision import MODEL, PROVIDER
from traffic_ai.violation_images import image_descriptors, image_fingerprint


class Files:
    def __init__(self):
        self.paths = {}

    def url(self, path, download=False):
        value = '/download/' + path.name if download else '/preview/' + path.parent.name + '/' + path.name
        self.paths[value] = path
        return value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Direct storage cannot connect to any network service.')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    root = tmp_path / 'project'
    source = root / 'uploads/synthetic.bmp'
    source.parent.mkdir(parents=True)
    Image.fromarray(np.full((32, 48, 3), 100, np.uint8)).save(source)
    files = Files()
    service = module.DirectReview(root, files)
    info = {'kind': 'image', 'source_revision': source_revision(source)}
    view = service.prepare(source, info, synthetic=True)
    data = {'review_id': view['review_id'], 'fingerprint': view['fingerprint']}
    return service, source, info, view, data, files


def result_for(service, data):
    images = service.load(data)['images']
    return {**data, 'state': 'reviewed', 'provider': PROVIDER, 'model': MODEL, 'mode': 'direct_images',
            'image_fingerprint': image_fingerprint(images), 'evidence': image_descriptors(images),
            'observations': [{'kind': 'head_protection_unclear', 'evidence_ids': [images[0]['id']],
                              'uncertainties': ['model_may_misread'], 'text': 'UNTRUSTED PROSE MUST NOT SURVIVE'}],
            'suspected_issues': [], 'limitations': ['UNTRUSTED LIMITATION']}


def test_direct_preview_is_exact_bounded_jpeg_with_no_detector_required(prepared):
    service, source, _, view, data, files = prepared
    original = source.read_bytes()
    loaded = service.load(data)
    frame, image = view['frames'][0], loaded['images'][0]
    assert frame['available'] is True and frame['position'] == 'selected'
    assert frame['timestamp'] == 0 and frame['role'] == 'context'
    assert base64.b64decode(image['data_url'].split(',')[1]) == files.paths[frame['url']].read_bytes()
    assert view['synthetic'] is True and view['model_result'] is None
    assert view['source_readiness']['available'] is True
    assert source.read_bytes() == original
    assert not (service.root / 'runs').exists()


def test_result_and_human_disposition_persist_separately_and_export(prepared):
    service, source, _, _, data, files = prepared
    original = source.read_bytes()
    saved = service.save_result(data, result_for(service, data))
    assert 'UNTRUSTED' not in json.dumps(saved['model_result'])
    model_result = saved['model_result']
    for action in ('accept', 'reject', 'reset'):
        result = service.decide({**data, 'decision': action, 'note': '<reviewer note>'})
        assert result['model_result'] == model_result
        assert result['review']['decision'] == (None if action == 'reset' else action)
    reopened = module.DirectReview(service.root, files).view(data['review_id'])
    assert reopened['model_result'] == model_result
    exported = service.export(data)
    report = json.loads(files.paths[exported['url']].read_text())
    assert report['external_reporting'] is False and report['model_result'] == model_result
    assert report['synthetic'] is True and source.read_bytes() == original
    assert service.recent()[0]['review_id'] == data['review_id']


@pytest.mark.parametrize('change', ['source', 'same_revision_source', 'missing_source', 'jpeg', 'manifest'])
def test_changed_source_or_prepared_bytes_cannot_be_loaded_or_saved(prepared, change):
    service, source, _, view, data, files = prepared
    model_result = result_for(service, data)
    if change in ('source', 'same_revision_source'):
        stat = source.stat()
        raw = bytearray(source.read_bytes()); raw[-1] ^= 1; source.write_bytes(raw)
        if change == 'same_revision_source':
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == 'missing_source':
        source.unlink()
    elif change == 'jpeg':
        files.paths[view['frames'][0]['url']].write_bytes(b'not a JPEG')
    else:
        path = service.root / 'direct-reviews' / data['review_id'] / 'manifest.json'
        manifest = json.loads(path.read_text()); manifest['requested_timestamp'] = 2; path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        service.load(data)
    with pytest.raises((ValueError, OSError)):
        service.save_result(data, model_result)
    if change in ('source', 'same_revision_source', 'missing_source'):
        assert service.view(data['review_id'])['source_readiness']['available'] is False


@pytest.mark.parametrize('change', ['model', 'state', 'fingerprint', 'review_id', 'image_fingerprint', 'evidence', 'citation'])
def test_mismatched_model_result_is_not_saved(prepared, change):
    service, _, _, _, data, _ = prepared
    result = result_for(service, data)
    if change == 'evidence':
        result['evidence'][0]['sha256'] = '0' * 64
    elif change == 'citation':
        result['observations'][0]['evidence_ids'] = ['other-frame']
    else:
        result[change] = 'wrong'
    with pytest.raises(ValueError):
        service.save_result(data, result)
    assert service.view(data['review_id'])['model_result'] is None


def test_video_preparation_uses_actual_decoded_timestamp(prepared, monkeypatch):
    service, source, _, _, _, _ = prepared
    calls = []
    def decode(path, info, timestamp):
        calls.append((path, timestamp))
        return np.full((20, 30, 3), 40, np.uint8), 1.3
    monkeypatch.setattr(module, 'preview_frame', decode)
    view = service.prepare(source, {'kind': 'video', 'duration': 3, 'source_revision': source_revision(source)}, timestamp=1.2)
    assert calls == [(source.resolve(), 1.2)]
    assert view['frames'][0]['timestamp'] == 1.3
    assert view['synthetic'] is None


def test_source_change_during_decode_is_rejected(prepared, monkeypatch):
    service, source, info, _, _, _ = prepared
    def decode(path):
        source.write_bytes(b'changed during decoding')
        return np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(module, 'load_image', decode)
    with pytest.raises(ValueError, match='source changed'):
        service.prepare(source, info)


@pytest.mark.parametrize('timestamp', [-1, float('nan'), float('inf'), True, 1])
def test_still_image_rejects_invalid_timestamp(prepared, timestamp):
    service, source, info, _, _, _ = prepared
    with pytest.raises(ValueError):
        service.prepare(source, info, timestamp=timestamp)


def test_path_bounds_stale_requests_and_size_limit(prepared, tmp_path, monkeypatch):
    service, source, info, _, data, _ = prepared
    external = tmp_path / 'outside.bmp'; external.write_bytes(source.read_bytes())
    with pytest.raises(ValueError):
        service.prepare(external, info)
    with pytest.raises(ValueError):
        service.view('../uploads')
    with pytest.raises(ValueError):
        service.load({**data, 'fingerprint': 'stale'})
    with pytest.raises(ValueError):
        service.decide({**data, 'decision': 'accept'})
    monkeypatch.setattr(module, 'MAX_SOURCE_BYTES', 8)
    with pytest.raises(ValueError, match='limit'):
        service.prepare(source, info)


def test_stored_result_revalidated_and_prose_canonical_on_restore(prepared):
    service, _, _, _, data, _ = prepared
    service.save_result(data, result_for(service, data))
    path = service.root / 'direct-reviews' / data['review_id'] / 'state.json'
    state = json.loads(path.read_text())
    state['model_result']['observations'][0]['text'] = 'UNTRUSTED GUILT CLAIM'
    path.write_text(json.dumps(state))
    assert 'UNTRUSTED' not in json.dumps(service.view(data['review_id']))
    state['model_result']['observations'][0]['evidence_ids'] = ['wrong']
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        service.view(data['review_id'])
