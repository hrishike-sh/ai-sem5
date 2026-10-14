"""Loopback plate workflow using explicitly synthetic pixels and stubbed predictions."""
import io
import json
import urllib.error
import urllib.request

from PIL import Image
import pytest

from tests.test_web_server import server, request


def bound(view):
    return {key: view[key] for key in ('review_id', 'fingerprint')}


@pytest.fixture
def plate_review(server, monkeypatch):
    calls = []
    def predict(rgb):
        calls.append(True)
        return [{'box': [15., 25., 45., 38.], 'detector_score': .9,
                 'ocr_score': .92, 'text': 'SYN1234'}]
    monkeypatch.setattr('traffic_ai.plate_models.predict_plates', predict)
    buffer = io.BytesIO()
    Image.new('RGB', (64, 48), 'gray').save(buffer, 'PNG')
    source = request(server, '/api/upload?name=synthetic-plate-api.png', raw=buffer.getvalue())
    view = request(server, '/api/direct-prepare', {'media': source['id'], 'source_revision': source['source_revision']})
    view = request(server, '/api/direct-annotate', {**bound(view), 'subjects': [
        {'kind': 'car', 'bbox': [5, 5, 58, 43]}]})
    return source, view, calls


def test_plate_route_correction_export_and_reopen_preserve_original_ocr(server, plate_review):
    _, initial, calls = plate_review
    view = request(server, '/api/direct-plates', bound(initial))
    assert calls == [True] and view['plate_evidence']['state'] == 'complete'
    vehicle = view['plate_evidence']['subjects'][0]
    assert vehicle['status'] == 'candidate'
    candidate = vehicle['candidates'][0]
    fixed = request(server, '/api/direct-plate-correction', {**bound(view),
                    'vehicle_id': vehicle['id'], 'plate_id': candidate['plate_id'], 'text': 'SYN5678'})
    assert fixed['fingerprint'] != view['fingerprint']
    assert fixed['plate_evidence']['subjects'][0]['candidates'][0]['text'] == 'SYN1234'
    assert fixed['plate_evidence']['subjects'][0]['manual_correction']['text'] == 'SYN5678'
    assert fixed['evidence_table']['summary']['suspected_issue_count'] == 0
    reopened = request(server, '/api/direct-review?id=' + fixed['review_id'])
    assert reopened['evidence_table'] == fixed['evidence_table'] and calls == [True]
    exported = request(server, '/api/direct-export', bound(fixed))
    path = server.lab.direct_reviews.root / 'direct-reviews' / fixed['review_id'] / exported['filename']
    payload = json.loads(path.read_text())
    assert payload['plate_evidence'] == fixed['plate_evidence']
    assert payload['evidence_table'] == fixed['evidence_table']
    assert payload['external_reporting'] is False
    with pytest.raises(urllib.error.HTTPError) as stale:
        request(server, '/api/direct-plate-correction', {**bound(view), 'vehicle_id': vehicle['id'],
                'plate_id': candidate['plate_id'], 'text': 'STALE123'})
    assert stale.value.code == 400


def test_plate_routes_require_csrf_and_do_not_run_on_changed_source(server, plate_review):
    source, view, calls = plate_review
    for route in ('/api/direct-plates', '/api/direct-plate-correction'):
        with pytest.raises(urllib.error.HTTPError) as denied:
            request(server, route, bound(view), token=False)
        assert denied.value.code == 403
    path, _ = server.lab.media[source['id']]
    path.write_bytes(b'Changed synthetic fixture')
    with pytest.raises(urllib.error.HTTPError) as changed:
        request(server, '/api/direct-plates', bound(view))
    assert changed.value.code == 400 and calls == []


def test_evidence_table_script_is_served_by_loopback_app(server):
    with urllib.request.urlopen(f'http://127.0.0.1:{server.server_port}/evidence_table.js') as response:
        assert response.status == 200
        assert 'javascript' in response.headers['Content-Type']
        assert b'root.ViolationEvidence=api' in response.read()
