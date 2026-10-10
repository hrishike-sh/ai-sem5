"""Loopback API integration using synthetic pictures and mocked Astra responses."""
import io
import json
import urllib.error
import threading

from PIL import Image
import pytest

from traffic_ai.violation_images import image_fingerprint
from traffic_ai.identified_vision import canonicalize_result, subjects_fingerprint
from tests.test_web_server import server, request


class SyntheticAstra:
    def __init__(self):
        self.calls = 0

    def review(self, images, consent=False, subjects=None, local_evidence=None, vehicle_type_evidence=None):
        self.calls += 1
        value = {'image_fingerprint': image_fingerprint(images),
                 'subjects_fingerprint': subjects_fingerprint(subjects),
                 'subject_results': [{'id': s['id'], 'role': 'person_unclear',
                     'associated_vehicle_id': None, 'status': 'unclear',
                     'observations': [{'kind': 'association_unclear', 'evidence_ids': s['evidence_ids']}],
                     'suspected_issues': [], 'uncertainties': ['model_may_misread']} for s in subjects]}
        if local_evidence is not None:
            from traffic_ai.identified_vision import local_evidence_fingerprint
            value['local_evidence_fingerprint'] = local_evidence_fingerprint(local_evidence)
        if vehicle_type_evidence is not None:
            from traffic_ai.identified_vision import vehicle_type_evidence_fingerprint
            value['vehicle_type_evidence_fingerprint'] = vehicle_type_evidence_fingerprint(vehicle_type_evidence)
        return canonicalize_result(value, images, subjects, local_evidence=local_evidence,
                                   vehicle_type_evidence=vehicle_type_evidence)


@pytest.fixture
def direct(server, monkeypatch):
    monkeypatch.setattr('traffic_ai.subject_models.predict_heads', lambda rgb: [])
    monkeypatch.setattr('traffic_ai.vehicle_types.predict_types', lambda rgb: [])
    pixels = io.BytesIO()
    Image.new('RGB', (64, 48), 'gray').save(pixels, 'PNG')
    source = request(server, '/api/upload?name=synthetic-direct.png', raw=pixels.getvalue())
    preview = request(server, '/api/direct-prepare', {'media': source['id'], 'source_revision': source['source_revision']})
    preview = request(server, '/api/direct-annotate', {**bound(preview),
                      'subjects': [{'kind': 'person', 'bbox': [10, 5, 35, 40]}]})
    preview = request(server, '/api/direct-local-models', bound(preview))
    model = SyntheticAstra()
    server.lab.direct_vision = model
    return source, preview, model


def bound(preview):
    return {key: preview[key] for key in ('review_id', 'fingerprint')}


def test_manually_numbered_flow_reopens_annotations_decision_and_export(server, direct):
    _, preview, model = direct
    assert preview['frames'][0]['available'] and preview['model_result'] is None and model.calls == 0
    result = request(server, '/api/direct-analyze', {**bound(preview), 'consent': True})
    assert model.calls == 1 and result['model_result']['model'] == 'gpt-6-astra'
    assert result['model_result']['subject_results'][0]['suspected_issues'] == []
    assert result['model_result']['subject_results'][0]['id'] == preview['annotations']['subjects'][0]['id']
    decided = request(server, '/api/direct-decision', {**bound(preview), 'decision': 'reject', 'note': 'Synthetic API fixture.'})
    assert decided['review']['decision'] == 'reject'
    reopened = request(server, '/api/direct-review?id=' + preview['review_id'])
    assert reopened['review']['note'] == 'Synthetic API fixture.' and model.calls == 1
    exported = request(server, '/api/direct-export', bound(preview))
    path = server.lab.direct_reviews.root / 'direct-reviews' / preview['review_id'] / exported['filename']
    payload = json.loads(path.read_text())
    assert payload['external_reporting'] is False and payload['model_result']['model'] == 'gpt-6-astra'
    assert payload['annotations']['subjects'] == preview['annotations']['subjects']
    assert 'data:image' not in path.read_text() and 'synthetic-oauth' not in path.read_text()
    assert any(item['review_id'] == preview['review_id'] for item in request(server, '/api/direct-library'))


@pytest.mark.parametrize('consent', [None, False, 'true', 1])
def test_no_direct_network_without_literal_consent(server, direct, consent):
    _, preview, model = direct
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(preview), 'consent': consent})
    assert model.calls == 0


def test_changed_original_blocks_direct_send_before_model(server, direct):
    source, preview, model = direct
    path, _ = server.lab.media[source['id']]
    path.write_bytes(b'Changed fixture after preview')
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(preview), 'consent': True})
    assert model.calls == 0
    assert request(server, '/api/direct-review?id=' + preview['review_id'])['source_readiness']['available'] is False


def test_direct_mutation_requires_csrf_and_ignores_client_supplied_images(server, direct):
    _, preview, model = direct
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/direct-analyze', {**bound(preview), 'consent': True}, token=False)
    assert denied.value.code == 403 and model.calls == 0
    result = request(server, '/api/direct-analyze', {**bound(preview), 'consent': True,
                     'images': [{'data_url': 'https://untrusted.invalid/image'}], 'model': 'gpt-6-sol'})
    assert result['model_result']['evidence'][0]['sha256'] == preview['frames'][0]['sha256']
    assert result['model_result']['model'] == 'gpt-6-astra' and model.calls == 1


def test_new_analysis_requires_numbered_subjects(server):
    pixels = io.BytesIO()
    Image.new('RGB', (64, 48), 'gray').save(pixels, 'PNG')
    source = request(server, '/api/upload?name=synthetic-no-boxes.png', raw=pixels.getvalue())
    view = request(server, '/api/direct-prepare', {'media': source['id'], 'source_revision': source['source_revision']})
    model = SyntheticAstra()
    server.lab.direct_vision = model
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/direct-analyze', {**bound(view), 'consent': True})
    assert denied.value.code == 400 and model.calls == 0


def test_annotation_changes_invalidate_old_send_and_saved_decision(server, direct):
    _, view, model = direct
    request(server, '/api/direct-analyze', {**bound(view), 'consent': True})
    request(server, '/api/direct-decision', {**bound(view), 'decision': 'accept'})
    subject = view['annotations']['subjects'][0]
    corrected = request(server, '/api/direct-annotate', {**bound(view), 'subjects': [
        {'id': subject['id'], 'kind': 'person', 'bbox': [11, 5, 36, 40]}]})
    assert corrected['annotations']['subjects'][0]['id'] == subject['id']
    assert corrected['fingerprint'] != view['fingerprint']
    assert corrected['model_result'] is None and corrected['review']['decision'] is None
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(view), 'consent': True})
    assert model.calls == 1


def test_annotation_route_requires_csrf_and_rejects_invalid_subjects(server, direct):
    _, view, model = direct
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/direct-annotate', {**bound(view), 'subjects': []}, token=False)
    assert denied.value.code == 403
    for subjects in (None, {}, [{'id': 'P999', 'kind': 'person', 'bbox': [1, 1, 5, 5]}]):
        with pytest.raises(urllib.error.HTTPError):
            request(server, '/api/direct-annotate', {**bound(view), 'subjects': subjects})
    assert model.calls == 0


def test_local_models_bind_revision_export_and_require_csrf(server, direct):
    _, view, model = direct
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/direct-local-models', bound(view), token=False)
    assert denied.value.code == 403
    fresh = request(server, '/api/direct-local-models', bound(view))
    assert fresh['fingerprint'] != view['fingerprint']
    assert fresh['local_evidence']['state'] == 'complete'
    assert fresh['local_evidence']['subjects'][0]['helmet']['status'] == 'not_observed'
    assert model.calls == 0
    exported = request(server, '/api/direct-export', bound(fresh))
    path = server.lab.direct_reviews.root / 'direct-reviews' / fresh['review_id'] / exported['filename']
    assert json.loads(path.read_text())['local_evidence'] == fresh['local_evidence']
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(view), 'consent': True})
    assert model.calls == 0


def test_saved_corrections_require_new_local_evidence(server, direct):
    _, view, model = direct
    changed = request(server, '/api/direct-annotate', {**bound(view), 'subjects': [
        {'id': 'P1', 'kind': 'person', 'bbox': [11, 5, 35, 40]}]})
    assert changed['local_evidence'] is None
    with pytest.raises(urllib.error.HTTPError):
        request(server, '/api/direct-analyze', {**bound(changed), 'consent': True})
    assert model.calls == 0


def test_vehicle_type_endpoint_independently_saves_report(server, direct):
    _, view, model = direct
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/direct-vehicle-types', bound(view), token=False)
    assert denied.value.code == 403
    fresh = request(server, '/api/direct-vehicle-types', bound(view))
    assert fresh['vehicle_type_evidence']['state'] == 'complete'
    assert fresh['local_evidence'] == view['local_evidence']
    assert fresh['feature_report']['features']['vehicle_types']['state'] == 'complete'
    assert fresh['fingerprint'] != view['fingerprint'] and model.calls == 0


def test_completed_post_releases_lock_before_client_can_start_next_stage(server, monkeypatch):
    original = server.RequestHandlerClass.respond
    observed = []
    def response(handler, value, *args, **kwargs):
        if value == {'pipeline_stage': 'completed'}:
            def next_request():
                acquired = server.lab.lock.acquire(blocking=False)
                observed.append(acquired)
                if acquired:
                    server.lab.lock.release()
            thread = threading.Thread(target=next_request)
            thread.start()
            thread.join(timeout=2)
        return original(handler, value, *args, **kwargs)
    monkeypatch.setattr(server.RequestHandlerClass, 'respond', response)
    monkeypatch.setattr(server.lab, 'post', lambda route, data: {'pipeline_stage': 'completed'})
    assert request(server, '/api/synthetic-stage', {'fixture': True}) == {'pipeline_stage': 'completed'}
    assert observed == [True]
