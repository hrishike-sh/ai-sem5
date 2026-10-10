"""Derived reports use synthetic evidence, with no network or model loading."""
from copy import deepcopy
import json

from traffic_ai.feature_report import build_feature_report
from tests.test_subject_models import subject, evidence
from tests.test_direct_review import prepared  # noqa: F401


def annotations(local=None):
    subjects = [subject(), subject('V1', 'motorcycle', origin='manual'), subject('V2', 'car')]
    return {'subjects': subjects, 'detector': {'truncated': True},
            **({'local_evidence': local(subjects)} if local else {})}


def test_pending_helmet_preserves_localization_and_counts():
    data = annotations()
    before = deepcopy(data)
    report = build_feature_report(data)
    assert data == before
    assert report['features']['helmet']['state'] == 'pending'
    assert report['features']['localization']['state'] == 'complete'
    counts = report['features']['vehicle_counts']
    assert counts['state'] == 'complete' and counts['data']['total'] == 2
    assert counts['data']['by_kind']['motorcycle'] == {'total': 1, 'manual': 1, 'detector': 0}
    assert report['summary']['pending_features'] == ['helmet', 'vehicle_types']
    assert report['summary']['vehicle_count'] == 2


def test_unavailable_helmet_does_not_block_other_features(monkeypatch):
    from traffic_ai.subject_models import build_evidence
    import numpy as np
    def fail(_):
        raise RuntimeError('private runtime path')
    monkeypatch.setattr('traffic_ai.subject_models.predict_heads', fail)
    data = annotations(lambda rows: build_evidence(np.zeros((200, 200, 3), dtype=np.uint8), rows, 200, 200, truncated=True))
    report = build_feature_report(data)
    assert report['features']['helmet']['state'] == 'unavailable'
    assert report['features']['helmet']['errors']
    assert report['summary']['completed_features'] == ['localization', 'vehicle_counts']
    assert report['summary']['vehicle_count'] == 2
    assert 'private runtime path' not in str(report)


def test_not_observed_remains_observation_not_no_helmet():
    report = build_feature_report(annotations(lambda rows: evidence(rows, truncated=True)))
    assert report['summary']['helmet_status_counts'] == {'not_observed': 1, 'not_applicable': 2}
    assert report['features']['helmet']['state'] == 'complete'
    assert report['summary']['unavailable_features'] == []


def test_empty_and_legacy_annotations_do_not_invent_detections():
    assert build_feature_report(None) is None
    report = build_feature_report({'subjects': []})
    assert report['features']['localization']['state'] == 'needs_input'
    assert report['summary']['subject_count'] == report['summary']['vehicle_count'] == 0
    assert report['summary']['helmet_status_counts'] == {}


def test_unavailable_localization_does_not_claim_completed_zero_count():
    report = build_feature_report({'subjects': [], 'detector': {'state': 'unavailable'}})
    assert report['features']['localization']['state'] == 'unavailable'
    assert report['features']['vehicle_counts']['state'] == 'needs_input'
    assert 'vehicle_counts' not in report['summary']['completed_features']
    manual = annotations()
    manual['detector']['state'] = 'unavailable'
    assert build_feature_report(manual)['features']['vehicle_counts']['state'] == 'complete'


def test_report_reopens_exports_and_does_not_change_manifest_binding(prepared, monkeypatch):
    service, _, _, legacy, data, files = prepared
    assert legacy['feature_report'] is None
    view = service.annotate(data, subjects=[{'kind': 'person', 'bbox': [2, 2, 16, 29]},
                                          {'kind': 'motorcycle', 'bbox': [3, 14, 35, 31]}])
    bound = {key: view[key] for key in ('review_id', 'fingerprint')}
    folder = service._folder(view['review_id'])
    manifest_before = (folder / 'manifest.json').read_bytes()
    assert service.view(view['review_id'])['feature_report'] == view['feature_report']
    exported = service.export(bound)
    saved = json.loads(files.paths[exported['url']].read_text())
    assert saved['feature_report'] == view['feature_report']
    assert (folder / 'manifest.json').read_bytes() == manifest_before
    monkeypatch.setattr('traffic_ai.subject_models.predict_heads', lambda _: [])
    completed = service.run_models(bound)
    assert completed['feature_report']['summary']['completed_features'] == ['localization', 'helmet', 'vehicle_counts']
    assert completed['feature_report']['summary']['vehicle_count'] == 1
    assert service.view(view['review_id'])['feature_report'] == completed['feature_report']


def test_report_does_not_alias_persisted_evidence():
    data = annotations(lambda rows: evidence(rows, truncated=True))
    report = build_feature_report(data)
    report['features']['helmet']['data']['evidence']['subjects'][0]['helmet']['status'] = 'no_helmet'
    report['features']['localization']['data']['subjects'][0]['bbox'][0] = 999
    assert data['local_evidence']['subjects'][0]['helmet']['status'] == 'not_observed'
    assert data['subjects'][0]['bbox'][0] == 10


def test_vehicle_type_feature_is_independent_and_validated(monkeypatch):
    import sys
    from types import SimpleNamespace
    checked = []
    def validate(value, subjects):
        checked.append((value, subjects))
        return value
    monkeypatch.setitem(sys.modules, 'traffic_ai.vehicle_types', SimpleNamespace(validate_vehicle_evidence=validate))
    data = annotations()
    specialist = {'state': 'unavailable', 'reason': 'Synthetic specialist unavailable.',
                  'score_notice': 'Raw scores.', 'limitations': ['Synthetic fixture.']}
    report = build_feature_report(data, specialist)
    assert checked == [(specialist, data['subjects'])]
    assert report['features']['vehicle_types']['state'] == 'unavailable'
    assert report['summary']['unavailable_features'] == ['vehicle_types']
    assert report['summary']['pending_features'] == ['helmet']
    assert report['summary']['completed_features'] == ['localization', 'vehicle_counts']
    report['features']['vehicle_types']['data']['evidence']['reason'] = 'changed'
    assert specialist['reason'] == 'Synthetic specialist unavailable.'
