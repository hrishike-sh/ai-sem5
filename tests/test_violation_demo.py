"""Real local encoding/rule/review integration on visibly synthetic geometry only."""
import json

import pytest
import urllib.error

from traffic_ai import violation_demo as demo
from traffic_ai.violation_review import ViolationReview
from traffic_ai.violation_preparation import EvidencePreparations
from tests.test_web_server import server, request


class Files:
    def url(self, path, download=False):
        return '/view/' + path.name


@pytest.mark.parametrize('scenario,verdict,status', [
    ('opposing', 'violation', 'supported'),
    ('following', 'no_violation_observed', 'insufficient_evidence'),
    ('unknown', 'cannot_assess', 'insufficient_evidence'),
])
def test_actual_rule_demo_encodes_and_reopens_review(tmp_path, scenario, verdict, status):
    result = demo.create_demo(tmp_path, {'scenario': scenario})
    run = tmp_path / 'runs' / result['id']
    request = json.loads((run / 'request.json').read_text())
    metadata = request['synthetic_demo']
    assert result['synthetic'] is True and metadata['detector_executed'] is False
    assert metadata['rule_engine_executed'] is True and metadata['assessment_scope'] == ['wrong_way']
    assert request['info']['frame_count'] == 30 and request['info']['fps'] == 10
    assert 'not legal violations' in result['notice']
    records = [json.loads(line) for line in (run / 'detections.jsonl').read_text().splitlines()]
    assert [r['timestamp'] for r in records] == [i / 10 for i in range(30)]
    assert all(r['synthetic'] and len(r['decisions']) == 1 for r in records)
    assert records[-1]['decisions'][0]['verdict'] == verdict
    assert not (run / 'events.jsonl').read_bytes()
    review = ViolationReview(tmp_path, Files())
    view = review.view(result['id'])
    assert view['synthetic'] is True and len(view['cards']) == 1
    card = view['cards'][0]
    assert card['recorded_verdict'] == verdict and card['status'] == status
    bound = {'id': result['id'], 'fingerprint': view['fingerprint'], 'card_id': card['id']}
    frames = review.frames(bound)['frames']
    if scenario != 'unknown':
        assert [f['timestamp'] for f in frames] == [0, .5, 1.0]
        assert all(f['available'] for f in frames)
    else:
        assert frames[0]['available'] is False  # Never fabricate pre-clip footage.
    preparation = EvidencePreparations(review).prepare(bound)
    assert preparation['mode'] == 'images_and_facts'
    loaded = EvidencePreparations(review).load({**bound, 'preparation_id': preparation['preparation_id']})
    assert len(loaded) == (2 if scenario == 'unknown' else 3)
    review.decide({**bound, 'decision': 'accept' if scenario == 'opposing' else 'reject',
                   'note': 'SYNTHETIC configured-rule demonstration only.'})
    reopened = ViolationReview(tmp_path, Files()).view(result['id'])
    assert reopened['cards'][0]['review']['note'].startswith('SYNTHETIC')
    exported = review.export(bound)
    payload = json.loads((run / exported['filename']).read_text())
    assert payload['synthetic'] is True and payload['external_reporting'] is False
    assert payload['cards'][0]['status'] == status
    assert 'data:image' not in json.dumps(payload)


@pytest.mark.parametrize('data', [None, {}, {'scenario': 'real'}, {'scenario': '../x'},
                                       {'scenario': []}, {'scenario': 'opposing', 'source': 'private.mp4'}])
def test_invalid_demo_cannot_select_external_source_or_write(tmp_path, data):
    with pytest.raises(ValueError):
        demo.create_demo(tmp_path, data)
    assert list(tmp_path.iterdir()) == []


def test_encoding_failure_does_not_publish_completed_demo(tmp_path, monkeypatch):
    def failed(*args):
        raise ValueError('Synthetic test encoder failure')
    monkeypatch.setattr(demo, '_encode', failed)
    with pytest.raises(ValueError, match='encoder failure'):
        demo.create_demo(tmp_path, {'scenario': 'opposing'})
    run = next((tmp_path / 'runs').iterdir())
    assert json.loads((run / 'status.json').read_text())['state'] == 'failed'
    with pytest.raises(ValueError):
        ViolationReview(tmp_path, Files()).view(run.name)


def test_new_demo_never_overwrites_saved_decision(tmp_path):
    first = demo.create_demo(tmp_path, {'scenario': 'opposing'})
    review = ViolationReview(tmp_path, Files())
    view = review.view(first['id'])
    review.decide({'id': first['id'], 'fingerprint': view['fingerprint'], 'card_id': view['cards'][0]['id'],
                   'decision': 'accept', 'note': 'Keep this saved review.'})
    second = demo.create_demo(tmp_path, {'scenario': 'opposing'})
    assert first['id'] != second['id']
    assert review.view(first['id'])['cards'][0]['review']['note'] == 'Keep this saved review.'


def test_demo_http_requires_session_token_and_returns_reopenable_run(server):
    with pytest.raises(urllib.error.HTTPError) as denied:
        request(server, '/api/violation-demo', {'scenario': 'opposing'}, token=False)
    assert denied.value.code == 403
    created = request(server, '/api/violation-demo', {'scenario': 'opposing'})
    run = request(server, '/api/run?id=' + created['id'])
    view = request(server, '/api/violation-review?id=' + created['id'])
    assert run['status']['state'] == 'complete'
    assert run['report']['synthetic_demo']['detector_executed'] is False
    assert view['synthetic'] is True and view['cards'][0]['status'] == 'supported'
