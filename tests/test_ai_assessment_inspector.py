"""Synthetic file-contract fixtures; never detector accuracy or replacement truth."""
import base64
from collections import Counter
import hashlib
import json
from urllib.parse import urlencode
import urllib.error

import pytest

from ivp import ai_assessments as module
from ivp.media import write_json
from tests.test_web_server import server as server, request


@pytest.fixture
def saved(tmp_path):
    run = tmp_path / 'runs/ai-inspector-fixture'
    run.mkdir(parents=True)
    decisions = [{'rule': 'footpath_riding' if i % 2 else 'prohibited_turn',
                  'verdict': 'no_violation_observed' if i % 9 == 0 else 'cannot_assess',
                  'reason': f'SYNTHETIC CONTRACT {i}: <img src=x onerror=alert(1)>',
                  'timestamp': i / 10, 'track_id': i, 'evidence': {'synthetic': True}}
                 for i in range(45)]
    records = [{'timestamp': stamp, 'scene_id': 1, 'decisions': decisions[i*15:(i+1)*15]}
               for i, stamp in enumerate([0., .2, .1])]
    records[1].update(frame=3, scene_revision=2, camera_stable=False, synthetic=True)
    (run / 'detections.jsonl').write_text('\n' + ''.join(json.dumps(r)+'\n' for r in records), encoding='utf8')
    write_json(run / 'status.json', {'state': 'complete'})
    write_json(run / 'report.json', {'ai': {'assessment_counts': dict(Counter(
        f"{d['rule']}/{d['verdict']}" for d in decisions))}})
    return tmp_path, run, decisions


def hashes(run):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.is_file()}


def page(saved, **kwargs):
    root, run, _ = saved
    return module.saved_ai_page(root, run.name, **kwargs)


def test_pages_preserve_exact_decisions_source_order_and_line_pointers(saved):
    _, run, decisions = saved
    before = hashes(run)
    current = page(saved, verdict='all')
    seen = []
    while True:
        assert len(current['decisions']) <= 20
        for card in current['decisions']:
            seen.append(card['decision'])
            record = json.loads((run / 'detections.jsonl').read_text().splitlines()[card['source']['line']-1])
            index = int(card['source']['pointer'].rsplit('/', 1)[1])
            assert record['decisions'][index] == card['decision']
        cursor = current['pagination']['next_cursor']
        if cursor is None:
            break
        current = page(saved, verdict='all', cursor=cursor)
    assert seen == decisions
    assert current['pagination']['start'] == 41 and current['pagination']['end'] == 45
    previous = page(saved, verdict='all', cursor=current['pagination']['previous_cursor'])
    assert previous['decisions'][0]['source_position'] == 21
    assert hashes(run) == before


def test_default_unknown_filter_and_recorded_context_do_not_invent_provenance(saved):
    view = page(saved)
    assert view['pagination']['filtered_total'] == 40 and view['pagination']['total'] == 45
    assert all(c['decision']['verdict'] == 'cannot_assess' for c in view['decisions'])
    first = view['decisions'][0]
    assert first['source_position'] == 2 and first['source']['line'] == 2
    assert first['frame_context']['frame'] is None and first['frame_context']['synthetic'] is None
    second_record = next(c for c in view['decisions'] if c['source']['line'] == 3)
    assert second_record['frame_context']['synthetic'] is True
    assert second_record['frame_context']['camera_stable'] is False
    assert view['report_counts'] == 'match' and 'authenticity is not established' in view['integrity']
    filtered = page(saved, verdict='no_violation_observed', rule='prohibited_turn')
    assert [x['decision']['track_id'] for x in filtered['decisions']] == [0, 18, 36]
    assert page(saved, verdict='violation')['pagination']['filtered_total'] == 0


@pytest.mark.parametrize('options', [{'verdict':'clear'}, {'rule':'speed'}, {'cursor':'broken!'}, {'cursor':'e30'}])
def test_malformed_filters_and_cursors_are_rejected(saved, options):
    with pytest.raises(ValueError):
        page(saved, **options)


def test_cursor_binds_source_run_filters_and_fingerprints(saved):
    _, run, _ = saved
    cursor = page(saved)['pagination']['next_cursor']
    with pytest.raises(ValueError, match='outdated|outside'):
        page(saved, cursor=cursor, verdict='all')
    forged = json.loads(base64.urlsafe_b64decode(cursor+'='*(-len(cursor)%4)))
    for field, value in [('offset', 21), ('offset', 400), ('run', 'different'), ('sha256', '0'*64)]:
        changed = {**forged, field:value}
        token = base64.urlsafe_b64encode(json.dumps(changed).encode()).decode().rstrip('=')
        with pytest.raises(ValueError):
            page(saved, cursor=token)
    with (run / 'detections.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='outdated'):
        page(saved, cursor=cursor)


def test_changed_between_parse_and_integrity_read_exposes_no_cards(saved, monkeypatch):
    _, run, _ = saved
    original = module._digest
    changed = False
    def mutate(path):
        nonlocal changed
        if not changed:
            changed = True
            with (run/'detections.jsonl').open('a') as stream:
                stream.write('\n')
        return original(path)
    monkeypatch.setattr(module, '_digest', mutate)
    with pytest.raises(ValueError, match='changed while reading'):
        page(saved)


def test_missing_report_counts_stay_explicitly_unrecorded(saved):
    _, run, _ = saved
    write_json(run/'report.json', {'ai':{}})
    view = page(saved)
    assert view['report_counts'] == 'not_recorded' and view['pagination']['total'] == 45


def test_card_limit_includes_frame_provenance_and_does_not_truncate_evidence(saved, monkeypatch):
    _, run, _ = saved
    records = [json.loads(line) for line in (run/'detections.jsonl').read_text().splitlines() if line]
    records[0]['provenance'] = 'Synthetic oversized context ' * 100
    (run/'detections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    monkeypatch.setattr(module, 'MAX_DECISION_BYTES', 2048)
    with pytest.raises(ValueError, match='evidence limit'):
        page(saved)


@pytest.mark.parametrize('problem', ['running', 'missing', 'counts', 'corrupt', 'nonfinite', 'oversize'])
def test_incomplete_corrupt_mismatched_and_oversize_sources_are_actionable(saved, monkeypatch, problem):
    _, run, _ = saved
    if problem == 'running':
        write_json(run/'status.json', {'state':'running'})
    elif problem == 'missing':
        (run/'detections.jsonl').unlink()
    elif problem == 'counts':
        write_json(run/'report.json', {'ai':{'assessment_counts':{}}})
    elif problem == 'corrupt':
        (run/'detections.jsonl').write_text('not JSON')
    elif problem == 'nonfinite':
        (run/'detections.jsonl').write_text('{"decisions": [], "timestamp": NaN}')
    else:
        monkeypatch.setattr(module, 'MAX_LINE_BYTES', 32)
    with pytest.raises(ValueError):
        page(saved)


def test_route_exposes_bounded_read_only_page_on_completed_AI_run(server):
    root = server.lab.library.root.parent
    run = root/'runs/inspector-route'
    run.mkdir(parents=True)
    decision = {'rule':'helmet_absence','verdict':'cannot_assess','reason':'Synthetic API boundary', 'timestamp':0}
    (run/'detections.jsonl').write_text(json.dumps({'decisions':[decision]})+'\n')
    write_json(run/'status.json', {'state':'complete'})
    write_json(run/'report.json', {'ai':{'assessment_counts':{'helmet_absence/cannot_assess':1}}})
    before = hashes(run)
    response = request(server, '/api/ai-assessments?'+urlencode({'id':run.name}))
    assert response['decisions'][0]['decision'] == decision
    assert response['pagination']['total'] == 1 and hashes(run) == before
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, '/api/ai-assessments?id=../outside')
    assert error.value.code == 400
