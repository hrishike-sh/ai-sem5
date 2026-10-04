"""Synthetic contract fixtures only; these tests make no detector accuracy claim."""
import hashlib
import json

import numpy as np
import pytest

from ivp.marking_policy import configured_geometry_policy, review_only_policy
from ivp.media import png_bytes, source_revision
from traffic_ai import violation_review as module


class Files:
    def __init__(self):
        self.paths = {}

    def url(self, path, download=False):
        url = '/download/' + path.name if download else '/view/' + path.name
        self.paths[url] = path
        return url


def decision(rule='helmet_absence', evidence=None, verdict='violation', stamp=0., track=1):
    return {'rule': rule, 'verdict': verdict, 'timestamp': stamp, 'track_id': track,
            'reason': 'SYNTHETIC CONTRACT fixture, not live inference.',
            'evidence': evidence if evidence is not None else {'heads': [[1, 1, 3, 3]], 'association': 'Recorded rider association'}}


def frame(*decisions, stamp=0., scene=0, **extra):
    return {'timestamp': stamp, 'scene_id': scene, 'synthetic': True,
            'camera_stable': True, 'decisions': list(decisions), **extra}


@pytest.fixture
def saved(tmp_path):
    run = tmp_path / 'runs/synthetic-review'
    run.mkdir(parents=True)
    source = tmp_path / 'uploads/synthetic.png'
    source.parent.mkdir()
    source.write_bytes(png_bytes(np.zeros((8, 8, 3), dtype=np.uint8)))
    request = {'source': str(source), 'info': {'kind': 'image', 'source_revision': source_revision(source)}}
    (run / 'request.json').write_text(json.dumps(request), encoding='utf-8')
    (run / 'status.json').write_text('{"state":"complete"}', encoding='utf-8')
    (run / 'report.json').write_text('{"ai":{}}', encoding='utf-8')
    (run / 'events.jsonl').write_bytes(b'{"original":"synthetic fixture"}\r\n')
    write_frames(run, [frame(decision())])
    files = Files()
    return module.ViolationReview(tmp_path, files), run, source, files


def write_frames(run, records):
    (run / 'detections.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records), encoding='utf-8')


def bound(saved):
    review, run, _, _ = saved
    view = review.view(run.name)
    return {'id': run.name, 'fingerprint': view['fingerprint'], 'card_id': view['cards'][0]['id']}


def set_video(saved):
    _, run, _, _ = saved
    request = json.loads((run / 'request.json').read_text())
    request['info'].update(kind='video', fps=25, stream_index=0, rotation=0)
    (run / 'request.json').write_text(json.dumps(request))


def test_explicit_helmet_evidence_supported_but_never_a_legal_finding(saved):
    review, run, _, _ = saved
    view = review.view(run.name)
    card = view['cards'][0]
    assert card['status'] == 'supported'
    assert card['recorded_verdict'] == 'violation'
    assert not card['missing_facts']
    assert view['synthetic'] is True and card['synthetic'] is True
    assert 'not live inference' in view['provenance_notice']
    assert 'not a legal finding' in view['limits']['notice']
    assert card['review'] == {'decision': None, 'note': ''}
    assert view['source_readiness']['available'] is True
    assert view['source_readiness']['revision_kind'] == 'metadata'


@pytest.mark.parametrize('evidence', [{}, {'helmet_count': 0}, {'heads': []}, {'heads': [[1, 1, 3, 3]]},
                                      {'heads': [[3, 3, 1, 1]], 'association': 'synthetic'}])
def test_missing_helmet_is_not_bare_head_evidence(saved, evidence):
    review, run, _, _ = saved
    write_frames(run, [frame(decision(evidence=evidence))])
    assert review.view(run.name)['cards'][0]['status'] == 'insufficient_evidence'


def direction_evidence():
    return {'observed_motion': [-20, 0], 'permitted_direction': [1, 0], 'direction_cosine': -1,
            'marking': {**configured_geometry_policy(), 'direction': [1, 0],
                        'polygon': [[0, 0], [30, 0], [30, 30], [0, 30]]}}


@pytest.mark.parametrize('change', ['none', 'no_authority', 'appearance', 'no_motion', 'no_camera', 'image', 'false_cosine'])
def test_wrong_way_requires_motion_and_independent_direction(saved, change):
    review, run, _, _ = saved
    if change != 'image':
        set_video(saved)
    evidence = direction_evidence()
    if change == 'no_authority':
        evidence.pop('marking')
    elif change == 'appearance':
        evidence['marking'] = {**review_only_policy(), 'direction': [1, 0]}
    elif change == 'no_motion':
        evidence['observed_motion'] = [0, 0]
    elif change == 'false_cosine':
        evidence['observed_motion'] = [20, 0]
    write_frames(run, [frame(decision('wrong_way', evidence), camera_stable=change != 'no_camera')])
    card = review.view(run.name)['cards'][0]
    assert card['status'] == ('supported' if change == 'none' else 'insufficient_evidence')


def test_stopped_traffic_does_not_establish_red_and_zebra_is_not_obstruction(saved):
    review, run, _, _ = saved
    set_video(saved)
    write_frames(run, [frame(decision('red_light_crossing', {'signal': 'red', 'stopped_traffic': True}),
                             decision('crossing_obstruction_during_red', {'crossing_association': {'polygon': [[0, 0], [1, 1]]}}),
                             decision('triple_riding', {'rider_count': 3}))])
    view = review.view(run.name)
    assert len(view['cards']) == 2
    assert all(card['status'] == 'insufficient_evidence' for card in view['cards'])
    assert all('associated_red_signal' in [fact['id'] for fact in card['missing_facts']] for card in view['cards'])
    assert view['limits']['deferred_triple_riding_observations'] == 1


def test_repeated_unknowns_group_by_track_rule_scene_and_cap_is_explicit(saved, monkeypatch):
    review, run, _, _ = saved
    monkeypatch.setattr(module, 'MAX_CARDS', 2)
    records = [frame(decision(verdict='cannot_assess', stamp=i), stamp=i) for i in range(30)]
    records += [frame(decision(track=2)), frame(decision(track=3)), frame(decision(), scene=1)]
    write_frames(run, records)
    view = review.view(run.name)
    assert len(view['cards']) == 2
    assert view['cards'][0]['observation_count'] == 30
    assert view['limits']['truncated'] is True and view['limits']['omitted_observations'] == 2


def test_priority_rules_are_retained_when_other_rules_fill_card_limit(saved, monkeypatch):
    review, run, _, _ = saved
    monkeypatch.setattr(module, 'MAX_CARDS', 2)
    write_frames(run, [frame(decision('prohibited_turn', {}), decision('restricted_entry', {}),
                             decision('wrong_way', {}), decision())])
    view = review.view(run.name)
    assert [card['rule'] for card in view['cards']] == ['helmet_absence', 'wrong_way']
    assert view['limits']['omitted_observations'] == 2


@pytest.mark.parametrize('verdicts,selected_index', [
    (['cannot_assess', 'no_violation_observed', 'cannot_assess'], 1),
    (['no_violation_observed', 'cannot_assess'], 0),
    (['cannot_assess', 'cannot_assess'], 0),
    (['no_violation_observed', 'violation', 'cannot_assess'], 1),
])
def test_group_preserves_completed_negative_over_initial_unknown(saved, verdicts, selected_index):
    review, run, _, _ = saved
    set_video(saved)
    write_frames(run, [frame(decision('wrong_way', {}, verdict=verdict, stamp=index), stamp=index)
                       for index, verdict in enumerate(verdicts)])
    original = (run / 'detections.jsonl').read_bytes()
    card = review.view(run.name)['cards'][0]
    assert card['recorded_verdict'] == verdicts[selected_index]
    assert card['timestamp'] == selected_index
    assert card['observation_count'] == len(verdicts)
    assert (card['first_timestamp'], card['last_timestamp']) == (0, len(verdicts)-1)
    assert card['status'] == 'insufficient_evidence'
    if verdicts[selected_index] == 'no_violation_observed':
        assert card['reason'] == ('The recorded rule did not observe its positive condition. '
                                  'This is one observation, not proof that the whole clip is clear.')
        assert card['missing_facts']
    assert (run / 'detections.jsonl').read_bytes() == original


@pytest.mark.parametrize('change', ['replace', 'remove'])
def test_source_readiness_warns_without_rewriting_historical_assessments(saved, change):
    review, run, source, files = saved
    data = bound(saved)
    before = review.decide({**data, 'decision': 'reject', 'note': 'Historical reviewer note'})
    original_exports = {name: (run / name).read_bytes() for name in before['provenance']['source_hashes']}
    if change == 'replace':
        source.write_bytes(png_bytes(np.full((8, 8, 3), 255, np.uint8)))
    else:
        source.unlink()
    after = review.view(run.name)
    assert after['source_readiness']['available'] is False
    assert 'Historical saved assessments remain available' in after['source_readiness']['reason']
    assert after['fingerprint'] == before['fingerprint']
    assert after['cards'] == before['cards']
    assert after['cards'][0]['status'] == 'supported'
    changed_note = review.decide({**data, 'decision': 'reject', 'note': 'Reviewed saved output; media unavailable'})
    assert changed_note['source_readiness']['available'] is False
    assert changed_note['cards'][0]['review']['note'] == 'Reviewed saved output; media unavailable'
    exported = review.export(data)
    report = json.loads(files.paths[exported['url']].read_text())
    assert report['source_readiness'] == after['source_readiness']
    assert report['cards'][0]['status'] == 'supported'
    assert report['fingerprint'] == before['fingerprint']
    assert all((run / name).read_bytes() == payload for name, payload in original_exports.items())


@pytest.mark.parametrize('association', [None, {'state': 'red', 'associated_approach': 'different', 'timestamp': 0, 'model_score': .9},
                                      {'state': 'red', 'associated_approach': 'approach-a', 'timestamp': -1, 'model_score': .9}])
def test_red_signal_must_be_current_and_associated_with_same_approach(saved, association):
    review, run, _, _ = saved
    set_video(saved)
    evidence = {'signal': 'red', 'signal_association': association, 'approach_id': 'approach-a',
                'observed_red_entry': {'timestamp': 0}, 'stop_line': [[0, 0], [10, 0]]}
    write_frames(run, [frame(decision('stop_line_encroachment', evidence))])
    card = review.view(run.name)['cards'][0]
    assert card['status'] == 'insufficient_evidence'
    assert 'associated_red_signal' in [fact['id'] for fact in card['missing_facts']]


def test_recorded_signal_and_entry_are_still_contextual_review(saved):
    review, run, _, _ = saved
    set_video(saved)
    evidence = {'signal': 'red', 'signal_association': {'state': 'red', 'associated_approach': 'approach-a',
                'timestamp': 0, 'model_score': .9}, 'approach_id': 'approach-a',
                'observed_red_entry': {'timestamp': 0}, 'stop_line': [[0, 0], [10, 0]]}
    write_frames(run, [frame(decision('stop_line_encroachment', evidence))])
    assert review.view(run.name)['cards'][0]['status'] == 'needs_review'


def test_review_state_independent_accept_reject_reset_and_original_bytes_unchanged(saved):
    review, run, _, _ = saved
    originals = {name: (run / name).read_bytes() for name in ('detections.jsonl', 'events.jsonl', 'request.json', 'report.json')}
    args = bound(saved)
    for action in ('accept', 'reject'):
        view = review.decide({**args, 'decision': action, 'note': '<unsafe> remains plain data'})
        assert view['cards'][0]['review']['decision'] == action
        assert view['cards'][0]['status'] == 'supported'
    view = review.decide({**args, 'decision': 'reset'})
    assert view['cards'][0]['review'] == {'decision': None, 'note': ''}
    assert all((run / name).read_bytes() == raw for name, raw in originals.items())
    assert not list(run.glob('*.tmp'))


@pytest.mark.parametrize('filename', ['detections.jsonl', 'events.jsonl', 'request.json', 'report.json'])
def test_changed_source_export_invalidates_cards_review_and_stale_actions(saved, filename):
    review, run, _, _ = saved
    args = bound(saved)
    review.decide({**args, 'decision': 'accept'})
    with (run / filename).open('ab') as stream:
        stream.write(b'\n')
    fresh = review.view(run.name)
    assert fresh['fingerprint'] != args['fingerprint']
    assert fresh['cards'][0]['id'] != args['card_id']
    assert fresh['cards'][0]['review']['decision'] is None
    for action in (review.decide, review.frames, review.export):
        with pytest.raises(ValueError, match='stale'):
            action({**args, 'decision': 'reject'})


def test_image_frames_have_no_invented_temporal_context_and_are_stably_bound(saved):
    review, _, _, files = saved
    args = bound(saved)
    result = review.frames(args)
    before, during, after = result['frames']
    assert not before['available'] and 'still image' in before['reason']
    assert not after['available'] and 'still image' in after['reason']
    assert during['available'] and during['timestamp'] == 0
    assert files.paths[during['url']].is_file()
    assert during['id'] == review.view(args['id'])['cards'][0]['frame_ids'][0]
    assert review.frames(args)['frames'][1]['id'] == during['id']
    with pytest.raises(ValueError, match='exact saved review'):
        review.frames({**args, 'card_id': 'another-card'})


def test_video_frames_show_actual_decoder_timestamps_and_missing_after(saved, monkeypatch):
    review, run, _, _ = saved
    set_video(saved)
    write_frames(run, [frame(decision(stamp=1), stamp=1)])
    calls = []
    def decode(source, info, seconds):
        calls.append(seconds)
        return np.zeros((8, 8, 3), dtype=np.uint8), {.5: .52, 1: 1.02, 1.5: 1}[seconds]
    monkeypatch.setattr(module, 'preview_frame', decode)
    frames = review.frames(bound(saved))['frames']
    assert calls == [.5, 1, 1.5]
    assert frames[0]['timestamp'] == .52 and frames[1]['timestamp'] == 1.02
    assert not frames[2]['available'] and frames[2]['url'] is None


def test_media_replaced_or_revision_missing_never_extracts(saved, monkeypatch):
    review, run, source, _ = saved
    source.write_bytes(source.read_bytes() + b'changed')
    monkeypatch.setattr(module, 'load_image', lambda *args: pytest.fail('must not decode replaced source'))
    assert all(not item['available'] for item in review.frames(bound(saved))['frames'])
    request = json.loads((run / 'request.json').read_text())
    del request['info']['source_revision']
    (run / 'request.json').write_text(json.dumps(request))
    assert all('unrecorded' in item['reason'] for item in review.frames(bound(saved))['frames'])


def test_source_replaced_during_decode_cannot_produce_bound_frame(saved, monkeypatch):
    review, _, source, _ = saved
    def decode(path):
        source.write_bytes(source.read_bytes() + b'changed during decode')
        return np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(module, 'load_image', decode)
    with pytest.raises(ValueError, match='changed during frame extraction'):
        review.frames(bound(saved))


def test_saved_output_changed_during_frame_decode_is_rejected(saved, monkeypatch):
    review, run, _, _ = saved
    def decode(path):
        with (run / 'events.jsonl').open('ab') as stream:
            stream.write(b'\n')
        return np.zeros((8, 8, 3), dtype=np.uint8)
    monkeypatch.setattr(module, 'load_image', decode)
    with pytest.raises(ValueError, match='Saved evidence changed during frame extraction'):
        review.frames(bound(saved))


def test_external_source_and_run_traversal_rejected(saved, tmp_path):
    review, run, source, _ = saved
    request = json.loads((run / 'request.json').read_text())
    external = tmp_path / 'external.png'
    external.write_bytes(source.read_bytes())
    request.update(source=str(external), info={'kind': 'image', 'source_revision': source_revision(external)})
    (run / 'request.json').write_text(json.dumps(request))
    assert all('uploads or demo' in item['reason'] for item in review.frames(bound(saved))['frames'])
    for name in ('../synthetic-review', 'a/b', 'a\\b', '', None, 'a.json'):
        with pytest.raises(ValueError):
            review.view(name)


def test_export_preserves_hashes_and_separates_review_from_algorithm(saved):
    review, run, _, files = saved
    args = bound(saved)
    review.decide({**args, 'decision': 'reject', 'note': 'SYNTHETIC human review fixture'})
    result = review.export(args)
    report = json.loads(files.paths[result['url']].read_bytes())
    assert report['cards'][0]['recorded_verdict'] == 'violation'
    assert report['cards'][0]['review']['decision'] == 'reject'
    assert report['synthetic'] is True and report['external_reporting'] is False
    assert report['deferred_rules'] == ['triple_riding']
    assert report['provenance']['source_hashes']['events.jsonl'] == hashlib.sha256((run / 'events.jsonl').read_bytes()).hexdigest()
    assert str(run.parent.parent) not in json.dumps(report)
    first_export = files.paths[result['url']].read_bytes()
    review.decide({**args, 'decision': 'accept'})
    second = review.export(args)
    assert second['filename'] != result['filename']
    assert files.paths[result['url']].read_bytes() == first_export


def test_accepting_insufficient_evidence_does_not_promote_status(saved):
    review, run, _, _ = saved
    write_frames(run, [frame(decision(evidence={'helmet_count': 0}))])
    accepted = review.decide({**bound(saved), 'decision': 'accept'})['cards'][0]
    assert accepted['review']['decision'] == 'accept'
    assert accepted['status'] == 'insufficient_evidence'
    assert accepted['missing_facts']


@pytest.mark.parametrize('raw', [b'not json\n', b'[]\n', b'{"decisions":[],"timestamp":NaN}\n',
                               b'{"decisions":[],"timestamp":1e999}\n',
                               b'{"decisions":[false],"timestamp":0}\n'])
def test_malformed_json_and_nonfinite_evidence_fail_closed(saved, raw):
    review, run, _, _ = saved
    (run / 'detections.jsonl').write_bytes(raw)
    with pytest.raises(ValueError):
        review.view(run.name)


def test_bounded_line_file_and_note_limits(saved, monkeypatch):
    review, run, _, _ = saved
    args = bound(saved)
    with pytest.raises(ValueError, match='4000'):
        review.decide({**args, 'decision': 'accept', 'note': 'x' * 4001})
    monkeypatch.setattr(module, 'MAX_LINE', 10)
    with pytest.raises(ValueError, match='line or file limit'):
        review.view(run.name)
    monkeypatch.setattr(module, 'MAX_FILE', 10)
    with pytest.raises(ValueError, match='size limit'):
        review.view(run.name)


def test_running_run_and_mismatched_frame_timestamp_fail_closed(saved):
    review, run, _, _ = saved
    write_frames(run, [frame(decision(stamp=1), stamp=0)])
    with pytest.raises(ValueError, match='frame binding'):
        review.view(run.name)
    (run / 'status.json').write_text('{"state":"running"}')
    with pytest.raises(ValueError, match='completes successfully'):
        review.view(run.name)
