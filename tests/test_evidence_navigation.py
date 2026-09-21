"""All saved assessments remain reachable through bounded, integrity-checked pages."""
import base64
from collections import Counter
from copy import deepcopy
import json
import urllib.error
from urllib.parse import urlencode

import pytest

from ivp.media import write_json
from tests.test_evidence_bridge import evidence_source, finished, fingerprints
from tests.test_web_server import server, request


@pytest.fixture
def many_assessments(server, evidence_source):
    source, template = evidence_source
    events = []
    for index in range(45):
        event = deepcopy(template)
        event.update(event_id=f'synthetic-page-{index+1:03}', timestamp_seconds=index/10,
                     status='candidate' if index == 24 else 'observation' if index % 3 == 0 else 'withheld',
                     reason='SYNTHETIC VIEW FIXTURE: tests navigation only; no detector or measurement truth.')
        event['thumbnail'] = 'not-exported.png' if index == 24 else None
        events.append(event)
    # Retain source order even when event timestamps go backwards.
    events[22]['timestamp_seconds'] = .05
    (source/'motion-incidents.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events), encoding='utf8')
    write_json(source/'motion-incidents.json', {'event_count':len(events), 'counts':dict(Counter(e['status'] for e in events)),
                                             'events':events, 'omitted_from_index':0})
    ready = finished(server, request(server, '/api/evidence-review', {'source_run':source.name}))
    return source, events, ready


def page(server, ready, cursor=None, status='all'):
    query = {'id':ready['id'], 'status':status}
    if cursor is not None:
        query['cursor'] = cursor
    return request(server, '/api/evidence-review?'+urlencode(query))


def test_late_candidate_and_all_source_events_are_reachable_without_raw_rows(server, many_assessments):
    source, events, ready = many_assessments
    original = fingerprints(source)
    artifact = source.parent/'evidence-reviews'/ready['id']/'offline'
    artifact_before = fingerprints(artifact)
    seen = []
    current = ready
    while True:
        view = current['review']
        assert len(view['events']) <= 20
        assert current['artifact_integrity']['state'] == 'verified'
        for item in view['events']:
            assert all('samples' not in item.get(field, {}) for field in ('track_context','stop_context'))
            seen.append(item['event'])
        cursor = view['pagination']['next_cursor']
        if cursor is None:
            break
        current = page(server, ready, cursor)
    assert seen == events
    assert current['review']['pagination']['start'] == 41
    assert current['review']['pagination']['end'] == 45
    previous = page(server, ready, current['review']['pagination']['previous_cursor'])
    assert previous['review']['events'][4]['event'] == events[24]
    assert previous['review']['events'][4]['source_position'] == 25
    assert original == fingerprints(source) and artifact_before == fingerprints(artifact)


def test_status_filter_finds_late_candidate_with_citations_and_missing_thumbnail_disclosure(server, many_assessments):
    _, events, ready = many_assessments
    filtered = page(server, ready, status='candidate')['review']
    assert filtered['pagination']['filtered_total'] == 1
    assert filtered['pagination']['start'] == filtered['pagination']['end'] == 1
    assert filtered['events'][0]['event'] == events[24]
    card = filtered['events'][0]
    assert card['source_position'] == 25
    assert card['facts'][0]['citations']
    assert not card.get('thumbnail_url')
    assert 'thumbnail' in card['thumbnail_notice'].lower()
    withheld = page(server, ready, status='withheld')['review']
    assert withheld['pagination']['filtered_total'] == sum(e['status']=='withheld' for e in events)
    assert all(c['event']['status']=='withheld' for c in withheld['events'])


@pytest.mark.parametrize('cursor,status', [('broken!', 'all'), ('e30', 'all'), (None, 'speeding')])
def test_invalid_navigation_is_rejected(server, many_assessments, cursor, status):
    _, _, ready = many_assessments
    with pytest.raises(urllib.error.HTTPError) as error:
        page(server, ready, cursor, status)
    assert error.value.code == 400


def test_cursor_is_bound_to_filter_artifact_and_valid_page_boundary(server, many_assessments):
    _, _, ready = many_assessments
    pagination = ready['review']['pagination']
    cursor = pagination['next_cursor']
    with pytest.raises(urllib.error.HTTPError):
        page(server, ready, cursor, 'candidate')
    for offset, fingerprint in [(21, pagination['artifact_sha256']), (10000, pagination['artifact_sha256']), (20, '0'*64)]:
        forged = base64.urlsafe_b64encode(json.dumps([offset,'all',fingerprint]).encode()).decode().rstrip('=')
        with pytest.raises(urllib.error.HTTPError) as error:
            page(server, ready, forged)
        assert error.value.code == 400


def test_late_page_suppresses_tampered_artifact_without_rewriting_sources(server, many_assessments):
    source, _, ready = many_assessments
    original = fingerprints(source)
    cursor = ready['review']['pagination']['next_cursor']
    artifact = source.parent/'evidence-reviews'/ready['id']/'offline'
    with (artifact/'review.json').open('ab') as stream:
        stream.write(b'\n')
    response = page(server, ready, cursor)
    assert response['artifact_integrity']['state'] == 'changed'
    assert not response['downloads'] and 'review' not in response
    assert original == fingerprints(source)


def test_json_changed_between_completion_check_and_read_never_becomes_a_page(server, many_assessments, monkeypatch):
    source, _, ready = many_assessments
    service = server.lab.evidence_reviews
    read = service.read
    edited = False
    def changed(path, *args, **kwargs):
        nonlocal edited
        if path.name == 'review.json' and not edited:
            edited = True
            with path.open('ab') as stream:
                stream.write(b'\n')
        return read(path, *args, **kwargs)
    monkeypatch.setattr(service, 'read', changed)
    response = page(server, ready, ready['review']['pagination']['next_cursor'])
    assert response['artifact_integrity']['state'] == 'changed'
    assert not response['downloads'] and 'review' not in response
