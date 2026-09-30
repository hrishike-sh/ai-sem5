"""Generated pixels and local fixtures only; no model or network calls."""
import base64
import hashlib
import json

import numpy as np
import pytest

from ivp.media import png_bytes
from traffic_ai.violation_preparation import EvidencePreparations
from traffic_ai.violation_images import validate_evidence_images
from tests.test_violation_review import saved, bound, write_frames, frame, decision


def test_preview_is_exact_bytes_bound_to_card_and_source(saved):
    review, run, _, files = saved
    prepared = EvidencePreparations(review)
    data = bound(saved)
    preview = prepared.prepare(data)
    assert preview['mode'] == 'images_and_facts'
    images = prepared.load({**data, 'preparation_id': preview['preparation_id']})
    card = review.view(run.name)['cards'][0]
    validate_evidence_images(images, card['frame_ids'])
    available = [f for f in preview['frames'] if f['available']]
    assert len(available) == len(images) == 1
    encoded = base64.b64decode(images[0]['data_url'].split(',')[1])
    assert files.paths[available[0]['url']].read_bytes() == encoded
    assert hashlib.sha256(encoded).hexdigest() == available[0]['sha256']
    assert available[0]['timestamp'] == 0
    assert preview['facts'] == card['facts']


def test_no_source_offers_explicit_facts_only(saved):
    review, _, source, _ = saved
    source.unlink()
    prepared = EvidencePreparations(review)
    data = bound(saved)
    result = prepared.prepare(data)
    assert result['mode'] == 'facts_only'
    assert not any(f['available'] for f in result['frames'])
    with pytest.raises((ValueError, OSError)):
        prepared.load({**data, 'preparation_id': result['preparation_id']})


@pytest.mark.parametrize('change', ['jpeg', 'manifest', 'source', 'run_evidence', 'card'])
def test_changed_preparation_is_rejected_before_model_use(saved, change):
    review, run, source, files = saved
    prepared = EvidencePreparations(review)
    data = bound(saved)
    preview = prepared.prepare(data)
    payload = {**data, 'preparation_id': preview['preparation_id']}
    if change == 'jpeg':
        path = files.paths[next(f['url'] for f in preview['frames'] if f['available'])]
        path.write_bytes(b'changed JPEG')
    elif change == 'manifest':
        path = run / ('violation-preparation-' + preview['preparation_id'] + '.json')
        body = json.loads(path.read_text()); body['images'][0]['image']['timestamp'] = 2.
        path.write_text(json.dumps(body))
    elif change == 'source':
        source.write_bytes(png_bytes(np.ones((8, 8, 3), dtype=np.uint8)*255))
    elif change == 'run_evidence':
        (run/'report.json').write_text('{"ai":{},"modified":true}')
    else:
        payload['card_id'] = 'another-card'
    with pytest.raises(ValueError):
        prepared.load(payload)


def test_only_trusted_source_geometry_allows_subject_crop(saved):
    review, run, _, _ = saved
    record = frame(decision())
    record.update(source_size=[100,100],processed_size=[50,50],
                  transform_to_source_pixel_centres=[[2,0,0],[0,2,0],[0,0,1]])
    record['decisions'][0]['box'] = [20,20,30,30]
    record['decisions'][0]['evidence']['heads'] = [[22,15,27,20]]
    write_frames(run,[record])
    prepared = EvidencePreparations(review)
    card = review.view(run.name)['cards'][0]
    crop = prepared._crop(run,card,0,(100,100,3))
    assert crop == (28,18,72,72)
    path=run/'detections.jsonl'
    path.write_bytes(b'\r\n'+path.read_bytes())
    assert prepared._crop(run,card,0,(100,100,3)) == crop
    assert prepared._crop(run,card,0,(200,200,3)) is None
    later = {**record, 'timestamp': .2, 'scene_id': 1, 'decisions': []}
    write_frames(run,[record,later])
    assert prepared._crop(run,card,.2,(100,100,3)) is None
    later['scene_id'] = 0
    write_frames(run,[record,later])
    assert prepared._crop(run,card,.2,(100,100,3)) is None
    record['scene_id'] = 1
    write_frames(run,[record])
    assert prepared._crop(run,card,0,(100,100,3)) is None
    record['scene_id'] = 0
    record.pop('transform_to_source_pixel_centres')
    write_frames(run,[record])
    assert prepared._crop(run,card,0,(100,100,3)) is None


def test_nonapplicable_helmet_observations_are_counted_not_presented_as_candidates(saved):
    review, run, _, _ = saved
    irrelevant = decision(verdict='cannot_assess')
    irrelevant['reason'] = 'This rider rule applies to motorcycles'
    write_frames(run,[frame(irrelevant,decision(track=2))])
    view = review.view(run.name)
    assert len(view['cards']) == 1 and view['cards'][0]['track_id'] == 2
    assert view['omitted_nonapplicable_count'] == 1
    assert view['limits']['recorded_decisions'] == 2
