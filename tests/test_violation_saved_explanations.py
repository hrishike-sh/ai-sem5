"""Saved explanation fixtures are generated locally; no live model requests."""
import json

import numpy as np
import pytest

from traffic_ai.violation_llm import card_fingerprint, VISUAL_OBSERVATIONS
from traffic_ai.violation_images import prepare_evidence_image, image_descriptors, image_fingerprint
from tests.test_violation_review import saved, bound


def explanation(card, images=False):
    result = {'state':'explained','provider':'chatgpt_subscription','model':'fixture-model',
              'card_id':card['id'],'card_fingerprint':card_fingerprint(card),'rule':card['rule'],'status':card['status'],
              'fact_ids':[f['id'] for f in card['facts']], 'missing_fact_ids':[f['id'] for f in card['missing_facts']],
              'frame_ids':card['frame_ids'],'vision_mode':'facts_only','visual_observations':[], 'images_sent':[],
              'explanation':'Untrusted replacement prose must never survive normalization.'}
    if images:
        image = prepare_evidence_image(np.zeros((8,8,3),dtype=np.uint8),frame_id=card['frame_ids'][0],timestamp=0.)
        result.update(vision_mode='selected_frames',images_sent=image_descriptors([image]),image_fingerprint=image_fingerprint([image]),
            visual_observations=[{'frame_id':image['id'],'category':category,'observation':next(iter(options)),
                                  'text':'Untrusted replacement observation text'} for category,options in VISUAL_OBSERVATIONS.items()])
    return result


@pytest.mark.parametrize('images',[False,True])
def test_saved_explanation_survives_review_decision_and_export(saved,images):
    review,run,_,files=saved
    data=bound(saved)
    original=review.view(run.name)['cards'][0]
    result=review.save_explanation(data,explanation(original,images))
    assert result['saved'] is True
    assert 'Untrusted' not in json.dumps(result)
    view=review.decide({**data,'decision':'reject','note':'Synthetic review disposition'})
    saved_result=view['explanations'][original['id']]
    assert saved_result['status']==original['status']
    assert view['cards'][0]['review']['decision']=='reject'
    assert saved_result['vision_mode']==('selected_frames' if images else 'facts_only')
    export=review.export(data)
    contents=json.loads(files.paths[export['url']].read_text())
    assert contents['explanations'][original['id']]['saved'] is True
    assert 'data_url' not in json.dumps(contents)


def test_saved_explanation_discarded_after_original_evidence_change(saved):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    review.save_explanation(data,explanation(card))
    (run/'report.json').write_text('{"ai":{},"changed":true}')
    assert review.view(run.name)['explanations']=={}


def test_invented_visual_claim_invalidates_only_saved_explanation(saved):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    review.save_explanation(data,explanation(card,True))
    path=run/'violation-explanations.json'
    content=json.loads(path.read_text())
    content['results'][card['id']]['visual_observations'][0]['observation']='guilty_speeding'
    path.write_text(json.dumps(content))
    view=review.view(run.name)
    assert view['explanations']=={} and view['cards'][0]['status']==card['status']


def test_cached_selection_cannot_invent_evidence(saved):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    value=explanation(card)
    value['fact_ids']=['invented-speed']
    with pytest.raises(ValueError): review.save_explanation(data,value)


@pytest.mark.parametrize('key',['local_path','raw_base64','token'])
def test_saved_image_descriptor_rejects_unexpected_fields(saved,key):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    value=explanation(card,True)
    value['images_sent'][0][key]='synthetic unexpected value'
    value['image_fingerprint']=image_fingerprint(value['images_sent'])
    with pytest.raises(ValueError): review.save_explanation(data,value)


def test_wrong_card_fingerprint_rejected_before_save(saved):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    value=explanation(card); value['card_fingerprint']='wrong-fingerprint'
    with pytest.raises(ValueError): review.save_explanation(data,value)


def test_explanation_storage_limit_preserves_previous_cache(saved,monkeypatch):
    import traffic_ai.violation_review as module
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    review.save_explanation(data,explanation(card))
    path=run/'violation-explanations.json'; previous=path.read_bytes()
    monkeypatch.setattr(module,'MAX_JSON',16)
    with pytest.raises(ValueError,match='storage limit'): review.save_explanation(data,explanation(card))
    assert path.read_bytes()==previous


@pytest.mark.parametrize('value',[{'unexpected':'private data'},'not a date','2026-10-21T19:00:00'])
def test_cached_timestamp_must_be_bounded_timezone_aware_iso(saved,value):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    review.save_explanation(data,explanation(card))
    path=run/'violation-explanations.json'; content=json.loads(path.read_text())
    content['results'][card['id']]['saved_at']=value
    path.write_text(json.dumps(content))
    assert review.view(run.name)['explanations']=={}


def test_cache_writer_uses_exact_measured_utf8_bytes(saved):
    review,run,_,_=saved
    data=bound(saved); card=review.view(run.name)['cards'][0]
    review.save_explanation(data,explanation(card))
    payload=(run/'violation-explanations.json').read_bytes()
    assert b'\r\n' not in payload
    assert len(payload)==len(json.dumps(json.loads(payload),indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8'))
