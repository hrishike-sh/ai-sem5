"""Synthetic predictions test aggregation; no weights, downloads or hosted inference."""
import copy
import numpy as np
import pytest
from traffic_ai import subject_models as m


def subject(id='P1', kind='person', bbox=None, origin='detector'):
    return {'id':id,'kind':kind,'bbox':bbox or [10,10,90,190], 'origin':origin,
            'evidence_ids':['context','crop-'+id]}


def head(label='no_helmet', score=.8, box=None):
    return {'label':label,'model_score':score,'box':box or [30,15,60,45]}


def evidence(subjects=None, detections=None, **kwargs):
    return m.build_evidence(np.zeros((200,200,3),dtype=np.uint8), subjects or [subject()],200,200,
                            detections=[] if detections is None else detections, **kwargs)


@pytest.mark.parametrize('detections,status', [([], 'not_observed'),([head()], 'no_helmet'),
    ([head('helmet')], 'helmet'), ([head(score=.499)], 'low_confidence'),
    ([head(score=.5)], 'no_helmet'), ([head(),head('helmet',.11)], 'unclear')])
def test_status_scores_and_conflicts(detections,status):
    value=evidence(detections=detections)
    assert value['subjects'][0]['helmet']['status']==status
    assert value['subjects'][0]['helmet']['detections']==[{**d,'evidence_id':'crop-P1'} for d in detections]
    assert 'not calibrated' in value['score_notice']
    assert 'Heuristic' in value['thresholds']['notice']
    assert m.validate_local_evidence(value,[subject()])==value


def test_ambiguous_overlapping_people_are_unassigned():
    subjects=[subject(),subject('P2',bbox=[15,10,95,190])]
    value=evidence(subjects,[head()])
    assert all(s['helmet']['status']=='not_observed' for s in value['subjects'])
    assert value['unassigned_detections'][0]['reason']=='ambiguous_person_overlap'


@pytest.mark.parametrize('box',[[30,120,60,150],[0,0,190,190],[150,10,180,40]])
def test_lower_body_oversize_or_outside_head_not_assigned(box):
    value=evidence(detections=[head(box=box)])
    assert value['subjects'][0]['helmet']['status']=='not_observed'
    assert value['unassigned_detections']


def test_original_resolution_scaling_and_geometric_motorcycle_candidates():
    subjects=[subject(),subject('V1','motorcycle',[5,100,150,195]),subject('V2','car',[20,80,160,195],'manual')]
    value=m.build_evidence(np.zeros((400,400,3),dtype=np.uint8),subjects,200,200,detections=[head(box=[60,30,120,90])],truncated=True)
    assert value['subjects'][0]['helmet']['status']=='no_helmet'
    assert value['subjects'][0]['vehicle_candidates']==[{'id':'V1','relationship':'geometric_unconfirmed'}]
    assert value['subjects'][1]['helmet']['status']=='not_applicable'
    assert value['vehicle_counts']['by_kind']['car']=={'total':1,'detector':0,'manual':1}
    assert value['vehicle_counts']['by_kind']['motorcycle']=={'total':1,'detector':1,'manual':0}
    assert value['vehicle_counts']['total']==2 and value['vehicle_counts']['truncated']
    m.validate_local_evidence(value,subjects)


def test_unavailable_is_explicit_and_vehicle_only_skips_inference(monkeypatch):
    def fail(rgb): raise RuntimeError('sensitive local path')
    monkeypatch.setattr(m,'predict_heads',fail)
    value=m.build_evidence(np.zeros((200,200,3),dtype=np.uint8),[subject()],200,200)
    assert value['state']=='unavailable' and 'sensitive' not in str(value)
    assert value['subjects'][0]['helmet']['status']=='unavailable'
    m.validate_local_evidence(value,[subject()])
    value=m.build_evidence(np.zeros((200,200,3),dtype=np.uint8),[subject('V1','car')],200,200)
    assert value['state']=='complete'
    m.validate_local_evidence(value,[subject('V1','car')])


@pytest.mark.parametrize('tamper',['score','status','citation','candidate','metadata','reason'])
def test_saved_evidence_rejects_tampering(tamper):
    value=copy.deepcopy(evidence(detections=[head()]))
    h=value['subjects'][0]['helmet']
    if tamper=='score': h['detections'][0]['model_score']=float('nan')
    elif tamper=='status': h['status']='helmet'
    elif tamper=='citation': h['detections'][0]['evidence_id']='wrong'
    elif tamper=='candidate': value['subjects'][0]['vehicle_candidates']=[{'id':'V42','relationship':'geometric_unconfirmed'}]
    elif tamper=='metadata': value['model']['sha256']='0'*64
    else: value['reason']='untrusted prose'
    with pytest.raises(ValueError):m.validate_local_evidence(value,[subject()])


def test_merged_person_distinct_heads_abstain_and_validator_derives_reason():
    subjects=[subject(bbox=[40,0,190,190])]
    value=evidence(subjects,[head(box=[64,10,126,55]),head(box=[114,12,158,52])])
    helmet=value['subjects'][0]['helmet']
    assert helmet['status']=='unclear'
    assert helmet['reason']=='Multiple distinct head candidates within one person box; correct/split boxes.'
    assert len(helmet['detections'])==2
    m.validate_local_evidence(value,subjects)
    helmet['status']='no_helmet'
    with pytest.raises(ValueError):m.validate_local_evidence(value,subjects)


def test_overlapping_duplicate_heads_remain_one_candidate():
    subjects=[subject(bbox=[0,0,150,190])]
    detections=[head(box=[15,10,75,70]),head(score=.54,box=[50,14,80,65]),head(score=.53,box=[30,15,80,65])]
    value=evidence(subjects,detections)
    assert value['subjects'][0]['helmet']['status']=='no_helmet'
    assert len(value['subjects'][0]['helmet']['detections'])==3
    m.validate_local_evidence(value,subjects)


def test_broad_bridge_prediction_cannot_merge_distinct_heads():
    detections=[head(box=[20,10,120,60]),head(score=.7,box=[20,10,60,60]),head(score=.6,box=[80,10,120,60])]
    assert m._status(detections)[0]=='unclear'
