import copy
import numpy as np
import pytest
from traffic_ai import vehicle_types as m


def subject(id='V1',kind='car',bbox=None):
    return {'id':id,'kind':kind,'bbox':bbox or [10,20,90,100],'origin':'detector','evidence_ids':['context','crop-'+id]}


def detection(class_id=0,score=.8,box=None):
    return {'class_id':class_id,'label':m.CLASSES[str(class_id)],'model_score':score,'box':box or [10,20,90,100]}


def build(subjects=None,detections=None):
    return m.build_evidence(np.zeros((200,200,3),np.uint8),[subject()] if subjects is None else subjects,200,200,detections=[] if detections is None else detections)


@pytest.mark.parametrize('predictions,status', [([], 'not_observed'),([detection()], 'candidate'),
    ([detection(score=.49)],'low_confidence'),([detection(score=.5)],'candidate'),
    ([detection(),detection(1)],'unclear')])
def test_candidate_scores_status_and_counts(predictions,status):
    evidence=build(detections=predictions)
    assert evidence['subjects'][0]['status']==status
    assert evidence['candidate_counts']['total']==sum(p['model_score']>=.5 for p in predictions)
    assert 'not calibrated' in evidence['score_notice']
    m.validate_vehicle_evidence(evidence,[subject()])


def test_ambiguous_numbered_vehicle_overlap_is_unassigned():
    subjects=[subject(),subject('V2',bbox=[12,22,92,102])]
    evidence=build(subjects,[detection()])
    assert all(row['status']=='not_observed' for row in evidence['subjects'])
    assert evidence['unassigned_detections'][0]['reason']=='ambiguous_vehicle_overlap'
    assert evidence['candidate_counts']['total']==1
    m.validate_vehicle_evidence(evidence,subjects)


def test_person_is_not_vehicle_and_unmatched_proposals_are_counted_separately():
    subjects=[subject('P1','person')]
    evidence=build(subjects,[detection(6)])
    assert evidence['subjects'][0]['status']=='not_applicable'
    assert evidence['unassigned_detections'][0]['reason']=='no_numbered_vehicle_match'
    assert evidence['candidate_counts']['by_type']['Three-wheeler']==1
    m.validate_vehicle_evidence(evidence,subjects)


def test_bound_original_coordinates_scaled_before_matching():
    subjects=[subject()]
    evidence=m.build_evidence(np.zeros((400,400,3),np.uint8),subjects,200,200,detections=[detection(box=[20,40,180,200])],truncated=True)
    assert evidence['subjects'][0]['candidates'][0]['box']==[20.,40.,180.,200.]
    assert evidence['subjects'][0]['candidates'][0]['evidence_id']=='crop-V1'


def test_unavailable_preserves_independent_feature_semantics(monkeypatch):
    def fail(rgb):raise RuntimeError('private path')
    monkeypatch.setattr(m,'predict_types',fail)
    subjects=[subject(),subject('P1','person')]
    evidence=m.build_evidence(np.zeros((200,200,3),np.uint8),subjects,200,200)
    assert evidence['state']=='unavailable' and evidence['subjects'][0]['status']=='unavailable'
    assert evidence['subjects'][1]['status']=='not_applicable'
    assert 'private path' not in str(evidence)
    m.validate_vehicle_evidence(evidence,subjects)


@pytest.mark.parametrize('change',['class','label','score','box','citation','status','count','metadata'])
def test_saved_tampering_rejected(change):
    evidence=copy.deepcopy(build(detections=[detection()]))
    row=evidence['subjects'][0];d=row['candidates'][0]
    if change=='class':d['class_id']=14
    elif change=='label':d['label']='Other guessed type'
    elif change=='score':d['model_score']=float('nan')
    elif change=='box':d['box'][0]=-1
    elif change=='citation':d['evidence_id']='wrong'
    elif change=='status':row['status']='not_observed'
    elif change=='count':evidence['candidate_counts']['total']=20
    else:evidence['model']['sha256']='0'*64
    with pytest.raises(ValueError):m.validate_vehicle_evidence(evidence,[subject()])


def test_invalid_detector_coordinates_or_class_discarded():
    bad=detection();bad['class_id']=100
    evidence=build(detections=[detection(box=[0,0,201,100]),detection(score=float('nan')),bad])
    assert evidence['candidate_counts']['total']==0
    assert evidence['subjects'][0]['status']=='not_observed'


def test_missing_weights_fail_before_model_import(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'configure_cache',lambda:None)
    monkeypatch.setattr(m,'model_path',lambda key,default:tmp_path/'missing.pt')
    with pytest.raises(RuntimeError,match='not available locally'):
        m.predict_types(np.zeros((20,20,3),np.uint8))


def test_pin_mismatch_rejected_before_model_construction(tmp_path,monkeypatch):
    path=tmp_path/'wrong.pt';path.write_bytes(b'bad')
    monkeypatch.setattr(m,'configure_cache',lambda:None)
    monkeypatch.setattr(m,'model_path',lambda key,default:path)
    monkeypatch.setattr(m,'BYTES',3)
    with pytest.raises(RuntimeError,match='do not match'):
        m.predict_types(np.zeros((20,20,3),np.uint8))
