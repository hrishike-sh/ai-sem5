import copy
import numpy as np
import pytest
from traffic_ai import plate_models as m


def subject(id='V1',kind='car',bbox=None):
    return {'id':id,'kind':kind,'bbox':bbox or [10,20,190,180],'evidence_ids':['context','crop-'+id]}


def detection(**changes):
    return {'box':[50,100,100,120],'detector_score':.9,'ocr_score':.95,'text':'MH12 AB1234',**changes}


def build(subjects=None,detections=None):
    return m.build_evidence(np.zeros((200,200,3),np.uint8),[subject()] if subjects is None else subjects,200,200,detections=[] if detections is None else detections)


@pytest.mark.parametrize('values,status', [([], 'not_observed'),([detection()],'candidate'),
    ([detection(text='',ocr_score=None)],'unreadable'),([detection(ocr_score=.79)],'unclear'),
    ([detection(text='A')],'unclear'),([detection(),detection()],'unclear')])
def test_status_and_roundtrip(values,status):
    result=build(detections=values)
    assert result['subjects'][0]['status']==status
    assert m.validate_plate_evidence(result,[subject()])==result


def test_unique_containment_and_original_coordinate_scaling():
    subjects=[subject(bbox=[10,10,60,90]),subject('V2',bbox=[70,10,95,90])]
    evidence=m.build_evidence(np.zeros((200,200,3),np.uint8),subjects,100,100,detections=[detection()])
    assert len(evidence['subjects'][0]['candidates'])==1
    m.validate_plate_evidence(evidence,subjects)
    subjects[1]['bbox']=[10,10,60,90]
    with pytest.raises(ValueError): m.validate_plate_evidence(evidence,subjects)


def test_ambiguous_containment_is_not_assigned_and_person_not_applicable():
    subjects=[subject(),subject('V2'),subject('P1','person')]
    result=build(subjects,[detection()])
    assert result['unassigned_detections'][0]['reason']=='ambiguous_vehicle_overlap'
    assert result['subjects'][2]['status']=='not_applicable'
    m.validate_plate_evidence(result,subjects)


@pytest.mark.parametrize('change',[{'text':'<script>'},{'text':'A'*17},{'ocr_score':float('nan')},
    {'ocr_score':None},{'detector_score':True},{'box':[0,0,201,20]},{'box':[0,0,0,2]}])
def test_bad_detector_output_is_rejected(change):
    result=build(detections=[detection(**change)])
    assert not result['subjects'][0]['candidates']


def test_manual_correction_keeps_raw_candidate_and_requires_specific_plate():
    result=build(detections=[detection(),detection(text='MH12 AB1235')])
    with pytest.raises(ValueError): m.correct_plate_evidence(result,[subject()],'V1','MH12 AB1236')
    corrected=m.correct_plate_evidence(result,[subject()],'V1','MH12 AB1236',plate_id='plate-2')
    assert corrected['subjects'][0]['candidates']==result['subjects'][0]['candidates']
    assert corrected['subjects'][0]['manual_correction']=={'text':'MH12 AB1236','source':'manual','plate_id':'plate-2'}
    assert result['subjects'][0]['manual_correction'] is None
    with pytest.raises(ValueError): m.correct_plate_evidence(result,[subject()],'V2','ABC123',plate_id='plate-2')


def test_unavailable_without_downloader(monkeypatch):
    def missing(_): raise RuntimeError('missing weights')
    monkeypatch.setattr(m,'predict_plates',missing)
    result=m.build_evidence(np.zeros((20,20,3),np.uint8),[subject()],20,20)
    assert result['state']=='unavailable' and result['subjects'][0]['status']=='unavailable'
    m.validate_plate_evidence(result,[subject()])


def test_bounds_duplicates_and_tampered_confidence_rejected():
    original=build(detections=[detection(),detection()])
    for field,value in [('plate_id','plate-1'),('ocr_score',float('nan')),('box',[0,0,500,500])]:
        modified=copy.deepcopy(original); modified['subjects'][0]['candidates'][1][field]=value
        with pytest.raises(ValueError): m.validate_plate_evidence(modified,[subject()])
    assert len(build(detections=[detection()]*30)['subjects'][0]['candidates'])==20


def test_missing_ocr_field_and_malformed_detector_batch():
    invalid=detection(text=''); invalid.pop('ocr_score')
    assert build(detections=[invalid])['subjects'][0]['status']=='not_observed'
    assert build(detections={})['state']=='unavailable'


def test_correction_without_candidate_or_on_person_rejected():
    for subjects in ([subject()],[subject('P1','person')]):
        with pytest.raises(ValueError):
            m.correct_plate_evidence(build(subjects),subjects,subjects[0]['id'],'MH12 AB1234')


def test_hash_pin_mismatch_does_not_load(tmp_path):
    path=tmp_path/'detector.pt'; path.write_bytes(b'wrong model')
    with pytest.raises(RuntimeError): m._check_file(path,m.DETECTOR_SHA256)
