"""Actual Worker/reader transport, with guarded synthetic producer and video mocks."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from ivp import ai_assessments
from ivp.media import write_json
from traffic_ai.contracts import Detection, Rule, Verdict
from tests.test_confirmation_export import (
    worker as worker, request as create_request, install_analyzer, records,
)


def test_vehicle_list_mapping_and_current_trace_survive_worker_and_inspector(worker,monkeypatch,tmp_path):
    rows=[(0.,[(Verdict.VIOLATION,7,Rule.HELMET)],50),
          (.1,[(Verdict.UNKNOWN,7,Rule.HELMET)],250)]
    instances=install_analyzer(worker,monkeypatch,rows)
    original=worker.Analyzer

    class IndexedAnalyzer(original):
        def analyze(self,*args,**kwargs):
            rgb,analysis,events=super().analyze(*args,**kwargs)
            analysis.detections.insert(0,Detection('person',.9,(0,0,10,20),99))
            self.event_vehicle_indices=[1]
            return rgb,analysis,events

    monkeypatch.setattr(worker,'Analyzer',IndexedAnalyzer)
    (tmp_path/'runs').mkdir()
    source,run=create_request(tmp_path/'runs','video')
    before=source.read_bytes()

    def video(run,frame_hook,final_hook):
        pixels=np.full((80,120,3),80,np.uint8)
        for index,(stamp,_,_) in enumerate(rows):
            frame_hook(SimpleNamespace(color_rgb=pixels,timestamp=stamp),pixels,index,0.)
        summary=final_hook()
        saved=records(run)
        summary['assessment_counts']=dict(Counter(d['rule']+'/'+d['verdict'] for r in saved for d in r['decisions']))
        write_json(run/'report.json',{'ai':summary,'synthetic':True})
        worker.update(run,'complete',progress=1,message='Synthetic transport contract')

    monkeypatch.setattr(worker,'process_video',video)
    worker.process_run(run)
    saved=records(run)
    assert saved[0]['event_confirmation']['pending'][0]['observations']==1
    boundary=saved[1]['event_confirmation']['lifecycle']['boundaries'][0]
    assert boundary['source_detection_index']==0 and boundary['discarded_states']['pending']==1
    for row in saved:
        assert row['event_vehicle_indices']==[1]
        assert row['detections'][row['event_vehicle_indices'][0]]['track_id']==7
        assert row['event_confirmation_indexing']['index_base']==0
    view=ai_assessments.saved_ai_page(tmp_path,run.name,verdict='all')
    for card,row in zip(view['decisions'],saved):
        assert card['decision']==row['decisions'][0]
        assert card['frame_context']['event_confirmation']==row['event_confirmation']
        assert card['frame_context']['event_vehicle_indices']==row['event_vehicle_indices']
    assert source.read_bytes()==before and instances[0].source_frames[1][0]['verdict']==Verdict.UNKNOWN


@pytest.mark.parametrize('failure',['nonfinite','invalid_index'])
def test_invalid_new_metadata_cannot_publish_complete_output(worker,monkeypatch,tmp_path,failure):
    install_analyzer(worker,monkeypatch,[(0.,[(Verdict.UNKNOWN,7,Rule.HELMET)],50)])
    original=worker.Analyzer

    class InvalidAnalyzer(original):
        def analyze(self,*args,**kwargs):
            result=super().analyze(*args,**kwargs)
            if failure=='nonfinite':
                self.events=SimpleNamespace(confirmation_snapshot=lambda:{'bad':float('nan')})
            else:
                self.event_vehicle_indices=[50]
            return result

    monkeypatch.setattr(worker,'Analyzer',InvalidAnalyzer)
    source,run=create_request(tmp_path,'image')
    before=source.read_bytes()
    with pytest.raises(ValueError):
        worker.process_run(run)
    assert not (run/'report.json').exists() and source.read_bytes()==before


def test_recorded_confirmation_keeps_existing_card_limit_instead_of_truncating(tmp_path,monkeypatch):
    run=tmp_path/'runs/large-trace'
    run.mkdir(parents=True)
    write_json(run/'status.json',{'state':'complete'})
    write_json(run/'report.json',{'ai':{'assessment_counts':{'helmet_absence/cannot_assess':1}}})
    import json
    (run/'detections.jsonl').write_text(json.dumps({'decisions':[{'rule':'helmet_absence','verdict':'cannot_assess'}],
        'event_confirmation':{'note':'SYNTHETIC oversized transport '+('x'*3000)}})+'\n')
    monkeypatch.setattr(ai_assessments,'MAX_DECISION_BYTES',2048)
    with pytest.raises(ValueError,match='evidence limit'):
        ai_assessments.saved_ai_page(tmp_path,run.name)
