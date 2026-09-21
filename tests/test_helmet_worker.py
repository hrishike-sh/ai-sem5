import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from ivp.media import png_bytes,source_revision
from ivp.pipeline import Settings
from ivp.helmet_worker import process_run
from tests.test_helmet_observations import candidates


@pytest.fixture
def frame(tmp_path,monkeypatch):
    source=tmp_path/'original.png';source.write_bytes(png_bytes(np.full((200,200,3),80,np.uint8)))
    run=tmp_path/'run';run.mkdir()
    request={'source':str(source),'info':{'kind':'image','source_revision':source_revision(source)},
             'settings':Settings().to_dict(),'helmet':{'media_id':'m','timestamp':0.}}
    (run/'request.json').write_text(json.dumps(request))
    monkeypatch.setattr('traffic_ai.local_models.verified_local_models',lambda:({},{}))
    class Model:
        def __init__(self,helmet):assert helmet is True;self.manifest={'stub':True}
        def detect(self,rgb,track):assert track is False;return candidates()
    monkeypatch.setitem(sys.modules,'traffic_ai.perception',SimpleNamespace(Perception=Model))
    return run,source


def test_frame_exports_and_binding(frame):
    run,source=frame;before=source.read_bytes();process_run(run)
    report=json.loads((run/'report.json').read_text());summary=json.loads((run/'helmet-observations.json').read_text())
    assert report['helmet']==summary and summary['counts']=={'helmet_present':1}
    assert summary['source_binding']['source_revision']==source_revision(source)
    assert b'helmet_present' in (run/'helmet-observations.csv').read_bytes()
    assert source.read_bytes()==before


def test_mutated_source_withholds_completion(frame,monkeypatch):
    from ivp import helmet_worker
    run,source=frame;write=helmet_worker.write_json
    def change(path,data):
        write(path,data)
        if path.name=='report.json':source.write_bytes(source.read_bytes()+b'changed')
    monkeypatch.setattr(helmet_worker,'write_json',change)
    with pytest.raises(ValueError,match='changed'):process_run(run)
    assert json.loads((run/'status.json').read_text())['state']=='failed'
    assert not (run/'report.json').exists()
    assert (run/'incomplete-report.json').exists()  # Recoverable diagnostic only.


def test_cancel_precedes_model_lookup(frame,monkeypatch):
    run,_=frame;(run/'cancel.request').touch()
    monkeypatch.setattr('traffic_ai.local_models.verified_local_models',lambda:pytest.fail('cancelled before model preflight'))
    process_run(run)
    assert json.loads((run/'status.json').read_text())['state']=='cancelled'
