"""Offline boundary tests: no real detector, network, worker or video launch."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ivp.media import png_bytes, source_revision
from ivp.pipeline import Settings
from ivp.zebra_worker import process_run
from traffic_ai.contracts import Detection
import web_server


def region(revision, stamp=0.):
    return {'schema_version':1,'semantic':'zebra_crossing','reviewed':True,'media_id':'m',
            'source_revision':revision,'reference_seconds':stamp,
            'points':[[.2,.3],[.8,.3],[.8,.55],[.2,.55]]}


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    source=tmp_path/'source.png'
    source.write_bytes(png_bytes(np.full((200,300,3),120,np.uint8)))
    revision=source_revision(source)
    run=tmp_path/'run';run.mkdir()
    info={'kind':'image','width':300,'height':200,'source_revision':revision}
    request={'source':str(source),'info':info,'settings':Settings().to_dict(),
             'zebra':{'region':region(revision),'timestamp':0.}}
    (run/'request.json').write_text(json.dumps(request))
    # The box bottom centre lies below the band, yet its upper body overlaps.
    detections=[Detection('car',.9,(125.,70.,180.,140.)),
                Detection('car',.9,(30.,130.,65.,170.))]
    class FakeDetector:
        def __init__(self,helmet):
            assert helmet is False
            self.manifest={'fixture':'offline detector stub, not accuracy evidence'}
        def detect(self,rgb):
            return detections
    monkeypatch.setitem(sys.modules,'traffic_ai.perception',SimpleNamespace(Perception=FakeDetector))
    home=tmp_path/'runtime';(home/'models').mkdir(parents=True)
    (home/'models'/'yolo11s.pt').write_bytes(b'offline stub marker, never loaded')
    monkeypatch.setattr('traffic_ai.runtime.runtime_root',lambda:home)
    return run,source,request,detections


def test_frame_worker_metrics_overlay_and_export(fixture):
    run,source,request,_=fixture
    before=source.read_bytes()
    process_run(run)
    report=json.loads((run/'report.json').read_text())
    summary=json.loads((run/'zebra-observations.json').read_text())
    assert report['zebra']==summary
    assert summary['source_binding']['source_revision']==source_revision(source)
    assert summary['counts']=={'observed':1,'no_overlap':1,'withheld':0}
    positive=summary['observations'][0]
    assert positive['label']=='on zebra crossing'
    assert positive['overlap_area_px2']==pytest.approx(2200)
    assert positive['vehicle_overlap_fraction']==pytest.approx(40/70)
    assert positive['legal_violation'] is False
    assert positive['eligible_for_rules'] is False
    assert b'on zebra crossing' in (run/'zebra-observations.csv').read_bytes()
    assert source.read_bytes()==before
    assert json.loads((run/'status.json').read_text())['state']=='complete'


def test_frame_worker_source_change_cannot_publish(fixture,monkeypatch):
    run,source,_,_=fixture
    from ivp import zebra_worker
    original=zebra_worker.write_json
    def write(path,data):
        original(path,data)
        if path.name=='report.json':
            source.write_bytes(source.read_bytes()+b'changed')
    monkeypatch.setattr(zebra_worker,'write_json',write)
    with pytest.raises(ValueError,match='changed'):
        process_run(run)
    assert json.loads((run/'status.json').read_text())['state']=='failed'


def test_frame_cancel_precedes_detector(fixture):
    run,_,_,_=fixture
    (run/'cancel.request').touch()
    process_run(run)
    assert json.loads((run/'status.json').read_text())['state']=='cancelled'
    assert not (run/'processed.png').exists()


def test_missing_model_never_constructs_detector(fixture,monkeypatch,tmp_path):
    run,_,_,_=fixture
    monkeypatch.setattr('traffic_ai.runtime.runtime_root',lambda:tmp_path/'absent-runtime')
    def forbidden(*a,**kw):
        pytest.fail('Missing local model must not construct a downloading detector')
    monkeypatch.setitem(sys.modules,'traffic_ai.perception',SimpleNamespace(Perception=forbidden))
    with pytest.raises(RuntimeError,match='model weights are missing'):
        process_run(run)
    assert json.loads((run/'status.json').read_text())['state']=='failed'
    assert not (run/'processed.png').exists()


def test_frame_api_handoff_and_rejects_stale_review(monkeypatch,tmp_path):
    revision='r'; path=tmp_path/'source.png'
    info={'kind':'image','source_revision':revision}
    lab=web_server.Lab.__new__(web_server.Lab)
    lab._source=lambda data:(path,info)
    lab.run=lambda name:{'id':name}
    calls=[]
    def launch(*args,**kwargs):
        calls.append((args,kwargs));return Path('fixture-run')
    monkeypatch.setattr(web_server,'start_video',launch)
    data={'media':'m','source_revision':revision,'timestamp':0.,'region':region(revision)}
    assert lab.post('/api/zebra',data)=={'id':'fixture-run'}
    assert calls[0][1]['zebra']['region']==data['region']
    data['region']['source_revision']='old'
    with pytest.raises(ValueError,match='outdated'):
        lab.post('/api/zebra',data)
    assert len(calls)==1


@pytest.mark.parametrize('bad_stamp',[True,'0',None,float('nan')])
def test_frame_api_bad_timestamp_never_launches(monkeypatch,tmp_path,bad_stamp):
    lab=web_server.Lab.__new__(web_server.Lab)
    lab._source=lambda data:(tmp_path/'unused',{'kind':'image','source_revision':'r'})
    monkeypatch.setattr(web_server,'start_video',lambda *a,**kw:pytest.fail('must not launch'))
    with pytest.raises(ValueError):
        lab.post('/api/zebra',{'media':'m','timestamp':bad_stamp,'region':region('r')})


def test_motion_worker_registered_coordinates_and_neutral_export(tmp_path,monkeypatch):
    from ivp import motion_worker
    from ivp.zebra import validate_region
    run=tmp_path/'run';run.mkdir()
    request={'source':'offline fixture','info':{'source_revision':'r','fps':10.,'duration':.3},
             'motion':{'tracking_mode':'vehicles','zebra':validate_region(region('r'),'m','r',0.)}}
    (run/'request.json').write_text(json.dumps(request))
    class Engine:
        def __init__(self,**kw):
            assert kw=={'tracking_mode':'vehicles','frame_interval':.1}
            self.stabilizer=SimpleNamespace(failed=False,matrix=np.float32([[1,0,-25],[0,1,0]]),
                valid_mask=np.full((200,300),255,np.uint8))
            self.last_rgb=np.zeros((200,300,3),np.uint8);self.heat=np.zeros((200,300));self.peak=None
            self.registration_observed=True
        def step(self,rgb,stamp,change):
            t={'box':(125,70,55,70),'model_label':'car','confidence':.9,'tracking_state':'observed',
               'centre':(152.5,105),'speed':None,'speed_unit':'px/s','speed_status':'withheld'}
            return self.last_rgb,{'timestamp_seconds':stamp,'state':'measured','moving_area_percent':1},[],[(7,t)]
        def summary(self):return {'fixture':'not a real tracking/codec test'}
        def heatmap(self):return self.last_rgb
    monkeypatch.setattr(motion_worker,'MotionAnalyzer',Engine)
    def video(run,hook,finish,**kw):
        (run/'measurements.csv').write_text('timestamp_seconds,duration_seconds\n0.2,0.1\n')
        (run/'coordinates.json').write_text('{}')
        for i in range(3):
            result=SimpleNamespace(timestamp=i*.1,color_rgb=np.zeros((200,300,3),np.uint8),
                                   transform_to_source=np.diag([2.,2.,1.]))
            hook(result,result.color_rgb,i,0.)
        summary=finish();assert summary['zebra']['episodes']==1
    monkeypatch.setattr(motion_worker,'process_video',video)
    motion_worker.run_motion(run)
    rows=[json.loads(s) for s in (run/'motion-zebra.jsonl').read_text().splitlines()]
    assert [r['status'] for r in rows]==['pending','pending','observed']
    binding=rows[-1]['source_binding']
    assert binding['current_frame_to_source_pixel_centres']==[[2.,0.,50.],[0.,2.,0.],[0.,0.,1.]]
    assert rows[-1]['label']=='on zebra crossing'
    assert rows[-1]['legal_violation'] is False
    assert json.loads((run/'motion-zebra.json').read_text())['source_revision']=='r'
