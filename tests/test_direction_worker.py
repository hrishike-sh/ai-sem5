import json
from types import SimpleNamespace
import numpy as np
from ivp import motion_worker
from tests.test_configured_direction import config


def test_direction_worker_stream_summary_mapping_and_archive(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    (run/'request.json').write_text(json.dumps({'source':'offline fixture',
        'info':{'source_revision':'r','fps':10.,'duration':1.2},
        'motion':{'tracking_mode':'vehicles','direction':config()}}))
    class Engine:
        def __init__(self,**kw):
            assert kw=={'tracking_mode':'vehicles','frame_interval':.1}
            self.registration_observed=True;self.stabilizer=SimpleNamespace(failed=False,
                matrix=np.float32([[1,0,-20],[0,1,0]]),valid_mask=np.full((200,200),255,np.uint8))
            self.last_rgb=np.zeros((200,200,3),np.uint8);self.heat=np.zeros((200,200));self.peak=None
        def step(self,rgb,stamp,change):
            y=50+stamp*50
            track={'box':(90,y-20,20,20),'model_label':'car','confidence':.9,'tracking_state':'observed',
                   'centre':(100,y-10),'speed':None,'speed_unit':'px/s','speed_status':'withheld'}
            return rgb,{'timestamp_seconds':stamp,'state':'measured','moving_area_percent':1.},[],[(7,track)]
        def summary(self):return {'fixture':'offline worker hook, not actual tracking or codec evidence'}
        def heatmap(self):return self.last_rgb
    def video(run,frame,finish,**kw):
        (run/'measurements.csv').write_text('timestamp_seconds,duration_seconds\n1.1,0.1\n')
        (run/'coordinates.json').write_text('{}')
        for i in range(12):
            r=SimpleNamespace(color_rgb=np.zeros((200,200,3),np.uint8),timestamp=i*.1,
                              transform_to_source=np.diag([2.,2.,1.]))
            frame(r,r.color_rgb,i,0.)
        summary=finish();assert summary['direction']['episodes']==1
    monkeypatch.setattr(motion_worker,'MotionAnalyzer',Engine);monkeypatch.setattr(motion_worker,'process_video',video)
    motion_worker.run_motion(run)
    rows=[json.loads(s) for s in (run/'motion-direction.jsonl').read_text().splitlines()]
    assert len(rows)==12 and rows[-1]['label']=='opposite configured direction'
    assert rows[-1]['source_binding']['current_frame_to_source_pixel_centres']==[[2.,0.,40.],[0.,2.,0.],[0.,0.,1.]]
    assert json.loads((run/'motion-direction.json').read_text())['episodes']==1
    import zipfile
    with zipfile.ZipFile(run/'motion-evidence.zip') as z:
        for name in ['motion-direction.json','motion-direction.jsonl']:assert z.read(name)==(run/name).read_bytes()
