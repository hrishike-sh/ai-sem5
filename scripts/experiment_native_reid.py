"""Offline experiment only. Writes H-drive artifacts, leaves app defaults intact.

python -m scripts.experiment_native_reid --source india-delhi-overhead.mp4
"""
import argparse
import hashlib
import json
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from ivp.jobs import ROOT,job_lock
from ivp.media import probe_video,write_json
from ivp.pipeline import Settings
from ivp.motion_worker import run_motion
from ivp.motion import MotionAnalyzer
from traffic_ai.vehicle_motion import VehicleTracker
from traffic_ai.runtime import configure_cache
from scripts.validate_tracking import verify


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='india-delhi-overhead.mp4',choices=['india-delhi-overhead.mp4','india-city.mp4'])
    args=parser.parse_args();home=configure_cache();folder=home/'experiments'/('native-reid-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    folder.mkdir(parents=True,exist_ok=False)
    tracker_config=folder/'botsort.yaml'
    tracker_config.write_text('''tracker_type: botsort
track_high_thresh: 0.35
track_low_thresh: 0.1
new_track_thresh: 0.4
track_buffer: 24
match_thresh: 0.75
fuse_score: true
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: true
model: auto
''',encoding='utf8')
    def load(self):
        import torch
        from ultralytics import YOLO
        from ultralytics.nn.modules.head import Detect
        torch.set_num_threads(2);path=home/'models/yolo11s.pt'
        if not path.is_file():raise RuntimeError('Existing local model required; no downloads in this experiment.')
        model=YOLO(str(path));head=model.model.model[-1]
        if not isinstance(head,Detect) or head.end2end:
            raise RuntimeError('Native features unavailable; refusing external ReID model fallback.')
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        class Adapter:
            max_frames_lost=24
            rows=np.empty((0,8),np.float32)
            def update(adapter,*unused):
                model.predictor.trackers[0].max_frames_lost=adapter.max_frames_lost
                return adapter.rows
            def reset(adapter):
                if getattr(model,'predictor',None) is not None:
                    for tracker in getattr(model.predictor,'trackers',[]):tracker.reset()
        adapter=Adapter()
        def predict(*a,**kw):
            result=model.track(*a,**kw,persist=True,tracker=str(tracker_config))
            boxes=result[0].boxes
            adapter.rows=np.empty((0,8),np.float32) if boxes.id is None else np.c_[boxes.xyxy.cpu().numpy(),boxes.id.cpu().numpy(),boxes.conf.cpu().numpy(),boxes.cls.cpu().numpy(),np.arange(len(boxes))]
            return result
        self.model=SimpleNamespace(predict=predict,names=model.names)
        self.association=adapter
        self.manifest={'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'device':self.device,'experimental_native_reid':True}
    VehicleTracker._load=load
    old_summary=VehicleTracker.summary
    def summary(self):
        return {**old_summary(self),'association':'EXPERIMENT: BoT-SORT with native YOLO11 feature ReID; not the app default'}
    VehicleTracker.summary=summary
    old_motion_summary=MotionAnalyzer.summary
    def motion_summary(self):
        return {**old_motion_summary(self),'method':'EXPERIMENT: local YOLO11s + BoT-SORT native-feature ReID + IVP registration and speed; not the app default'}
    MotionAnalyzer.summary=motion_summary
    source=ROOT/'demo'/args.source
    config={'axis':'horizontal','position':.7 if args.source=='india-city.mp4' else .65,
            'rectangle':[.04,.28,.94,.98] if args.source=='india-city.mp4' else [0,.4,1,.98],
            'tracking_mode':'vehicles','output_view':'tracks'}
    write_json(folder/'request.json',{'source':str(source),'info':probe_video(source),'settings':Settings(max_width=960).to_dict(),'selected_masks':[],'motion':config})
    print('EXPERIMENT:',folder,flush=True)
    with job_lock():run_motion(folder)
    result=verify(folder)
    if args.source=='india-delhi-overhead.mp4':
        from scripts.check_delhi_tracking import check
        result['delhi_check']=check(folder)
    write_json(folder/'experiment-result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='delhi_check'},indent=2),flush=True)
    if 'delhi_check' in result:
        print(json.dumps({k:result['delhi_check'][k] for k in ['matched_observations','objects_with_identity_changes','distinct_ids_per_object']},indent=2),flush=True)


if __name__=='__main__':main()
