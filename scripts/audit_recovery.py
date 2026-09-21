"""Render observed-before/recovered source crops for the offline recovery trial."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from traffic_ai.runtime import configure_cache
from ivp.media import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--robust',action='store_true');args=parser.parse_args()
    configure_cache()
    from ultralytics.engine.results import Boxes
    from traffic_ai.recovery import RepeatedWeakRecovery
    if args.robust:
        from traffic_ai.robust_recovery import RobustObservedRecovery
        tracker_type=RobustObservedRecovery
    else:tracker_type=RepeatedWeakRecovery
    request=json.loads((args.folder/'request.json').read_text())
    size=json.loads((args.folder/'coordinates.json').read_text())['processed_size']
    matrices={float(r['timestamp_seconds']):np.array([float(r[f'camera_m{i}']) for i in range(6)]).reshape(2,3)
              for r in csv.DictReader((args.folder/'motion-activity.csv').open())}
    engine=tracker_type(SimpleNamespace(track_high_thresh=.35,track_low_thresh=.1,new_track_thresh=.4,
        track_buffer=24,match_thresh=.75,fuse_score=True))
    cap=cv2.VideoCapture(request['source']);observed={};events=[];panels=[];last=None
    def crop(stamp,box):
        cap.set(cv2.CAP_PROP_POS_MSEC,stamp*1000);ok,frame=cap.read()
        if not ok:raise RuntimeError(f'Cannot read {stamp}')
        frame=cv2.resize(frame,tuple(size))
        x,y,x2,y2=box;inv=cv2.invertAffineTransform(matrices[stamp])
        points=np.array([[x,y],[x2,y],[x2,y2],[x,y2]])@inv[:,:2].T+inv[:,2]
        cv2.polylines(frame,[points.astype(np.int32)],True,(80,255,80),2)
        low=np.maximum(0,points.min(axis=0)-40).astype(int)
        high=np.minimum(size,points.max(axis=0)+40).astype(int)
        patch=frame[low[1]:high[1],low[0]:high[0]]
        scale=min(320/patch.shape[1],240/patch.shape[0]);patch=cv2.resize(patch,None,fx=scale,fy=scale)
        tile=np.zeros((270,320,3),np.uint8);tile[30:30+patch.shape[0],:patch.shape[1]]=patch
        cv2.putText(tile,f'{stamp:.3f}s',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1)
        return tile
    for line in (args.folder/'association.jsonl').open():
        frame=json.loads(line);stamp=frame['timestamp'];gap=None if last is None else stamp-last
        if last is not None and (frame.get('tracker_frame_id')==1 or gap>.2):engine.reset();observed.clear()
        if gap is not None and gap>0:engine.max_frames_lost=max(2,min(120,round(.8/gap)))
        last=stamp
        engine.observation_timestamp=stamp
        rows=engine.update(Boxes(np.asarray(frame['detections'],np.float32).reshape(-1,6),tuple(reversed(size))))
        for event in engine.recovery_events:
            key=event['track_id'];previous=observed.get(key);current=next(r for r in rows if int(r[4])==key)
            record={**event,'timestamp':stamp,'previous_observation':previous[0] if previous else None,
                    'previous_box':previous[1].tolist() if previous else None,'recovered_box':current[:4].tolist()}
            events.append(record)
            if previous and (not args.robust or event.get('geometry')):
                pair=cv2.hconcat([crop(*previous),crop(stamp,current[:4])])
                cv2.putText(pair,f'ID {key}: last observed / recovered',(8,264),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                panels.append(pair)
        for row in rows:observed[int(row[4])]=(stamp,row[:4].copy())
    cap.release()
    prefix='robust-recovery' if args.robust else 'recovery'
    write_json(args.folder/f'{prefix}-observations.json',events)
    if panels:cv2.imwrite(str(args.folder/f'{prefix}-contact-sheet.jpg'),cv2.vconcat(panels))
    print(json.dumps({'events':len(events),'rendered_events':len(panels),'sheet':str(args.folder/f'{prefix}-contact-sheet.jpg')}))


if __name__=='__main__':main()
