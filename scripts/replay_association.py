"""Offline geometry experiment over recorded detections; never changes app defaults."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from traffic_ai.runtime import configure_cache
from ivp.jobs import ROOT
from ivp.media import write_json
from scripts.check_city_identities import iou


from traffic_ai.proposal_geometry import grouped_proposals


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    parser.add_argument('--weak-recovery',action='store_true')
    parser.add_argument('--robust-recovery',action='store_true');args=parser.parse_args()
    configure_cache()
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.byte_tracker import BYTETracker
    from traffic_ai.association import GroundAnchorTracker
    frames=[json.loads(line) for line in (args.folder/'association.jsonl').read_text().splitlines()]
    request=json.loads((args.folder/'request.json').read_text())
    fixture_name={'india-city.mp4':'city-tracking-check.json','india-delhi-overhead.mp4':'delhi-tracking-check.json',
                  'india-intersection.mp4':'intersection-tracking-check.json',
                  'india-rickshaws.mp4':'rickshaw-tracking-check.json'}[Path(request['source']).name]
    fixture=json.loads((ROOT/'demo'/fixture_name).read_text())
    matrices={}
    with (args.folder/'motion-activity.csv').open() as f:
        for row in csv.DictReader(f):
            matrices[float(row['timestamp_seconds'])]=np.array([float(row[f'camera_m{i}']) for i in range(6)]).reshape(2,3)
    summaries={}
    variants=['baseline','grouped_proposals','ground_anchor']
    if args.weak_recovery or args.robust_recovery:variants.append('repeated_weak_recovery')
    if args.robust_recovery:variants.append('robust_observed_recovery')
    for variant in variants:
        if variant=='robust_observed_recovery':
            from traffic_ai.robust_recovery import RobustObservedRecovery
            tracker_type=RobustObservedRecovery
        elif variant=='repeated_weak_recovery':
            from traffic_ai.recovery import RepeatedWeakRecovery
            tracker_type=RepeatedWeakRecovery
        else:tracker_type=GroundAnchorTracker if variant=='ground_anchor' else BYTETracker
        tracker=tracker_type(SimpleNamespace(track_high_thresh=.35,track_low_thresh=.1,new_track_thresh=.4,track_buffer=24,match_thresh=.75,fuse_score=True))
        captured={};rejected=0;exact=True;last_stamp=None;recovery_events=[]
        for frame in frames:
            gap=None if last_stamp is None else frame['timestamp']-last_stamp
            if last_stamp is not None and (frame.get('tracker_frame_id')==1 or gap>.2):tracker.reset()
            if gap is not None and gap>0:tracker.max_frames_lost=max(2,min(120,round(.8/gap)))
            last_stamp=frame['timestamp']
            data=np.asarray(frame['detections'],np.float32).reshape(-1,6)
            rejected_indices=grouped_proposals(data) if variant=='grouped_proposals' else set()
            rejected+=len(rejected_indices)
            if rejected_indices:data=data[[i for i in range(len(data)) if i not in rejected_indices]]
            tracker.observation_timestamp=frame['timestamp']
            rows=tracker.update(Boxes(data,tuple(reversed(fixture['coordinate_size']))))
            recovery_events.extend(getattr(tracker,'recovery_events',[]))
            exact &= np.array_equal(rows,np.asarray(frame['rows'],np.float32))
            captured[frame['timestamp']]=rows
        checks=[];identities={}
        for obs in fixture['observations']:
            inv=cv2.invertAffineTransform(matrices[obs['time']]);candidates=[]
            for row in captured[obs['time']]:
                x,y,x2,y2=row[:4]
                points=np.array([[x,y],[x2,y],[x2,y2],[x,y2]])@inv[:,:2].T+inv[:,2]
                score=iou(np.array(obs['box']),np.r_[points.min(axis=0),points.max(axis=0)])
                candidates.append((score,int(row[4])))
            score,key=max(candidates,default=(0,None));matched=score>=.5
            checks.append({**obs,'matched':matched,'id':key if matched else None,'iou':float(score)})
            if matched:identities.setdefault(obs['object'],set()).add(key)
        summaries[variant]={'matched':sum(x['matched'] for x in checks),'identity_changes':sum(len(ids)>1 for ids in identities.values()),
                            'ids':{k:sorted(v) for k,v in identities.items()},'checks':checks,'suppressed':rejected,
                            'recorded_output_exact':exact,'recovery_events':recovery_events}
    write_json(args.folder/('replay-robust-recovery.json' if args.robust_recovery else 'replay-weak-recovery.json' if args.weak_recovery else 'replay-results.json'),summaries)
    print(json.dumps({k:{n:v for n,v in result.items() if n!='checks'} for k,result in summaries.items()},indent=2))


if __name__=='__main__':main()
