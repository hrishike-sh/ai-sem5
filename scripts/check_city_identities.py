"""Evaluate a small, transparent real-video development spot check.

python -m scripts.check_city_identities runs/<completed-city-run>
Does not supply annotations to the tracker or claim full-scene accuracy.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from ivp.jobs import ROOT
from ivp.media import write_json


def iou(a,b):
    lower=np.maximum(a[:2],b[:2]);upper=np.minimum(a[2:],b[2:])
    intersection=np.prod(np.maximum(0,upper-lower))
    return float(intersection/max(1,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection))


def check(run, fixture_name='city-tracking-check.json'):
    reference=json.loads((ROOT/'demo'/fixture_name).read_text(encoding='utf8'))
    request=json.loads((run/'request.json').read_text(encoding='utf8'))
    if Path(request['source']).name!=reference['source']:
        raise ValueError('This spot check applies only to its credited source video.')
    if hashlib.sha256(Path(request['source']).read_bytes()).hexdigest()!=reference['source_sha256']:
        raise ValueError('The sample changed; the saved annotations are no longer valid.')
    width=request['info']['width'];limit=request['settings'].get('max_width',0)
    if min(width,limit or width)!=reference['coordinate_size'][0]:
        raise ValueError('Run this development check at its documented analysis resolution.')
    targets={o['time'] for o in reference['observations']}
    cameras={}
    with (run/'motion-activity.csv').open(encoding='utf8') as f:
        for row in csv.DictReader(f):
            stamp=float(row['timestamp_seconds'])
            if stamp in targets:cameras[stamp]=np.array([float(row[f'camera_m{i}']) for i in range(6)]).reshape(2,3)
    tracks={t:[] for t in targets}
    with (run/'motion-tracks.csv').open(encoding='utf8') as f:
        for row in csv.DictReader(f):
            stamp=float(row['timestamp_seconds'])
            if stamp not in targets:continue
            x,y,w,h=[float(row[name]) for name in ['bbox_x','bbox_y','bbox_width','bbox_height']]
            inverse=cv2.invertAffineTransform(cameras[stamp])
            points=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])@inverse[:,:2].T+inverse[:,2]
            box=np.r_[points.min(axis=0),points.max(axis=0)]
            tracks[stamp].append((row['track_id'],box))
    observations=[];identities={}
    for obs in reference['observations']:
        options=[(iou(np.array(obs['box']),box),key) for key,box in tracks[obs['time']]]
        score,key=max(options,default=(0,None))
        matched=score>=.5
        observations.append({**obs,'matched':matched,'iou':round(score,3),'track_id':key if matched else None})
        if matched:identities.setdefault(obs['object'],set()).add(key)
    result={'scope':reference['note'],'observations':observations,
            'matched_observations':sum(o['matched'] for o in observations),'total_observations':len(observations),
            'distinct_ids_per_object':{k:sorted(v) for k,v in identities.items()},
            'objects_with_identity_changes':sum(len(v)>1 for v in identities.values())}
    write_json(run/'identity-spot-check.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(check(Path(sys.argv[1])),indent=2))
