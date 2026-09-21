"""Source-frame identities and static-landmark registration checks, independent
of inference. python -m scripts.check_delhi_tracking runs/<Delhi-run>
"""
import csv
import hashlib
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from ivp.jobs import ROOT
from ivp.media import preview_frame,write_json
from scripts.check_city_identities import iou


def check(run):
    fixture=json.loads((ROOT/'demo/delhi-tracking-check.json').read_text(encoding='utf8'))
    request=json.loads((run/'request.json').read_text(encoding='utf8'));source=Path(request['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=fixture['source_sha256']:
        raise ValueError('This development fixture requires the unchanged credited Delhi sample.')
    if request['settings'].get('max_width',960)<540:
        raise ValueError('This check requires the original 540x960 analysis resolution.')
    times={o['time'] for o in fixture['observations']};matrices={};tracks={t:[] for t in times}
    with (run/'motion-activity.csv').open(encoding='utf8') as f:
        for row in csv.DictReader(f):
            t=float(row['timestamp_seconds'])
            if t in times:matrices[t]=np.array([float(row[f'camera_m{i}']) for i in range(6)]).reshape(2,3)
    with (run/'motion-tracks.csv').open(encoding='utf8') as f:
        for row in csv.DictReader(f):
            t=float(row['timestamp_seconds'])
            if t not in times:continue
            x,y,w,h=[float(row[k]) for k in ['bbox_x','bbox_y','bbox_width','bbox_height']]
            inv=cv2.invertAffineTransform(matrices[t])
            points=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])@inv[:,:2].T+inv[:,2]
            tracks[t].append((row,np.r_[points.min(axis=0),points.max(axis=0)]))
    observations=[];ids={}
    for obs in fixture['observations']:
        candidates=[(iou(np.array(obs['box']),box),row) for row,box in tracks[obs['time']]]
        score,row=max(candidates,key=lambda item:item[0],default=(0,{}))
        matched=score>=.5;key=row.get('track_id') if matched else None
        observations.append({**obs,'matched':matched,'iou':round(score,3),'track_id':key,
            'speed':row.get('speed') if matched else None,'speed_status':row.get('speed_status') if matched else None})
        if matched:ids.setdefault(obs['object'],set()).add(key)
    # Find the same small texture patch in human-selected source-frame windows.
    # These searches do not use the tracker's camera transform as a search prior.
    frames={t:cv2.cvtColor(preview_frame(source,request['info'],t)[0],cv2.COLOR_RGB2GRAY) for t in times|{0}}
    landmarks=[]
    for item in fixture['landmarks']:
        x,y=item['reference'];template=frames[0][y-8:y+8,x-8:x+8]
        for stamp,centre in item['search_centres'].items():
            t=int(stamp);cx,cy=centre;x0,y0=cx-25,cy-25
            search=frames[t][y0:cy+25,x0:cx+25]
            corr=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
            _,score,_,point=cv2.minMaxLoc(corr)
            measured=np.array([x0+point[0]+8,y0+point[1]+8])
            transformed=matrices[t][:,:2]@measured+matrices[t][:,2]
            landmarks.append({'landmark':item['name'],'time':t,'source_point':measured.tolist(),
                'template_correlation':round(score,4),'match_accepted':score>=.8,
                'registration_error_px':round(float(np.linalg.norm(transformed-[x,y])),3) if score>=.8 else None})
    errors=[p['registration_error_px'] for p in landmarks if p['match_accepted']]
    result={'scope':fixture['note'],'observations':observations,'matched_observations':sum(o['matched'] for o in observations),
        'total_observations':len(observations),'distinct_ids_per_object':{k:sorted(v) for k,v in ids.items()},
        'objects_with_identity_changes':sum(len(v)>1 for v in ids.values()),'landmarks':landmarks,
        'registration':{'accepted_checks':len(errors),'total_checks':len(landmarks),
            'median_error_px':float(np.median(errors)) if errors else None,'maximum_error_px':max(errors) if errors else None,
            'scope':'Integer-pixel patch matching on four stationary background landmarks. Not road-plane calibration or a speed-accuracy test; local parallax can differ.'}}
    write_json(run/'delhi-spot-check.json',result)
    return result


if __name__=='__main__':
    result=check(Path(sys.argv[1]))
    print(json.dumps({k:v for k,v in result.items() if k not in ['observations','landmarks']},indent=2))
