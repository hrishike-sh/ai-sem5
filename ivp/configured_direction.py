"""Explicit source-bound direction review through the existing RuleEngine."""
import hashlib
import json
import math
from collections import Counter
import cv2
import numpy as np
from .marking_policy import configured_geometry_policy
from traffic_ai.arrows import apply_configured_direction
from traffic_ai.contracts import Detection,Rule,Verdict
from traffic_ai.geometry import point_in_polygon
from traffic_ai.rules import RuleEngine,SceneEvidence

NOTICE=('Tracked movement compared with a user-configured direction, using the existing direction evaluator. '
        'The arrow is an explicit assumption, not automatic recognition of a legal road direction. '
        'Opposite movement is a review candidate, never a legal finding.')


def validate_config(value,media_id,revision):
    if not isinstance(value,dict) or set(value)!={'schema_version','type','reviewed','media_id','source_revision','reference_seconds','points','arrow'}:
        raise ValueError('Mark and review a direction region and arrow on the first frame.')
    if (type(value['schema_version']) is not int or value['schema_version']!=1 or value['type']!='configured_direction'
            or value['reviewed'] is not True or value['media_id']!=media_id or value['source_revision']!=revision
            or isinstance(value['reference_seconds'],bool) or value['reference_seconds']!=0):
        raise ValueError('Direction review belongs to different footage or a different reference frame.')
    for key,length in [('points',4),('arrow',2)]:
        pts=value[key]
        if (not isinstance(pts,list) or len(pts)!=length or any(not isinstance(p,list) or len(p)!=2 for p in pts)
            or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<1 for p in pts for v in p)):
            raise ValueError('Direction geometry needs finite normalized interior points.')
    polygon=np.asarray(value['points'],np.float32)
    if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon)<.005:
        raise ValueError('Direction region must be convex and cover useful road area.')
    a,b=value['arrow']
    if math.dist(a,b)<.05 or not all(point_in_polygon(p,value['points']) for p in (a,b)):
        raise ValueError('Draw a clear expected-travel arrow inside the reviewed region.')
    result=json.loads(json.dumps(value))
    result['config_id']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()[:20]
    return result


class DirectionReview:
    def __init__(self,config):
        self.config=json.loads(json.dumps(config));self.engine=RuleEngine();self.shape=None;self.last_stamp=None
        self.invalidated=False;self.episodes={};self.counts=Counter();self.events=0
        self.frames=0;self.absent_history_resets=0

    def step(self,tracks,stamp,shape,*,registration_valid=True,scene_changed=False,valid_mask=None):
        duplicate=self.last_stamp is not None and stamp==self.last_stamp
        if (scene_changed or self.last_stamp is not None and stamp<self.last_stamp or self.shape is not None and tuple(shape)!=self.shape):
            self.invalidated=True
        self.last_stamp=stamp;self.shape=tuple(shape);h,w=shape
        self.frames+=1
        seen={key for key,_ in tracks if key!='SELECTED'}
        for key in set(self.engine.history)-seen:
            self.engine.history.pop(key,None);self.engine.track_context.pop(key,None)
            self.engine.stop_entries.pop(key,None);self.absent_history_resets+=1
            if key in self.episodes:self.episodes[key]['clear']=0
        if not registration_valid or self.invalidated:
            self.engine.reset(self.config['config_id'])
        polygon=[[p[0]*w,p[1]*h] for p in self.config['points']]
        a,b=np.asarray(self.config['arrow'])*(w,h)
        marking={**configured_geometry_policy(),'polygon':polygon,'direction':(b-a).tolist(),
                 'source_revision':self.config['source_revision'],'config_id':self.config['config_id']}
        base=SceneEvidence(scene_id=self.config['config_id'],stationary_camera=True,confidence=1.,approach_polygon=polygon)
        rows=[]
        for key,t in tracks:
            if key=='SELECTED':continue
            x,y,bw,bh=t['box'];box=(float(x),float(y),float(x+bw),float(y+bh))
            reason=('Direction review invalidated by scene/time/size discontinuity' if self.invalidated else
                    'Current registration unverified' if not registration_valid else
                    'Duplicate frame timestamp' if duplicate else t.get('speed_blocked') or
                    (None if t.get('tracking_state')=='observed' else 'Track is not currently unambiguously observed'))
            if min(t.get('confidence',0),1)<.25:reason=reason or 'Weak vehicle detection'
            if x<=2 or y<=2 or x+bw>=w-2 or y+bh>=h-2:reason=reason or 'Vehicle clipped by frame'
            if valid_mask is not None and any(not valid_mask[min(h-1,max(0,int(py))),min(w-1,max(0,int(px)))] for px,py in [(x,y),(x+bw,y),(x+bw,y+bh),(x,y+bh)]):
                reason=reason or 'Vehicle touches a camera-warp border'
            row={'timestamp_seconds':float(stamp),'track_id':key,'vehicle_box_xyxy':box,'status':'withheld',
                 'label':None,'reason':reason,'config_id':self.config['config_id'],
                 'source_revision':self.config['source_revision'],'legal_violation':False,'eligible_for_rules':False}
            if reason:
                if key in self.episodes:self.episodes[key]['clear']=0
                if not duplicate:
                    self.engine.history.pop(key,None);self.engine.track_context.pop(key,None)
            else:
                vehicle=Detection(t.get('model_label','vehicle'),float(t['confidence']),box,key)
                scene=apply_configured_direction([marking],vehicle,base)
                decision=next(d for d in self.engine.evaluate(vehicle,stamp,scene,video=True) if d.rule==Rule.WRONG_WAY)
                row['reason']=decision.reason
                row['measurements']=decision.evidence
                if decision.verdict==Verdict.UNKNOWN and key in self.episodes:self.episodes[key]['clear']=0
                if decision.verdict==Verdict.VIOLATION:
                    episode=self.episodes.get(key)
                    if episode is None:
                        self.events+=1;episode={'id':self.events,'clear':0};self.episodes[key]=episode
                        row['new_episode']=True
                    else:row['new_episode']=False
                    episode['clear']=0
                    row.update(status='opposite_direction_observed',label='opposite configured direction',episode_id=episode['id'])
                elif decision.verdict==Verdict.CLEAR:
                    aligned=decision.evidence.get('direction_cosine',0)>=.7
                    row['status']='consistent_with_configured_direction' if aligned else 'opposite_direction_not_established'
                    if key in self.episodes and aligned:
                        self.episodes[key]['clear']+=1
                        if self.episodes[key]['clear']>=2:del self.episodes[key]
                    elif key in self.episodes:self.episodes[key]['clear']=0
            self.counts[row['status']]+=1;rows.append(row)
        while len(self.episodes)>1024:del self.episodes[next(iter(self.episodes))]
        return rows

    def summary(self):
        return {'schema_version':1,'notice':NOTICE,'configuration':self.config,'sample_counts':dict(self.counts),
                'episodes':self.events,'review_invalidated':self.invalidated,'legal_violation':False,'eligible_for_rules':False,
                'decoded_frames':self.frames,'absent_track_history_resets':self.absent_history_resets,
                'method':'Existing apply_configured_direction + RuleEngine.WRONG_WAY; complete segment in region, >=15 analysis px over >=0.5 seconds, cosine < -0.7. No speed calibration.'}


def draw(rgb,config,rows):
    out=rgb.copy();h,w=out.shape[:2]
    cv2.polylines(out,[np.rint(np.asarray(config['points'])*(w,h)).astype(np.int32)],True,(190,140,245),2)
    a,b=np.rint(np.asarray(config['arrow'])*(w,h)).astype(int)
    cv2.arrowedLine(out,tuple(a),tuple(b),(190,140,245),3,tipLength=.15)
    for row in rows:
        if row['label']:
            x,y,x2,y2=map(round,row['vehicle_box_xyxy']);cv2.rectangle(out,(x,y),(x2,y2),(255,180,40),2)
            pos=(max(2,min(w-245,x)),max(18,y-7))
            cv2.putText(out,row['label'],pos,cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(out,row['label'],pos,cv2.FONT_HERSHEY_SIMPLEX,.5,(255,180,40),1,cv2.LINE_AA)
    return out
