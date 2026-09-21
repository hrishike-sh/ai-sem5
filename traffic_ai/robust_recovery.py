"""Robust observed-box geometry for short gaps after prediction drift.

Only new observed detections can restore an ID. Missing frames have no output.
"""
from collections import deque
import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching
from traffic_ai.recovery import RepeatedWeakRecovery
from traffic_ai.proposal_geometry import grouped_proposals


def geometry_costs(a,b):
    overlap=1-matching.iou_distance(list(a),list(b))
    anchors_a=np.c_[(a[:,0]+a[:,2])/2,a[:,3]];anchors_b=np.c_[(b[:,0]+b[:,2])/2,b[:,3]]
    scale=np.maximum(1,np.minimum((a[:,2:]-a[:,:2])[:,None],(b[:,2:]-b[:,:2])[None]))
    offset=np.linalg.norm((anchors_a[:,None]-anchors_b[None])/scale,axis=2)
    return overlap,offset,.5*(1-overlap)+.5*np.minimum(1,offset)


class RobustObservedRecovery(RepeatedWeakRecovery):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observations={}
        self.robust_pending={}
        self.robust_recovery_count=0

    def reset(self):
        super().reset()
        self.observations.clear();self.robust_pending.clear()

    def update(self,results,*args,**kwargs):
        self.current_detections=np.asarray(results.data)
        self.frame_shape=results.orig_shape
        rows=super().update(results,*args,**kwargs)
        for row in rows:
            history=self.observations.setdefault(int(row[4]),deque(maxlen=12))
            history.append((self.frame_id,np.asarray(results.xyxy[int(row[-1])],dtype=float).copy(),
                            getattr(self,'observation_timestamp',None)))
        live={t.track_id for t in self.tracked_stracks+self.lost_stracks}
        self.observations={key:value for key,value in self.observations.items() if key in live}
        self.current_detections=None
        return rows

    def robust_box(self,track):
        history=list(self.observations.get(track.track_id,[]))
        # Require enough observations from approximately half a second, rather
        # than extrapolating old low-frame-rate positions through acceleration.
        now=getattr(self,'observation_timestamp',None)
        use_time=now is not None and history and all(p[2] is not None for p in history)
        coordinate=2 if use_time else 0
        window=.5 if use_time else max(1,self.max_frames_lost*.625)
        history=[p for p in history if history[-1][coordinate]-p[coordinate]<=window+1e-6] if history else []
        if len(history)<8:return track.xyxy
        times=np.array([p[coordinate] for p in history],dtype=float)
        target=now if use_time else self.frame_id
        if not np.isfinite(times).all() or not np.isfinite(target) or np.any(np.diff(times)<=0):
            return track.xyxy
        boxes=np.array([p[1] for p in history])
        values=np.c_[(boxes[:,0]+boxes[:,2])/2,boxes[:,3],boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]]
        slopes=np.diff(values,axis=0)/np.diff(times)[:,None]
        velocity=np.median(slopes,axis=0)
        predicted=np.median(values+(target-times)[:,None]*velocity,axis=0)
        x,bottom,width,height=predicted
        if width<6 or height<6:return track.xyxy
        return np.array([x-width/2,bottom-height,x+width/2,bottom])

    def _second_association(self,pool,unmatched,detections,activated,refind,lost):
        super()._second_association(pool,unmatched,detections,activated,refind,lost)
        used={t.idx for t in activated+refind}
        h,w=self.frame_shape
        available=[d for d in detections if d.idx not in used and d.score>=.2
                   and d.xyxy[0]>2 and d.xyxy[1]>2
                   and d.xyxy[2]<w-2 and d.xyxy[3]<h-2]
        recent=[t for t in pool if t.state==TrackState.Lost
                # Leave one-frame churn to the primary matcher. This fallback
                # needs a sustained gap followed by repeated observations.
                and 3<=self.frame_id-t.end_frame<=max(2,self.max_frames_lost//2)
                and t.end_frame-t.start_frame>=self.max_frames_lost]
        if not available or not recent:
            self.robust_pending={};return
        # This diagnostic is needed only for the extra fallback, not for every
        # frame or for ByteTrack's normal detection/association path.
        grouped_indices=grouped_proposals(self.current_detections)
        available=[d for d in available if d.idx not in grouped_indices]
        if not available:
            self.robust_pending={};return
        others=[t for t in self.tracked_stracks+self.lost_stracks if t not in recent]
        all_tracks=recent+others
        a=np.array([self.robust_box(t) for t in all_tracks]);b=np.array([d.xyxy for d in available])
        overlap,offset,cost=geometry_costs(a,b)
        _,_,raw_cost=geometry_costs(np.array([t.xyxy for t in all_tracks]),b)
        # A smoothed history must not conceal a competitor whose current
        # prediction is close. Consider both geometries when refusing a match.
        competing_cost=np.minimum(cost,raw_cost)
        pending={}
        for i,track in enumerate(recent):
            if a[i,0]<=2 or a[i,1]<=2 or a[i,2]>=w-2 or a[i,3]>=h-2:continue
            j=int(np.argmin(cost[i]))
            if overlap[i,j]<.5 or offset[i,j]>.15 or cost[i,j]>.3:continue
            competitors=np.delete(competing_cost[:,j],i);alternatives=np.delete(competing_cost[i],j)
            if ((len(competitors) and competitors.min()-cost[i,j]<.2)
                    or (len(alternatives) and alternatives.min()-cost[i,j]<.2)):continue
            detection=available[j];previous=self.robust_pending.get(track.track_id)
            if previous and previous[0]==self.frame_id-1:
                agreement=1-matching.iou_distance([previous[1]],[detection.xyxy])[0,0]
                if agreement>=.5:
                    self.recovery_count+=1
                    self.robust_recovery_count+=1
                    self.recovery_events.append({'frame':self.frame_id,'track_id':track.track_id,
                        'score':float(detection.score),'overlap':float(overlap[i,j]),'offset':float(offset[i,j]),
                        'geometry':'robust observed history'})
                    self._apply_match(track,detection,activated,refind)
                    continue
            pending[track.track_id]=(self.frame_id,detection.xyxy.copy())
        self.robust_pending=pending
