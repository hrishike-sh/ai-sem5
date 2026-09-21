"""Explicit hybrid mode: existing local vehicle detector + classical tracking.

No helmet/LLM/segmentation models are loaded. IVP owns registration and speed.
"""
from collections import deque
from types import SimpleNamespace
import hashlib
import numpy as np
from ivp.timing import FrameTiming
from ivp.performance import configure_opencv_threads, configure_torch_threads


def ambiguous_overlaps(rows):
    """Nested hypotheses sharing a bottom contact point cannot support speed.

    Ordinary overlaps with distinct bottom points remain usable. This does not
    merge IDs or pretend to solve occlusion; it withholds uncertain measurements.
    """
    if len(rows)<2:return set()
    boxes=np.asarray(rows)[:,:4];extent=boxes[:,2:]-boxes[:,:2]
    area=np.prod(extent,axis=1)
    intersection=np.prod(np.maximum(0,np.minimum(boxes[:,None,2:],boxes[None,:,2:])-np.maximum(boxes[:,None,:2],boxes[None,:,:2])),axis=2)
    covered=intersection/np.maximum(1,np.minimum(area[:,None],area[None,:]))>.85
    bottom=np.abs(boxes[:,None,3]-boxes[None,:,3])<=np.minimum(extent[:,None,1],extent[None,:,1])*.12
    centres=(boxes[:,0]+boxes[:,2])/2
    aligned=np.abs(centres[:,None]-centres[None,:])<=np.minimum(extent[:,None,0],extent[None,:,0])*.3
    ambiguous=covered&bottom&aligned
    np.fill_diagonal(ambiguous,False)
    return set(np.flatnonzero(ambiguous.any(axis=1)).tolist())


class VehicleTracker:
    def __init__(self, axis='horizontal', position=.65, min_age=.3, frame_interval=1/30):
        self.timing = FrameTiming(frame_interval)
        self.axis,self.position,self.min_age=axis,position,min_age
        self.tracks={};self.counts={'positive':0,'negative':0}
        self.ambiguous_boxes=0
        self.model=None;self.last_stamp=None;self.confirmed=0;self.recovered=0
        self.observed_samples=0;self.longest_seconds=0.;self.epoch=0
        self.ambiguous_samples=0;self.ambiguous_frames=0

    def _load(self):
        from .runtime import configure_cache
        path=configure_cache()/'models'/'yolo11s.pt'
        if not path.is_file():
            raise RuntimeError('Local vehicle model is missing. Run Setup-AI.ps1, or choose IVP-only moving objects.')
        import torch
        from ultralytics import YOLO
        from .robust_recovery import RobustObservedRecovery
        # Ultralytics disables OpenCV threading on import for its DataLoader.
        # This worker processes frames directly and has no DataLoader workers.
        opencv_runtime = configure_opencv_threads()
        cpu_threads = configure_torch_threads(torch)
        self.device='cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model=YOLO(str(path))
        self.association=RobustObservedRecovery(SimpleNamespace(track_high_thresh=.35,track_low_thresh=.1,
            new_track_thresh=.4,track_buffer=24,match_thresh=.75,fuse_score=True))
        self.manifest={'filename':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'device':self.device,'cpu_threads':cpu_threads,**opencv_runtime,
                       'gpu_name':torch.cuda.get_device_name(0) if self.device=='cuda:0' else None}

    def reset(self):
        if self.model is not None:self.association.reset()
        self.tracks.clear();self.epoch+=1;self.last_stamp=None

    def update(self, boxes, stamp, shape, gray, mask, rgb):
        if self.model is None:self._load()
        gap=None if self.last_stamp is None else stamp-self.last_stamp
        if gap is not None and (gap<=0 or gap>self.timing.discontinuity):
            self.reset()
        if gap is not None and gap>0:
            self.association.max_frames_lost=max(2,min(120,round(.8/gap)))
        self.last_stamp=stamp
        result=self.model.predict(np.ascontiguousarray(rgb[:,:,::-1]),conf=.1,iou=.6,imgsz=960,
            classes=[1,2,3,5,7],agnostic_nms=True,max_det=150,device=self.device,verbose=False)[0]
        self.association.observation_timestamp=stamp
        rows=self.association.update(result.boxes.cpu().numpy(),rgb)
        weak_recovered={event['track_id'] for event in getattr(self.association,'recovery_events',[])}
        ambiguous=ambiguous_overlaps(rows)
        self.ambiguous_samples+=len(ambiguous);self.ambiguous_frames+=bool(ambiguous)
        self.tracks={k:t for k,t in self.tracks.items() if stamp-t['time']<=max(1.,2*self.timing.interval)+1e-6}
        visible,events=[],[];h,w=shape;coordinate=1 if self.axis=='horizontal' else 0
        line=self.position*(h if coordinate else w)
        for index,row in enumerate(rows):
            x,y,x2,y2,identity,confidence,cls=row[:7]
            key=self.epoch*1000000+int(identity)
            if min(x2-x,y2-y)<6:continue
            centre=np.array([(x+x2)/2,(y+y2)/2])
            roi=getattr(self,'rectangle',[0,0,1,1])
            if not roi[0]*w<=centre[0]<=roi[2]*w or not roi[1]*h<=centre[1]<=roi[3]*h:continue
            if key not in self.tracks:
                self.tracks[key]={'start':stamp,'time':stamp,'hits':0,'trail':deque(maxlen=90),
                    'confirmed':False,'side':0,'counted':False,'centre':centre}
            t=self.tracks[key]
            elapsed=stamp-t['time']
            weak_recovery=int(identity) in weak_recovered
            if elapsed>self.timing.recovery or weak_recovery:
                self.recovered+=1;t.pop('speed_history',None);t['trail'].clear();t['side']=0
            t.update(box=(float(x),float(y),float(x2-x),float(y2-y)),centre=centre,time=stamp,hits=t['hits']+1,
                     label='vehicle',model_label=str(self.model.names[int(cls)]),confidence=float(confidence),
                     tracking_state='recovered from weak observations' if weak_recovery else 'observed')
            t['speed_blocked']='vehicle clipped by frame' if x<=2 or y<=2 or x2>=w-2 or y2>=h-2 else 'weak detection' if confidence<.2 else None
            if getattr(self,'valid_mask',None) is not None:
                corners=[(x-2,y-2),(x2+2,y-2),(x2+2,y2+2),(x-2,y2+2)]
                if any(not self.valid_mask[int(np.clip(cy,0,h-1)),int(np.clip(cx,0,w-1))] for cx,cy in corners):
                    t['speed_blocked']='vehicle touches a camera-warp border'
            if index in ambiguous:
                t['speed_blocked']='overlapping vehicle hypotheses'
                t['tracking_state']='ambiguous overlap'
                t.pop('speed_history',None);t['side']=0
            # COCO has no auto-rickshaw class. Do not turn its car/truck guesses
            # into confident Indian vehicle-type labels in the product.
            if not t['confirmed'] and t['hits']>=4 and stamp-t['start']>=self.min_age:
                t['confirmed']=True;self.confirmed+=1
            if not t['confirmed']:continue
            t['trail'].append(tuple(map(int,centre)))
            side=1 if centre[coordinate]>line+5 else -1 if centre[coordinate]<line-5 else 0
            if index in ambiguous:side=0
            if side and t['side'] and side!=t['side'] and not t['counted']:
                direction='positive' if side>0 else 'negative';self.counts[direction]+=1;t['counted']=True
                events.append({'timestamp_seconds':stamp,'track_id':key,'direction':direction,'x':float(centre[0]),'y':float(centre[1])})
            if side:t['side']=side
            visible.append((key,t));self.observed_samples+=1
            self.longest_seconds=max(self.longest_seconds,stamp-t['start'])
        return visible,events

    def summary(self):
        return {'mode':'vehicles','confirmed_tracks':self.confirmed,'recovered_after_short_occlusion':self.recovered,
                'recoveries_from_repeated_weak_observations':getattr(getattr(self,'association',None),'recovery_count',0),
                'recoveries_from_robust_geometry':getattr(getattr(self,'association',None),'robust_recovery_count',0),
                'recovery_counter_scope':'All detector candidates in the frame, including outside the selected analysis region.',
                'robust_geometry_clock':'Source presentation timestamps; no constant-frame-rate assumption.',
                'ambiguous_overlap_frames':self.ambiguous_frames,'withheld_ambiguous_track_samples':self.ambiguous_samples,
                'observed_track_samples':self.observed_samples,'longest_track_seconds':round(self.longest_seconds,3),
                'detector':getattr(self,'manifest',None),'association':'ByteTrack with overlap + bottom-centre continuity; IoU for low-score tracked matches; two consecutive unambiguous weak observations can recover a recent established ID; robust observed geometry is a guarded fallback for sustained prediction drift; 0.8 second lost buffer with a two-frame minimum',
                'meaning':'Automatic vehicle candidates; COCO labels are grouped as vehicle because auto-rickshaw type is not supported. Misses, duplicates and ID swaps remain possible.'}
