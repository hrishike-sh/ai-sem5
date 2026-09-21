"""Automatic foreground discovery with independent, verified KLT object tracks.

No neural classifier: identities refer to moving objects, not guaranteed vehicles.
Foreground supplies seeds, never replaces an established object's box during a merge.
"""
from collections import deque
import cv2
import numpy as np
from .timing import FrameTiming


def overlap(a, b):
    x, y = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0, right-x)*max(0, bottom-y)
    return intersection/max(1, min(a[2]*a[3], b[2]*b[3]))


class AutomaticTracker:
    def __init__(self, axis='horizontal', position=.65, min_age=.3, frame_interval=1/30):
        self.timing = FrameTiming(frame_interval)
        self.axis, self.position, self.min_age = axis, position, min_age
        self.tracks, self.next_id = {}, 1
        self.previous = None
        self.last_stamp = None
        self.counts = {'positive': 0, 'negative': 0}
        self.ambiguous_boxes = 0
        self.confirmed = self.recovered = self.merge_frames = 0
        self.observed_samples = self.occluded_samples = 0
        self.longest_seconds = 0.
        self.max_tracks = 96

    @staticmethod
    def _features(gray, mask, box):
        h, w = gray.shape
        x, y, bw, bh = box
        region = np.zeros_like(gray)
        region[max(0,int(y)):min(h,int(y+bh)+1), max(0,int(x)):min(w,int(x+bw)+1)] = 255
        if mask is not None:
            region &= mask
        points = cv2.goodFeaturesToTrack(gray, 32, .015, 3, mask=region, blockSize=3)
        return np.empty((0,1,2),np.float32) if points is None else points

    @staticmethod
    def _appearance(rgb, box):
        x,y,bw,bh = map(int, box); h,w = rgb.shape[:2]
        patch = rgb[max(0,y):min(h,y+bh),max(0,x):min(w,x+bw)]
        if not patch.size:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256])
        return cv2.normalize(hist,hist).ravel()

    def update(self, boxes, stamp, shape, gray, mask, rgb):
        h,w = shape
        dt = 0 if self.last_stamp is None else stamp-self.last_stamp
        if dt <= 0 or dt > self.timing.discontinuity:
            self.tracks.clear()
            self.previous = None
        self.tracks = {key:t for key,t in self.tracks.items()
                       if stamp-t['time'] <= max(.65,2.5*self.timing.interval)+1e-6
                       and (t['confirmed'] or stamp-t['start'] < max(1.2,4*self.timing.interval))}
        for t in self.tracks.values():
            t['observed'] = False
        # Batch all objects into two LK calls rather than two calls per object.
        candidates = [(key,t) for key,t in self.tracks.items() if len(t['points']) >= 4]
        if self.previous is not None and candidates:
            points = np.concatenate([t['points'] for _,t in candidates])
            moved,status,error = cv2.calcOpticalFlowPyrLK(self.previous,gray,points,None,winSize=(21,21),maxLevel=3)
            if moved is not None:
                back,back_status,_ = cv2.calcOpticalFlowPyrLK(gray,self.previous,moved,None,winSize=(21,21),maxLevel=3)
                good = (status.ravel()==1)&(back_status.ravel()==1)&(error.ravel()<25)&(np.linalg.norm(back-points,axis=2).ravel()<.8)
                offset = 0
                for _,t in candidates:
                    size = len(t['points']); valid = good[offset:offset+size]
                    old = points[offset:offset+size,0][valid]; new = moved[offset:offset+size,0][valid]
                    offset += size
                    if len(new) < 4:
                        t['points'] = np.empty((0,1,2),np.float32); continue
                    vectors = new-old
                    shift = np.median(vectors,axis=0)
                    coherent = np.linalg.norm(vectors-shift,axis=1) < max(1.2,np.linalg.norm(shift)*.25)
                    if coherent.sum()<4 or coherent.sum()<size*.35:
                        t['points'] = np.empty((0,1,2),np.float32); continue
                    old,new = old[coherent],new[coherent]
                    shift = np.median(new-old,axis=0)
                    if np.linalg.norm(shift) > min(w,h)*.12:
                        t['points'] = np.empty((0,1,2),np.float32); continue
                    # Translation keeps the measured anchor stable. Scale changes are
                    # accepted only from enough spatially distributed correspondences.
                    scale = 1.
                    if len(new)>=8 and np.min(np.ptp(old,axis=0))>8:
                        matrix,inliers = cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=1.)
                        if matrix is not None and inliers.sum()>=len(new)*.8:
                            estimate = np.linalg.norm(matrix[0,:2])
                            if .98<estimate<1.02:
                                scale = 1.+(estimate-1.)*.5
                    centre = t['centre']+shift
                    bw,bh = np.asarray(t['box'][2:])*scale
                    t['box'] = (centre[0]-bw/2,centre[1]-bh/2,bw,bh)
                    t['centre'] = centre
                    t['velocity'] = .6*t['velocity']+.4*shift/dt
                    t['points'] = new.reshape(-1,1,2)
                    t['observed'] = True

        self.ambiguous_boxes = 0
        blocked = set()
        for index,box in enumerate(boxes):
            owners = [t for t in self.tracks.values() if overlap(box,t['box'])>.35]
            if len(owners)>1:
                blocked.add(index); self.ambiguous_boxes += 1
            elif owners:
                # A live feature track owns this component even when it fragments.
                if owners[0]['observed']:
                    blocked.add(index)

        # Conservative recovery after a short occlusion: require unique geometric
        # AND colour agreement. No speed or line crossing is inferred in the gap.
        for index,box in enumerate(boxes):
            if index in blocked:
                continue
            options=[]
            for key,t in self.tracks.items():
                if t['observed'] or not t['confirmed']:
                    continue
                gap=stamp-t['time']
                predicted=np.asarray(t['box'])+np.r_[t['velocity']*gap,[0,0]]
                ratio=box[2]*box[3]/max(1,t['box'][2]*t['box'][3])
                if overlap(box,predicted)>.5 and .6<ratio<1.6:
                    appearance=self._appearance(rgb,box)
                    if appearance is not None and cv2.compareHist(t['appearance'],appearance,cv2.HISTCMP_BHATTACHARYYA)<.25:
                        options.append(key)
            if len(options)==1:
                t=self.tracks[options[0]]
                features=self._features(gray,mask,box)
                if len(features)>=4:
                    x,y,bw,bh=box
                    t.update(box=box,centre=np.array([x+bw/2,y+bh/2]),points=features,observed=True,side=0)
                    t.pop('speed_history',None)
                    t['trail'].clear(); self.recovered+=1; blocked.add(index)

        # Discover all suitable moving components, without per-object clicks.
        for index,box in sorted(enumerate(boxes),key=lambda item:item[1][2]*item[1][3],reverse=True):
            if index in blocked or len(self.tracks)>=self.max_tracks:
                continue
            if any(overlap(box,t['box'])>.25 for t in self.tracks.values()):
                continue
            points=self._features(gray,mask,box)
            if len(points)<4:
                continue
            x,y,bw,bh=box; centre=np.array([x+bw/2,y+bh/2])
            self.tracks[self.next_id]={'box':box,'centre':centre,'origin':centre.copy(),
                'velocity':np.zeros(2),'time':stamp,'start':stamp,'hits':0,'points':points,
                'appearance':self._appearance(rgb,box),'confirmed':False,'observed':True,
                'side':0,'counted':False,'trail':deque(maxlen=90),'last_seed':stamp}
            self.next_id+=1

        visible,events=[],[]
        coordinate=1 if self.axis=='horizontal' else 0
        line=self.position*(h if coordinate else w)
        for key,t in list(self.tracks.items()):
            centre=t['centre']
            if not 0<=centre[0]<w or not 0<=centre[1]<h:
                del self.tracks[key]; continue
            if not t['observed']:
                t['points']=np.empty((0,1,2),np.float32)
                t.pop('speed_history',None); t['side']=0
                self.occluded_samples+=int(t['confirmed'])
                continue
            t['time']=stamp; t['hits']+=1
            if not t['confirmed'] and t['hits']>=4 and stamp-t['start']>=self.min_age and np.linalg.norm(centre-t['origin'])>=2:
                t['confirmed']=True; self.confirmed+=1
            t['feature_count']=len(t['points'])
            t['tracking_state']='observed'
            t['trail'].append(tuple(map(int,centre)))
            # Replenish only from foreground within an isolated track. Avoid
            # importing a neighbour's texture while a merged blob covers both.
            if len(t['points'])<12 and stamp-t['last_seed']>.25 and not any(overlap(t['box'],other['box'])>.15 for other in self.tracks.values() if other is not t):
                seeds=self._features(gray,mask,t['box'])
                if len(seeds):
                    distances=np.linalg.norm(seeds[:,None,0,:]-t['points'][None,:,0,:],axis=2)
                    seeds=seeds[np.min(distances,axis=1)>3]
                    t['points']=np.concatenate([t['points'],seeds])[:32]
                t['last_seed']=stamp
            if not t['confirmed']:
                continue
            side=1 if centre[coordinate]>line+5 else -1 if centre[coordinate]<line-5 else 0
            if side and t['side'] and side!=t['side'] and not t['counted']:
                direction='positive' if side>0 else 'negative'
                self.counts[direction]+=1; t['counted']=True
                events.append({'timestamp_seconds':stamp,'track_id':key,'direction':direction,'x':float(centre[0]),'y':float(centre[1])})
            if side:t['side']=side
            visible.append((key,t)); self.observed_samples+=1
            self.longest_seconds=max(self.longest_seconds,stamp-t['start'])
        if self.ambiguous_boxes:self.merge_frames+=1
        self.previous=gray.copy(); self.last_stamp=stamp
        return visible,events

    def summary(self):
        return {'mode':'automatic_ivp','confirmed_tracks':self.confirmed,
                'recovered_after_short_occlusion':self.recovered,'observed_track_samples':self.observed_samples,
                'withheld_occluded_samples':self.occluded_samples,'merged_foreground_frames':self.merge_frames,
                'longest_track_seconds':round(self.longest_seconds,3),'max_active_tracks':self.max_tracks,
                'meaning':'Automatically seeded moving-object tracks. No vehicle classifier; a pedestrian or a merged group can also form a track.'}
