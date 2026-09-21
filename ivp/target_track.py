"""User-initialized feature tracking; no neural detector or semantic guess."""
from collections import deque
import cv2
import numpy as np
from .timing import FrameTiming


class SelectedTarget:
    def __init__(self, rectangle, frame_interval=1/30):
        self.timing = FrameTiming(frame_interval)
        points=np.asarray(rectangle,dtype=float)
        if (points.shape!=(4,) or not np.isfinite(points).all() or np.any(points<0) or np.any(points>1)
                or points[2]-points[0]<.02 or points[3]-points[1]<.02):
            raise ValueError('Select a target at least 2% wide and high in the first frame.')
        self.rectangle=points
        self.previous=None
        self.points=None
        self.box=None
        self.track={'trail':deque(maxlen=60)}
        self.status='waiting'
        self.last_stamp=None
        self.frames=0
        self.lost_at=None

    def step(self,rgb,stamp,registration_ok=True):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY);h,w=gray.shape
        if self.previous is None:
            x1,y1,x2,y2=self.rectangle*[w,h,w,h]
            self.box=np.array([x1,y1,x2-x1,y2-y1],float)
            self.quad=np.float32([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])
            mask=np.zeros((h,w),np.uint8);mask[int(y1):int(y2),int(x1):int(x2)]=255
            self.points=cv2.goodFeaturesToTrack(gray,100,.02,4,mask=mask)
            self.status='tracking' if self.points is not None and len(self.points)>=8 else 'not enough target texture'
            self.previous=gray
        elif self.status=='tracking':
            moved,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.points,None,winSize=(31,31),maxLevel=3)
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,moved,None,winSize=(31,31),maxLevel=3)
            valid=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back-self.points,axis=2).ravel()<.8)
            old=self.points.reshape(-1,2)[valid];new=moved.reshape(-1,2)[valid]
            matrix,inliers=(None,None) if len(old)<8 else cv2.estimateAffinePartial2D(old,new,method=cv2.RANSAC,ransacReprojThreshold=1.5)
            if matrix is None or inliers.sum()<8 or not registration_ok or stamp-self.last_stamp>self.timing.discontinuity:
                self.status='lost - select a clearer target or shorter clip';self.lost_at=stamp
            else:
                scale=float(np.linalg.norm(matrix[0,:2]));shift=np.linalg.norm(matrix[:,2])
                if not .9<scale<1.1 or shift>min(h,w)*.15:
                    self.status='lost - inconsistent feature motion';self.lost_at=stamp
                else:
                    x,y,bw,bh=self.box
                    corners=self.quad@matrix[:,:2].T+matrix[:,2]
                    self.quad=corners
                    lower=corners.min(axis=0);upper=corners.max(axis=0)
                    self.box=np.r_[lower,upper-lower]
                    self.points=new[inliers.ravel()==1].reshape(-1,1,2)
                    self.previous=gray
                    if np.any(lower<0) or upper[0]>w or upper[1]>h:
                        self.status='target left frame';self.lost_at=stamp
        self.last_stamp=stamp
        if self.status!='tracking':
            return None
        x,y,bw,bh=self.box;centre=np.array([x+bw/2,y+bh/2])
        self.track.update(box=tuple(self.box),centre=centre,feature_count=len(self.points))
        self.track['trail'].append(tuple(map(int,centre)));self.frames+=1
        return self.track

    def summary(self):
        return {'method':'User-selected first-frame patch, Shi-Tomasi corners, forward/backward Lucas-Kanade and RANSAC similarity tracking.',
                'rectangle':self.rectangle.tolist(),'tracked_frames':self.frames,'status':self.status,'lost_at_seconds':self.lost_at,
                'limitation':'The user identifies the vehicle. Visual feature drift, occlusion and changing apparent shape can still affect the box and speed. No automatic reacquisition.'}
