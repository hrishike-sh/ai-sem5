"""Conservative first-frame registration from texture outside the traffic ROI."""
import cv2
import numpy as np


class TranslationStabilizer:
    # Retained class name for callers; registration now also handles mild rotation/scale.
    def __init__(self, rectangle):
        self.rectangle = rectangle
        self.previous = None
        self.corners = None
        self.offset = np.zeros(2, dtype=np.float64)
        self.matrix = np.float32([[1,0,0],[0,1,0]])
        self.corrected_frames = 0
        self.maximum = 0.
        self.failed = False
        self.rejected_frames = 0
        self.valid_mask = None
        self.last_residual = None
        self.orb = cv2.ORB_create(nfeatures=1600, fastThreshold=12)
        self.descriptors = None
        self.orb_recoveries = 0
        self.keyframe_matrix=self.matrix.copy()
        self.frames=0

    @staticmethod
    def acceptable(matrix,h,w):
        if not np.isfinite(matrix).all():return False
        determinant=np.linalg.det(matrix[:,:2])
        angle=abs(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
        if not .8<=determinant<=1.2 or angle>=8:return False
        # Translation alone is not evidence of failure: a pan can retain a
        # well-measured road patch. Require substantial real image overlap.
        frame=np.float32([[0,0],[w,0],[w,h],[0,h]])
        warped=(frame@matrix[:,:2].T+matrix[:,2]).astype(np.float32)
        area,_=cv2.intersectConvexConvex(frame,warped)
        return area>=h*w*.6

    def step(self, rgb):
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        h,w=gray.shape
        accepted=False
        self.failed=False
        if self.previous is None:
            self.previous=gray
            mask=np.full((h,w),255,np.uint8)
            x1,y1,x2,y2=self.rectangle
            mask[int(y1*h):int(np.ceil(y2*h)),int(x1*w):int(np.ceil(x2*w))]=0
            self.corners=cv2.goodFeaturesToTrack(gray,200,.02,10,mask=mask)
            self.context_mask=mask
            self.keypoints,self.descriptors=self.orb.detectAndCompute(gray,mask)
        elif self.corners is not None and len(self.corners)>=8:
            moved,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,self.corners,None,winSize=(31,31),maxLevel=4)
            back,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,moved,None,winSize=(31,31),maxLevel=4)
            valid=(status.ravel()==1)&(back_status.ravel()==1)&(np.linalg.norm(back-self.corners,axis=2).ravel()<.8)
            source=moved.reshape(-1,2)[valid]
            reference=self.corners.reshape(-1,2)[valid]
            target=reference@self.keyframe_matrix[:,:2].T+self.keyframe_matrix[:,2]
            if len(source)>=8:
                matrix,inliers=cv2.estimateAffine2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=1.8)
                if matrix is not None:
                    if inliers.mean()>=.65 and self.acceptable(matrix,h,w):
                        self.matrix=matrix.astype(np.float32);self.offset=matrix[:,2].copy();accepted=True
                        transformed=source@matrix[:,:2].T+matrix[:,2]
                        self.last_residual=float(np.median(np.linalg.norm(transformed-target,axis=1)[inliers.ravel()==1]))
            if not accepted and self.descriptors is not None:
                # Long clips can lose first-frame LK correspondences despite a
                # nearly fixed camera. Descriptor matching recovers first-frame
                # coordinates directly, without integrating drifting transforms.
                current_mask=cv2.warpAffine(self.context_mask,cv2.invertAffineTransform(self.matrix),(w,h),flags=cv2.INTER_NEAREST)
                keys,descriptors=self.orb.detectAndCompute(gray,current_mask)
                if descriptors is not None:
                    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors,self.descriptors,k=2)
                    matches=[pair[0] for pair in pairs if len(pair)==2 and pair[0].distance<.72*pair[1].distance]
                    if len(matches)>=16:
                        source=np.float32([keys[m.queryIdx].pt for m in matches])
                        target=np.float32([self.keypoints[m.trainIdx].pt for m in matches])
                        candidate,inliers=cv2.estimateAffine2D(source,target,method=cv2.RANSAC,ransacReprojThreshold=1.8)
                        if candidate is not None and inliers.mean()>=.55 and inliers.sum()>=12:
                            if self.acceptable(candidate,h,w):
                                self.matrix=candidate.astype(np.float32);self.offset=candidate[:,2].copy();accepted=True
                                errors=np.linalg.norm(source@candidate[:,:2].T+candidate[:,2]-target,axis=1)
                                self.last_residual=float(np.median(errors[inliers.ravel()==1]));self.orb_recoveries+=1
            if not accepted:
                self.failed=True;self.rejected_frames+=1
        self.frames+=1
        if accepted and self.frames%20==0:
            # Refresh short-range LK texture while retaining the transform into
            # the original coordinate system. This avoids large first-frame
            # brightness changes breaking every subsequent frame.
            current_mask=cv2.warpAffine(self.context_mask,cv2.invertAffineTransform(self.matrix),(w,h),flags=cv2.INTER_NEAREST)
            fresh=cv2.goodFeaturesToTrack(gray,200,.02,10,mask=current_mask)
            if fresh is not None and len(fresh)>=12:
                self.previous=gray.copy();self.corners=fresh;self.keyframe_matrix=self.matrix.copy()
        self.maximum=max(self.maximum,float(np.linalg.norm(self.offset)))
        identity=np.float32([[1,0,0],[0,1,0]])
        if np.linalg.norm(self.matrix-identity)>.15:
            self.corrected_frames+=1
        self.valid_mask=cv2.warpAffine(np.full((h,w),255,np.uint8),self.matrix,(w,h),flags=cv2.INTER_NEAREST)
        rgb=cv2.warpAffine(rgb,self.matrix,(w,h),borderMode=cv2.BORDER_CONSTANT)
        return rgb,accepted

    def summary(self):
        return {'method':'Keyframe Lucas-Kanade registration into first-frame coordinates, with ORB reference recovery outside the road ROI and RANSAC affine transforms.',
                'descriptor_recoveries':self.orb_recoveries,
                'corrected_frames':self.corrected_frames,'rejected_frames':self.rejected_frames,
                'maximum_translation_px':self.maximum,'last_registration_residual_px':self.last_residual,
                'coordinate_frame':'First-frame image coordinates. Per-frame raw-to-analysis affine transform is in motion-activity.csv; empty borders excluded from measurement.',
                'limitation':'Needs textured stationary context outside the ROI. Corrects mild pan, rotation and zoom; not large perspective changes or parallax.'}
