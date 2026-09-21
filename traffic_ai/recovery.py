"""Conservative recovery from repeated observed boxes after short gaps.

No weak detection creates a new identity. Scores retain their detector values;
competition, age and geometry checks can refuse a match even when it is close.
"""
import numpy as np
from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.utils import matching
from traffic_ai.association import GroundAnchorTracker


class RepeatedWeakRecovery(GroundAnchorTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_recovery = {}
        self.recovery_count = 0
        self.recovery_events = []

    def reset(self):
        super().reset()
        self.pending_recovery.clear()
        self.recovery_events.clear()

    def _second_association(self, pool, unmatched, detections, activated, refind, lost):
        # Diagnostics are per-frame, keeping long-video memory bounded.
        self.recovery_events = []
        super()._second_association(pool, unmatched, detections, activated, refind, lost)
        used = {t.idx for t in activated+refind}
        available = [d for d in detections if d.idx not in used and d.score >= .2]
        recent = [t for t in pool if t.state == TrackState.Lost
                  and self.frame_id-t.end_frame <= max(2, self.max_frames_lost//2)
                  and t.end_frame-t.start_frame >= self.max_frames_lost]
        if not available or not recent:
            self.pending_recovery = {}
            return
        # Include all other existing hypotheses when checking competition.
        others = [t for t in self.tracked_stracks+self.lost_stracks if t not in recent]
        all_tracks = recent+others
        a = np.array([t.xyxy for t in all_tracks])
        b = np.array([d.xyxy for d in available])
        overlap = 1-matching.iou_distance(all_tracks, available)
        anchors_a = np.c_[(a[:,0]+a[:,2])/2, a[:,3]]
        anchors_b = np.c_[(b[:,0]+b[:,2])/2, b[:,3]]
        scale = np.maximum(1, np.minimum((a[:,2:]-a[:,:2])[:,None], (b[:,2:]-b[:,:2])[None]))
        offset = np.linalg.norm((anchors_a[:,None]-anchors_b[None])/scale, axis=2)
        cost = .5*(1-overlap)+.5*np.minimum(1,offset)
        pending = {}
        for i,track in enumerate(recent):
            j = int(np.argmin(cost[i]))
            if overlap[i,j] < .5 or offset[i,j] > .15 or cost[i,j] > .3:
                continue
            competitors = np.delete(cost[:,j],i)
            alternatives = np.delete(cost[i],j)
            if ((len(competitors) and competitors.min()-cost[i,j] < .2)
                    or (len(alternatives) and alternatives.min()-cost[i,j] < .2)):
                continue
            detection = available[j]
            previous = self.pending_recovery.get(track.track_id)
            if previous and previous[0] == self.frame_id-1:
                agreement = 1-matching.iou_distance([previous[1]], [detection.xyxy])[0,0]
                if agreement >= .5:
                    self.recovery_count += 1
                    self.recovery_events.append({'frame':self.frame_id,'track_id':track.track_id,
                        'score':float(detection.score),'overlap':float(overlap[i,j]),'offset':float(offset[i,j])})
                    self._apply_match(track,detection,activated,refind)
                    continue
            pending[track.track_id] = (self.frame_id, detection.xyxy.copy())
        self.pending_recovery = pending
