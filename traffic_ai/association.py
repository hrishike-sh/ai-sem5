"""Vehicle association using overlap and road-contact continuity.

Imported only by optional vehicle tracking. No appearance model or weights.
"""
import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.utils import matching


class GroundAnchorTracker(BYTETracker):
    """Keep ByteTrack's lifecycle and low-score stage; refine high-score cost.

    Whole-box IoU is fragile when a detector changes the visible top of a
    vehicle. Its bottom centre can remain stable. Neither cue alone proves
    identity, so require some overlap and combine both, with confidence fusion.
    """
    def get_dists(self, tracks, detections):
        overlap = matching.iou_distance(tracks, detections)
        if not tracks or not detections:
            return overlap
        a = np.asarray([track.xyxy for track in tracks])
        b = np.asarray([detection.xyxy for detection in detections])
        anchors_a = np.c_[(a[:, 0] + a[:, 2]) / 2, a[:, 3]]
        anchors_b = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
        scale = np.maximum(1, np.minimum((a[:, 2:] - a[:, :2])[:, None],
                                        (b[:, 2:] - b[:, :2])[None]))
        distance = np.minimum(1, np.linalg.norm((anchors_a[:, None] - anchors_b[None]) / scale, axis=2))
        cost = np.where(overlap < .9, .5 * overlap + .5 * distance, overlap)
        return matching.fuse_score(cost, detections) if self.args.fuse_score else cost
