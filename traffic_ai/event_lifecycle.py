"""Keep event aggregation within explicitly observed continuity boundaries.

This uses the RuleEngine's existing 0.75-second / five-box-width track policy.
It does not establish identity, camera stability, scene semantics or a verdict.
Call with all current eligible vehicle detections, including unknown assessments.
"""
from copy import deepcopy
from math import hypot

from .event_continuity import ContradictionAwareEventCollector


class LifecycleAwareEventCollector(ContradictionAwareEventCollector):
    def __init__(self, persistence=.4, gap=1.0):
        super().__init__(persistence=persistence, gap=gap)
        self._observations = {}
        self._lifecycle = {"state": "not_observed"}

    def _discard(self, reason, boundaries, track_id=None, **evidence):
        counts = {}
        for name in ("pending", "active", "reported_turn_entries"):
            states = getattr(self, name)
            removed = [key for key in states if track_id is None or key[0] == track_id]
            for key in removed:
                del states[key]
            counts[name] = len(removed)
        boundaries.append({"reason": reason, "track_id": track_id, "discarded_states": counts, **evidence})

    def update(self, decisions, timestamp, scene_id=0, image=False, *,
               detections=None, camera_stable=None, scene_revision=None):
        boundaries = []
        observed = None if detections is None else list(detections)
        scene_reset = self.scene_id != scene_id
        if scene_reset:
            self._discard("scene_identity_changed", boundaries, previous_scene_id=self.scene_id)
            self._observations.clear()
        if image:
            self._observations.clear()
        else:
            if observed is not None:
                # Absence is not a clear verdict. It only ends correspondence
                # after the same gap that already invalidates producer history.
                for identity, (last, point) in list(self._observations.items()):
                    if timestamp - last > .75:
                        self._discard("tracking_observation_gap", boundaries, identity,
                                      last_observed_at=last, elapsed_seconds=timestamp - last)
                        del self._observations[identity]
                for index, detection in enumerate(observed):
                    identity = detection.track_id
                    if identity is None:
                        continue
                    point = detection.ground_point
                    previous = self._observations.get(identity)
                    if previous:
                        distance = hypot(point[0] - previous[1][0], point[1] - previous[1][1])
                        limit = 5 * max(8., detection.box[2] - detection.box[0])
                        if distance > limit:
                            self._discard("tracking_position_jump", boundaries, identity,
                                          previous_point=list(previous[1]), current_point=list(point),
                                          distance_pixels=distance, limit_pixels=limit,
                                          source_detection_index=index)
                    self._observations[identity] = (timestamp, point)
        created, displayed = super().update(decisions, timestamp, scene_id, image)
        self._lifecycle = {"policy": "observed_event_lifecycle_v1", "timestamp": timestamp,
                           "image": bool(image), "scene_id": scene_id, "scene_revision": scene_revision,
                           "camera_stable": camera_stable, "boundaries": boundaries,
                           "detection_observations": "not_recorded" if observed is None else len(observed),
                           "source_detection_index_space": "supplied eligible vehicle detection list",
                           "retained_track_observations": len(self._observations),
                           "track_gap_seconds": .75, "position_jump_box_widths": 5,
                           "context_metadata_role": "Recorded only; camera/revision changes do not infer identity loss",
                           "scope": "Aggregation continuity only; raw decisions and historical event evidence unchanged"}
        return created, displayed

    def lifecycle_snapshot(self):
        return deepcopy(self._lifecycle)

    def confirmation_snapshot(self):
        return {**super().confirmation_snapshot(), "lifecycle": self.lifecycle_snapshot()}
