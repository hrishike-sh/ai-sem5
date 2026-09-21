"""Do not confirm an event using support contradicted by an explicit clear.

Raw rule assessments are not changed. The existing collector still owns timing,
active episode deduplication, transition immediacy and unknown/missing gap policy.
Only unconfirmed support and contradictory current-frame emission are filtered.
"""
from copy import deepcopy

from .contracts import Verdict
from .events import EventCollector


def _frame_key(decision, image):
    if not image and decision.track_id is None:
        return None
    if decision.track_id is not None:
        return decision.track_id, decision.rule
    return (tuple(decision.box), decision.rule) if decision.box is not None else None


def _describe(key):
    identity, rule = key
    return {"identity": list(identity) if isinstance(identity, tuple) else identity,
            "identity_basis": "image_box" if isinstance(identity, tuple) else "track_id",
            "rule": getattr(rule, "value", str(rule))}


class ContradictionAwareEventCollector(EventCollector):
    def __init__(self, persistence=.4, gap=1.0):
        super().__init__(persistence=persistence, gap=gap)
        self._confirmation = {"state": "not_observed"}

    def update(self, decisions, timestamp, scene_id=0, image=False):
        decisions = list(decisions)
        keys = [_frame_key(decision, image) for decision in decisions]
        clear_indices, positive_indices, unknown_keys = {}, {}, set()
        for index, (decision, key) in enumerate(zip(decisions, keys)):
            if key is None:
                continue
            if decision.verdict == Verdict.CLEAR:
                clear_indices.setdefault(key, []).append(index)
            elif decision.verdict == Verdict.VIOLATION and decision.box is not None:
                positive_indices.setdefault(key, []).append(index)
            elif decision.verdict == Verdict.UNKNOWN:
                unknown_keys.add(key)
        conflicts = set(clear_indices) & set(positive_indices)
        cleared = []
        scene_reset = self.scene_id != scene_id
        if not scene_reset:
            for key, support in list(self.pending.items()):
                frame_key = key[:2]
                if frame_key not in clear_indices:
                    continue
                first, last, count = self.pending.pop(key)
                cleared.append({**_describe(frame_key), "first_observed_at": first,
                                "last_observed_at": last, "observations": count,
                                "reason": "explicit_clear_assessment",
                                "source_decision_indices": clear_indices[frame_key][:]})
        # A clear assessment and a violation for the same identity/rule cannot
        # be resolved by iteration order. Preserve both source decisions, while
        # withholding the contradictory positive from event emission/rendering.
        filtered = [decision for decision, key in zip(decisions, keys)
                    if not (key in conflicts and decision.verdict == Verdict.VIOLATION)]
        events, displayed = super().update(filtered, timestamp, scene_id=scene_id, image=image)
        pending = []
        for key, (first, last, count) in self.pending.items():
            frame_key = key[:2]
            reason = ("persistence_not_yet_satisfied" if frame_key in positive_indices
                      else "current_assessment_unknown_gap_tolerance" if frame_key in unknown_keys
                      else "no_current_violation_gap_tolerance")
            pending.append({**_describe(frame_key), "first_observed_at": first, "last_observed_at": last,
                            "observations": count, "elapsed_seconds": timestamp - first,
                            "required_persistence_seconds": self.persistence, "reason": reason,
                            "source_decision_indices": positive_indices.get(frame_key, [])[:]})
        displayed_ids = {id(decision) for decision in displayed}
        self._confirmation = {"policy": "explicit_clear_confirmation_v1", "timestamp": timestamp,
                              "scene_id": scene_id, "image": bool(image), "scene_reset": scene_reset,
                              "cleared_pending": cleared, "pending": pending,
                              "contradictions": [{**_describe(key), "reason": "conflicting_frame_assessments",
                                                   "clear_decision_indices": clear_indices[key][:],
                                                   "violation_decision_indices": positive_indices[key][:]}
                                                  for key in positive_indices if key in conflicts],
                              "displayed_decision_indices": [i for i, d in enumerate(decisions) if id(d) in displayed_ids],
                              "new_events": len(events),
                              "scope": "Confirmation-state audit only; source verdicts/evidence and existing gap policy are unchanged"}
        return events, displayed

    def confirmation_snapshot(self):
        return deepcopy(self._confirmation)
