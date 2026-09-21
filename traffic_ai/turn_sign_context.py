"""Bounded context-padding fallback for an injected existing turn-sign detector.

No model is constructed here. Predictions are mapped back to original RGB and
must pass the existing score/shape checks there. Additional representations do
not establish sign applicability, legal direction or calibrated confidence.
"""
from copy import deepcopy
import hashlib
import json
from math import ceil, isfinite

import numpy as np

from .geometry import area, intersection
from .turn_signs import turn_sign_observation


POLICY = "bounded_context_padding_v1"


def _fingerprint(rgb):
    header = json.dumps({"shape": list(rgb.shape), "dtype": str(rgb.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + np.ascontiguousarray(rgb).tobytes()).hexdigest()


def _iou(a, b):
    overlap = intersection(a, b)
    return overlap / max(1e-9, area(a) + area(b) - overlap)


class ContextPaddedTurnDetector:
    """At most three model calls; original first, then two fixed padded views.

    Fallback runs only when the original yields no recognised prohibition. The
    detector must expose detect(rgb) and may expose manifest. Cancellation raises
    normally: no partial list is returned. Inputs and underlying scores remain
    unchanged. Pixel budget limits only additional views, never silently resizes.
    """

    def __init__(self, detector, cancel=lambda: None, max_canvas_pixels=4_000_000):
        if type(max_canvas_pixels) is not int or max_canvas_pixels < 1:
            raise ValueError("Canvas budget must be a positive integer")
        self.detector = detector
        self.cancel = cancel
        self.max_canvas_pixels = max_canvas_pixels
        self.last_audit = None

    @property
    def manifest(self):
        return {**deepcopy(getattr(self.detector, "manifest", {})),
                "context_adapter": {"policy": POLICY, "padding_factors": [1., 2.], "padding_rgb": [114, 114, 114],
                                    "max_model_calls": 3, "max_extra_canvas_pixels": self.max_canvas_pixels,
                                    "acceptance": "Existing original-pixel score>=0.8, crop-size and colour/shape checks; applicability unverified"}}

    def _pass(self, original, padding, index):
        self.cancel()
        height, width = original.shape[:2]
        canvas = np.full((height + 2 * padding, width + 2 * padding, 3), 114, np.uint8)
        canvas[padding:padding + height, padding:padding + width] = original
        record = {"index": index, "padding_pixels": padding, "canvas_shape": list(canvas.shape),
                  "canvas_sha256": _fingerprint(canvas), "status": "running"}
        self.last_audit["passes"].append(record)
        raw = self.detector.detect(canvas)
        self.cancel()
        if not isinstance(raw, list):
            raise ValueError("Detector must return an observation list")
        candidates = []
        for item in raw:
            self.cancel()
            box = item.get("box") if isinstance(item, dict) else None
            score = item.get("model_score") if isinstance(item, dict) else None
            label = item.get("label") if isinstance(item, dict) else None
            if (not isinstance(box, (tuple, list)) or len(box) != 4
                    or any(not isinstance(v, (int, float)) or isinstance(v, bool) or not isfinite(v) for v in box)
                    or area(box) <= 0 or not isinstance(score, (int, float)) or isinstance(score, bool)
                    or not isfinite(score) or not 0 <= score <= 1 or not isinstance(label, str)):
                record.setdefault("rejected", []).append({"reason": "malformed_detector_observation"})
                raise ValueError("Malformed detector observation; no candidate result is available")
            mapped = [box[0] - padding, box[1] - padding, box[2] - padding, box[3] - padding]
            clipped = [max(0., mapped[0]), max(0., mapped[1]), min(float(width), mapped[2]), min(float(height), mapped[3])]
            support = area(clipped) / max(1e-9, area(mapped))
            trace = {"pass_index": index, "padding_pixels": padding, "raw_box": list(box), "label": label,
                     "model_score": float(score), "mapped_unclipped_box": mapped, "original_area_fraction": support}
            trace["source_provenance"] = {key: deepcopy(item[key]) for key in ("synthetic", "provenance") if key in item}
            if area(clipped) <= 0 or support < .9:
                record.setdefault("rejected", []).append({**trace, "reason": "prediction_not_supported_by_original_pixels"})
                continue
            observation = turn_sign_observation(original, clipped, label, score)
            candidates.append({"observation": observation, "trace": trace})
        record.update(status="complete", raw_observations=len(raw), retained_observations=len(candidates))
        return candidates

    def _combine(self, candidates):
        groups = []
        for candidate in candidates:
            matches = [g for g in groups if candidate["trace"]["pass_index"] not in {m["trace"]["pass_index"] for m in g["members"]}
                       and any(_iou(candidate["observation"]["box"], m["observation"]["box"]) > .5 for m in g["members"])]
            if len(matches) == 1:
                matches[0]["members"].append(candidate)
            else:
                if matches:
                    for group in matches:
                        group["spatial_ambiguity"] = True
                groups.append({"members": [candidate], "spatial_ambiguity": bool(matches)})
        result = []
        for group in groups:
            accepted = [m for m in group["members"] if m["observation"]["status"] == "recognised_candidate"]
            chosen = max(accepted or group["members"], key=lambda m: m["observation"]["model_score"])
            observation = deepcopy(chosen["observation"])
            strong_labels = {m["observation"]["label"] for m in group["members"] if m["observation"]["model_score"] >= .8}
            if group["spatial_ambiguity"] or len(strong_labels) > 1:
                observation.update(status="cannot_assess", prohibited_turns=[],
                                   reason="Conflicting high-score meanings or ambiguous cross-pass object grouping")
            observation.update(source_frame_sha256=self.last_audit["source_frame_sha256"], proposal_adapter=POLICY,
                               detector_hypotheses=[deepcopy(m["trace"]) for m in group["members"]])
            result.append(observation)
        return result

    def detect(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Expected a nonempty uint8 RGB image")
        self.cancel()
        source_hash = _fingerprint(rgb)
        self.last_audit = {"policy": POLICY, "state": "running", "source_frame_sha256": source_hash,
                           "source_shape": list(rgb.shape), "passes": [], "skipped": []}
        candidates = self._pass(rgb, 0, 0)
        if not any(c["observation"]["status"] == "recognised_candidate" for c in candidates):
            height, width = rgb.shape[:2]
            for index, factor in enumerate((1., 2.), 1):
                self.cancel()
                padding = ceil(max(height, width) * factor)
                pixels = (height + 2 * padding) * (width + 2 * padding)
                if pixels > self.max_canvas_pixels:
                    self.last_audit["skipped"].append({"pass_index": index, "padding_pixels": padding,
                                                       "reason": "canvas_pixel_budget", "requested_pixels": pixels})
                    continue
                candidates.extend(self._pass(rgb, padding, index))
        self.cancel()
        if _fingerprint(rgb) != source_hash:
            raise ValueError("Original RGB changed during detector execution")
        result = self._combine(candidates)
        self.last_audit.update(state="complete", model_calls=len(self.last_audit["passes"]),
                               recognised_candidates=sum(o["status"] == "recognised_candidate" for o in result),
                               scope="Candidate recognition only; no sign applicability or legal verdict")
        return result
