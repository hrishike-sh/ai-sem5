"""Monotonic footpath gate preserving source raster topology and uncertainty.

Mask labels remain unverified model evidence, not legal ground truth. This gate
can only withhold existing footpath geometry; it never creates a polygon or a
violation. Other scene fields and deterministic measurements are preserved.
"""
from collections import deque
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from math import ceil, floor, hypot, isfinite

import numpy as np

from .contracts import Rule
from .geometry import point_in_polygon


POLICY = "raster_footpath_contact_v1"
WINDOW = 1.5
GAP = .75


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


class RasterFootpathSupport:
    def __init__(self, road_mask, sidewalk_mask, *, frame_shape, scene_id, revision, scores,
                 max_tracks=512, max_samples=256):
        if type(max_tracks) is not int or max_tracks < 1 or type(max_samples) is not int or max_samples < 2:
            raise ValueError("Positive track capacity and at least two samples are required")
        self.max_tracks, self.max_samples = max_tracks, max_samples
        self.scene_id, self.revision = scene_id, revision
        self.scores = deepcopy(scores) if isinstance(scores, dict) else {}
        self.errors, self.hashes, self._tracks = [], {}, {}
        self._polygon_signature = None
        shape_valid = (isinstance(frame_shape, (tuple, list)) and len(frame_shape) == 2
                       and all(type(v) is int and v > 0 for v in frame_shape))
        self.shape = tuple(frame_shape) if shape_valid else None
        if not shape_valid:
            self.errors.append("invalid_processed_frame_shape")
        if type(scene_id) is not int or scene_id < 0 or type(revision) is not int or revision < 1:
            self.errors.append("invalid_scene_binding")
        self._road = self._mask(road_mask, "road")
        self._sidewalk = self._mask(sidewalk_mask, "sidewalk")

    def _mask(self, value, name):
        if (not isinstance(value, np.ndarray) or value.shape != self.shape
                or value.dtype not in (np.dtype(bool), np.dtype(np.uint8))
                or not np.isin(value, (0, 1, 255)).all()):
            self.errors.append(name + "_mask_unavailable_or_invalid")
            return None
        mask = (value > 0).copy()
        mask.flags.writeable = False
        header = json.dumps({"shape": list(mask.shape), "encoding": "boolean_support"}, sort_keys=True).encode()
        self.hashes[name] = hashlib.sha256(header + mask.tobytes()).hexdigest()
        return mask

    def source_snapshot(self):
        return {"policy": POLICY, "frame_shape": list(self.shape) if self.shape else None,
                "scene_id": self.scene_id, "revision": self.revision, "errors": self.errors[:],
                "semantic_mask_sha256": self.hashes.copy(), "scores": deepcopy(self.scores),
                "polygon_signature_sha256": self._polygon_signature,
                "scope": "Source raster support only; semantic/legal boundary accuracy unverified"}

    def spatial_support(self, vehicle, polygons):
        trace = {"status": "withheld", "reason": "mask_source_unavailable", "category": None}
        if self.errors:
            return trace
        box = vehicle.box
        if len(box) != 4 or not all(_finite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            return {**trace, "reason": "invalid_vehicle_box"}
        point = vehicle.ground_point
        radius = max(1, ceil(.05 * (box[2] - box[0])))
        x, y = floor(point[0]), floor(point[1])
        left, top, right, bottom = x - radius, y - radius, x + radius + 1, y + radius + 1
        trace.update(ground_point=list(point), contact_radius_pixels=radius,
                     contact_patch=[left, top, right, bottom])
        if left < 0 or top < 0 or right > self.shape[1] or bottom > self.shape[0]:
            return {**trace, "reason": "contact_patch_outside_processed_frame"}
        road, sidewalk = self._road[top:bottom, left:right], self._sidewalk[top:bottom, left:right]
        cells = road.size
        trace.update(patch_pixels=cells, road_pixels=int(road.sum()), sidewalk_pixels=int(sidewalk.sum()),
                     unknown_pixels=int(np.count_nonzero(~road & ~sidewalk)),
                     conflicting_pixels=int(np.count_nonzero(road & sidewalk)))
        try:
            inside = any(point_in_polygon(point, polygon) for polygon in polygons)
        except (TypeError, ValueError, IndexError):
            return {**trace, "reason": "invalid_footpath_geometry"}
        trace["inside_existing_footpath_polygon"] = inside
        if not polygons:
            return {**trace, "reason": "footpath_polygons_unavailable"}
        if trace["conflicting_pixels"]:
            return {**trace, "reason": "contradictory_semantic_masks"}
        if sidewalk.all() and inside:
            return {**trace, "status": "supported", "category": "sidewalk", "reason": "original_sidewalk_patch_supported"}
        if road.all() and not inside:
            return {**trace, "status": "supported", "category": "road", "reason": "original_road_patch_supported"}
        if sidewalk.all() or road.all():
            return {**trace, "reason": "polygon_raster_disagreement"}
        return {**trace, "reason": "boundary_or_unresolved_contact_patch"}

    def scene_for(self, vehicle, timestamp, scene, *, video, scene_revision):
        trace = {"source": self.source_snapshot(), "timestamp": timestamp if _finite(timestamp) else None,
                 "status": "withheld", "reason": "context_unavailable",
                 "source_provenance": {key: deepcopy(scene.source_evidence[key]) for key in ("synthetic", "provenance")
                                       if key in scene.source_evidence} or {"status": "not_recorded"}}
        identity = vehicle.track_id
        eligible_id = type(identity) is int and identity >= 0
        stable = (scene.stationary_camera and _finite(scene.confidence) and scene.confidence >= .8)
        scores_ready = all(_finite(self.scores.get(k)) and .8 <= self.scores[k] <= 1 for k in ("road", "sidewalk"))
        reason = ("not_a_two_wheeler" if vehicle.label not in ("motorcycle", "bicycle")
                  else "video_required" if not video else "tracked_identity_required" if not eligible_id
                  else "invalid_timestamp" if not _finite(timestamp) or timestamp < 0
                  else "scene_revision_mismatch" if scene.scene_id != self.scene_id or scene_revision != self.revision
                  else "camera_or_scene_unresolved" if not stable else "mask_source_unavailable" if self.errors
                  else "semantic_scores_unresolved" if not scores_ready else None)
        if not stable or reason == "scene_revision_mismatch":
            self._tracks.clear()
        if reason is not None:
            if eligible_id:
                self._tracks.pop(identity, None)
            trace["reason"] = reason
        else:
            try:
                signature = hashlib.sha256(json.dumps(scene.footpath_polygons, sort_keys=True, allow_nan=False).encode()).hexdigest()
            except (TypeError, ValueError):
                self._tracks.clear()
                trace["reason"] = "invalid_footpath_geometry"
                evidence = {**scene.source_evidence, "footpath_contact_support": trace}
                return replace(scene, footpath_polygons=[], source_evidence=evidence)
            if signature != self._polygon_signature:
                self._tracks.clear()
                self._polygon_signature = signature
            trace["source"]["polygon_signature_sha256"] = signature
            spatial = self.spatial_support(vehicle, scene.footpath_polygons)
            trace["contact"] = spatial
            self._tracks = {key: value for key, value in self._tracks.items() if timestamp - value["last"] <= WINDOW + GAP}
            state = self._tracks.get(identity)
            if state and timestamp <= state["last"]:
                self._tracks.pop(identity, None)
                trace["reason"] = "nonincreasing_timestamp"
            else:
                if state and (timestamp - state["last"] > GAP or
                              hypot(vehicle.ground_point[0] - state["point"][0], vehicle.ground_point[1] - state["point"][1])
                              > 5 * max(8., vehicle.box[2] - vehicle.box[0])):
                    state = None
                if state is None:
                    state = {"first": timestamp, "samples": deque(maxlen=self.max_samples)}
                    self._tracks[identity] = state
                state.update(last=timestamp, point=vehicle.ground_point)
                state["samples"].append((timestamp, spatial["status"] == "supported", spatial["category"]))
                cutoff = timestamp - WINDOW
                while len(state["samples"]) > 1 and state["samples"][1][0] <= cutoff:
                    state["samples"].popleft()
                recent = [p for p in state["samples"] if p[0] >= cutoff]
                coverage = state["first"] <= cutoff and state["samples"][0][0] <= cutoff
                trace["history"] = {"first_observed_at": state["first"], "window_seconds": WINDOW,
                                    "retained_samples": len(state["samples"]), "recent_samples": len(recent),
                                    "unresolved_recent_samples": sum(not p[1] for p in recent), "covers_full_window": coverage}
                if not coverage:
                    trace["reason"] = "full_contact_window_not_observed"
                elif not recent or not all(p[1] for p in recent):
                    trace["reason"] = "unresolved_contact_history"
                else:
                    trace.update(status="supported", reason="source_supported_contact_window")
                if len(self._tracks) > self.max_tracks:
                    oldest = min(self._tracks, key=lambda key: self._tracks[key]["last"])
                    self._tracks.pop(oldest)
        evidence = {**scene.source_evidence, "footpath_contact_support": trace}
        return replace(scene, footpath_polygons=scene.footpath_polygons if trace["status"] == "supported" else [],
                       source_evidence=evidence)


def annotate_footpath_support(decisions, scene):
    """Add source trace without changing verdicts, reasons or other rule evidence."""
    trace = scene.source_evidence.get("footpath_contact_support")
    return [replace(d, evidence={**deepcopy(d.evidence), "footpath_contact_support": deepcopy(trace)})
            if d.rule == Rule.FOOTPATH else d for d in decisions]
