"""Invalidate automatic scene context when camera tracking changes reference.

The existing camera's measured return values are preserved. A separate refresh
flag prevents an unquantified reference loss from silently reviving old masks,
sign applicability or approach geometry after tracking becomes stable again.
"""
from copy import deepcopy
from math import isfinite

from .segmentation import CameraStability


class RefreshAwareCameraStability(CameraStability):
    """Existing camera observations plus an explicit scene-refresh handshake.

Call acknowledge_scene after scene inference succeeds. Once invalidated, scene
context is unavailable until the camera is stable and a newer revision is
acknowledged. The camera observation can be stable while context_ready is false;
stability of the new reference does not validate geometry from the old one.
"""

    def __init__(self):
        super().__init__()
        self._reset_context()

    def _reset_context(self):
        self._pending = False
        self._pending_since = None
        self._reasons = set()
        self._last_stable = False
        self._last_timestamp = None
        self._binding = None
        self._last_observation = None
        self._requests = 0
        self._completed_refreshes = 0

    def reset(self):
        super().reset()
        self._reset_context()

    @property
    def scene_refresh_required(self):
        return self._pending

    @property
    def context_ready(self):
        return self._last_stable and not self._pending and self._binding is not None

    def _invalidate(self, timestamp, reason):
        if not self._pending:
            self._pending_since = timestamp
            self._requests += 1
        self._pending = True
        self._reasons.add(reason)

    def observe(self, rgb, timestamp, dynamic_boxes=()):
        valid_time = (type(timestamp) in (int, float) and isfinite(timestamp) and timestamp >= 0)
        if not valid_time or self._last_timestamp is not None and timestamp <= self._last_timestamp:
            self._invalidate(timestamp if valid_time else None,
                             "nonincreasing_timestamp" if valid_time else "invalid_timestamp")
            super().reset()
            self._last_stable = False
            self._last_timestamp = timestamp if valid_time else None
            self._last_observation = {"timestamp": self._last_timestamp, "stable": False,
                                      "measured_displacement": None, "reason": "timestamp_discontinuity"}
            return False, None
        before = self.reference
        stable, displacement = super().observe(rgb, timestamp, dynamic_boxes)
        rebased = before is not None and self.reference is not before
        if rebased:
            self._invalidate(timestamp, "unquantified_reference_loss" if displacement is None else "camera_reference_rebased")
        elif self._last_stable and not stable:
            self._invalidate(timestamp, "camera_stability_lost")
        self._last_stable = bool(stable)
        self._last_timestamp = timestamp
        self._last_observation = {"timestamp": timestamp, "stable": bool(stable),
                                  "measured_displacement": displacement, "reference_rebased": rebased}
        return stable, displacement

    def acknowledge_scene(self, scene_id, revision):
        """Acknowledge successful scene inference, never a request to skip it.

        Initial scene proposals can be registered before camera warmup. A pending
        refresh cannot be acknowledged while unstable or with the old revision.
        """
        if type(scene_id) is not int or scene_id < 0 or type(revision) is not int or revision < 1:
            raise ValueError("Scene identity and positive revision must be integers")
        if self._pending:
            newer = (self._binding is None or scene_id != self._binding[0] or revision > self._binding[1])
            if not self._last_stable or not newer:
                return False
            self._completed_refreshes += 1
        elif self._binding is not None and scene_id == self._binding[0] and revision < self._binding[1]:
            return False
        self._binding = (scene_id, revision)
        self._pending = False
        self._pending_since = None
        self._reasons.clear()
        return True

    def lifecycle_snapshot(self):
        return {"policy": "refresh_after_camera_reference_loss_v1", "context_ready": self.context_ready,
                "scene_refresh_required": self.scene_refresh_required, "pending_since": self._pending_since,
                "reasons": sorted(self._reasons), "scene_binding": list(self._binding) if self._binding else None,
                "refresh_requests": self._requests, "completed_refreshes": self._completed_refreshes,
                "last_camera_observation": deepcopy(self._last_observation),
                "scope": "Camera-reference continuity only; no scene semantics, calibration or model accuracy inferred"}
