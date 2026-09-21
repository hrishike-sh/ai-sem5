from collections import deque
from dataclasses import asdict, dataclass
from math import atan2, degrees

import numpy as np


@dataclass
class TurnObservation:
    track_id: int
    timestamp: float
    angle_degrees: float | None = None
    projected_motion: str = "unknown"
    rule_turn: str | None = None
    entry_heading_ended_at: float | None = None
    exit_heading_started_at: float | None = None
    reason: str = "Insufficient tracked movement"

    def to_dict(self):
        return asdict(self)


class Trajectories:
    def __init__(self, horizon=12., max_samples=300):
        self.horizon = horizon
        self.max_samples = max_samples
        self.tracks = {}
        self.calibrations = {}

    def reset(self):
        self.tracks.clear()
        self.calibrations.clear()

    def observe(self, detection, timestamp, stable, road_transform=None):
        if detection.track_id is None:
            return None
        identity = detection.track_id
        observation = TurnObservation(identity, timestamp)
        transform = None
        signature = None
        if road_transform is not None:
            transform = np.asarray(road_transform, dtype=float)
            if transform.shape != (3, 3) or not np.isfinite(transform).all():
                raise ValueError("Road-plane transform must be a finite, invertible 3 by 3 matrix")
            scale = np.max(np.abs(transform))
            if scale == 0:
                raise ValueError("Road-plane transform must be a finite, invertible 3 by 3 matrix")
            transform = transform / scale
            if abs(np.linalg.det(transform)) < 1e-12:
                raise ValueError("Road-plane transform must be a finite, invertible 3 by 3 matrix")
            transform *= np.sign(transform.flat[np.flatnonzero(transform)[0]])
            signature = tuple(transform.flat)
        if not stable:
            self.tracks.pop(identity, None)
            self.calibrations.pop(identity, None)
            observation.reason = "Camera is moving or stability is unresolved"
            return observation
        if identity in self.calibrations and self.calibrations[identity] != signature:
            self.tracks.pop(identity, None)
        self.calibrations[identity] = signature
        track = self.tracks.setdefault(identity, deque(maxlen=self.max_samples))
        point = detection.ground_point
        extent = max(8., detection.box[2] - detection.box[0])
        if track:
            elapsed = timestamp - track[-1][0]
            if elapsed <= 0:
                raise ValueError("Trajectory timestamps must increase")
            if elapsed > .75 or np.linalg.norm(np.asarray(point) - track[-1][1]) > 5 * extent:
                track.clear()
        track.append((timestamp, point))
        while track and timestamp - track[0][0] > self.horizon:
            track.popleft()
        self.tracks = {key: value for key, value in self.tracks.items() if timestamp - value[-1][0] <= self.horizon}
        self.calibrations = {key: self.calibrations[key] for key in self.tracks}
        if len(track) < 8 or timestamp - track[0][0] < 1.:
            return observation
        times = np.asarray([sample[0] for sample in track])
        points = np.asarray([sample[1] for sample in track], dtype=float)
        pixel_points = points.copy()
        if transform is not None:
            homogeneous = np.c_[points, np.ones(len(points))] @ transform.T
            denominators = homogeneous[:, 2]
            if np.any(np.abs(denominators) < 1e-8) or (denominators.min() < 0 < denominators.max()):
                observation.reason = "Trajectory intersects the road-plane horizon"
                return observation
            points = homogeneous[:, :2] / homogeneous[:, 2, None]
        count = max(3, len(points) // 3)
        if min(np.linalg.norm(pixel_points[count - 1] - pixel_points[0]), np.linalg.norm(pixel_points[-1] - pixel_points[-count])) < .5 * extent:
            observation.reason = "Entry or exit movement is too small relative to the vehicle"
            return observation

        def velocity(t, p):
            centred = t - t.mean()
            return (centred[:, None] * (p - p.mean(axis=0))).sum(axis=0) / max(1e-9, (centred ** 2).sum())

        entry, exit = velocity(times[:count], points[:count]), velocity(times[-count:], points[-count:])
        if min(np.linalg.norm(entry), np.linalg.norm(exit)) < 1e-8:
            return observation
        angle = degrees(atan2(entry[0] * exit[1] - entry[1] * exit[0], np.dot(entry, exit)))
        observation.angle_degrees = angle
        observation.entry_heading_ended_at = float(times[count - 1])
        observation.exit_heading_started_at = float(times[-count])
        observation.projected_motion = "straight" if abs(angle) < 25 else "reversal" if abs(angle) > 140 else "counterclockwise" if angle > 45 else "clockwise" if angle < -45 else "uncertain_curve"
        observation.reason = "Entry and exit headings measured in the image plane; road-plane calibration is required for legal turn interpretation"
        if road_transform is not None:
            observation.rule_turn = "straight" if abs(angle) < 25 else "u_turn" if abs(angle) > 140 else "left" if angle > 45 else "right" if angle < -45 else None
            observation.reason = "Entry and exit headings measured in a calibrated right-handed road plane"
        return observation
