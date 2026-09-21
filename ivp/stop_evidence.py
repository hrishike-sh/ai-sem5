"""Bounded, observed road-position evidence for a low-motion stop interval."""
from collections import deque
import math

import numpy as np
from .timing import FrameTiming


class ObservedStopWindow:
    """Use actual observations, without backdating a smoothed speed threshold.

    The caller must reset on any missing/uncertain observation or approach exit.
    A positive result additionally needs a current accepted calibrated speed.
    """
    seconds = 1.
    maximum_kmh = 2.
    maximum_diameter_m = maximum_kmh / 3.6 * seconds

    def __init__(self, frame_interval=1/30):
        self.samples = deque(maxlen=256)
        self.window_seconds = max(self.seconds, 7*FrameTiming(frame_interval).interval)

    def update(self, stamp, point, *, accepted_speed):
        self.samples.append((float(stamp), np.asarray(point, dtype=float)))
        # Keep the one real observation bracketing the start of a full second.
        # Never interpolate a position or infer a stop through missing frames.
        while len(self.samples) > 1 and self.samples[1][0] <= stamp-self.window_seconds:
            self.samples.popleft()
        span = stamp-self.samples[0][0]
        if len(self.samples) < 8 or span < self.seconds-1e-6:
            return None
        if (accepted_speed is None or not math.isfinite(accepted_speed)
                or not 0 <= accepted_speed <= self.maximum_kmh):
            return None
        points = np.array([p for _, p in self.samples])
        # The road-coordinate bounding-box diagonal is a conservative upper
        # bound on any pairwise displacement, in linear time and bounded memory.
        diameter = float(np.linalg.norm(points.max(axis=0)-points.min(axis=0)))
        times = np.array([t for t, _ in self.samples])
        times -= times.mean()
        velocity = (times[:, None]*(points-points.mean(axis=0))).sum(axis=0)/(times*times).sum()
        fitted_kmh = float(np.linalg.norm(velocity)*3.6)
        if diameter > self.maximum_diameter_m or fitted_kmh > self.maximum_kmh:
            return None
        return {'method': 'Continuous observed calibrated bottom-centre positions; bounded displacement and fitted velocity',
                'start_seconds': self.samples[0][0], 'end_seconds': float(stamp),
                'duration_seconds': float(span), 'observed_samples': len(self.samples),
                'position_diameter_bound_m': diameter, 'maximum_position_diameter_m': self.maximum_diameter_m,
                'fitted_speed_kmh': fitted_kmh, 'maximum_fitted_speed_kmh': self.maximum_kmh}
