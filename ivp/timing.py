"""Cadence-aware observation limits; velocity always uses actual timestamps."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class FrameTiming:
    interval: float = 1 / 30

    def __post_init__(self):
        if (isinstance(self.interval, bool) or not isinstance(self.interval, (int, float))
                or not math.isfinite(self.interval) or self.interval <= 0):
            raise ValueError('Frame interval must be a positive finite number of seconds.')

    @property
    def discontinuity(self):
        return max(.2, 1.5 * self.interval) + 1e-6

    @property
    def recovery(self):
        return max(.1, 1.5 * self.interval) + 1e-6

    @property
    def speed_window(self):
        # Preserve eight observed samples, even on low-frame-rate CCTV.
        return max(1.2, 8 * self.interval)
