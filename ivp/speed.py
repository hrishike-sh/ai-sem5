"""Road-plane calibration and timestamp-based region speed estimates."""
from collections import Counter, deque

import cv2
import numpy as np

from .timing import FrameTiming


class RoadCalibration:
    def __init__(self, config):
        self.config = config
        try:
            points = np.asarray(config['points'], dtype=np.float32)
            width, length = float(config['width_m']), float(config['length_m'])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Calibration needs four road corners and measured width/length in metres.') from exc
        if (points.shape != (4, 2) or not np.isfinite(points).all() or
                np.any(points < 0) or np.any(points > 1) or
                not .1 <= width <= 1000 or not .1 <= length <= 1000):
            raise ValueError('Use four normalized image points and road dimensions between 0.1 and 1000 metres.')
        if not cv2.isContourConvex(points) or abs(cv2.contourArea(points)) < .005:
            raise ValueError('Road corners must form a convex quadrilateral in perimeter order, covering at least 0.5% of the image.')
        self.points = points
        self.matrix = cv2.getPerspectiveTransform(points, np.float32([[0,0],[width,0],[width,length],[0,length]]))

    def project(self, point, shape):
        h, w = shape
        p = np.asarray(point, np.float32) / [w, h]
        if cv2.pointPolygonTest(self.points, tuple(map(float, p)), False) < 0:
            return None  # Never extrapolate beyond the measured road patch.
        result = cv2.perspectiveTransform(p.astype(np.float32).reshape(1,1,2), self.matrix)[0,0]
        return result if np.isfinite(result).all() else None


class SpeedEstimator:
    def __init__(self, calibration=None, frame_interval=1/30):
        self.timing = FrameTiming(frame_interval)
        self.calibration = RoadCalibration(calibration) if calibration is not None else None
        self.accepted = 0
        self.sum_speed = 0.
        self.maximum = 0.
        self.invalid_reason = None
        self.position_jumps = 0
        self.status_counts = Counter()

    def invalidate(self):
        if self.calibration:
            self.invalid_reason = 'camera/scene changed; calibration no longer valid'

    def update(self, track, stamp, shape):
        self._update(track, stamp, shape)
        self.status_counts[track['speed_status']] += 1

    def _update(self, track, stamp, shape):
        history = track.setdefault('speed_history', deque(maxlen=180))
        x,y,w,h = track['box']
        point = np.array([x+w/2, y+h], dtype=float)
        # The window describes this estimate, so a withheld observation must
        # not inherit the last accepted estimate's metadata.
        track.update(speed=None, speed_window_seconds=None,
                     speed_unit='km/h' if self.calibration else 'px/s', speed_status='collecting stable track')
        if track.get('speed_blocked'):
            history.clear();track['speed_status']=track['speed_blocked'];return
        if self.invalid_reason:
            history.clear(); track['speed_status']=self.invalid_reason; return
        if self.calibration:
            point = self.calibration.project(point, shape)
            if point is None:
                history.clear(); track['speed_status']='outside calibrated road'; return
        if history:
            elapsed = stamp-history[-1][0]
            ratio = (w*h)/history[-1][2]
            if elapsed <= 0 or elapsed > self.timing.discontinuity or not .67 <= ratio <= 1.5:
                history.clear()
        history.append((stamp, point, w*h))
        while history and stamp-history[0][0] > self.timing.speed_window + 1e-6:
            history.popleft()
        if len(history) < 8 or stamp-history[0][0] < .5:
            return
        if self._position_jump(history):
            history.clear()
            self.position_jumps += 1
            track['speed_status'] = 'abrupt position change; collecting a new speed window'
            return
        t = np.array([p[0] for p in history]); xy = np.array([p[1] for p in history])
        t -= t.mean()
        velocity = (t[:,None]*(xy-xy.mean(axis=0))).sum(axis=0)/(t*t).sum()
        residual = np.median(np.linalg.norm(xy-(xy.mean(axis=0)+t[:,None]*velocity), axis=1))
        distance = np.linalg.norm(velocity)*(t[-1]-t[0])
        tolerance = .25 if self.calibration else 2.
        if residual > max(tolerance, distance*.18):
            track['speed_status']='unstable trajectory'; return
        speed = float(np.linalg.norm(velocity))*(3.6 if self.calibration else 1.)
        if self.calibration and speed > 160:
            track['speed_status']='implausible estimate'; return
        track.update(speed=round(speed,2), speed_status='estimated', speed_window_seconds=round(float(t[-1]-t[0]),3))
        self.accepted += 1; self.sum_speed += speed; self.maximum = max(self.maximum, speed)

    def _position_jump(self, history):
        # A median residual over the full window can hide one bad latest point.
        # Predict it using only earlier observations. Local curvature permits
        # smooth acceleration/turning; this is a measurement gate, not a new ID.
        previous = list(history)[:-1][-8:]
        times = np.array([p[0] for p in previous])
        points = np.array([p[1] for p in previous])
        span = times[-1] - times[0]
        normalized = (times-times[-1]) / span
        design = np.c_[np.ones(len(times)), normalized, normalized**2]
        coefficients = np.linalg.lstsq(design, points, rcond=None)[0]
        ahead = (history[-1][0]-times[-1]) / span
        predicted = np.array([1., ahead, ahead**2]) @ coefficients
        noise = np.median(np.linalg.norm(points-design @ coefficients, axis=1))
        step = np.linalg.norm(predicted-points[-1])
        floor = .5 if self.calibration else 4.
        return np.linalg.norm(history[-1][1]-predicted) > max(floor, 4*noise, .35*step)

    def summary(self):
        return {'unit':'km/h' if self.calibration else 'px/s', 'calibrated':self.calibration is not None,
                'invalid_reason':self.invalid_reason,
                'calibration':self.calibration.config if self.calibration else None,
                'accepted_track_samples':self.accepted,
                'observed_track_samples':sum(self.status_counts.values()),
                'withheld_track_samples':sum(self.status_counts.values())-self.accepted,
                'status_counts':dict(self.status_counts),
                'sample_accounting':'One outcome per observed track sample sent to this estimator; frames with no visible track are not included. Counts are not vehicles or an accuracy score.',
                'rejected_position_jumps':self.position_jumps,
                'sample_mean':self.sum_speed/self.accepted if self.accepted else None,
                'maximum':self.maximum if self.accepted else None,
                'nominal_frame_interval_seconds':self.timing.interval,
                'maximum_window_seconds':self.timing.speed_window,
                'method':f'Least-squares velocity over at least 8 observed bottom-centre positions and 0.5 seconds, up to {self.timing.speed_window:g} seconds; local quadratic prediction rejects abrupt position changes; real presentation timestamps.',
                'meaning':'Region-motion estimate, not a validated vehicle speed. Calibration assumes a flat rectangular road patch with measured dimensions and real-time footage (not slow motion or time-lapse).'}
