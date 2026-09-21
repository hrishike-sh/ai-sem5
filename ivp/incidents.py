"""Review candidates from observed trajectories and explicitly supplied scene rules.

No signal recognition, legal judgement, or inference of missing positions occurs here.
Coordinates belong to the stabilized first-frame image, normalized to its dimensions.
"""
from collections import Counter
from copy import deepcopy
import math

from traffic_ai.geometry import crossed_segment, signed_distance
from .timing import FrameTiming
from .speed import RoadCalibration
from .stop_evidence import ObservedStopWindow


STATES = {'observe', 'red', 'green', 'stop_required', 'unknown'}


def validate_config(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('Incident settings must be an object.')
    allowed = {'enabled', 'line', 'direction', 'zone', 'zone_role', 'approach_zone', 'condition',
               'condition_intervals', 'speed_limit_kmh', 'calibration_confirmed',
               'real_time_confirmed', 'synthetic'}
    if set(value) - allowed:
        raise ValueError('Unknown incident settings: '+', '.join(sorted(set(value)-allowed)))
    config = {'enabled': False, 'line': None, 'direction': 'both', 'zone': None,
              'zone_role': 'observe', 'approach_zone': None, 'condition': 'observe',
              'condition_intervals': [], 'speed_limit_kmh': None,
              'calibration_confirmed': False, 'real_time_confirmed': False, 'synthetic': False,
              **deepcopy(value)}
    for name in ('enabled', 'calibration_confirmed', 'real_time_confirmed', 'synthetic'):
        if type(config[name]) is not bool:
            raise ValueError(name+' must be true or false.')
    def number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    line = config['line']
    if line is not None:
        if (not isinstance(line, list) or len(line) != 2 or
                any(not isinstance(p, list) or len(p) != 2 or
                    any(not number(v) or not 0 <= v <= 1 for v in p) for p in line) or
                math.dist(*line) < .02):
            raise ValueError('Incident line needs two distinct normalized endpoints at least 2% apart.')
    for name in ('zone', 'approach_zone'):
        rect = config[name]
        if rect is not None and (not isinstance(rect, list) or len(rect) != 4 or
                any(not number(v) or not 0 <= v <= 1 for v in rect) or
                rect[2]-rect[0] < .02 or rect[3]-rect[1] < .02):
            raise ValueError(name+' needs [left, top, right, bottom] normalized coordinates, at least 2% wide/high.')
    if not isinstance(config['direction'],str) or config['direction'] not in ('positive', 'negative', 'both'):
        raise ValueError('Incident direction must be positive, negative or both.')
    if (not isinstance(config['condition'],str) or config['condition'] not in STATES or
            not isinstance(config['zone_role'],str) or config['zone_role'] not in ('observe', 'restricted')):
        raise ValueError('Choose a supported condition and zone role.')
    intervals = config['condition_intervals']
    if not isinstance(intervals, list) or len(intervals) > 100:
        raise ValueError('Use at most 100 condition intervals.')
    end = -1
    for item in intervals:
        if (not isinstance(item, dict) or set(item) != {'start', 'end', 'state'} or
                not number(item['start']) or not number(item['end']) or
                not 0 <= item['start'] < item['end'] or item['start'] < end or
                not isinstance(item['state'],str) or item['state'] not in STATES):
            raise ValueError('Condition intervals must be sorted, nonoverlapping {start, end, state} entries in seconds.')
        end = item['end']
    limit = config['speed_limit_kmh']
    if limit is not None and (not number(limit) or not 1 <= limit <= 160):
        raise ValueError('Speed threshold must be between 1 and 160 km/h.')
    if config['enabled'] and line is None and config['zone'] is None and limit is None:
        raise ValueError('Enable a line, zone or speed threshold for incident review.')
    return config


class IncidentAnalyzer:
    """Bounded per-track state; the caller streams returned events to disk."""
    def __init__(self, config, calibration=None, frame_interval=1/30):
        self.config = validate_config(config)
        self.enabled = bool(self.config and self.config['enabled'])
        self.calibration = deepcopy(calibration)
        self.road_calibration = RoadCalibration(calibration) if calibration else None
        self.timing = FrameTiming(frame_interval)
        self.tracks = {}
        self.counts = Counter()
        self.serial = 0
        self.invalid_reason = None
        self.evicted = 0
        self.frame_index = 0
        self.excluded_geometry_frames = 0

    def condition(self, stamp):
        if self.config['condition_intervals']:
            return next((v['state'] for v in self.config['condition_intervals']
                         if v['start'] <= stamp < v['end']), 'unknown')
        return self.config['condition']

    @staticmethod
    def inside(point, rect, margin=0):
        return rect is not None and rect[0]+margin <= point[0] <= rect[2]-margin and rect[1]+margin <= point[1] <= rect[3]-margin

    def metric_reason(self, track):
        c = self.config
        if not self.calibration or not c['calibration_confirmed']:
            return 'Measured road calibration has not been supplied and confirmed.'
        if not c['real_time_confirmed']:
            return 'Real-time recording has not been confirmed.'
        if track.get('speed_status') != 'estimated' or track.get('speed_unit') != 'km/h' or track.get('speed') is None:
            return 'No reliable calibrated speed at this observation: '+str(track.get('speed_status', 'unavailable'))
        return None

    def event(self, key, track, stamp, kind, status, reason, point, **measurements):
        self.serial += 1
        self.counts[status] += 1
        return {'event_id': f'incident-{self.serial:06d}', 'timestamp_seconds': round(float(stamp), 6),
                'track_id': key, 'type': kind, 'status': status, 'reason': reason,
                'synthetic': self.config['synthetic'],
                'basis': 'Synthetic declared scale and conditions' if self.config['synthetic'] else 'User-configured scene conditions; requires human review',
                'quality': {'tracking_state': track.get('tracking_state'), 'detector_confidence': track.get('confidence'),
                            'confidence_meaning': 'Detector score, not probability that an incident occurred.',
                            'speed_status': track.get('speed_status'), 'speed_window_seconds': track.get('speed_window_seconds')},
                'measurements': {'speed': track.get('speed'), 'speed_unit': track.get('speed_unit'), **measurements},
                'coordinates': {'frame': 'stabilized first-frame image', 'anchor_normalized': list(map(float, point)),
                                'bbox_analysis_xywh': list(map(float, track['box']))},
                'configuration': deepcopy(self.config), 'calibration': self.calibration,
                'thumbnail': None}

    def step(self, visible, stamp, shape, *, registration_valid=True, scene_changed=False):
        if not self.enabled:
            return []
        self.frame_index += 1
        events = []
        if scene_changed:
            self.invalid_reason = 'Camera/scene changed; configured geometry must be set again.'
        if not registration_valid or self.invalid_reason:
            self.excluded_geometry_frames += 1
            self.tracks.clear()
            return events
        self.tracks = {k: s for k, s in self.tracks.items() if stamp-s['time'] <= max(2, self.timing.discontinuity*2)}
        h, w = shape
        line = self.config['line']
        pixel_line = [[p[0]*w, p[1]*h] for p in line] if line else None
        for key, track in visible:
            if key == 'SELECTED':
                continue
            x,y,bw,bh = track['box']
            point = ((x+bw/2)/w, (y+bh)/h)
            pixel = (point[0]*w, point[1]*h)
            state = self.tracks.get(key)
            if state is None and len(self.tracks) >= 512:
                oldest = min(self.tracks, key=lambda k: self.tracks[k]['time'])
                del self.tracks[oldest]; self.evicted += 1
            bad = (track.get('speed_blocked') or not track.get('confirmed', False) or
                   str(track.get('speed_status') or '').startswith('abrupt position change') or
                   track.get('tracking_state') != 'observed' or float(track.get('confidence') or 0) < .35 or
                   x <= 1 or y <= 1 or x+bw >= w-1 or y+bh >= h-1)
            gap = state and (state['frame_index'] != self.frame_index-1 or
                             stamp <= state['time'] or stamp-state['time'] > self.timing.discontinuity)
            if bad or gap:
                prior_sent = state['sent'] if state else set()
                if 'quality' not in prior_sent:
                    events.append(self.event(key, track, stamp, 'track_quality', 'withheld',
                        'Uncertain, clipped or discontinuous track; crossing and speed history reset.', point))
                    prior_sent.add('quality')
                state = {'time': stamp, 'sent': prior_sent, 'frame_index': self.frame_index}
                self.tracks[key] = state
                if bad:
                    continue
            if state is None:
                if len(self.tracks) >= 512:
                    oldest = min(self.tracks, key=lambda k: self.tracks[k]['time'])
                    del self.tracks[oldest]; self.evicted += 1
                state = {'time': stamp, 'sent': set(), 'frame_index': self.frame_index}
                self.tracks[key] = state
            state['time'] = stamp
            state['frame_index'] = self.frame_index
            metric_reason = self.metric_reason(track)
            # Stop evidence is accumulated only while reliably observed inside the approach.
            if self.inside(point, self.config['approach_zone']):
                if not state.get('in_approach'):
                    state['approach_entered'] = bool(state.get('approach_seen_outside'))
                    state['approach_complete'] = True
                    for name in ('metric_start', 'stopped', 'stop_evidence'):
                        state.pop(name, None)
                    state['stop_window'] = ObservedStopWindow(self.timing.interval)
                state['in_approach'] = True
                if metric_reason:
                    state['approach_complete'] = False
                    state.pop('metric_start', None)
                else:
                    state.setdefault('metric_start', stamp)
                # A decelerating speed window can be temporarily unstable even
                # while every calibrated position remains continuously seen.
                # Keep those positions, but publish proof only with an accepted
                # low-speed estimate and a complete observed low-motion window.
                eligible = (self.road_calibration is not None and self.config['calibration_confirmed']
                            and self.config['real_time_confirmed'] and track.get('speed_status') in
                            ('estimated', 'collecting stable track', 'unstable trajectory'))
                ground = self.road_calibration.project(pixel, shape) if eligible else None
                if ground is None:
                    state['stop_window'] = ObservedStopWindow(self.timing.interval)
                else:
                    proof = state['stop_window'].update(stamp, ground,
                        accepted_speed=track['speed'] if metric_reason is None else None)
                    if proof and not state.get('stopped'):
                        state['stopped'] = True
                        state['stop_evidence'] = proof
            else:
                state['approach_seen_outside'] = True
                state['in_approach'] = False
                state.pop('stop_window', None)
            if pixel_line:
                distance = signed_distance(pixel, pixel_line)
                side = 1 if distance > 3 else -1 if distance < -3 else 0
                anchor = state.get('anchor')
                if side and anchor and anchor['side'] != side:
                    direction = 'positive' if side > 0 else 'negative'
                    a,b = (anchor['pixel'], pixel) if side > 0 else (pixel, anchor['pixel'])
                    if (self.config['direction'] in (direction, 'both') and
                            crossed_segment(a, b, pixel_line, 3) and 'line' not in state['sent']):
                        da = signed_distance(anchor['pixel'], pixel_line)
                        fraction = abs(da)/(abs(da)+abs(distance))
                        crossed_at = anchor['time']+fraction*(stamp-anchor['time'])
                        condition = self.condition(crossed_at)
                        # A transition within the observed crossing bracket is ambiguous.
                        boundaries = [v[k] for v in self.config['condition_intervals'] for k in ('start','end')]
                        if any(anchor['time'] < t <= stamp for t in boundaries):
                            condition = 'unknown'
                        status, reason = self.crossing_decision(condition, state, stamp, metric_reason)
                        events.append(self.event(key, track, crossed_at, 'line_crossing', status, reason, point,
                            condition=condition, direction=direction, observation_timestamp_seconds=stamp,
                            crossing_bracket_seconds=[anchor['time'],stamp], stopped_in_approach=bool(state.get('stopped')),
                            stop_evidence=deepcopy(state.get('stop_evidence'))))
                        state['sent'].add('line')
                    for name in ('metric_start', 'stopped', 'stop_evidence'):
                        state.pop(name, None)
                if side:
                    state['anchor'] = {'pixel': pixel, 'time': stamp, 'side': side}
            zone = self.config['zone']
            if zone:
                inner = self.inside(point, zone, .005)
                outer = self.inside(point, [max(0,zone[0]-.005),max(0,zone[1]-.005),min(1,zone[2]+.005),min(1,zone[3]+.005)])
                if inner and state.get('outside_zone') and 'zone' not in state['sent']:
                    condition = self.condition(stamp)
                    if any(state['zone_outside_time'] < interval[k] <= stamp
                           for interval in self.config['condition_intervals'] for k in ('start','end')):
                        condition = 'unknown'
                    if self.config['zone_role'] == 'observe' or condition in ('observe', 'green', 'stop_required'):
                        status, reason = 'observation', 'Entered configured zone. A crossing or zebra marking alone does not establish a violation.'
                    elif condition == 'red':
                        status, reason = 'candidate', 'Entered explicitly restricted zone during user-supplied red condition; review scene applicability.'
                    else:
                        status, reason = 'withheld', 'Zone entry observed, but applicable condition is unknown.'
                    events.append(self.event(key, track, stamp, 'zone_entry', status, reason, point, condition=condition))
                    state['sent'].add('zone')
                if not outer:
                    state['outside_zone'] = True
                    state['zone_outside_time'] = stamp
                elif inner:
                    state['outside_zone'] = False
            limit = self.config['speed_limit_kmh']
            guard = max(2.,limit*.05) if limit is not None else None
            if limit is not None and 'speeding' not in state['sent']:
                if metric_reason:
                    state.pop('speed_start', None); state['speed_samples'] = 0
                    if 'speed_unavailable' not in state['sent']:
                        events.append(self.event(key, track, stamp, 'speed_assessment', 'withheld', metric_reason, point, threshold_kmh=limit))
                        state['sent'].add('speed_unavailable')
                elif track['speed'] > limit+guard:
                    state.setdefault('speed_start', stamp)
                    state['speed_samples'] = state.get('speed_samples', 0)+1
                    if stamp-state['speed_start'] >= .5-1e-6 and state['speed_samples'] >= 3:
                        events.append(self.event(key, track, stamp, 'speeding', 'candidate',
                            'Estimated speed remained above the threshold plus the conservative measurement guard for at least 0.5 seconds; review calibration and track.',
                            point, threshold_kmh=limit, measurement_guard_kmh=guard,
                            sustained_seconds=stamp-state['speed_start'], observed_samples=state['speed_samples']))
                        state['sent'].add('speeding')
                else:
                    state.pop('speed_start', None); state['speed_samples'] = 0
                    if track['speed'] > limit and 'borderline_speed' not in state['sent']:
                        events.append(self.event(key,track,stamp,'speed_assessment','withheld',
                            'Estimate is above the threshold but inside the conservative measurement guard; no speeding conclusion.',
                            point,threshold_kmh=limit,measurement_guard_kmh=guard))
                        state['sent'].add('borderline_speed')
        return events

    def crossing_decision(self, condition, state, stamp, metric_reason):
        if condition in ('observe', 'green'):
            return 'observation', 'Configured line crossed under an unrestricted/observation condition.'
        if condition == 'red':
            return 'candidate', 'Configured stop line crossed during a user-supplied red condition; verify signal applicability.'
        if condition == 'stop_required':
            if state.get('stopped'):
                return 'observation', 'A continuous one-second low-motion interval was observed in the configured approach, with calibrated displacement and fitted speed within the stop thresholds.'
            if (metric_reason or not state.get('approach_entered') or not state.get('approach_complete') or
                    'metric_start' not in state or stamp-state['metric_start'] < 1):
                return 'withheld', 'Insufficient continuous calibrated approach history to establish whether a stop occurred.'
            return 'candidate', 'Crossed configured stop line without a one-second stop in the observed approach; review required.'
        return 'withheld', 'Line crossing observed, but applicable condition is unknown or changes within its timing bracket.'

    def summary(self):
        return {'enabled': self.enabled, 'configuration': self.config, 'calibration': self.calibration,
                'counts': {k: self.counts[k] for k in ('candidate','observation','withheld')},
                'event_count': self.serial, 'invalid_reason': self.invalid_reason, 'evicted_track_states': self.evicted,
                'excluded_geometry_frames': self.excluded_geometry_frames,
                'thresholds': {'line_deadband_px': 3, 'minimum_detector_score': .35,
                    'speed_sustain_seconds': .5, 'speed_minimum_samples': 3, 'stop_seconds': 1, 'stop_max_kmh': 2,
                    'stop_minimum_samples': 8, 'stop_position_diameter_m': ObservedStopWindow.maximum_diameter_m,
                    'speed_measurement_guard': 'max(2 km/h, 5% of configured threshold); conservative review policy guard, not a validated measurement-accuracy bound, confidence interval or legal tolerance'},
                'limitations': ['Review candidates, not legal findings. Conditions and dimensions are supplied by the user.',
                    'No inferred crossing across an observation gap; missed vehicles and ID changes can miss or duplicate events.',
                    'Each event type is emitted once per active track. Track state is capped at 512 and expires after a short absence.',
                    'A zebra crossing alone is never a violation. Synthetic results do not validate real-world accuracy.']}
