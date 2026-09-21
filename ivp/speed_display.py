"""Optional rendering filter over existing speed estimates; never a traffic verdict."""
from collections import Counter
import math


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_speed_display(value, calibration=None):
    if value is None:
        return None  # Legacy requests retain their exact rendering path.
    if not isinstance(value, dict) or set(value) != {'enabled', 'limit', 'unit'}:
        raise ValueError('Speed display needs enabled, limit and unit fields.')
    if type(value['enabled']) is not bool:
        raise ValueError('Speed display enabled must be true or false.')
    unit = 'km/h' if calibration is not None else 'px/s'
    if value['unit'] != unit:
        raise ValueError('Speed limit units do not match this calibration. Re-enter the limit in '+unit+'.')
    limit = value['limit']
    if value['enabled']:
        if not finite_number(limit) or limit <= 0:
            raise ValueError('Enter a positive finite speed limit in '+unit+'.')
    elif limit is not None:
        raise ValueError('A disabled speed display must have a null limit.')
    return {'enabled': value['enabled'], 'limit': limit, 'unit': unit}


def display_decision(track, config):
    if not config or not config['enabled']:
        return True, 'filter off'
    if (track.get('confirmed') is not True or
            track.get('tracking_state') not in ('observed', 'recovered from weak observations') or
            track.get('observed') is False):
        return False, 'track is not a confirmed current observation'
    if track.get('speed_blocked') or track.get('tracking_state') == 'ambiguous overlap':
        return False, 'tracking evidence withheld'
    if track.get('speed_status') != 'estimated':
        return False, 'speed unavailable or withheld'
    speed = track.get('speed')
    if not finite_number(speed) or speed < 0:
        return False, 'speed unavailable or withheld'
    if track.get('speed_unit') != config['unit']:
        return False, 'unit mismatch; no conversion'
    return (True, 'strictly above limit') if speed > config['limit'] else (False, 'equal to or below limit')


class SpeedDisplay:
    def __init__(self, value, calibration=None, tracking_mode='ivp'):
        self.config = validate_speed_display(value, calibration) or {'enabled':False,'limit':None,'unit':'km/h' if calibration is not None else 'px/s'}
        self.tracking_mode = tracking_mode
        self.counts = Counter()
        self.frame_size = None

    def observe(self, track, shape):
        self.frame_size = [int(shape[1]), int(shape[0])]
        drawn, reason = display_decision(track, self.config)
        self.counts[reason] += 1
        track['speed_display_drawn'] = drawn
        track['speed_display_reason'] = reason
        return drawn

    def summary(self):
        return {**self.config, 'comparison': 'strictly_greater_than',
                'comparison_value': 'Existing per-track speed rounded to two decimals, as exported in motion-tracks.csv.',
                'scope': 'vehicle tracks' if self.tracking_mode == 'vehicles' else 'moving texture regions; not established vehicles',
                'coordinate_space': 'Camera-registered first-frame analysis image after aspect-preserving resize; not source pixels, screen pixels or mosaic-panel pixels.',
                'analysis_frame_size': self.frame_size,
                'position': 'Observed track box bottom-centre; existing estimator and withholding gates unchanged.',
                'sample_counts': dict(self.counts),
                'meaning': 'Display filter only. No legal speed limit, violation verdict or measurement-accuracy claim. All measurements remain in the trajectories export.'}
