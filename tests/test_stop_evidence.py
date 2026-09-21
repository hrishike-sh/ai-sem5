"""Stop evidence uses observed positions; synthetic truth never enters inference."""
import numpy as np
import pytest

from ivp.incidents import IncidentAnalyzer
from ivp.speed import SpeedEstimator
from ivp.stop_evidence import ObservedStopWindow


CAL = {'points': [[0, 0], [1, 0], [1, 1], [0, 1]], 'width_m': 20, 'length_m': 40}
CONFIG = {'enabled': True, 'line': [[.1, .65], [.9, .65]], 'direction': 'positive',
          'condition': 'stop_required', 'approach_zone': [.1, .4, .9, .65],
          'calibration_confirmed': True, 'real_time_confirmed': True}


@pytest.mark.parametrize('fps,pause,status', [(5, 2.5, 'withheld'), (15, 1.5, 'observation'),
                                           (30, 1.5, 'observation'), (120, 1.5, 'observation')])
def test_completed_short_stop_uses_observed_window_without_speed_lag(fps, pause, status):
    a = IncidentAnalyzer(CONFIG, CAL, 1/fps)
    speed = SpeedEstimator(CAL, 1/fps)
    t = {}
    events = []
    low_speed_times = []
    for i in range(fps*7):
        stamp = i/fps
        # At low cadence, abrupt braking/departure can trip the existing
        # position gate. That uncertainty must discard even earlier stop proof.
        travelled = 5*min(stamp, 2.5)+5*max(0, stamp-2.5-pause)
        y = 8+travelled
        t.update(box=(480, y/40*1000-40, 40, 40), confirmed=True,
                 tracking_state='observed', confidence=.9)
        speed.update(t, stamp, (1000, 1000))
        if t['speed'] is not None and t['speed'] <= 2:
            low_speed_times.append(stamp)
        events += a.step([(1, t)], stamp, (1000, 1000))
    crossing = [e for e in events if e['type'] == 'line_crossing']
    assert len(crossing) == 1 and crossing[0]['status'] == status
    proof = crossing[0]['measurements']['stop_evidence']
    if status == 'withheld':
        assert proof is None
        assert any(e['type'] == 'track_quality' for e in events)
        return
    assert proof['duration_seconds'] >= 1-1e-6 and proof['observed_samples'] >= 8
    assert proof['position_diameter_bound_m'] <= 2/3.6
    assert proof['start_seconds'] >= 2.4  # No invented stationary approach.
    assert proof['end_seconds'] <= 2.6+pause
    assert low_speed_times[-1]-low_speed_times[0] < pause


@pytest.mark.parametrize('problem', ['short_pause', 'gap', 'weak', 'ambiguous', 'no_scale',
                                    'unconfirmed_scale', 'no_timing', 'not_a_stop'])
def test_incomplete_or_unsupported_stop_never_gets_positive_proof(problem):
    config = dict(CONFIG)
    if problem == 'unconfirmed_scale':
        config['calibration_confirmed'] = False
    if problem == 'no_timing':
        config['real_time_confirmed'] = False
    a = IncidentAnalyzer(config, None if problem == 'no_scale' else CAL, 1/15)
    speed = SpeedEstimator(CAL, 1/15)
    t = {}
    events = []
    for i in range(105):
        stamp = i/15
        pause = .5 if problem == 'short_pause' else 1.5
        travelled = 5*min(stamp, 2.5)+5*max(0, stamp-2.5-pause)
        if problem == 'not_a_stop':
            travelled = 5*stamp
        t.update(box=(480, (8+travelled)/40*1000-40, 40, 40), confirmed=True,
                 tracking_state='observed', confidence=.9)
        speed.update(t, stamp, (1000, 1000))
        if 2.8 < stamp < 3.5:
            if problem == 'weak':
                t['confidence'] = .2
            if problem == 'ambiguous':
                t['tracking_state'] = 'ambiguous overlap'
            if problem == 'gap':
                events += a.step([], stamp, (1000, 1000))
                continue
        events += a.step([(1, t)], stamp, (1000, 1000))
    crossing = [e for e in events if e['type'] == 'line_crossing']
    assert len(crossing) == 1
    assert crossing[0]['status'] != 'observation'
    assert crossing[0]['measurements']['stop_evidence'] is None
    if problem in ('gap', 'weak', 'ambiguous', 'no_scale', 'unconfirmed_scale', 'no_timing'):
        assert crossing[0]['status'] == 'withheld'


@pytest.mark.parametrize('motion', ['steady', 'out_and_back', 'jump'])
def test_small_or_cancelled_fitted_speed_cannot_hide_position_movement(motion):
    window = ObservedStopWindow()
    proof = None
    for i in range(16):
        t = i/15
        x = t if motion == 'steady' else np.sin(t*np.pi) if motion == 'out_and_back' else float(i >= 8)
        proof = window.update(t, [x, 0], accepted_speed=0)
    assert proof is None


def test_stop_window_is_bounded_and_requires_a_current_accepted_speed():
    window = ObservedStopWindow()
    for i in range(500):
        assert window.update(i/30, [2, 2], accepted_speed=None) is None
    assert len(window.samples) <= 32
    assert window.update(500/30, [2, 2], accepted_speed=0)['observed_samples'] >= 8


def test_low_cadence_stop_needs_eight_actual_observations():
    window = ObservedStopWindow(.2)
    for i in range(7):
        assert window.update(i*.2, [2, 2], accepted_speed=0) is None
    proof = window.update(1.4, [2, 2], accepted_speed=0)
    assert proof['observed_samples'] == 8 and proof['duration_seconds'] == 1.4


def test_approach_reentry_does_not_reuse_an_earlier_stop():
    a = IncidentAnalyzer(CONFIG, CAL, 1/15)
    events = []
    positions = [.35]+[.5]*24+[.35]+[.45+i*.01 for i in range(23)]
    for i, y in enumerate(positions):
        t = {'box': (480, y*1000-40, 40, 40), 'confirmed': True,
             'tracking_state': 'observed', 'confidence': .9,
             'speed': 0 if 1 <= i <= 24 else 18, 'speed_unit': 'km/h', 'speed_status': 'estimated'}
        events += a.step([(1, t)], i/15, (1000, 1000))
    crossing = [e for e in events if e['type'] == 'line_crossing']
    assert len(crossing) == 1 and crossing[0]['status'] == 'candidate'
    assert crossing[0]['measurements']['stop_evidence'] is None


@pytest.mark.parametrize('problem', ['gap', 'weak', 'registration', 'scene'])
def test_uncertainty_after_a_completed_stop_discards_its_proof(problem):
    a = IncidentAnalyzer(CONFIG, CAL, 1/15)
    events = []
    for i in range(50):
        t = {'box': (480, (.5 if i < 30 else .5+(i-30)*.01)*1000-40, 40, 40),
             'confirmed': True, 'tracking_state': 'observed', 'confidence': .9,
             'speed': 0 if i < 30 else 18, 'speed_unit': 'km/h', 'speed_status': 'estimated'}
        if i == 25:
            assert a.tracks[1]['stopped']
            if problem == 'weak':
                t['confidence'] = .2
            if problem == 'gap':
                events += a.step([], i/15, (1000, 1000))
                continue
        events += a.step([(1, t)], i/15, (1000, 1000),
                         registration_valid=not (problem == 'registration' and i == 25),
                         scene_changed=problem == 'scene' and i == 25)
    crossing = [e for e in events if e['type'] == 'line_crossing']
    if problem == 'scene':
        assert not crossing and a.invalid_reason
    else:
        assert len(crossing) == 1 and crossing[0]['status'] == 'withheld'
        assert crossing[0]['measurements']['stop_evidence'] is None


def test_zero_reported_speed_cannot_confirm_a_visibly_moving_track():
    a = IncidentAnalyzer(CONFIG, CAL, 1/15)
    events = []
    for i in range(50):
        y = .35+i*.007
        t = {'box': (480, y*1000-40, 40, 40), 'confirmed': True,
             'tracking_state': 'observed', 'confidence': .9,
             'speed': 0, 'speed_unit': 'km/h', 'speed_status': 'estimated'}
        events += a.step([(1, t)], i/15, (1000, 1000))
    crossing = [e for e in events if e['type'] == 'line_crossing']
    assert len(crossing) == 1 and crossing[0]['status'] == 'candidate'
    assert crossing[0]['measurements']['stop_evidence'] is None
