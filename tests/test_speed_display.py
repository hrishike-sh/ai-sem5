from collections import deque
from copy import deepcopy
import math
import numpy as np
import pytest

from ivp.speed import SpeedEstimator
from ivp.speed_display import SpeedDisplay, display_decision, validate_speed_display
from ivp.motion import MotionAnalyzer
from ivp.feature_scope import validate_feature_scope


def config(limit=30, unit='px/s', enabled=True):
    return {'enabled':enabled,'limit':limit if enabled else None,'unit':unit}


def observation(speed=30, unit='px/s', **extra):
    return {'speed':speed,'speed_status':'estimated','speed_unit':unit,'confirmed':True,
            'tracking_state':'observed','observed':True,**extra}


@pytest.mark.parametrize('speed,expected',[(0,False),(29.99,False),(30,False),(30.01,True),(math.nan,False),(math.inf,False),(-1,False),(None,False),(True,False)])
@pytest.mark.parametrize('unit',['px/s','km/h'])
def test_strict_boundary_and_invalid_measurements(speed, expected, unit):
    assert display_decision(observation(speed,unit),config(unit=unit))[0] is expected


@pytest.mark.parametrize('change',[
    {'speed_status':'collecting stable track'},{'speed_status':'outside calibrated road'},
    {'speed_status':None},{'speed_blocked':'overlap'},{'tracking_state':'ambiguous overlap'},
    {'tracking_state':'predicted'},{'tracking_state':'lost'},{'tracking_state':None},
    {'confirmed':False},{'confirmed':None},{'observed':False},{'speed_unit':'km/h'}])
def test_unknown_stale_gap_or_wrong_unit_never_highlight(change):
    assert display_decision(observation(50,**change),config())[0] is False


@pytest.mark.parametrize('limit',[0,-1,True,None,'30',math.nan,math.inf])
def test_invalid_enabled_limit_rejected(limit):
    with pytest.raises(ValueError):validate_speed_display(config(limit))


def test_units_must_match_calibration_and_disabled_limit_is_null():
    with pytest.raises(ValueError):validate_speed_display(config(unit='km/h'))
    with pytest.raises(ValueError):validate_speed_display(config(),{'points':[]})
    assert validate_speed_display(config(enabled=False))==config(enabled=False)
    with pytest.raises(ValueError):validate_speed_display({'enabled':False,'unit':'px/s','limit':10})
    assert SpeedDisplay(None).summary()['enabled'] is False


def test_pixel_filter_uses_tracking_and_does_not_bypass_legacy_incident_gate():
    validate_feature_scope('tracking',{'speed_display':config(),'incidents':None})
    with pytest.raises(ValueError):validate_feature_scope('speed_review',{'speed_display':config(),'incidents':None})
    with pytest.raises(ValueError):validate_feature_scope('speed_review',{'incidents':None})
    with pytest.raises(ValueError):validate_feature_scope('speed_review',{'speed_display':config(),'direction':{'points':[]}})


def test_actual_estimator_gaps_and_calibration_invalidation_withhold_filter():
    for calibrated in (False,True):
        road={'points':[[0,0],[1,0],[1,1],[0,1]],'width_m':10,'length_m':10} if calibrated else None
        estimator=SpeedEstimator(road)
        t=observation(None,'km/h' if calibrated else 'px/s')
        cfg=config(5,t['speed_unit'])
        for i in range(35):t['box']=(20+i,20,10,10);estimator.update(t,i/30,(100,100))
        assert display_decision(t,cfg)[0]
        before=deepcopy(t);display_decision(t,cfg)
        assert t['speed']==before['speed'] and t['speed_window_seconds']==before['speed_window_seconds']
        if calibrated:estimator.invalidate()
        estimator.update(t,3,(100,100))
        assert t['speed'] is None and not display_decision(t,cfg)[0]


def test_analysis_pixels_depend_on_analysis_resolution_not_source_or_mosaic():
    measured=[]
    for size in (200,400):
        estimator=SpeedEstimator();t=observation(None)
        for i in range(35):t['box']=(size*.2+i*size/200,20,10,10);estimator.update(t,i/30,(size,size))
        measured.append(t['speed'])
    assert measured==[30,60]
    d=SpeedDisplay(config(40));assert not d.observe(observation(30),(200,200));assert d.observe(observation(60),(400,400))
    assert d.summary()['analysis_frame_size']==[400,400]
    assert 'not source pixels' in d.summary()['coordinate_space']


class FixtureTracker:
    def __init__(self):
        self.tracks={};self.counts={'positive':0,'negative':0};self.ambiguous_boxes=0
    def update(self,*args):
        tracks=[]
        for i,speed in enumerate([29.99,30,30.01,None]):
            t=observation(speed);t.update(box=(35+i*55,75,25,25),centre=np.array([47+i*55,87]),trail=deque([(40+i*55,82),(47+i*55,87)]))
            if speed is None:t['speed_status']='collecting stable track'
            tracks.append((i+1,t))
        return tracks,[]
    def summary(self):return {}


def render_fixture(display):
    # No model: actual renderer with controlled already-estimated observations.
    e=MotionAnalyzer(speed_display=display);e.tracker=FixtureTracker();e.warm_until=0
    e.speed.update=lambda *args:None
    rgb=np.random.default_rng(8).integers(25,100,(180,320,3),dtype=np.uint8)
    for i in range(6):e.step(rgb,i/30)
    return e,e.step(rgb,6/30)


def test_renderer_filters_boxes_only_and_off_is_pixel_identical_to_legacy():
    legacy,old=render_fixture(None);off,disabled=render_fixture(config(enabled=False));on,filtered=render_fixture(config())
    assert np.array_equal(old[0],disabled[0])
    assert old[1]==disabled[1]==filtered[1] and old[2]==filtered[2]
    assert [t['speed'] for _,t in old[3]]==[t['speed'] for _,t in filtered[3]]
    assert [t['speed_display_drawn'] for _,t in filtered[3]]==[False,False,True,False]
    # The accepted track's text can overlap its neighbour; inspect lower box edges, away from labels.
    for i in (0,1,3):assert np.array_equal(filtered[0][95:100,35+i*55:60+i*55],on.last_rgb[95:100,35+i*55:60+i*55])
    assert not np.array_equal(filtered[0][75:100,145:170],on.last_rgb[75:100,145:170])
    assert np.array_equal(old[0][116:119],filtered[0][116:119])  # Counting geometry remains.
    assert on.summary()['speed_display']['limit']==30
