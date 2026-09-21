import pytest
from ivp.incidents import IncidentAnalyzer, validate_config


CAL = {'points': [[.1,.1],[.9,.1],[.9,.9],[.1,.9]], 'width_m': 16, 'length_m': 40}
BASE = {'enabled': True, 'line': [[.15,.65],[.85,.65]], 'direction': 'positive', 'condition': 'red'}


def test_withheld_incident_serializes_no_previous_speed_window():
    import json
    from ivp.speed import SpeedEstimator
    calibration = {'points': [[0,0],[1,0],[1,1],[0,1]], 'width_m': 10, 'length_m': 10}
    estimator = SpeedEstimator(calibration)
    observation = {'confirmed': True, 'tracking_state': 'observed', 'confidence': .8}
    for index in range(35):
        observation['box'] = (20+index,20,10,10)
        estimator.update(observation,index/30,(100,100))
    assert observation['speed'] is not None
    observation.update(box=(55,20,10,10), speed_blocked='overlapping vehicle hypotheses')
    estimator.update(observation,35/30,(100,100))
    analyzer = IncidentAnalyzer({**BASE, 'synthetic': True},calibration)
    events = analyzer.step([(1,observation)],35/30,(100,100))
    saved = json.loads(json.dumps(events))
    assert len(saved) == 1 and saved[0]['type'] == 'track_quality'
    assert saved[0]['status'] == 'withheld' and saved[0]['synthetic']
    assert saved[0]['measurements']['speed'] is None
    assert saved[0]['quality']['speed_window_seconds'] is None


def track(y, x=.5, speed=36, **extra):
    return {'box': (x*1000-20,y*1000-40,40,40), 'confirmed': True,
            'tracking_state': 'observed', 'confidence': .8, 'speed': speed,
            'speed_unit': 'km/h', 'speed_status': 'estimated', 'speed_window_seconds': 1, **extra}


def run(analyzer, observations):
    result = []
    for t, obj in observations:
        result += analyzer.step([(1,obj)],t,(1000,1000))
    return result


@pytest.mark.parametrize('condition,status', [('red','candidate'),('green','observation'),('observe','observation'),('unknown','withheld')])
def test_line_conditions_deadband_and_dedup(condition,status):
    a=IncidentAnalyzer({**BASE,'condition':condition})
    events=run(a,[(i/10,track(y)) for i,y in enumerate([.64,.649,.651,.66,.64,.66])])
    assert len(events)==1 and events[0]['status']==status
    assert events[0]['timestamp_seconds']==pytest.approx(.15)
    assert events[0]['measurements']['crossing_bracket_seconds']==[0,.3]


@pytest.mark.parametrize('x,direction',[(.05,'positive'),(.95,'positive'),(.5,'negative')])
def test_segment_endpoints_and_direction(x,direction):
    a=IncidentAnalyzer({**BASE,'direction':direction})
    assert run(a,[(0,track(.64,x)),(.1,track(.66,x))])==[]


def test_condition_transition_and_uncovered_intervals_are_unknown():
    a=IncidentAnalyzer({**BASE,'condition_intervals':[{'start':0,'end':.05,'state':'green'}, {'start':.05,'end':1,'state':'red'}]})
    assert run(a,[(0,track(.64)),(.1,track(.66))])[0]['status']=='withheld'
    assert a.condition(1)=='unknown'


@pytest.mark.parametrize('bad',[{'tracking_state':'ambiguous overlap'},{'confidence':.2},{'speed_blocked':'frame clipped'}])
def test_uncertain_track_cannot_create_crossing(bad):
    a=IncidentAnalyzer(BASE)
    events=run(a,[(0,track(.64)),(.1,track(.65,**bad)),(.2,track(.66))])
    assert not any(e['status']=='candidate' for e in events)
    assert events[0]['type']=='track_quality'


def test_missing_observation_camera_change_and_registration_reset():
    a=IncidentAnalyzer(BASE)
    assert not any(e['status']=='candidate' for e in run(a,[(0,track(.64)),(.5,track(.66))]))
    a=IncidentAnalyzer(BASE);a.step([(1,track(.64))],0,(1000,1000))
    a.step([], .1,(1000,1000),registration_valid=False)
    assert a.step([(1,track(.66))],.2,(1000,1000))==[]
    a.step([], .3,(1000,1000),scene_changed=True)
    assert run(a,[(.4,track(.64)),(.5,track(.66))])==[]
    assert 'Camera/scene' in a.summary()['invalid_reason']


def test_one_missing_frame_resets_crossing_even_at_high_frame_rate():
    a=IncidentAnalyzer(BASE)
    a.step([(1,track(.64))],0,(1000,1000))
    a.step([],1/30,(1000,1000))
    events=a.step([(1,track(.66))],2/30,(1000,1000))
    assert not any(e['type']=='line_crossing' for e in events)
    assert events[0]['status']=='withheld'


@pytest.mark.parametrize('speed,expected',[(18,False),(30,False),(36,True)])
def test_sustained_speed_and_synthetic_label(speed,expected):
    a=IncidentAnalyzer({'enabled':True,'speed_limit_kmh':30,'calibration_confirmed':True,
        'real_time_confirmed':True,'synthetic':True},CAL,frame_interval=.1)
    events=run(a,[(i/10,track(.3+i*.01,speed=speed)) for i in range(20)])
    assert len(events)==int(expected)
    if expected:
        assert events[0]['type']=='speeding' and events[0]['synthetic']
        assert events[0]['measurements']['sustained_seconds']>=.5


@pytest.mark.parametrize('missing',['calibration','calibration_confirmed','real_time_confirmed','speed'])
def test_speed_requirements_withhold(missing):
    config={'enabled':True,'speed_limit_kmh':30,'calibration_confirmed':True,'real_time_confirmed':True}
    if missing in config:config[missing]=False
    a=IncidentAnalyzer(config,None if missing=='calibration' else CAL)
    events=run(a,[(i/10,track(.4,speed=None if missing=='speed' else 36)) for i in range(12)])
    assert len(events)==1 and events[0]['status']=='withheld'


def test_speed_requires_sustain_after_below_limit_and_gap():
    a=IncidentAnalyzer({'enabled':True,'speed_limit_kmh':30,'calibration_confirmed':True,'real_time_confirmed':True},CAL)
    events=run(a,[(0,track(.4)),(.1,track(.4)),(.2,track(.4,speed=18)),(.3,track(.4)),(.4,track(.4)),(.7,track(.4)),(.8,track(.4))])
    assert not any(e['type']=='speeding' for e in events)


@pytest.mark.parametrize('speed',[30.01,30.5,31.99,32])
def test_borderline_speed_is_withheld_with_explicit_guard(speed):
    a=IncidentAnalyzer({'enabled':True,'speed_limit_kmh':30,'calibration_confirmed':True,'real_time_confirmed':True},CAL)
    events=run(a,[(i/10,track(.4,speed=speed)) for i in range(15)])
    assert len(events)==1 and events[0]['status']=='withheld'
    assert events[0]['measurements']['measurement_guard_kmh']==2


@pytest.mark.parametrize('mode,status',[('no_stop','candidate'),('stop','observation'),('born_inside','withheld'),('no_scale','withheld'),('missing_speed','withheld')])
def test_stop_requires_fully_observed_approach(mode,status):
    a=IncidentAnalyzer({**BASE,'condition':'stop_required','approach_zone':[.15,.45,.85,.65],
        'calibration_confirmed':True,'real_time_confirmed':True},None if mode=='no_scale' else CAL)
    observations=[(0,track(.44))] if mode!='born_inside' else []
    observations += [(i/10,track(.56 if mode=='stop' else .46+.008*i,
        speed=0 if mode=='stop' else None if mode=='missing_speed' and i==2 else 36)) for i in range(1,21)]
    observations += [(2.1,track(.66))]
    events=run(a,observations)
    assert len(events)==1 and events[0]['status']==status


def test_rejected_position_jump_is_not_line_crossing_evidence():
    a=IncidentAnalyzer(BASE)
    events=run(a,[(0,track(.63)),(.1,track(.68,speed=None,
        speed_status='abrupt position change; collecting a new speed window')),(.2,track(.69))])
    assert not any(e['type']=='line_crossing' for e in events)
    assert any(e['type']=='track_quality' and e['status']=='withheld' for e in events)


def test_rejected_position_jump_discards_a_previously_observed_stop():
    a=IncidentAnalyzer({**BASE,'condition':'stop_required','approach_zone':[.15,.45,.85,.65],
        'calibration_confirmed':True,'real_time_confirmed':True},CAL)
    events=run(a,[(i/15,track(.55,speed=0)) for i in range(30)])
    assert a.tracks[1]['stopped']
    events+=run(a,[(2,track(.56,speed=None,
        speed_status='abrupt position change; collecting a new speed window')),(2.0666667,track(.6)),(2.1333333,track(.66))])
    crossing=[e for e in events if e['type']=='line_crossing']
    assert len(crossing)==1 and crossing[0]['status']=='withheld'
    assert crossing[0]['measurements']['stop_evidence'] is None


@pytest.mark.parametrize('role,condition,status',[('observe','red','observation'),('restricted','red','candidate'),('restricted','green','observation'),('restricted','unknown','withheld')])
def test_zone_entry_requires_outside_observation_and_explicit_restriction(role,condition,status):
    config={'enabled':True,'zone':[.15,.65,.85,.78],'zone_role':role,'condition':condition}
    a=IncidentAnalyzer(config)
    assert run(a,[(0,track(.7)),(.1,track(.71))])==[]
    events=run(a,[(.2,track(.63)),(.3,track(.7)),(.4,track(.72))])
    assert len(events)==1 and events[0]['status']==status


def test_zone_condition_change_during_entry_is_withheld():
    a=IncidentAnalyzer({'enabled':True,'zone':[.15,.65,.85,.78],'zone_role':'restricted',
        'condition_intervals':[{'start':0,'end':.05,'state':'green'},{'start':.05,'end':1,'state':'red'}]})
    events=run(a,[(0,track(.63)),(.1,track(.7))])
    assert len(events)==1 and events[0]['status']=='withheld'


def test_disabled_and_bounded_track_state():
    assert IncidentAnalyzer(None).step([],0,(100,100))==[]
    a=IncidentAnalyzer(BASE)
    a.step([(i,track(.4,confidence=.1)) for i in range(700)],0,(1000,1000))
    assert len(a.tracks)<=512
    a.step([],3,(1000,1000));assert a.tracks=={}


@pytest.mark.parametrize('config',[{'enabled':True},{**BASE,'line':[[0,0],[0,0]]},
    {**BASE,'zone':[.2,.5,.1,.7]},{**BASE,'speed_limit_kmh':float('nan')},
    {**BASE,'synthetic':'false'},{**BASE,'condition':[]},{**BASE,'condition_intervals':[{'start':0,'end':1,'state':[]}]},
    {**BASE,'condition_intervals':[{'start':2,'end':1,'state':'red'}]},
    {**BASE,'condition_intervals':[{'start':0,'end':2,'state':'red'},{'start':1,'end':3,'state':'green'}]}])
def test_invalid_config(config):
    with pytest.raises(ValueError):validate_config(config)


def test_evidence_coordinates_stream_and_thumbnail_limits(tmp_path):
    import json
    import numpy as np
    from ivp.incident_export import IncidentEvidence
    a=IncidentAnalyzer(BASE)
    event=run(a,[(0,track(.64)),(.1,track(.66))])[0]
    with IncidentEvidence(tmp_path,a) as sink:
        sink.record([event],np.zeros((1000,1000,3),np.uint8),np.array([[1,0,10],[0,1,20]]),np.diag([2,2,1]),2,'revision')
        assert event['coordinates']['anchor_source_px']==pytest.approx([980,1280])
        assert event['evidence_timestamp_seconds']==.1
        assert event['source_revision']=='revision'
        assert (tmp_path/event['thumbnail']).is_file()
        sink.thumbnails=40
        next_event=a.event(2,track(.66),.2,'line_crossing','observation','test',(.5,.66))
        sink.record([next_event],np.zeros((1000,1000,3),np.uint8),np.eye(2,3),np.eye(3),3,'revision')
        assert next_event['thumbnail'] is None
        summary=sink.finish()
    assert len((tmp_path/'motion-incidents.jsonl').read_text().splitlines())==2
    assert len(summary['events'])==2
    assert json.loads((tmp_path/'motion-incidents.json').read_text())['configuration']['condition']=='red'
