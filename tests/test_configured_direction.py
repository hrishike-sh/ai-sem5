import pytest
import numpy as np
from ivp.configured_direction import DirectionReview,validate_config


def config(arrow=None):
    return validate_config({'schema_version':1,'type':'configured_direction','reviewed':True,
        'media_id':'m','source_revision':'r','reference_seconds':0.,
        'points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
        'arrow':arrow or [[.5,.8],[.5,.2]]},'m','r')


def track(y,**extra):
    return [(7,{'box':(90,y-20,20,20),'model_label':'car','confidence':.95,'tracking_state':'observed',**extra})]


def replay(engine,start=0.,count=12):
    return [engine.step(track(50+i*5),start+i*.1,(200,200))[0] for i in range(count)]


def test_existing_evaluator_opposite_direction_and_one_episode():
    engine=DirectionReview(config());rows=replay(engine)
    assert all(r['status']=='withheld' for r in rows[:5])
    assert all(r['label']=='opposite configured direction' for r in rows[5:])
    assert engine.events==1 and sum(r.get('new_episode',False) for r in rows)==1
    assert all(r['legal_violation'] is False and r['eligible_for_rules'] is False for r in rows)
    assert rows[-1]['measurements']['direction_cosine']==pytest.approx(-1.)


def test_same_direction_is_consistent_and_no_legal_claim():
    rows=replay(DirectionReview(config([[.5,.2],[.5,.8]])))
    assert rows[-1]['status']=='consistent_with_configured_direction'
    assert rows[-1]['label'] is None


@pytest.mark.parametrize('change',[
    {'reviewed':False},{'source_revision':'old'},{'reference_seconds':1.},
    {'type':'paint_arrow'},{'eligible_for_rules':True},
    {'arrow':[[.5,.5],[.51,.5]]},{'arrow':[[.01,.01],[.5,.5]]},
])
def test_source_bound_explicit_configuration_only(change):
    value={k:v for k,v in config().items() if k!='config_id'};value.update(change)
    with pytest.raises(ValueError):validate_config(value,'m','r')


def test_duplicate_gap_and_ambiguity_do_not_manufacture_history_or_episodes():
    engine=DirectionReview(config());replay(engine)
    row=engine.step(track(110),1.1,(200,200))[0];assert row['status']=='withheld'
    row=engine.step(track(115,tracking_state='predicted'),1.2,(200,200))[0];assert row['status']=='withheld'
    rows=[engine.step(track(115+i*2),1.3+i*.1,(200,200))[0] for i in range(10)]
    assert rows[0]['status']=='withheld'
    assert rows[-1]['episode_id']==1 and not rows[-1]['new_episode']
    row=engine.step(track(150),4.,(200,200))[0];assert row['status']=='withheld'


@pytest.mark.parametrize('gate',[{'scene_changed':True},{'registration_valid':False}])
def test_current_registration_and_scene_gates(gate):
    engine=DirectionReview(config());replay(engine)
    row=engine.step(track(115),1.2,(200,200),**gate)[0]
    assert row['status']=='withheld' and row['label'] is None


def test_new_configuration_has_fresh_history_and_resize_invalidates_old_context():
    original=config();engine=DirectionReview(original);replay(engine)
    original['arrow']=[[.5,.2],[.5,.8]]
    assert engine.config['arrow']!=original['arrow']
    fresh=DirectionReview(config([[.5,.2],[.5,.8]]))
    assert fresh.step(track(115),1.2,(200,200))[0]['status']=='withheld'
    row=engine.step(track(115),1.2,(200,300))[0]
    assert row['status']=='withheld' and engine.invalidated


def test_camera_warp_border_withholds():
    engine=DirectionReview(config());mask=np.zeros((200,200),np.uint8)
    row=engine.step(track(60),0.,(200,200),valid_mask=mask)[0]
    assert row['label'] is None and 'warp border' in row['reason']


def test_decoded_empty_frame_clears_prior_motion_support():
    engine=DirectionReview(config())
    for i in range(3):engine.step(track(50+i*5),i*.1,(200,200))
    assert engine.step([],.3,(200,200))==[]
    assert not engine.engine.history
    for i in range(5):
        row=engine.step(track(70+i*5),.4+i*.1,(200,200))[0]
        assert row['status']=='withheld'
    row=engine.step(track(95),.9,(200,200))[0]
    assert row['status']=='opposite_direction_observed'


def test_perpendicular_motion_is_not_claimed_consistent():
    engine=DirectionReview(config())
    for i in range(10):
        t=track(100)[0][1];t['box']=(40+i*5,80,20,20)
        row=engine.step([(7,t)],i*.1,(200,200))[0]
    assert row['status']=='opposite_direction_not_established'
    assert row['label'] is None
