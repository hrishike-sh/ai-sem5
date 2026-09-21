import pytest
from ivp.feature_scope import validate_feature_scope


def config(feature):
    c={'enabled':True,'line':None,'zone':None,'approach_zone':None,'speed_limit_kmh':None,
       'zone_role':'observe','condition':'unknown','condition_intervals':[]}
    m={'tracking_mode':'vehicles','incidents':c,'calibration':None}
    if feature in {'red_line','required_stop'}:c['line']=[[.1,.6],[.9,.6]]
    if feature=='restricted_red':c.update(zone=[.1,.5,.9,.8],zone_role='restricted')
    if feature=='speed_review':c.update(speed_limit_kmh=30,condition='observe');m['calibration']={'measured':True}
    if feature=='required_stop':c.update(approach_zone=[.1,.3,.9,.6],condition_intervals=[{'start':0,'end':2,'state':'stop_required'}]);m['calibration']={'measured':True}
    return m


@pytest.mark.parametrize('feature',['red_line','restricted_red','speed_review','required_stop'])
def test_explicit_feature_scoping_and_conditions(feature):
    m=config(feature);validate_feature_scope(feature,m)
    m['incidents']['speed_limit_kmh']=20 if feature!='speed_review' else None
    with pytest.raises(ValueError):validate_feature_scope(feature,m)


@pytest.mark.parametrize('feature',['red_line','restricted_red','required_stop'])
def test_other_feature_condition_cannot_be_hidden_in_timeline(feature):
    m=config(feature);m['incidents']['condition_intervals']=[{'start':0,'end':2,'state':'red' if feature=='required_stop' else 'stop_required'}]
    with pytest.raises(ValueError):validate_feature_scope(feature,m)


def test_legacy_without_identity_is_compatible_but_unknown_identity_rejects():
    validate_feature_scope(None,config('red_line'))
    with pytest.raises(ValueError):validate_feature_scope('made_up',{})
