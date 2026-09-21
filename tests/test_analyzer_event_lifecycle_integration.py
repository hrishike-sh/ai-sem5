"""Current Analyzer wiring, true collector/gates and explicitly synthetic source masks."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_ai import analyzer as module
from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.event_lifecycle import LifecycleAwareEventCollector
from tests.test_analyzer_scene_lifecycle_integration import isolated_analyzer as isolated_analyzer
from tests.test_analyzer_footpath_integration import configure, analyze


@pytest.mark.parametrize('eligible',[True,False])
def test_all_eligible_unknown_observations_use_exact_full_detection_indices(isolated_analyzer,eligible):
    instance=isolated_analyzer
    assert module.EventCollector is LifecycleAwareEventCollector
    rgb,_,_=configure(instance)
    full=[Detection('person',.9,(10,10,30,50),99),Detection('traffic_light',.9,(40,10,55,35),100)]
    if eligible:
        full.extend([Detection('car',.9,(80,100,110,140),7),Detection('motorcycle',.9,(150,110,170,140),8)])
    instance.perception.detect=lambda *args,**kwargs: full
    _,analysis,_=instance.analyze(SimpleNamespace(color_rgb=rgb,timestamp=0.),rgb,video=True)
    trace=instance.events.confirmation_snapshot()['lifecycle']
    assert trace['detection_observations']==(2 if eligible else 0)
    assert trace['source_detection_index_space']=='supplied eligible vehicle detection list'
    assert trace['camera_stable'] is False and trace['scene_revision']==1
    assert instance.event_vehicle_indices==([2,3] if eligible else [])
    assert all(d.verdict==Verdict.UNKNOWN for d in analysis.decisions)
    assert analysis.detections==full


def warm_positive(instance):
    rgb,road,sidewalk=configure(instance)
    sidewalk[:]=0
    sidewalk[20:180,20:300]=255
    road[:]=np.where(sidewalk>0,0,255)
    events=[]
    for index in range(35):
        x=45+index*.7
        _,analysis,created=analyze(instance,index/10,rgb,Detection('motorcycle',.9,(x-10,35,x+10,60),1))
        events.extend(created)
    assert len(events)==1 and events[0].rule==Rule.FOOTPATH
    trace=next(d for d in analysis.decisions if d.rule==Rule.FOOTPATH).evidence['footpath_contact_support']
    assert trace['status']=='supported' and trace['history']['covers_full_window']
    return rgb,events[0]


@pytest.mark.parametrize('boundary',['jump','gap'])
def test_real_pipeline_restarts_contact_and_event_state_but_preserves_history(isolated_analyzer,boundary):
    instance=isolated_analyzer
    assert isinstance(instance.events,LifecycleAwareEventCollector)
    rgb,historical=warm_positive(instance)
    original=deepcopy(asdict(historical))
    stamp=3.5 if boundary=='jump' else 4.2
    x=220 if boundary=='jump' else 70
    _,analysis,created=analyze(instance,stamp,rgb,Detection('motorcycle',.9,(x-10,35,x+10,60),1))
    assert not created and not instance.events.active
    trace=instance.events.lifecycle_snapshot()
    expected='tracking_position_jump' if boundary=='jump' else 'tracking_observation_gap'
    assert any(b['reason']==expected and b['track_id']==1 for b in trace['boundaries'])
    if boundary=='jump':
        assert trace['boundaries'][0]['source_detection_index']==0
    support=next(d for d in analysis.decisions if d.rule==Rule.FOOTPATH).evidence['footpath_contact_support']
    assert support['history']['first_observed_at']==stamp and not support['history']['covers_full_window']
    subsequent=[]
    for index in range(1,22):
        position=x+index*.7
        _,_,created=analyze(instance,stamp+index/10,rgb,Detection('motorcycle',.9,(position-10,35,position+10,60),1))
        subsequent.extend(created)
    assert len(subsequent)==1 and subsequent[0].event_id!=historical.event_id
    assert asdict(historical)==original


def test_scene_reset_retains_completed_event_and_uses_new_source_binding(isolated_analyzer):
    instance=isolated_analyzer
    assert isinstance(instance.events,LifecycleAwareEventCollector)
    rgb,historical=warm_positive(instance)
    original=deepcopy(asdict(historical))
    _,_,created=analyze(instance,3.5,rgb,Detection('motorcycle',.9,(60,35,80,60),1),scene_change=True)
    assert not created and not instance.events.active
    assert asdict(historical)==original
    assert instance.footpath_context.source_snapshot()['revision']==2
    assert any(b['reason']=='scene_identity_changed' for b in instance.events.lifecycle_snapshot()['boundaries'])
