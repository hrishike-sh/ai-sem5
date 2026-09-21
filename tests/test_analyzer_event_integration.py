"""Actual Analyzer-selected collector with real rule output and guarded model mocks."""
from copy import deepcopy

import numpy as np
import pytest

from traffic_ai import analyzer as module
from traffic_ai.contracts import Decision, Detection, Rule, Verdict
from traffic_ai.event_continuity import ContradictionAwareEventCollector
from traffic_ai.render import annotate
from traffic_ai.rules import SceneEvidence
from tests.test_analyzer_scene_lifecycle_integration import isolated_analyzer as isolated_analyzer


def test_live_collector_binding_does_not_bridge_actual_rule_engine_clear(isolated_analyzer):
    instance=isolated_analyzer
    assert issubclass(module.EventCollector,ContradictionAwareEventCollector)
    assert isinstance(instance.events,ContradictionAwareEventCollector)
    polygon=[[0,0],[200,0],[200,200],[0,200]]
    scene=SceneEvidence(stationary_camera=True,confidence=.95,approach_polygon=polygon,
                        direction_polygon=polygon,permitted_direction=(1.,0.),source_evidence={'synthetic':True})
    verdicts=[]
    for stamp,x in [(0.,100),(.6,70),(.8,130),(1.1,60)]:
        vehicle=Detection('car',.95,(x-10,80,x+10,100),1)
        decisions=instance.rules.evaluate(vehicle,stamp,scene,video=True)
        before=deepcopy([d.to_dict() for d in decisions])
        verdicts.append(next(d.verdict for d in decisions if d.rule==Rule.WRONG_WAY))
        events,displayed=instance.events.update(decisions,stamp)
        assert before==[d.to_dict() for d in decisions]
        if stamp==.8:
            clear=instance.events.confirmation_snapshot()
            assert clear['cleared_pending'][0]['source_decision_indices']==[5]
    assert verdicts==[Verdict.UNKNOWN,Verdict.VIOLATION,Verdict.CLEAR,Verdict.VIOLATION]
    assert not events and not displayed
    trace=instance.summary()['event_confirmation']
    assert trace['pending'][0]['first_observed_at']==1.1
    assert trace['pending'][0]['source_decision_indices']==[5]
    trace['pending'].clear()
    assert instance.events.confirmation_snapshot()['pending']


@pytest.mark.parametrize('image,rule,identity',[(True,Rule.HELMET,None),(False,Rule.RED_LIGHT,1)])
def test_current_frame_contradiction_keeps_raw_decisions_and_no_rendered_green(isolated_analyzer,image,rule,identity):
    instance=isolated_analyzer
    assert isinstance(instance.events,ContradictionAwareEventCollector)
    values=[Decision(rule,v,'SYNTHETIC controlled collector input',0.,identity,(5,5,30,30),{'synthetic':True})
            for v in (Verdict.VIOLATION,Verdict.CLEAR)]
    before=deepcopy([d.to_dict() for d in values])
    events,displayed=instance.events.update(values,0.,image=image)
    assert not events and not displayed
    rgb=np.zeros((50,50,3),np.uint8)
    assert np.array_equal(annotate(rgb,displayed),rgb)
    assert before==[d.to_dict() for d in values]
    conflict=instance.events.confirmation_snapshot()['contradictions'][0]
    assert conflict['violation_decision_indices']==[0] and conflict['clear_decision_indices']==[1]
