from dataclasses import replace

from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.rules import RuleEngine, SceneEvidence
from traffic_ai.turn_context import TurnContexts


def vehicle(x, y, identity=1):
    return Detection('car', .95, (x-10, y-20, x+10, y), identity)


def scene():
    polygon = [[0, 0], [100, 0], [100, 130], [0, 130]]
    sign = {'associated_approach': 2, 'prohibited_turns': ['left'], 'status': 'recognised_candidate'}
    return SceneEvidence(scene_id=1, stationary_camera=True, confidence=.95,
                         approach_polygon=polygon, stop_line=((0, 100), (100, 100)), prohibited_turns={'left'},
                         source_evidence={'approach': {'id': 2, 'polygon': polygon}, 'applicable_sign': sign})


def enter(contexts):
    current = scene()
    assert contexts.observe(vehicle(50, 90), 0., current) is None
    assert contexts.observe(vehicle(50, 100), .1, current) is None
    return contexts.observe(vehicle(50, 110), .2, current)


def test_entry_prohibition_survives_exit_without_reusing_signal_context():
    contexts = TurnContexts()
    entry = enter(contexts)
    outside = replace(scene(), approach_polygon=[], stop_line=None, prohibited_turns=set(), source_evidence={})
    evidence = contexts.observe(vehicle(110, 140), .3, outside)
    assert evidence == entry
    rules = RuleEngine()
    results = rules.evaluate(vehicle(110, 140), .3, outside, video=True, observed_turn='left', turn_context=evidence)
    turn = next(x for x in results if x.rule == Rule.TURN)
    assert turn.verdict == Verdict.VIOLATION
    assert turn.evidence['observed_entry']['approach_id'] == 2
    assert outside.prohibited_turns == set()


def test_sign_without_observed_entry_cannot_report_turn():
    contexts = TurnContexts()
    assert contexts.observe(vehicle(50, 110), 0., scene()) is None
    decisions = RuleEngine().evaluate(vehicle(50, 110), 0., scene(), video=True, observed_turn='left')
    assert next(x for x in decisions if x.rule == Rule.TURN).verdict == Verdict.UNKNOWN


def test_entry_without_calibrated_turn_remains_unknown():
    context = enter(TurnContexts())
    decisions = RuleEngine().evaluate(vehicle(50, 110), .2, scene(), video=True, turn_context=context)
    assert next(x for x in decisions if x.rule == Rule.TURN).verdict == Verdict.UNKNOWN


def test_gap_camera_change_and_scene_change_invalidate_entry():
    for timestamp, current in [(1.1, scene()), (.3, replace(scene(), stationary_camera=False)),
                                (.3, replace(scene(), scene_id=2))]:
        contexts = TurnContexts()
        enter(contexts)
        assert contexts.observe(vehicle(50, 120), timestamp, current) is None


def test_track_cannot_inherit_another_vehicles_entry():
    context = enter(TurnContexts())
    decisions = RuleEngine().evaluate(vehicle(50, 110, 2), .3, scene(), video=True,
                                      observed_turn='left', turn_context=context)
    assert next(x for x in decisions if x.rule == Rule.TURN).verdict == Verdict.UNKNOWN


def test_entry_expires_even_with_continuous_observations():
    contexts = TurnContexts(lifetime=.5)
    enter(contexts)
    assert contexts.observe(vehicle(50, 120), .4, scene()) is not None
    assert contexts.observe(vehicle(50, 120), .6, scene()) is None
