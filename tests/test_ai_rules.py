import numpy as np
import pytest

from traffic_ai.contracts import Decision, Detection, Rule, Verdict
from traffic_ai.geometry import crossed_segment, point_in_polygon
from traffic_ai.render import annotate
from traffic_ai.rules import RiderEvidence, RuleEngine, SceneEvidence


def vehicle(x=50, y=50, label="motorcycle", track=1):
    return Detection(label, .9, (x - 10, y - 20, x + 10, y), track)


def lookup(decisions):
    return {d.rule: d.verdict for d in decisions}


def scene(timestamp=0, state="red"):
    return SceneEvidence(stationary_camera=True, confidence=.95,
                         approach_polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
                         stop_line=((0, 50), (100, 50)), signal_state=state,
                         signal_associated=True, signal_confidence=.95, signal_observed_at=timestamp)


def test_missing_context_never_becomes_green_box():
    decisions = RuleEngine().evaluate(vehicle(), 0, SceneEvidence())
    assert len(decisions) == len(Rule)
    assert all(d.verdict == Verdict.UNKNOWN for d in decisions)
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    assert np.array_equal(annotate(rgb, decisions), rgb)


def test_absent_helmet_detection_is_not_evidence():
    riders = RiderEvidence(associated_people=[vehicle(label="person")], association_confident=True)
    results = lookup(RuleEngine().evaluate(vehicle(), 0, scene(), riders))
    assert results[Rule.HELMET] == Verdict.UNKNOWN
    riders.bare_heads = [Detection("no_helmet", .95, (45, 20, 55, 30))]
    assert lookup(RuleEngine().evaluate(vehicle(), 0, scene(), riders))[Rule.HELMET] == Verdict.VIOLATION


def test_three_riders_require_confident_association():
    riders = RiderEvidence(associated_people=[vehicle(label="person") for _ in range(3)])
    engine = RuleEngine()
    assert lookup(engine.evaluate(vehicle(), 0, scene(), riders))[Rule.RIDERS] == Verdict.UNKNOWN
    riders.association_confident = True
    assert lookup(engine.evaluate(vehicle(), 0, scene(), riders))[Rule.RIDERS] == Verdict.VIOLATION


def test_red_crossing_requires_prior_red_and_video():
    engine = RuleEngine()
    assert lookup(engine.evaluate(vehicle(y=40), 0, scene(), video=True))[Rule.RED_LIGHT] == Verdict.UNKNOWN
    assert lookup(engine.evaluate(vehicle(y=60), .2, scene(.2), video=True))[Rule.RED_LIGHT] == Verdict.VIOLATION
    engine.reset(0)
    engine.evaluate(vehicle(y=40), 0, scene(0, "green"), video=True)
    assert lookup(engine.evaluate(vehicle(y=60), .2, scene(.2), video=True))[Rule.RED_LIGHT] == Verdict.CLEAR
    assert lookup(engine.evaluate(vehicle(y=60), .3, scene(.3)))[Rule.RED_LIGHT] == Verdict.UNKNOWN


def test_cut_and_stale_signal_remove_evidence():
    engine = RuleEngine()
    engine.evaluate(vehicle(y=40), 0, scene(), video=True)
    changed = scene(.2)
    changed.scene_id = 1
    assert lookup(engine.evaluate(vehicle(y=60), .2, changed, video=True))[Rule.RED_LIGHT] == Verdict.UNKNOWN
    assert lookup(engine.evaluate(vehicle(y=65), 2, changed, video=True))[Rule.STOP_LINE] == Verdict.UNKNOWN


def test_no_majority_direction_fallback():
    engine = RuleEngine()
    known = scene()
    engine.evaluate(vehicle(x=80), 0, known, video=True)
    assert lookup(engine.evaluate(vehicle(x=50), .5, known, video=True))[Rule.WRONG_WAY] == Verdict.UNKNOWN
    known.permitted_direction = (1, 0)
    assert lookup(engine.evaluate(vehicle(x=20), 1, known, video=True))[Rule.WRONG_WAY] == Verdict.VIOLATION


def test_boundary_is_segment_not_infinite_line():
    assert crossed_segment((50, 40), (50, 60), ((0, 50), (100, 50)))
    assert not crossed_segment((150, 40), (150, 60), ((0, 50), (100, 50)))
    assert point_in_polygon((0, 5), [(0, 0), (10, 0), (10, 10), (0, 10)])


def test_green_annotation_preserves_source():
    rgb = np.zeros((100, 100, 3), np.uint8)
    decision = Decision(Rule.HELMET, Verdict.VIOLATION, "test", 0, box=(20, 20, 60, 60))
    output = annotate(rgb, [decision])
    assert not rgb.any()
    assert output[40, 20].tolist() == [0, 255, 0]


def test_duplicate_timestamp_rejected():
    engine = RuleEngine()
    engine.evaluate(vehicle(), 0, scene(), video=True)
    with pytest.raises(ValueError, match="increase"):
        engine.evaluate(vehicle(), 0, scene(), video=True)


def test_slow_crossing_survives_boundary_deadband():
    engine = RuleEngine()
    states = []
    for index, y in enumerate([44, 47, 49, 51, 53, 55]):
        timestamp = index * .1
        states.append(lookup(engine.evaluate(vehicle(y=y), timestamp, scene(timestamp), video=True)))
    assert states[-1][Rule.RED_LIGHT] == Verdict.VIOLATION


def test_green_between_anchor_and_crossing_breaks_red_evidence():
    engine = RuleEngine()
    engine.evaluate(vehicle(y=44), 0, scene(), video=True)
    engine.evaluate(vehicle(y=49), .1, scene(.1, "green"), video=True)
    assert lookup(engine.evaluate(vehicle(y=55), .2, scene(.2), video=True))[Rule.RED_LIGHT] == Verdict.CLEAR


def test_stop_line_requires_observed_red_entry_and_stopped_occupancy():
    engine = RuleEngine()
    engine.evaluate(vehicle(y=40), 0, scene(), video=True)
    states = []
    for index in range(1, 21):
        timestamp = index * .1
        states.append(lookup(engine.evaluate(vehicle(y=56), timestamp, scene(timestamp), video=True)))
    assert states[0][Rule.STOP_LINE] != Verdict.VIOLATION
    assert states[-1][Rule.STOP_LINE] == Verdict.VIOLATION


@pytest.mark.parametrize("initial_state", ["green", "red"])
def test_vehicle_already_beyond_line_is_not_an_encroachment(initial_state):
    engine = RuleEngine()
    engine.evaluate(vehicle(y=60), 0, scene(0, initial_state), video=True)
    for index in range(1, 21):
        timestamp = index * .1
        result = lookup(engine.evaluate(vehicle(y=60), timestamp, scene(timestamp), video=True))
        assert result[Rule.STOP_LINE] == Verdict.UNKNOWN


def test_moving_red_crossing_is_not_stopped_encroachment():
    engine = RuleEngine()
    results = []
    for index in range(20):
        timestamp = index * .1
        results.append(lookup(engine.evaluate(vehicle(y=40 + index * 3), timestamp, scene(timestamp), video=True)))
    assert any(result[Rule.RED_LIGHT] == Verdict.VIOLATION for result in results)
    assert all(result[Rule.STOP_LINE] != Verdict.VIOLATION for result in results)


def test_tracking_gap_cannot_bridge_a_red_crossing():
    engine = RuleEngine()
    engine.evaluate(vehicle(y=40), 0, scene(), video=True)
    result = lookup(engine.evaluate(vehicle(y=60), 2, scene(2), video=True))
    assert result[Rule.RED_LIGHT] == Verdict.UNKNOWN
    assert result[Rule.STOP_LINE] == Verdict.UNKNOWN


def wide_scene(timestamp):
    current = scene(timestamp)
    current.approach_polygon = [(0, 0), (400, 0), (400, 300), (0, 300)]
    current.stop_line = ((0, 50), (400, 50))
    current.permitted_direction = (1, 0)
    current.restricted_polygon = [(100, 0), (400, 0), (400, 300), (100, 300)]
    current.restricted_classes = {"motorcycle"}
    return current


@pytest.mark.parametrize("rule,before,after,elapsed", [
    (Rule.RED_LIGHT, (50, 40), (50, 180), .2),
    (Rule.WRONG_WAY, (200, 75), (50, 75), .6),
    (Rule.RESTRICTED, (40, 75), (180, 75), .2),
], ids=["red-crossing", "wrong-way", "restricted-entry"])
def test_discontinuous_track_cannot_report_motion_violation(rule, before, after, elapsed):
    engine = RuleEngine()
    engine.evaluate(vehicle(*before), 0, wide_scene(0), video=True)
    results = lookup(engine.evaluate(vehicle(*after), elapsed, wide_scene(elapsed), video=True))
    assert results[rule] == Verdict.UNKNOWN


def test_discontinuous_track_cannot_inherit_a_previous_red_entry():
    engine = RuleEngine()
    engine.evaluate(vehicle(y=40), 0, wide_scene(0), video=True)
    engine.evaluate(vehicle(y=56), .1, wide_scene(.1), video=True)
    engine.evaluate(vehicle(x=200, y=180), .2, wide_scene(.2), video=True)
    states = []
    for index in range(1, 16):
        timestamp = .2 + index * .1
        states.append(lookup(engine.evaluate(vehicle(x=200, y=180), timestamp, wide_scene(timestamp), video=True)))
    assert states[-1][Rule.STOP_LINE] == Verdict.UNKNOWN
    assert all(result[Rule.STOP_LINE] != Verdict.VIOLATION for result in states)


def test_continuous_motion_after_a_jump_can_establish_new_evidence():
    engine = RuleEngine()
    engine.evaluate(vehicle(x=200, y=75), 0, wide_scene(0), video=True)
    jumped = lookup(engine.evaluate(vehicle(x=50, y=75), .6, wide_scene(.6), video=True))
    assert jumped[Rule.WRONG_WAY] == Verdict.UNKNOWN
    engine.evaluate(vehicle(x=45, y=75), .8, wide_scene(.8), video=True)
    recovered = lookup(engine.evaluate(vehicle(x=30, y=75), 1.2, wide_scene(1.2), video=True))
    assert recovered[Rule.WRONG_WAY] == Verdict.VIOLATION


def test_jump_in_one_track_does_not_discard_another_tracks_crossing():
    engine = RuleEngine()
    for identity in (1, 2):
        engine.evaluate(vehicle(y=40, track=identity), 0, wide_scene(0), video=True)
    jumped = lookup(engine.evaluate(vehicle(y=180, track=1), .2, wide_scene(.2), video=True))
    continuous = lookup(engine.evaluate(vehicle(y=60, track=2), .2, wide_scene(.2), video=True))
    assert jumped[Rule.RED_LIGHT] == Verdict.UNKNOWN
    assert continuous[Rule.RED_LIGHT] == Verdict.VIOLATION
