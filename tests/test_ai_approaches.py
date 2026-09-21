from ivp.marking_policy import configured_geometry_policy
import numpy as np

from traffic_ai.approaches import ApproachResolver, aligned_approaches, infer_approaches
from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.geometry import signed_distance
from traffic_ai.rules import RuleEngine, SceneEvidence


def approaches():
    arrow = {**configured_geometry_policy(), "direction": [0, -1], "arrow_polygon": [[90, 250], [110, 250], [100, 210]]}
    return infer_approaches([arrow], [{**configured_geometry_policy(), "line": [[50, 150], [150, 150]], "model_score": .5}], np.full((400, 300), 255, np.uint8))


def lamp(state="red", x=100):
    return {"box": (x-10, 50, x+10, 100), "state": state, "model_score": .9, "associated_approach": None}


def test_stop_line_oriented_by_arrow_and_road_support_required():
    result = approaches()
    assert len(result) == 1
    line = result[0]["stop_line"]
    assert signed_distance((100, 200), line) < 0
    assert signed_distance((100, 130), line) > 0
    arrow = result[0]["arrow"]
    assert not infer_approaches([arrow], [{**configured_geometry_policy(), "line": [[50, 150], [150, 150]]}], np.zeros((400, 300), np.uint8))
    assert not infer_approaches([arrow], [{**configured_geometry_policy(), "line": [[100, 150], [100, 50]]}], np.full((400, 300), 255, np.uint8))


def test_unique_signal_requires_temporal_support_and_expires():
    resolver = ApproachResolver(approaches())
    for timestamp in [0, .2, .4]:
        observations = [lamp()]
        resolver.update_signals(observations, timestamp, True)
    assert observations[0]["associated_approach"] == 0
    vehicle = Detection("car", .9, (80, 160, 120, 200), 1)
    scene = resolver.scene_for(vehicle, SceneEvidence(), .4)
    assert scene.signal_associated and scene.signal_state == "red"
    resolver.update_signals([], .5, True)
    assert not resolver.scene_for(vehicle, SceneEvidence(), .5).signal_associated


def test_multiple_lamps_or_overlapping_approaches_remain_ambiguous():
    resolver = ApproachResolver(approaches())
    for timestamp in [0, .2, .4]:
        resolver.update_signals([lamp(x=90), lamp(x=110)], timestamp, True)
    assert not resolver.resolved_signals
    first = approaches()[0]
    assert len(aligned_approaches(lamp()["box"], [first, {**first, "id": 1}])) == 2


def test_changing_light_or_camera_motion_resets_confirmation():
    resolver = ApproachResolver(approaches())
    for timestamp in [0, .2, .4]:
        resolver.update_signals([lamp()], timestamp, True)
    resolver.update_signals([lamp("green")], .5, True)
    assert not resolver.resolved_signals
    resolver.update_signals([lamp()], .6, False)
    assert not resolver.pending


def test_supported_sign_applies_only_to_unique_corridor():
    resolver = ApproachResolver(approaches())
    sign = {"box": (80, 50, 120, 100), "status": "recognised_candidate", "restriction": "no_entry", "prohibited_turns": []}
    resolver.assign_signs([sign])
    vehicle = Detection("car", .9, (80, 160, 120, 200), 1)
    scene = resolver.scene_for(vehicle, SceneEvidence(), 0)
    assert scene.restricted_polygon and "car" in scene.restricted_classes
    assert sign["associated_approach"] == 0
    resolver.assign_signs([sign, dict(sign)])
    assert not resolver.scene_for(vehicle, SceneEvidence(), 0).restricted_polygon


def test_signal_age_is_not_refreshed_by_reading_it():
    resolver = ApproachResolver(approaches())
    for timestamp in [0, .2, .4]:
        resolver.update_signals([lamp()], timestamp, True)
    vehicle = Detection("car", .9, (80, 160, 120, 200), 1)
    assert not resolver.scene_for(vehicle, SceneEvidence(), 2).signal_associated


def test_resolved_approach_feeds_signal_and_restriction_rules():
    resolver = ApproachResolver(approaches())
    resolver.assign_signs([{"box": (80, 50, 120, 100), "status": "recognised_candidate",
                            "restriction": "no_entry", "prohibited_turns": []}])
    for timestamp in [0, .2, .4]:
        resolver.update_signals([lamp()], timestamp, True)
    engine = RuleEngine()
    base = SceneEvidence(stationary_camera=True, confidence=.95)
    before = Detection("car", .9, (90, 150, 110, 170), 1)
    engine.evaluate(before, .4, resolver.scene_for(before, base, .4), video=True)
    resolver.update_signals([lamp()], .6, True)
    after = Detection("car", .9, (90, 110, 110, 130), 1)
    decisions = {d.rule: d for d in engine.evaluate(after, .6, resolver.scene_for(after, base, .6), video=True)}
    assert decisions[Rule.RED_LIGHT].verdict == Verdict.VIOLATION
    assert decisions[Rule.RESTRICTED].verdict == Verdict.VIOLATION
    assert decisions[Rule.RED_LIGHT].evidence["signal_association"]
    assert decisions[Rule.RESTRICTED].evidence["sign"]


def test_crossing_requires_unique_corridor_and_position_beyond_line():
    resolver = ApproachResolver(approaches())
    crossing = {**configured_geometry_policy(), "polygon": [[60, 110], [140, 110], [140, 140], [60, 140]]}
    resolver.assign_crossings([crossing])
    vehicle = Detection("car", .9, (80, 110, 120, 130), 1)
    base = SceneEvidence(crossing_polygon=[[1, 1], [2, 1], [2, 2]])
    assert resolver.scene_for(vehicle, base, 0).crossing_polygon == crossing["polygon"]
    behind = {**configured_geometry_policy(), "polygon": [[60, 170], [140, 170], [140, 190], [60, 190]]}
    resolver.assign_crossings([behind])
    assert not resolver.scene_for(vehicle, base, 0).crossing_polygon
    resolver.assign_crossings([crossing, dict(crossing)])
    assert not resolver.resolved_crossings
    first = approaches()[0]
    ambiguous = ApproachResolver([first, {**first, "id": 1}])
    ambiguous.assign_crossings([crossing])
    assert not ambiguous.resolved_crossings
