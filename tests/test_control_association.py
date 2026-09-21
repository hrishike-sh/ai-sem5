"""Synthetic association contracts, not scene recognition/traffic ground truth."""
from copy import deepcopy

from ivp.marking_policy import configured_geometry_policy

import numpy as np

from traffic_ai.approaches import ApproachResolver, infer_approaches
from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.control_association import ConflictAwareApproachResolver
from traffic_ai.rules import RuleEngine, SceneEvidence


def corridor():
    # Same geometry as existing test_ai_approaches; inputs are explicitly synthetic.
    arrow = {**configured_geometry_policy(), "direction": [0, -1], "arrow_polygon": [[90, 250], [110, 250], [100, 210]]}
    return infer_approaches([arrow], [{**configured_geometry_policy(), "line": [[50, 150], [150, 150]]}], np.full((400, 300), 255, np.uint8))[0]


def shifted(first, offset=60):
    other = deepcopy(first)
    other["id"] = 1
    for key in ("polygon", "stop_line"):
        for point in other[key]:
            point[0] += offset
    return other


def lamp(x=100, state="red", score=.9):
    return {"box": [x - 10, 50, x + 10, 100], "state": state, "model_score": score, "synthetic": True}


def sign(x=100, recognised=True):
    return {"box": [x - 10, 50, x + 10, 100], "status": "recognised_candidate" if recognised else "cannot_assess",
            "restriction": "no_entry" if recognised else None, "prohibited_turns": [], "synthetic": True}


def support(resolver, observations):
    for timestamp in (0., .2, .4):
        resolver.update_signals(observations, timestamp, True)


def test_unknown_shared_lamp_blocks_otherwise_unique_red_and_rules():
    first = corridor()
    legacy = ApproachResolver([first, shifted(first)])
    support(legacy, [lamp(80), lamp(130, "unknown")])
    assert 0 in legacy.resolved_signals  # Reproduced ignored-competitor gap.
    resolver = ConflictAwareApproachResolver([first, shifted(first)])
    observations = [lamp(80), lamp(130, "unknown")]
    support(resolver, observations)
    assert not resolver.resolved_signals and not resolver.pending
    assert observations[0]["association"]["reason"] == "competing_observations"
    assert observations[1]["association"]["reason"] == "ambiguous_approach"
    assert observations[0]["association"]["competing_observation_indices"] == [1]
    engine = RuleEngine()
    base = SceneEvidence(stationary_camera=True, confidence=.95)
    for timestamp, y in ((.4, 170), (.6, 130)):
        vehicle = Detection("car", .9, (70, y - 20, 90, y), 1)
        decisions = engine.evaluate(vehicle, timestamp, resolver.scene_for(vehicle, base, timestamp), video=True)
        assert all(d.verdict == Verdict.UNKNOWN for d in decisions if d.rule in (Rule.RED_LIGHT, Rule.STOP_LINE, Rule.CROSSING))


def test_unreadable_sign_blocks_recognised_competitor_and_restriction():
    resolver = ConflictAwareApproachResolver([corridor()])
    observations = [sign(), sign(recognised=False)]
    resolver.assign_signs(observations)
    assert not resolver.resolved_signs
    assert all(o["association"]["reason"] == "competing_observations" for o in observations)
    vehicle = Detection("car", .9, (80, 110, 120, 130), 1)
    scene = resolver.scene_for(vehicle, SceneEvidence(stationary_camera=True, confidence=.95), 1.)
    assert not scene.restricted_polygon and not scene.prohibited_turns
    restricted = next(d for d in RuleEngine().evaluate(vehicle, 1., scene, video=True) if d.rule == Rule.RESTRICTED)
    assert restricted.verdict == Verdict.UNKNOWN


def test_unique_candidates_still_feed_existing_temporal_rule_engine():
    resolver = ConflictAwareApproachResolver([corridor()])
    resolver.assign_signs([sign()])
    observations = [lamp()]
    support(resolver, observations)
    assert observations[0]["associated_approach"] == 0
    engine = RuleEngine()
    base = SceneEvidence(stationary_camera=True, confidence=.95)
    before = Detection("car", .9, (90, 150, 110, 170), 1)
    after = Detection("car", .9, (90, 110, 110, 130), 1)
    engine.evaluate(before, .4, resolver.scene_for(before, base, .4), video=True)
    resolver.update_signals([lamp()], .6, True)
    decisions = {d.rule: d for d in engine.evaluate(after, .6, resolver.scene_for(after, base, .6), video=True)}
    assert decisions[Rule.RED_LIGHT].verdict == Verdict.VIOLATION
    assert decisions[Rule.RESTRICTED].verdict == Verdict.VIOLATION
    association = decisions[Rule.RED_LIGHT].evidence["signal_association"]["association"]
    assert association["source_pointer"] == "/signal_observations/0"
    assert association["status"] == "associated"
    assert decisions[Rule.RESTRICTED].evidence["sign"]["synthetic"] is True


def test_conflict_resets_pending_support_and_needs_fresh_confirmation():
    resolver = ConflictAwareApproachResolver([corridor()])
    support(resolver, [lamp()])
    resolver.update_signals([lamp(), lamp(score=.1)], .5, True)
    assert not resolver.resolved_signals and not resolver.pending
    resolver.update_signals([lamp()], .6, True)
    resolver.update_signals([lamp()], .8, True)
    assert not resolver.resolved_signals
    resolver.update_signals([lamp()], 1., True)
    assert resolver.resolved_signals
    vehicle = Detection("car", .9, (80, 160, 120, 200), 1)
    base = SceneEvidence(stationary_camera=True, confidence=.95)
    assert resolver.scene_for(vehicle, base, 1.).signal_associated
    assert not resolver.scene_for(vehicle, base, 1.51).signal_associated


def test_camera_time_and_scene_revision_cannot_reuse_lamp_support():
    resolver = ConflictAwareApproachResolver([corridor()])
    support(resolver, [lamp()])
    resolver.update_signals([lamp()], .5, False)
    assert not resolver.pending and not resolver.resolved_signals
    resolver.update_signals([lamp()], .6, True)
    resolver.update_signals([lamp()], .6, True)
    assert not resolver.pending
    assert resolver.graphs["signal"]["observations"][0]["reason"] == "nonincreasing_timestamp"
    resolver.update_signals([lamp()], float("nan"), True)
    assert not resolver.pending
    assert resolver.graphs["signal"]["timestamp"] is None
    new_scene = ConflictAwareApproachResolver([corridor()])
    new_scene.update_signals([lamp()], 3., True)
    assert not new_scene.resolved_signals


def test_raw_evidence_preserved_and_bad_timestamp_or_unknown_meaning_withheld():
    resolver = ConflictAwareApproachResolver([corridor()])
    observation = {**lamp(), "timestamp": 0., "nested_metadata": {"synthetic": True}}
    original = deepcopy(observation)
    resolver.update_signals([observation], 1., True)
    assert all(observation[key] == value for key, value in original.items())
    assert observation["association"]["reason"] == "observation_timestamp_mismatch"
    unknown = sign()
    unknown.update(restriction=None, prohibited_turns=[{"unsupported": True}])
    resolver.assign_signs([unknown])
    assert not resolver.resolved_signs
    assert unknown["association"]["reason"] == "unresolved_sign_meaning"
    snapshot = resolver.association_snapshot()
    snapshot["graphs"].clear()
    assert resolver.graphs  # Snapshot does not expose mutable resolver state.


def test_unlocatable_observation_withholds_instead_of_silently_dropping_it():
    resolver = ConflictAwareApproachResolver([corridor()])
    observations = [lamp(), {**lamp(400), "box": [0, 0, float("nan"), 1]}]
    support(resolver, observations)
    assert not resolver.resolved_signals
    assert all(o["association"]["reason"] == "invalid_observation_geometry" for o in observations)
    invalid = corridor()
    invalid["direction"] = [0, 0]
    resolver = ConflictAwareApproachResolver([corridor(), invalid])
    resolver.assign_signs([sign()])
    assert resolver.scene_errors and not resolver.resolved_signs


def test_vehicle_ambiguity_or_unstable_scene_clears_inherited_control_fields():
    first = corridor()
    resolver = ConflictAwareApproachResolver([first, shifted(first)])
    support(resolver, [lamp(80)])
    inherited = SceneEvidence(stationary_camera=True, confidence=.95, signal_state="red", signal_associated=True,
                              signal_confidence=.9, signal_observed_at=.4, restricted_polygon=first["polygon"],
                              restricted_classes={"car"}, prohibited_turns={"left"},
                              source_evidence={"applicable_sign": sign(), "retained": "raw"})
    overlap = Detection("car", .9, (120, 160, 140, 200), 1)
    scene = resolver.scene_for(overlap, inherited, .4)
    assert not scene.signal_associated and not scene.restricted_polygon and not scene.prohibited_turns
    assert scene.source_evidence["control_association"]["vehicle_reason"] == "ambiguous_vehicle_approach"
    assert scene.source_evidence["retained"] == "raw" and "applicable_sign" not in scene.source_evidence
    assert inherited.signal_associated and inherited.restricted_polygon
    inherited.stationary_camera = False
    unique = Detection("car", .9, (70, 160, 90, 200), 1)
    assert not resolver.scene_for(unique, inherited, .4).signal_associated


def test_independent_corridors_and_off_corridor_noise_do_not_block_each_other():
    first = corridor()
    resolver = ConflictAwareApproachResolver([first, shifted(first, 200)])
    observations = [lamp(100), lamp(300, "green"), lamp(800, "unknown")]
    support(resolver, observations)
    assert set(resolver.resolved_signals) == {0, 1}
    assert observations[-1]["association"]["reason"] == "no_compatible_approach"


def test_actual_analyzer_hook_with_mocked_perception_exports_withheld_and_no_green_boxes(monkeypatch):
    """Rehearse primary's import substitution; this is explicitly mocked inference."""
    from types import SimpleNamespace
    from traffic_ai import analyzer as module

    class Perception:
        device, manifest = "cpu", {"test_double": True}

        def reset_tracks(self):
            pass

        def detect(self, rgb, track):
            y = 170 if instance.frames < 3 else 130
            return [Detection("car", .9, (70, y - 20, 90, y), 1),
                    Detection("traffic_light", .9, tuple(lamp(80)["box"])),
                    Detection("traffic_light", .9, tuple(lamp(130)["box"]))]

    class SceneModel:
        MODEL, REVISION, device = "synthetic-test-double", "none", "cpu"

        def __init__(self, **kwargs):
            pass

        def propose(self, rgb):
            return SimpleNamespace(detections=[])

    class Segmenter(SceneModel):
        def segment(self, rgb):
            return {"road": np.full(rgb.shape[:2], 255, np.uint8)}, {"road": .95}

    class TurnSigns(SceneModel):
        manifest = {"test_double": True}

        def detect(self, rgb):
            return []

    class Signs(SceneModel):
        def classify(self, rgb, detection):
            return sign(80, recognised=detection.track_id == 1)

    class Camera:
        def reset(self):
            pass

        def observe(self, *args):
            return True, 0.

    first = corridor()
    monkeypatch.setattr(module, "Perception", Perception)
    monkeypatch.setattr(module, "SceneProposer", SceneModel)
    monkeypatch.setattr(module, "RoadSegmenter", Segmenter)
    monkeypatch.setattr(module, "TurnSignDetector", TurnSigns)
    # These are no-entry classifier observations, not raw turn-model outputs.
    # Route the same competing evidence through its actual production interface.
    monkeypatch.setattr(module, "SignClassifier", Signs)
    monkeypatch.setattr(module, "sign_candidates", lambda *args: [
        Detection("traffic sign", .9, tuple(sign(80)["box"]), identity) for identity in (1, 2)])
    monkeypatch.setattr(module, "CameraStability", Camera)
    monkeypatch.setattr(module, "ApproachResolver", ConflictAwareApproachResolver)
    monkeypatch.setattr(module, "infer_approaches", lambda *args: [first, shifted(first)])
    monkeypatch.setattr(module, "grounded_markings", lambda rgb, *args: ({"stop_lines": [], "crossings": []}, rgb[:, :, 0]))
    monkeypatch.setattr(module, "road_arrows", lambda *args: [])
    monkeypatch.setattr(module, "signal_observation", lambda rgb, detection, timestamp:
                        {**lamp(80 if detection.box[0] == 70 else 130,
                                "red" if detection.box[0] == 70 else "unknown"), "timestamp": timestamp})
    instance = module.Analyzer()
    pixels = np.zeros((400, 300, 3), np.uint8)
    for timestamp in (0., .2, .4, .6, .8):
        result = SimpleNamespace(color_rgb=pixels, timestamp=timestamp)
        rendered, analysis, events = instance.analyze(result, pixels, video=True)
        assert not events and np.array_equal(rendered, pixels)
        assert len(analysis.decisions) == 9
        assert all(d.verdict == Verdict.UNKNOWN for d in analysis.decisions
                   if d.rule in (Rule.RED_LIGHT, Rule.STOP_LINE, Rule.CROSSING, Rule.RESTRICTED, Rule.TURN))
    summary = instance.summary()
    assert summary["frames"] == 5 and not summary["events_by_rule"]
    assert summary["signal_observations"][0]["association"]["reason"] == "competing_observations"
    assert all(s["associated_approach"] is None for s in summary["sign_observations"])
