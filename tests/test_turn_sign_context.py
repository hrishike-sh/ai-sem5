"""Mocked scale-context contracts; no trained model or real-accuracy claims."""
import cv2
import numpy as np
import pytest

from traffic_ai.turn_sign_context import ContextPaddedTurnDetector


def pixels():
    rgb = np.full((100, 100, 3), 255, np.uint8)
    cv2.circle(rgb, (50, 50), 40, (255, 0, 0), 10)
    return rgb


def prediction(padding=0, label="do_not_turn_l", score=.95, box=None):
    return {"box": box or [padding, padding, padding + 100, padding + 100], "label": label,
            "model_score": score, "synthetic": True}


class Detector:
    manifest = {"test_double": True, "synthetic": True}

    def __init__(self, callback):
        self.callback, self.calls = callback, []

    def detect(self, rgb):
        self.calls.append(rgb.shape)
        return self.callback(rgb)


def test_original_recognition_skips_fallback_and_preserves_raw_score_and_pixels():
    original = pixels()
    before = original.copy()
    detector = Detector(lambda rgb: [prediction()])
    adapter = ContextPaddedTurnDetector(detector)
    observations = adapter.detect(original)
    assert len(detector.calls) == 1 and np.array_equal(original, before)
    assert observations[0]["status"] == "recognised_candidate"
    assert observations[0]["model_score"] == .95
    assert observations[0]["associated_approach"] is None
    assert observations[0]["detector_hypotheses"][0]["source_provenance"] == {"synthetic": True}
    assert adapter.manifest["test_double"] and adapter.manifest["synthetic"]
    assert adapter.manifest["context_adapter"]["max_model_calls"] == 3
    assert "context_adapter" not in detector.manifest  # Existing model export fields are additive and unchanged.


def test_padded_fallback_maps_to_original_and_merges_consistent_cross_pass_duplicates():
    detector = Detector(lambda rgb: [] if rgb.shape[0] == 100 else [prediction(
        100 if rgb.shape[0] == 300 else 200, score=.85 if rgb.shape[0] == 300 else .91)])
    adapter = ContextPaddedTurnDetector(detector)
    observations = adapter.detect(pixels())
    assert detector.calls == [(100, 100, 3), (300, 300, 3), (500, 500, 3)]
    assert len(observations) == 1 and observations[0]["box"] == [0., 0., 100., 100.]
    assert observations[0]["prohibited_turns"] == ["left"]
    assert observations[0]["model_score"] == .91  # Existing raw score, never averaged/inflated.
    assert [h["model_score"] for h in observations[0]["detector_hypotheses"]] == [.85, .91]
    assert adapter.last_audit["state"] == "complete"


def test_strong_cross_pass_meaning_conflict_withholds_all_turn_constraints():
    detector = Detector(lambda rgb: [] if rgb.shape[0] == 100 else [prediction(
        100 if rgb.shape[0] == 300 else 200, "do_not_turn_l" if rgb.shape[0] == 300 else "do_not_turn_r")])
    observation = ContextPaddedTurnDetector(detector).detect(pixels())[0]
    assert observation["status"] == "cannot_assess" and not observation["prohibited_turns"]
    assert observation["associated_approach"] is None
    assert {h["label"] for h in observation["detector_hypotheses"]} == {"do_not_turn_l", "do_not_turn_r"}


def test_padding_only_predictions_are_excluded_and_shape_uses_original_pixels():
    detector = Detector(lambda rgb: [] if rgb.shape[0] == 100 else [prediction(box=[0, 0, 30, 30])])
    adapter = ContextPaddedTurnDetector(detector)
    assert adapter.detect(pixels()) == []
    assert all(p["rejected"][0]["reason"] == "prediction_not_supported_by_original_pixels" for p in adapter.last_audit["passes"][1:])
    detector = Detector(lambda rgb: [] if rgb.shape[0] == 100 else [prediction(100 if rgb.shape[0] == 300 else 200)])
    observations = ContextPaddedTurnDetector(detector).detect(np.zeros((100, 100, 3), np.uint8))
    assert all(o["status"] == "cannot_assess" for o in observations)


def test_canvas_budget_skips_additional_views_without_resizing_or_extra_inference():
    detector = Detector(lambda rgb: [])
    adapter = ContextPaddedTurnDetector(detector, max_canvas_pixels=80_000)
    assert adapter.detect(pixels()) == []
    assert detector.calls == [(100, 100, 3)]
    assert len(adapter.last_audit["skipped"]) == 2


def test_cancellation_returns_no_partial_result_or_extra_model_call():
    detector = Detector(lambda rgb: [])
    calls = []

    def cancel():
        calls.append(1)
        if len(calls) == 4:
            raise InterruptedError("Synthetic cancellation")

    adapter = ContextPaddedTurnDetector(detector, cancel=cancel)
    with pytest.raises(InterruptedError, match="cancellation"):
        adapter.detect(pixels())
    assert len(detector.calls) == 1 and adapter.last_audit["state"] != "complete"


def test_detector_cannot_mutate_original_and_malformed_output_does_not_promote():
    original = pixels()
    before = original.copy()

    def mutate_copy(rgb):
        rgb[:] = 0
        return [prediction()]

    ContextPaddedTurnDetector(Detector(mutate_copy)).detect(original)
    assert np.array_equal(original, before)
    detector = Detector(lambda rgb: [prediction(), prediction(box=[0, 0, float("nan"), 10])])
    adapter = ContextPaddedTurnDetector(detector)
    with pytest.raises(ValueError, match="Malformed"):
        adapter.detect(original)
    assert adapter.last_audit["state"] != "complete"


def test_ambiguous_cross_pass_object_grouping_is_not_silently_merged():
    def observe(rgb):
        if rgb.shape[0] == 100:
            return [prediction(score=.5, box=[0, 0, 70, 100]), prediction(score=.5, box=[30, 0, 100, 100])]
        return [prediction(100)] if rgb.shape[0] == 300 else []

    observations = ContextPaddedTurnDetector(Detector(observe)).detect(pixels())
    assert len(observations) == 3
    assert all(o["status"] == "cannot_assess" and not o["prohibited_turns"] for o in observations)


def test_adapter_feeds_control_context_but_cannot_create_a_turn_violation_without_entry():
    from tests.test_control_association import corridor
    from traffic_ai.control_association import ConflictAwareApproachResolver
    from traffic_ai.contracts import Detection, Rule, Verdict
    from traffic_ai.rules import RuleEngine, SceneEvidence

    approach = corridor()
    for key in ("polygon", "stop_line"):
        for point in approach[key]:
            point[0] -= 50
    detector = Detector(lambda rgb: [] if rgb.shape[0] == 100 else [prediction(100 if rgb.shape[0] == 300 else 200)])
    observations = ContextPaddedTurnDetector(detector).detect(pixels())
    resolver = ConflictAwareApproachResolver([approach])
    resolver.assign_signs(observations)
    vehicle = Detection("car", .9, (40, 160, 60, 180), 1)
    scene = resolver.scene_for(vehicle, SceneEvidence(stationary_camera=True, confidence=.95), 1.)
    assert scene.prohibited_turns == {"left"}
    assert scene.source_evidence["applicable_sign"]["source_frame_sha256"] == observations[0]["source_frame_sha256"]
    decision = next(d for d in RuleEngine().evaluate(vehicle, 1., scene, video=True, observed_turn="left") if d.rule == Rule.TURN)
    assert decision.verdict == Verdict.UNKNOWN  # Candidate recognition does not establish observed entry/applicability.
