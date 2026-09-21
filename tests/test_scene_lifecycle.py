"""Camera-context lifecycle contracts using known synthetic pixel transforms."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from traffic_ai.scene_lifecycle import RefreshAwareCameraStability
from traffic_ai.segmentation import CameraStability


def controlled_sequence():
    # Same deterministic textured-image construction as existing test_ai_scene.
    original = np.random.default_rng(22).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    translated = cv2.warpAffine(original, np.float32([[1, 0, 24], [0, 1, 0]]), (320, 240))
    return [(0., original), (.1, original), (1.2, original), (1.3, original * 0),
            (1.4, translated), (1.5, translated), (2.6, translated)]


def test_reference_loss_preserves_measurements_but_latches_context_refresh():
    base, camera = CameraStability(), RefreshAwareCameraStability()
    assert camera.acknowledge_scene(0, 1)
    for timestamp, pixels in controlled_sequence():
        measured = base.observe(pixels, timestamp)
        observed = camera.observe(pixels, timestamp)
        assert bool(observed[0]) == bool(measured[0])
        assert observed[1] is None if measured[1] is None else observed[1] == pytest.approx(measured[1])
        if timestamp == 1.2:
            assert camera.context_ready and not camera.scene_refresh_required
        if timestamp >= 1.3:
            assert camera.scene_refresh_required and not camera.context_ready
    assert observed[0] and observed[1] == pytest.approx(0.)
    assert "unquantified_reference_loss" in camera.lifecycle_snapshot()["reasons"]
    assert not camera.acknowledge_scene(0, 1)
    assert camera.acknowledge_scene(0, 2) and camera.context_ready
    assert camera.lifecycle_snapshot()["completed_refreshes"] == 1


def test_reference_loss_before_first_stable_frame_still_requires_refresh():
    sequence = controlled_sequence()
    camera = RefreshAwareCameraStability()
    camera.acknowledge_scene(0, 1)
    camera.observe(sequence[0][1], 0.)
    assert not camera.scene_refresh_required
    camera.observe(sequence[3][1], .1)
    assert camera.scene_refresh_required
    assert not camera.acknowledge_scene(0, 2)  # Fresh inference while unstable cannot validate a context.


def test_measured_motion_and_scene_reset_do_not_reuse_old_binding():
    camera = RefreshAwareCameraStability()
    camera.acknowledge_scene(0, 1)
    sequence = controlled_sequence()
    for timestamp, pixels in sequence[:3]:
        camera.observe(pixels, timestamp)
    camera.observe(sequence[-1][1], 1.3)
    observation = camera.lifecycle_snapshot()["last_camera_observation"]
    assert observation["measured_displacement"] > 1.5 and camera.scene_refresh_required
    camera.reset()
    assert not camera.context_ready and not camera.scene_refresh_required
    assert camera.lifecycle_snapshot()["scene_binding"] is None
    assert camera.acknowledge_scene(1, 1)
    assert not camera.observe(sequence[0][1], 0.)[0]


def test_bad_time_invalidates_and_snapshot_does_not_expose_mutable_state():
    camera = RefreshAwareCameraStability()
    camera.acknowledge_scene(0, 1)
    sequence = controlled_sequence()
    for timestamp, pixels in sequence[:3]:
        camera.observe(pixels, timestamp)
    assert camera.observe(sequence[0][1], 1.2) == (False, None)
    assert camera.scene_refresh_required and not camera.context_ready
    assert "nonincreasing_timestamp" in camera.lifecycle_snapshot()["reasons"]
    assert camera.observe(sequence[0][1], float("nan")) == (False, None)
    snapshot = camera.lifecycle_snapshot()
    assert snapshot["last_camera_observation"]["timestamp"] is None
    snapshot["reasons"].clear()
    assert camera.lifecycle_snapshot()["reasons"]
    with pytest.raises(ValueError, match="integers"):
        camera.acknowledge_scene(True, 2)


def build_analyzer(monkeypatch, refresh_hook):
    """Actual Analyzer, actual camera; perception/scene inference explicitly mocked."""
    from traffic_ai import analyzer as module

    class Perception:
        device, manifest = "cpu", {"test_double": True}

        def reset_tracks(self):
            pass

        def detect(self, *args, **kwargs):
            return []

    class SceneModel:
        MODEL, REVISION, device = "synthetic-test-double", "none", "cpu"

        def __init__(self, **kwargs):
            pass

        def propose(self, rgb):
            return SimpleNamespace(detections=[])

    class Segmenter(SceneModel):
        calls = 0
        fail_next = False

        def segment(self, rgb):
            self.calls += 1
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("Synthetic scene-inference failure")
            return {"road": np.full(rgb.shape[:2], 255, np.uint8)}, {"road": .95}

    class TurnSigns(SceneModel):
        manifest = {"test_double": True}

        def detect(self, rgb):
            return []

    class ForbiddenSignClassifier(SceneModel):
        def __init__(self, **kwargs):
            raise AssertionError("Scene lifecycle tests must not load a sign model")

    class BoundCamera(RefreshAwareCameraStability):
        def observe(self, *args, **kwargs):
            result = super().observe(*args, **kwargs)
            # Rehearse the exact primary-owned dirty-gate addition at observation.
            self.owner.scene_dirty |= self.scene_refresh_required
            return result

    class HookedAnalyzer(module.Analyzer):
        def _record_scene(self, *args):
            super()._record_scene(*args)
            assert self.camera.acknowledge_scene(self.scene_id, self.scene_revision)

    monkeypatch.setattr(module, "Perception", Perception)
    monkeypatch.setattr(module, "SceneProposer", SceneModel)
    monkeypatch.setattr(module, "RoadSegmenter", Segmenter)
    monkeypatch.setattr(module, "TurnSignDetector", TurnSigns)
    monkeypatch.setattr(module, "SignClassifier", ForbiddenSignClassifier)
    monkeypatch.setattr(module, "sign_candidates", lambda *args: [])
    monkeypatch.setattr(module, "CameraStability", BoundCamera if refresh_hook else CameraStability)
    instance = HookedAnalyzer() if refresh_hook else module.Analyzer()
    if refresh_hook:
        instance.camera.owner = instance
    return instance


def test_actual_analyzer_rebuilds_scene_after_unquantified_loss_only_with_refresh_hook(monkeypatch):
    old = build_analyzer(monkeypatch, False)
    improved = build_analyzer(monkeypatch, True)
    for timestamp, pixels in controlled_sequence():
        result = SimpleNamespace(color_rgb=pixels, timestamp=timestamp)
        old.analyze(result, pixels, video=True)
        improved.analyze(result, pixels, video=True)
        if timestamp == 1.3:
            assert not old.scene_dirty  # Reproduced gap: displacement was None.
            assert improved.scene_dirty and improved.camera.scene_refresh_required
    assert old.scene_revision == 1 and old.segmenter.calls == 1
    assert improved.scene_revision == 2 and improved.segmenter.calls == 2
    assert improved.camera.context_ready and not improved.scene_dirty


def test_failed_scene_refresh_stays_pending_until_successful_retry(monkeypatch):
    instance = build_analyzer(monkeypatch, True)
    sequence = controlled_sequence()
    for timestamp, pixels in sequence[:-1]:
        instance.analyze(SimpleNamespace(color_rgb=pixels, timestamp=timestamp), pixels, video=True)
    instance.segmenter.fail_next = True
    timestamp, pixels = sequence[-1]
    with pytest.raises(RuntimeError, match="scene-inference failure"):
        instance.analyze(SimpleNamespace(color_rgb=pixels, timestamp=timestamp), pixels, video=True)
    assert instance.camera.scene_refresh_required and not instance.camera.context_ready
    instance.analyze(SimpleNamespace(color_rgb=pixels, timestamp=2.7), pixels, video=True)
    assert instance.scene_revision == 2 and instance.camera.context_ready
