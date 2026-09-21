"""Real Analyzer bindings; all perception is mocked and model loading forbidden."""
import builtins
import json
import socket
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_ai import analyzer as module
from traffic_ai.control_association import ConflictAwareApproachResolver
from traffic_ai.scene_lifecycle import RefreshAwareCameraStability
from tests.test_scene_lifecycle import controlled_sequence


@pytest.fixture
def isolated_analyzer(monkeypatch):
    # Guard actual integration tests, not just standalone wrapper contracts.
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'transformers', 'ultralytics', 'requests', 'httpx'}:
            raise AssertionError(f'Lifecycle acceptance cannot import model/network library: {name}')
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('Lifecycle acceptance cannot construct a model, connect or launch a subprocess')

    class Perception:
        device, manifest = 'cpu', {'test_double': True}

        def reset_tracks(self):
            pass

        def detect(self, *args, **kwargs):
            return []

    class SceneModel:
        MODEL, REVISION, device = 'synthetic-test-double', 'none', 'cpu'

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
                raise RuntimeError('Synthetic segmentation failure')
            return {'road': np.full(rgb.shape[:2], 255, np.uint8)}, {'road': .95}

    class TurnSigns(SceneModel):
        manifest = {'test_double': True}
        fail_next = False

        def detect(self, rgb):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError('Synthetic late scene failure')
            return []

    class SignClassifier(SceneModel):
        __init__ = forbidden

    monkeypatch.setattr(module, 'Perception', Perception)
    monkeypatch.setattr(module, 'SceneProposer', SceneModel)
    monkeypatch.setattr(module, 'RoadSegmenter', Segmenter)
    monkeypatch.setattr(module, 'TurnSignDetector', TurnSigns)
    monkeypatch.setattr(module, 'SignClassifier', SignClassifier)
    monkeypatch.setattr(module, 'sign_candidates', lambda *args: [])
    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket.socket, 'connect_ex', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    # Crucially neither production CameraStability nor ApproachResolver is patched.
    assert module.CameraStability is RefreshAwareCameraStability
    assert module.ApproachResolver is ConflictAwareApproachResolver
    return module.Analyzer()


def analyze(instance, timestamp, pixels, **kwargs):
    return instance.analyze(SimpleNamespace(color_rgb=pixels, timestamp=timestamp), pixels, video=True, **kwargs)


def before_reacquisition(instance):
    for timestamp, pixels in controlled_sequence()[:-1]:
        analyze(instance, timestamp, pixels)
    assert instance.camera.scene_refresh_required and instance.scene_dirty
    assert not instance.camera.context_ready


def test_production_camera_refresh_composes_with_conflict_resolver_and_exports(isolated_analyzer):
    instance = isolated_analyzer
    for timestamp, pixels in controlled_sequence():
        _, analysis, events = analyze(instance, timestamp, pixels)
        assert not events and all(d.verdict.value == 'cannot_assess' for d in analysis.decisions)
        if timestamp == 1.2:
            old_resolver = instance.approach_resolver
            assert instance.camera.context_ready and instance.scene_revision == 1
        if 1.3 <= timestamp <= 1.5:
            assert instance.scene_dirty and not instance.camera.context_ready
    assert instance.scene_revision == 2 and instance.segmenter.calls == 2
    assert instance.approach_resolver is not old_resolver
    assert isinstance(instance.approach_resolver, ConflictAwareApproachResolver)
    assert instance.camera.context_ready and not instance.scene_dirty
    for exported in (instance.scene_snapshot(), instance.summary()):
        assert exported['control_association']['policy'] == 'mutual_unique_control_v1'
        lifecycle = exported['camera_context']
        assert lifecycle['scene_binding'] == [0, 2]
        assert lifecycle['completed_refreshes'] == 1 and lifecycle['context_ready']
        assert json.loads(json.dumps(exported, allow_nan=False)) == exported


@pytest.mark.parametrize('failure', ['segmentation', 'late_scene', 'cancellation'])
def test_failed_or_cancelled_refresh_is_not_acknowledged(isolated_analyzer, failure):
    instance = isolated_analyzer
    before_reacquisition(instance)
    if failure == 'segmentation':
        instance.segmenter.fail_next = True
    elif failure == 'late_scene':
        detector = getattr(instance.turn_sign_detector, 'detector', instance.turn_sign_detector)
        detector.fail_next = True
    else:
        def cancel_during_record():
            if instance.scene_revision > 1:
                raise InterruptedError('Synthetic cancellation')
        instance.cancel = cancel_during_record
    with pytest.raises((RuntimeError, InterruptedError), match='Synthetic'):
        analyze(instance, *controlled_sequence()[-1])
    pending = instance.camera.lifecycle_snapshot()
    assert pending['scene_binding'] == [0, 1]
    assert pending['scene_refresh_required'] and not pending['context_ready']
    assert instance.scene_dirty
    instance.cancel = lambda: None
    analyze(instance, 2.7, controlled_sequence()[-1][1])
    assert instance.camera.context_ready and not instance.scene_dirty
    assert instance.camera.lifecycle_snapshot()['scene_binding'] == [0, instance.scene_revision]


def test_explicit_scene_change_clears_old_camera_binding(isolated_analyzer):
    instance = isolated_analyzer
    before_reacquisition(instance)
    analyze(instance, 3., controlled_sequence()[0][1], scene_change=True)
    state = instance.camera.lifecycle_snapshot()
    assert state['scene_binding'] == [1, 2]
    assert not state['scene_refresh_required'] and not state['context_ready']
    assert state['refresh_requests'] == 0 and not instance.scene.stationary_camera


def test_timestamp_reversal_withholds_context_until_new_revision(isolated_analyzer):
    instance = isolated_analyzer
    sequence = controlled_sequence()
    for timestamp, pixels in sequence[:3]:
        analyze(instance, timestamp, pixels)
    analyze(instance, 1.2, sequence[0][1])
    assert instance.scene_dirty and not instance.scene.stationary_camera
    assert not instance.camera.context_ready
    for timestamp in [1.3, 1.4, 2.5]:
        analyze(instance, timestamp, sequence[0][1])
    assert instance.scene_revision == 2 and instance.camera.context_ready
