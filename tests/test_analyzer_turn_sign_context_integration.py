"""Production adapter construction with mocked perception and no model access."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_ai import analyzer as module
from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.control_association import ConflictAwareApproachResolver
from traffic_ai.scene_lifecycle import RefreshAwareCameraStability
from traffic_ai.turn_sign_context import ContextPaddedTurnDetector
from tests.test_analyzer_scene_lifecycle_integration import isolated_analyzer as isolated_analyzer
from tests.test_turn_sign_context import pixels, prediction


def frame(instance, rgb):
    instance.perception.detect = lambda *args, **kwargs: [Detection('car', .95, (20, 40, 60, 80), 1)]
    return instance.analyze(SimpleNamespace(color_rgb=rgb, timestamp=0.), rgb, video=False)


def test_production_constructs_adapter_and_preserves_existing_policies(isolated_analyzer):
    instance = isolated_analyzer
    frame(instance, pixels())
    adapter = instance.turn_sign_detector
    assert isinstance(adapter, ContextPaddedTurnDetector)
    assert isinstance(adapter.detector, module.TurnSignDetector)
    assert adapter.cancel is instance.cancel
    assert isinstance(instance.camera, RefreshAwareCameraStability)
    assert isinstance(instance.approach_resolver, ConflictAwareApproachResolver)
    assert adapter.last_audit['model_calls'] == 3
    summary = instance.summary()
    assert summary['models']['turn_signs']['test_double']
    assert summary['models']['turn_signs']['context_adapter']['max_model_calls'] == 3
    assert summary['turn_sign_context'] == adapter.last_audit
    assert instance.scene_snapshot()['turn_sign_context'] == adapter.last_audit
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


@pytest.mark.parametrize('conflict', [False, True])
def test_fallback_coordinates_and_meanings_reach_analyzer_without_creating_verdict(monkeypatch, isolated_analyzer, conflict):
    instance = isolated_analyzer
    calls = []

    def detect(self, rgb):
        calls.append(rgb.shape)
        if rgb.shape[0] == 100:
            return []
        padding = 100 if rgb.shape[0] == 300 else 200
        label = 'do_not_turn_r' if conflict and padding == 200 else 'do_not_turn_l'
        return [prediction(padding, label, .85 if padding == 100 else .91)]

    monkeypatch.setattr(module.TurnSignDetector, 'detect', detect)
    rgb = pixels()
    before = rgb.copy()
    rendered, analysis, events = frame(instance, rgb)
    assert calls == [(100, 100, 3), (300, 300, 3), (500, 500, 3)]
    assert np.array_equal(rgb, before) and np.array_equal(rendered, before)
    sign = instance.sign_observations[0]
    assert sign['box'] == [0., 0., 100., 100.] and sign['model_score'] == .91
    assert [h['model_score'] for h in sign['detector_hypotheses']] == [.85, .91]
    assert sign['status'] == ('cannot_assess' if conflict else 'recognised_candidate')
    assert sign['prohibited_turns'] == ([] if conflict else ['left'])
    assert sign['associated_approach'] is None
    assert next(d for d in analysis.decisions if d.rule == Rule.TURN).verdict == Verdict.UNKNOWN
    assert not events and instance.summary()['events_by_rule'] == {}
    snapshot = instance.scene_snapshot()
    assert snapshot['sign_observations'][0] == sign
    assert snapshot['control_association']['policy'] == 'mutual_unique_control_v1'


def test_adapter_cancellation_cannot_acknowledge_incomplete_scene(monkeypatch, isolated_analyzer):
    instance = isolated_analyzer
    stopped = False
    calls = []

    def cancel():
        if stopped:
            raise InterruptedError('Synthetic adapter cancellation')

    def detect(self, rgb):
        nonlocal stopped
        calls.append(rgb.shape)
        stopped = True
        return []

    instance.cancel = cancel
    monkeypatch.setattr(module.TurnSignDetector, 'detect', detect)
    with pytest.raises(InterruptedError, match='adapter cancellation'):
        frame(instance, pixels())
    assert calls == [(100, 100, 3)]
    assert isinstance(instance.turn_sign_detector, ContextPaddedTurnDetector)
    assert instance.turn_sign_detector.last_audit['state'] != 'complete'
    assert instance.camera.lifecycle_snapshot()['scene_binding'] is None
    assert instance.sign_observations == [] and instance.event_counts == {}
