"""Actual Analyzer/gate composition using explicit synthetic masks and mock models."""
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_ai import analyzer as module
from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.control_association import ConflictAwareApproachResolver
from traffic_ai.footpath_support import RasterFootpathSupport
from traffic_ai.rules import RuleEngine
from traffic_ai.scene_lifecycle import RefreshAwareCameraStability
from traffic_ai.turn_sign_context import ContextPaddedTurnDetector
from tests.test_analyzer_scene_lifecycle_integration import isolated_analyzer as isolated_analyzer
from tests.test_scene_lifecycle import controlled_sequence


def configure(instance, layout='sidewalk'):
    rgb = np.random.default_rng(22).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    sidewalk = np.zeros(rgb.shape[:2], np.uint8)
    sidewalk[20:111, 20:111] = 255
    if layout in ('hole', 'unknown'):
        sidewalk[30:100, 30:100] = 0
    road = np.where(sidewalk > 0, 0, 255).astype(np.uint8)
    if layout == 'unknown':
        road[30:100, 30:100] = 0
    if layout == 'conflict':
        road[20:111, 20:111] = 255
    scores = {'road': .95, 'sidewalk': .6 if layout == 'score' else .95}
    if layout == 'shape':
        sidewalk = sidewalk[:, :-1]
    instance.segmenter.segment = lambda pixels: ({'road': road, 'sidewalk': sidewalk}, scores)
    return rgb, road, sidewalk


def analyze(instance, timestamp, rgb, vehicle=None, scene_change=False):
    instance.perception.detect = lambda *args, **kwargs: [] if vehicle is None else [vehicle]
    return instance.analyze(SimpleNamespace(color_rgb=rgb, timestamp=timestamp), rgb,
                            video=True, scene_change=scene_change)


def footpath(analysis):
    return next(d for d in analysis.decisions if d.rule == Rule.FOOTPATH)


@pytest.mark.parametrize('layout', ['sidewalk', 'road', 'hole', 'unknown', 'boundary', 'conflict', 'shape', 'score'])
def test_actual_raster_binding_and_only_footpath_outcomes_change(isolated_analyzer, layout):
    instance = isolated_analyzer
    rgb, road, sidewalk = configure(instance, layout)
    sources_before = (rgb.copy(), road.copy(), sidewalk.copy())
    baseline = RuleEngine()
    first_supported = None
    for index in range(41):
        timestamp = index / 10
        x = 20 if layout == 'boundary' else (160 if layout == 'road' else 45) + index * .7
        y = 40 + index * .7 if layout == 'boundary' else 130 if layout == 'road' else 60
        vehicle = Detection('motorcycle', .9, (x-10, y-25, x+10, y), 1)
        rendered, actual, events = analyze(instance, timestamp, rgb, vehicle)
        assert isinstance(instance.footpath_context, RasterFootpathSupport)
        assert module.RasterFootpathSupport is RasterFootpathSupport
        assert isinstance(instance.camera, RefreshAwareCameraStability)
        assert isinstance(instance.approach_resolver, ConflictAwareApproachResolver)
        assert isinstance(instance.turn_sign_detector, ContextPaddedTurnDetector)
        # Same real rule engine, same source scene and observations, no gate on baseline.
        raw_scene = instance.approach_resolver.scene_for(vehicle, instance.scene, timestamp)
        original = baseline.evaluate(vehicle, timestamp, raw_scene, video=True)
        assert [d.to_dict() for d in actual.decisions if d.rule != Rule.FOOTPATH] == [
            d.to_dict() for d in original if d.rule != Rule.FOOTPATH]
        decision = footpath(actual)
        trace = decision.evidence['footpath_contact_support']
        if 'history' in trace and timestamp - trace['history']['first_observed_at'] < 1.5:
            assert decision.verdict == Verdict.UNKNOWN
        if trace['status'] == 'supported' and first_supported is None:
            first_supported = timestamp
        if layout not in ('sidewalk', 'road'):
            assert decision.verdict == Verdict.UNKNOWN and not events
            assert np.array_equal(rendered, rgb)
    assert np.array_equal(rgb, sources_before[0])
    assert np.array_equal(road, sources_before[1]) and np.array_equal(sidewalk, sources_before[2])
    if layout in ('sidewalk', 'road'):
        assert first_supported is not None
        assert decision.verdict == (Verdict.VIOLATION if layout == 'sidewalk' else Verdict.CLEAR)
        assert trace['history']['covers_full_window']
        assert timestamp - trace['history']['first_observed_at'] >= 1.5
    else:
        assert first_supported is None
    if layout == 'hole':
        assert next(d for d in original if d.rule == Rule.FOOTPATH).verdict == Verdict.VIOLATION
        assert trace['contact']['reason'] == 'polygon_raster_disagreement'
    if layout == 'conflict':
        assert trace['contact']['conflicting_pixels'] > 0
    if layout == 'shape':
        assert trace['source']['errors'] == ['sidewalk_mask_unavailable_or_invalid']
    if layout == 'score':
        assert trace['reason'] == 'semantic_scores_unresolved'
    snapshot = instance.scene_snapshot()['footpath_support']
    assert snapshot == instance.footpath_context.source_snapshot()
    assert snapshot['scene_id'] == instance.scene_id and snapshot['revision'] == instance.scene_revision
    assert instance.summary()['footpath_support'] == snapshot


def test_actual_camera_loss_rebuilds_gate_and_cannot_borrow_previous_window(isolated_analyzer):
    instance = isolated_analyzer
    rgb, _, _ = configure(instance)
    for index in range(29):
        x = 45 + index*.7
        _, analysis, _ = analyze(instance, index/10, rgb, Detection('motorcycle', .9, (x-10,35,x+10,60),1))
    assert footpath(analysis).evidence['footpath_contact_support']['status'] == 'supported'
    old = instance.footpath_context
    shifted = controlled_sequence()[-1][1]
    for timestamp, pixels in [(2.9, rgb*0), (3., shifted), (3.1, shifted), (4.2, shifted)]:
        _, analysis, _ = analyze(instance, timestamp, pixels, Detection('motorcycle', .9, (55,35,75,60),1))
    assert instance.scene_revision == 2 and instance.footpath_context is not old
    trace = footpath(analysis).evidence['footpath_contact_support']
    assert trace['source']['revision'] == 2 and trace['status'] == 'withheld'
    assert trace['history']['first_observed_at'] == 4.2
    assert not trace['history']['covers_full_window']


def test_explicit_scene_change_replaces_raster_binding(isolated_analyzer):
    instance = isolated_analyzer
    rgb, _, _ = configure(instance)
    analyze(instance, 0., rgb)
    old = instance.footpath_context
    analyze(instance, 1., rgb, scene_change=True)
    assert instance.footpath_context is not old
    source = instance.footpath_context.source_snapshot()
    assert source['scene_id'] == 1 and source['revision'] == 2
    assert instance.camera.lifecycle_snapshot()['scene_binding'] == [1,2]


@pytest.mark.parametrize('failure', ['segmentation', 'late_scene', 'cancellation'])
def test_failed_or_cancelled_refresh_cannot_bind_partial_rasters(isolated_analyzer, failure):
    instance = isolated_analyzer
    rgb, road, sidewalk = configure(instance)
    # Restore a failing segmenter interface while retaining the same source rasters.
    segment = instance.segmenter.segment
    def segment_with_failure(pixels):
        if instance.segmenter.fail_next:
            instance.segmenter.fail_next = False
            raise RuntimeError('Synthetic raster segmentation failure')
        return segment(pixels)
    instance.segmenter.segment = segment_with_failure
    for timestamp, pixels in controlled_sequence()[:-1]:
        analyze(instance, timestamp, pixels)
    old = instance.footpath_context
    source_before = (road.copy(), sidewalk.copy())
    if failure == 'segmentation':
        instance.segmenter.fail_next = True
    elif failure == 'late_scene':
        instance.turn_sign_detector.detector.fail_next = True
    else:
        def cancel():
            if instance.scene_revision > 1:
                raise InterruptedError('Synthetic raster cancellation')
        instance.cancel = cancel
    with pytest.raises((RuntimeError,InterruptedError), match='Synthetic'):
        analyze(instance, *controlled_sequence()[-1])
    assert instance.footpath_context is old
    assert instance.camera.scene_refresh_required and not instance.camera.context_ready
    assert old.source_snapshot()['revision'] == 1
    assert np.array_equal(road, source_before[0]) and np.array_equal(sidewalk, source_before[1])
    instance.cancel = lambda: None
    analyze(instance, 2.7, controlled_sequence()[-1][1])
    assert instance.footpath_context is not old
    assert instance.footpath_context.source_snapshot()['revision'] == instance.scene_revision
    assert instance.camera.context_ready
