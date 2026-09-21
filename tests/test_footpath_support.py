"""Raster/temporal source contracts; synthetic masks are not traffic truth."""
from dataclasses import replace

import numpy as np

from traffic_ai.contracts import Decision, Detection, Rule, Verdict
from traffic_ai.footpath_support import RasterFootpathSupport, annotate_footpath_support
from traffic_ai.rules import RuleEngine, SceneEvidence
from traffic_ai.segmentation import mask_polygons


def fixture(hole=False, max_samples=256, max_tracks=512):
    sidewalk = np.zeros((180, 180), np.uint8)
    sidewalk[20:111, 20:111] = 255
    if hole:
        sidewalk[30:100, 30:100] = 0
    road = np.where(sidewalk > 0, 0, 255).astype(np.uint8)
    scene = SceneEvidence(scene_id=0, stationary_camera=True, confidence=.95,
                          footpath_polygons=mask_polygons(sidewalk), source_evidence={"synthetic": True})
    gate = RasterFootpathSupport(road, sidewalk, frame_shape=sidewalk.shape, scene_id=0, revision=1,
                                scores={"road": .95, "sidewalk": .95}, max_samples=max_samples, max_tracks=max_tracks)
    return gate, scene, road, sidewalk


def bike(x, y=60, identity=1):
    return Detection("motorcycle", .9, (x - 10, y - 25, x + 10, y), identity)


def footpath(decisions):
    return next(d for d in decisions if d.rule == Rule.FOOTPATH)


def test_source_hole_filled_by_outer_polygon_cannot_support_a_violation():
    gate, scene, road, sidewalk = fixture(hole=True)
    before = (road.copy(), sidewalk.copy())
    old, improved = RuleEngine(), RuleEngine()
    for index in range(20):
        timestamp = index / 10
        vehicle = bike(45 + index * .7)
        old_decisions = old.evaluate(vehicle, timestamp, scene, video=True)
        old_decision = footpath(old_decisions)
        gated = gate.scene_for(vehicle, timestamp, scene, video=True, scene_revision=1)
        decisions = annotate_footpath_support(improved.evaluate(vehicle, timestamp, gated, video=True), gated)
        assert footpath(decisions).verdict == Verdict.UNKNOWN
        assert [d.to_dict() for d in old_decisions if d.rule != Rule.FOOTPATH] == [d.to_dict() for d in decisions if d.rule != Rule.FOOTPATH]
    assert old_decision.verdict == Verdict.VIOLATION
    trace = footpath(decisions).evidence["footpath_contact_support"]
    assert trace["source_provenance"] == {"synthetic": True}
    assert trace["contact"]["reason"] == "polygon_raster_disagreement"
    assert trace["contact"]["road_pixels"] == trace["contact"]["patch_pixels"]
    assert trace["history"]["unresolved_recent_samples"] > 0
    assert np.array_equal(road, before[0]) and np.array_equal(sidewalk, before[1])
    assert scene.footpath_polygons and scene.source_evidence == {"synthetic": True}


def test_full_supported_window_retains_existing_positive_and_observed_clear_paths():
    for location, expected in (("sidewalk", Verdict.VIOLATION), ("road", Verdict.CLEAR)):
        gate, scene, _, _ = fixture()
        engine = RuleEngine()
        for index in range(20):
            vehicle = bike((45 if location == "sidewalk" else 130) + index * .7, 60 if location == "sidewalk" else 130)
            gated = gate.scene_for(vehicle, index / 10, scene, video=True, scene_revision=1)
            result = footpath(engine.evaluate(vehicle, index / 10, gated, video=True))
            if index < 15:
                assert result.verdict == Verdict.UNKNOWN
        assert result.verdict == expected
        assert gated.source_evidence["footpath_contact_support"]["status"] == "supported"


def test_boundary_unknown_and_contradictory_pixels_remain_unresolved():
    gate, scene, road, sidewalk = fixture()
    assert gate.spatial_support(bike(20), scene.footpath_polygons)["reason"] == "boundary_or_unresolved_contact_patch"
    assert gate.spatial_support(bike(0), scene.footpath_polygons)["reason"] == "contact_patch_outside_processed_frame"
    assert gate.spatial_support(bike(45), [])["reason"] == "footpath_polygons_unavailable"
    road[20:111, 20:111] = 255
    conflict = RasterFootpathSupport(road, sidewalk, frame_shape=road.shape, scene_id=0, revision=1,
                                    scores={"road": .95, "sidewalk": .95})
    assert conflict.spatial_support(bike(45), scene.footpath_polygons)["reason"] == "contradictory_semantic_masks"
    invalid = RasterFootpathSupport(road.astype(float), sidewalk, frame_shape=road.shape, scene_id=0, revision=1,
                                   scores={"road": .95, "sidewalk": .95})
    assert invalid.errors and not invalid.scene_for(bike(45), 0., scene, video=True, scene_revision=1).footpath_polygons


def test_unknown_contact_blocks_until_it_leaves_the_entire_rule_window():
    gate, scene, _, _ = fixture()
    for index in range(16):
        result = gate.scene_for(bike(45 + index * .7), index / 10, scene, video=True, scene_revision=1)
    assert result.footpath_polygons
    result = gate.scene_for(bike(20), 1.6, scene, video=True, scene_revision=1)
    assert not result.footpath_polygons
    for index in range(17, 32):
        result = gate.scene_for(bike(50 + (index - 17) * .7), index / 10, scene, video=True, scene_revision=1)
        assert not result.footpath_polygons
    result = gate.scene_for(bike(61), 3.2, scene, video=True, scene_revision=1)
    assert result.footpath_polygons
    repeated = gate.scene_for(bike(61), 3.2, scene, video=True, scene_revision=1)
    assert repeated.source_evidence["footpath_contact_support"]["reason"] == "nonincreasing_timestamp"


def test_scene_camera_geometry_and_track_discontinuity_discard_old_support():
    gate, scene, _, _ = fixture()
    for index in range(20):
        result = gate.scene_for(bike(45 + index * .7), index / 10, scene, video=True, scene_revision=1)
    assert result.footpath_polygons
    unknown = gate.scene_for(bike(60), 2., scene, video=True, scene_revision=2)
    assert not unknown.footpath_polygons and not gate._tracks
    gate.scene_for(bike(60), 2.1, scene, video=True, scene_revision=1)
    changed = replace(scene, footpath_polygons=[[[21, 20], [110, 20], [110, 110], [21, 110]]])
    reset = gate.scene_for(bike(61), 2.2, changed, video=True, scene_revision=1)
    assert not reset.footpath_polygons
    assert reset.source_evidence["footpath_contact_support"]["history"]["first_observed_at"] == 2.2
    gate.scene_for(bike(61), 2.3, replace(scene, stationary_camera=False), video=True, scene_revision=1)
    assert not gate._tracks
    gap = gate.scene_for(bike(62), 4., scene, video=True, scene_revision=1)
    assert not gap.footpath_polygons
    assert not gate.scene_for(bike(62, identity=True), 4.1, scene, video=True, scene_revision=1).footpath_polygons


def test_masks_are_copied_capacity_is_bounded_and_annotations_preserve_measurements():
    gate, scene, _, sidewalk = fixture(max_samples=2, max_tracks=2)
    signature = gate.source_snapshot()["semantic_mask_sha256"]["sidewalk"]
    sidewalk[:] = 0
    assert gate.spatial_support(bike(45), scene.footpath_polygons)["category"] == "sidewalk"
    assert gate.source_snapshot()["semantic_mask_sha256"]["sidewalk"] == signature
    for index in range(30):
        result = gate.scene_for(bike(45 + index * .1), index / 10, scene, video=True, scene_revision=1)
    assert not result.footpath_polygons
    assert len(gate._tracks[1]["samples"]) <= 2
    for identity in (2, 3):
        gate.scene_for(bike(45, identity=identity), 3., scene, video=True, scene_revision=1)
    assert len(gate._tracks) <= 2
    original = Decision(Rule.FOOTPATH, Verdict.UNKNOWN, "source reason", 3., 1, bike(45).box,
                        {"synthetic": True, "measured_value": 7})
    other = Decision(Rule.WRONG_WAY, Verdict.UNKNOWN, "unchanged", 3.)
    annotated = annotate_footpath_support([original, other], result)
    assert annotated[1] is other and annotated[0].reason == original.reason and annotated[0].timestamp == 3.
    assert annotated[0].evidence["synthetic"] and annotated[0].evidence["measured_value"] == 7
    assert "footpath_contact_support" not in original.evidence
