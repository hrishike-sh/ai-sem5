"""Actual appearance-to-geometry boundaries, without downloading/running models."""
from copy import deepcopy
import csv
import io
import json
from types import SimpleNamespace
import urllib.error
import urllib.request
import zipfile

import numpy as np
from PIL import Image
import pytest

from ivp.marking_policy import configured_geometry_policy, geometry_is_configured, review_only_policy
from ivp.media import source_revision
from ivp.media import png_bytes
from ivp.pipeline import process_frame
from ivp.survey import survey
from traffic_ai.approaches import ApproachResolver, infer_approaches
from traffic_ai.arrows import apply_configured_direction, road_arrows
from traffic_ai.contracts import Detection, Rule, Verdict
from traffic_ai.control_association import ConflictAwareApproachResolver
from traffic_ai.markings import grounded_markings
from traffic_ai.rules import RuleEngine, SceneEvidence
from tests.test_web_server import server as server, request


def pixels():
    image = np.full((400, 300, 3), 50, np.uint8)
    for y in (80, 100, 120):
        image[y:y+8, 60:140] = 230
    image[146:154, 50:150] = 230
    return image


def arrow():
    return {**configured_geometry_policy(), 'direction': [0, -1], 'arrow_polygon': [[90, 250], [110, 250], [100, 210]]}


def detections():
    return [Detection('a crossing', .9, (50, 70, 150, 135)),
            Detection('a stop line', .9, (40, 140, 160, 160))]


def road():
    return np.full((400, 300), 255, np.uint8)


def configured_approach():
    return infer_approaches([arrow()], [{**configured_geometry_policy(), 'line': [[50, 150], [150, 150]]}], road())[0]


def test_actual_model_assisted_white_support_is_visible_but_cannot_configure_geometry():
    result, white = grounded_markings(pixels(), detections(), road())
    assert np.count_nonzero(white) > 0
    assert result['stop_lines'] and result['crossings']
    assert result['status'] == 'review_only'
    assert not infer_approaches([arrow()], result['stop_lines'], road())
    for resolver_type in (ApproachResolver, ConflictAwareApproachResolver):
        resolver = resolver_type([configured_approach()])
        resolver.assign_crossings(deepcopy(result['crossings']))
        assert not resolver.resolved_crossings
    for item in result['stop_lines']+result['crossings']:
        assert item['eligible_for_geometry'] is False and item['eligible_for_rules'] is False


@pytest.mark.parametrize('metadata', [{}, {'eligible_for_geometry': True, 'eligible_for_rules': True},
                                    {'status': 'review_only', **{k:v for k,v in configured_geometry_policy().items() if k != 'status'}},
                                    {**configured_geometry_policy(), 'eligible_for_geometry': 1}])
def test_missing_legacy_partial_or_appearance_authority_cannot_be_promoted(metadata):
    assert not geometry_is_configured(metadata)
    assert not infer_approaches([arrow()], [{**metadata, 'line': [[50, 150], [150, 150]]}], road())
    for resolver_type in (ApproachResolver, ConflictAwareApproachResolver):
        base = configured_approach()
        for key in configured_geometry_policy():
            base.pop(key, None)
        base.update(metadata)
        assert not resolver_type([base]).approaches
        resolver = resolver_type([configured_approach()])
        crossing = {**metadata, 'polygon': [[60, 100], [140, 100], [140, 130], [60, 130]]}
        resolver.assign_crossings([crossing])
        assert not resolver.resolved_crossings


@pytest.mark.parametrize('field', ['marking', 'arrow'])
@pytest.mark.parametrize('metadata', [{}, review_only_policy()])
def test_nested_appearance_cannot_hide_in_an_explicit_looking_approach(field, metadata):
    candidate = configured_approach()
    for key in configured_geometry_policy():
        candidate[field].pop(key, None)
    candidate[field].update(metadata)
    assert not ConflictAwareApproachResolver([candidate]).approaches


def test_explicit_configuration_path_retains_independent_geometry():
    resolver = ConflictAwareApproachResolver([configured_approach()])
    resolver.assign_crossings([{**configured_geometry_policy(), 'polygon': [[60, 100], [140, 100], [140, 130], [60, 130]]}])
    assert len(resolver.approaches) == 1 and 0 in resolver.resolved_crossings


def test_actual_analyzer_scene_path_cannot_create_rule_geometry_from_plausible_proposals(monkeypatch):
    import traffic_ai.analyzer as module
    # Bypass expensive model constructors, but execute the actual _record_scene
    # -> grounded_markings -> infer_approaches -> resolver/RuleEngine path.
    instance = module.Analyzer.__new__(module.Analyzer)
    instance.turn_contexts = SimpleNamespace(reset=lambda: None)
    instance.scene_revision = 0
    instance.scene_id = 0
    instance.scene = SceneEvidence(stationary_camera=True, confidence=.95)
    instance.sign_classifier = None
    instance.turn_sign_detector = SimpleNamespace(detect=lambda rgb: [])
    instance.cancel = lambda: None
    instance.camera = SimpleNamespace()
    monkeypatch.setattr(module, 'road_arrows', lambda *args: [arrow()])
    instance._record_scene(pixels(), SimpleNamespace(detections=detections()), {'road': road()})
    assert instance.markings['stop_lines'] and instance.markings['crossings']
    assert not instance.approach_resolver.approaches
    assert not instance.approach_resolver.resolved_crossings
    vehicle = Detection('car', .95, (90, 110, 110, 130), 1)
    # Even if an old scene has semantic fields, the current consumer clears
    # them rather than inheriting stale automatic geometry.
    stale = SceneEvidence(stationary_camera=True, confidence=.95, stop_line=((50, 150), (150, 150)),
                          crossing_polygon=[[60, 100], [140, 100], [140, 130]],
                          signal_associated=True, signal_confidence=.99, signal_state='red', signal_observed_at=0.)
    clean = instance.approach_resolver.scene_for(vehicle, stale, 0.)
    assert clean.stop_line is None and clean.crossing_polygon == [] and not clean.signal_associated
    decisions = RuleEngine().evaluate(vehicle, 0., clean, video=True)
    assert all(d.verdict == Verdict.UNKNOWN for d in decisions if d.rule in (Rule.RED_LIGHT, Rule.STOP_LINE, Rule.CROSSING))


def test_pipeline_and_survey_keep_review_only_json_and_raw_evidence():
    image = pixels()
    result = process_frame(image)
    assert result.candidates['status'] == 'review_only'
    assert result.candidates['eligible_for_geometry'] is False
    assert {'Paint colour seed', 'Paint shape proposals', 'Paint unbracketed proposals'} <= set(result.features)
    report, images = survey(image, [0, 0, 1, 1])
    roundtrip = json.loads(json.dumps(report))
    assert roundtrip['eligible_for_rules'] is False
    assert all(not geometry_is_configured(r) for r in roundtrip['fragments'])
    assert report['component_profile_evidence']
    assert {'colour-seed', 'shape-proposals', 'unbracketed-proposals'} <= set(images)


def test_api_survey_export_binds_source_and_keeps_ineligible_csv_and_stage_masks(server):
    buffer = io.BytesIO()
    Image.fromarray(pixels()).save(buffer, 'PNG')
    media = request(server, '/api/upload?name=appearance.png', raw=buffer.getvalue())
    run = request(server, '/api/survey', {'media': media['id'], 'rectangle': [0, 0, 1, 1]})
    report = run['report']
    assert report['source_binding']['media_id'] == media['id']
    assert report['source_binding']['source_revision'] == media['source_revision']
    assert report['source_binding']['timestamp_seconds'] == 0
    assert report['eligible_for_geometry'] is False and report['eligible_for_rules'] is False
    link = next(d['url'] for d in run['downloads'] if d['name'] == 'survey-evidence.zip')
    with urllib.request.urlopen(link) as response, zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        assert {'colour-seed.png', 'shape-proposals.png', 'unbracketed-proposals.png'} <= set(archive.namelist())
        exported = json.loads(archive.read('report.json'))
        assert exported == report
        rows = list(csv.DictReader(io.StringIO(archive.read('fragments.csv').decode())))
        assert rows and all(row['status'] == 'review_only' and row['eligible_for_geometry'] == 'False'
                            and row['eligible_for_rules'] == 'False' for row in rows)


def test_api_changed_source_is_rejected_before_survey_or_run(server, monkeypatch):
    buffer = io.BytesIO()
    Image.fromarray(pixels()).save(buffer, 'PNG')
    media = request(server, '/api/upload?name=stale.png', raw=buffer.getvalue())
    path, _ = server.lab.media[media['id']]
    Image.new('RGB', (300, 400), 'red').save(path)
    monkeypatch.setattr('web_server.new_run', lambda: pytest.fail('Stale source must not create a run'))
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(server, '/api/survey', {'media': media['id'], 'rectangle': [0, 0, 1, 1]})
    assert 'source changed' in exc.value.read().decode().lower()


def test_actual_motion_api_rejects_survey_as_incident_configuration(server, tmp_path, monkeypatch):
    report, _ = survey(pixels(), [0, 0, 1, 1])
    path = tmp_path/'bound-video.mp4'
    path.write_bytes(b'Not decoded: validation-only request, no job may start.')
    info = {'kind': 'video', 'source_revision': source_revision(path)}
    server.lab.media['validation-video'] = (path, info)
    monkeypatch.setattr('web_server.start_video', lambda *a, **k: pytest.fail('Appearance cannot start incident work'))
    with pytest.raises(urllib.error.HTTPError) as exc:
        request(server, '/api/motion', {'media': 'validation-video', 'incidents': report})
    assert 'Unknown incident settings' in exc.value.read().decode()


def test_standalone_image_mask_zip_has_bound_policy_and_unchanged_png_payloads(server):
    image = pixels()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, 'PNG')
    media = request(server, '/api/upload?name=mask-policy.png', raw=buffer.getvalue())
    masks = ['Paint candidate mask', 'White marking mask']
    run = request(server, '/api/export', {'media': media['id'], 'masks': masks})
    expected = process_frame(image)
    link = next(d['url'] for d in run['downloads'] if d['name'] == 'masks.zip')
    with urllib.request.urlopen(link) as response, zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['selected_masks'] == masks
        assert manifest['eligible_for_geometry'] is False and manifest['eligible_for_rules'] is False
        assert manifest['source_binding']['media_id'] == media['id']
        assert manifest['source_binding']['source_revision'] == media['source_revision']
        assert manifest['source_binding']['timestamp_seconds'] == 0
        for name in masks:
            assert archive.read(name.replace(' ', '_')+'.png') == png_bytes(expected.features[name])


@pytest.mark.parametrize('metadata', [{}, review_only_policy()])
def test_unflagged_or_review_only_arrow_cannot_supply_approach_direction_or_wrong_way(metadata):
    marking = {**metadata, 'direction': [0, -1], 'arrow_polygon': [[90, 250], [110, 250], [100, 210]],
               'polygon': [[0, 0], [300, 0], [300, 400], [0, 400]]}
    assert not infer_approaches([marking], [{**configured_geometry_policy(), 'line': [[50, 150], [150, 150]]}], road())
    scene = SceneEvidence(stationary_camera=True, confidence=.95, approach_polygon=marking['polygon'])
    engine = RuleEngine()
    for stamp, y in [(0., 100), (.5, 125), (1., 150)]:
        vehicle = Detection('car', .95, (90, y-20, 110, y), 1)
        current = apply_configured_direction([marking], vehicle, scene)
        assert current.permitted_direction is None and not current.direction_polygon
        assert 'direction_marking' not in current.source_evidence
        decisions = engine.evaluate(vehicle, stamp, current, video=True)
        assert next(d for d in decisions if d.rule == Rule.WRONG_WAY).verdict == Verdict.UNKNOWN
    stale = SceneEvidence(permitted_direction=(0, -1), direction_polygon=marking['polygon'], source_evidence={'direction_marking': marking})
    assert apply_configured_direction([], vehicle, stale).permitted_direction is None


def test_independently_configured_direction_keeps_its_own_legitimate_path():
    region = {**arrow(), 'polygon': [[0, 0], [300, 0], [300, 400], [0, 400]]}
    vehicle = Detection('car', .95, (90, 120, 110, 140), 1)
    scene = SceneEvidence(approach_polygon=region['polygon'])
    configured = apply_configured_direction([region], vehicle, scene)
    assert configured.permitted_direction == (0, -1)
    assert configured.source_evidence['direction_marking'] == region


def test_actual_arrow_pixels_are_review_only_even_with_a_geometric_tip():
    import cv2
    image = np.zeros((180, 180, 3), np.uint8)
    shape = np.array([[80, 150], [100, 150], [100, 75], [120, 75], [90, 20], [60, 75], [80, 75]])
    cv2.fillPoly(image, [shape], (230, 230, 230))
    proposals = [Detection('a road arrow', .95, (40, 10, 140, 160))]
    rows = road_arrows(image, proposals, np.full(image.shape[:2], 255, np.uint8))
    assert rows and all(row['status'] == 'review_only' and not geometry_is_configured(row) for row in rows)
