import io
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import cv2
import numpy as np
from PIL import Image
import pytest

from ivp.paint import paint_support
from ivp.pipeline import Settings, process_frame
from ivp.presentation import image_presentation, pixel_change
from ivp.stages import build_stages
from ivp.survey import survey
from tests.test_web_server import server, request
from web_server import demo_view, save_demo_assets


def scene():
    rng = np.random.default_rng(42)
    rgb = np.clip(rng.normal(65, 7, (120, 180, 3)), 0, 255).astype(np.uint8)
    rgb[45:85, 20:40] = 230
    rgb[45:85, 60:80] = 230
    rgb[45:85, 100:120] = 230
    return rgb


def test_enabled_but_unchanged_operations_are_not_presented_as_contributions():
    settings = Settings(denoise='Gaussian', gamma=.85)
    result = process_frame(np.full((72, 96, 3), 230, np.uint8), settings)
    doc, assets = image_presentation(result, settings, build_stages(result, settings), 'colour')
    assert [s['id'] for s in doc['steps']] == ['brightness']
    assert {s['id'] for s in doc['omitted_steps']} == {'denoised', 'colour', 'final'}
    assert np.array_equal(assets['final'], result.color_rgb)
    assert doc['summary']['changed_pixels_from_original'] == 72 * 96


@pytest.mark.parametrize('goal,allowed', [('regions', {'gray','smooth','contrast','selected','opening','closing','components'}), ('edges', {'gray','smooth','contrast','canny'})])
def test_only_executed_ancestors_of_selected_result_are_shown(goal, allowed):
    settings = Settings()
    result = process_frame(scene(), settings)
    graph = build_stages(result, settings)
    doc, assets = image_presentation(result, settings, graph, goal)
    assert {s['id'] for s in doc['steps']} <= allowed
    assert 'otsu' not in {s['id'] for s in doc['steps']}
    for step in doc['steps']:
        assert step['effect'] == pixel_change(assets[step['before']], assets[step['after']])
        assert step['effect']['changed_pixels'] > 0
    expected = next(s['image'] for s in graph if s['id'] == ('components' if goal == 'regions' else 'canny'))
    assert np.array_equal(assets[doc['final']], expected)


def test_identity_colour_retains_a_single_honest_result_without_forced_stages():
    result = process_frame(scene(), Settings())
    doc, assets = image_presentation(result, Settings(), build_stages(result, Settings()), 'colour')
    assert doc['steps'] == []
    assert doc['summary']['changed_pixels_from_original'] == 0
    assert np.array_equal(assets['original'], assets['final'])


def test_optional_road_capture_preserves_all_original_masks_parameters_counts_and_profiles():
    rgb = scene()
    normal, captured = paint_support(rgb), paint_support(rgb, capture_stages=True)
    for key in ['raw','mask','background','contrast','rejected','shape_mask','uncertain']:
        assert np.array_equal(getattr(normal,key), getattr(captured,key))
    assert normal.parameters == captured.parameters and normal.counts == captured.counts and normal.profiles == captured.profiles
    assert normal.trace == {}
    opened = cv2.morphologyEx(captured.trace['supported'], cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    assert np.array_equal(captured.trace['opened'], opened)
    assert np.array_equal(captured.trace['closed'], cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)))


def test_survey_capture_is_additive_and_noop_morphology_or_flanks_are_omitted():
    rgb = np.full((120, 180, 3), 65, np.uint8)
    rgb[45:85, 30:50] = 230
    old, old_images = survey(rgb, [0,0,1,1])
    new, images = survey(rgb, [0,0,1,1], capture_stages=True)
    doc = new.pop('ivp_demo')
    assert old == new
    assert all(np.array_equal(value, images[key]) for key,value in old_images.items())
    assert {s['id'] for s in doc['steps']}.isdisjoint({'opened','closed','mask'})
    assert all(s['effect']['changed_pixels'] > 0 for s in doc['steps'])
    shape = next((s for s in doc['steps'] if s['id']=='shapes'), None)
    if shape:
        assert 'bridge_opening_kernel' not in shape['parameters']
    assert old['eligible_for_geometry'] is False and old['eligible_for_rules'] is False


def test_saved_demo_requires_its_own_source_binding_and_unchanged_safe_assets(tmp_path):
    result = process_frame(scene(), Settings())
    doc, assets = image_presentation(result, Settings(), build_stages(result, Settings()), 'edges')
    binding = {'media_id':'bound','source_revision':'original','timestamp_seconds':0}
    save_demo_assets(tmp_path, doc, assets, binding)
    files = SimpleNamespace(url=lambda p, download=False: 'local://' + p.name)
    report = {'source_binding':binding,'ivp_demo':doc}
    assert demo_view(tmp_path, report, files)['source_binding'] == binding
    assert demo_view(tmp_path, {'source_binding':{**binding,'source_revision':'other'},'ivp_demo':doc}, files) is None
    (tmp_path/'executed-ivp'/next(iter(assets))).with_suffix('.png').write_bytes(b'changed')
    assert demo_view(tmp_path, report, files) is None
    assert demo_view(tmp_path, {'source_binding':binding}, files) is None
    doc['assets'] = {'../report':{'sha256':'anything'}}
    assert demo_view(tmp_path, report, files) is None


def test_http_saved_image_goal_has_durable_same_run_assets_and_legacy_export_is_unchanged(server):
    buffer = io.BytesIO();Image.fromarray(scene()).save(buffer,'PNG')
    media = request(server,'/api/upload?name=real-shaped-fixture.png',raw=buffer.getvalue())
    run = request(server,'/api/export',{'media':media['id'],'source_revision':media['source_revision'],'settings':{},'ivp_goal':'regions'})
    assert run['status']['state'] == 'complete'
    demo = run['demonstration']
    assert demo['source_binding'] == run['report']['source_binding']
    for key,url in demo['urls'].items():
        with urllib.request.urlopen(url) as response:
            Image.open(io.BytesIO(response.read())).load()
    assert request(server,'/api/run?id='+run['id'])['demonstration']['assets'] == demo['assets']


def test_http_road_capture_keeps_review_only_binding_and_actual_mask(server):
    buffer=io.BytesIO();Image.fromarray(scene()).save(buffer,'PNG')
    media=request(server,'/api/upload?name=synthetic-ui-proof.png',raw=buffer.getvalue())
    run=request(server,'/api/survey',{'media':media['id'],'source_revision':media['source_revision'],'rectangle':[0,0,1,1],'capture_presentation':True})
    assert run['demonstration']['source_binding'] == run['report']['source_binding']
    assert run['report']['eligible_for_rules'] is False and run['report']['eligible_for_geometry'] is False
    assert run['demonstration']['goal'] == 'road_appearance'
