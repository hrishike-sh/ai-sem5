import csv
import io
import json
import urllib.request
import zipfile

import numpy as np
import pytest
from PIL import Image

from ivp.survey import survey
from tests.test_web_server import request
from tests.test_web_server import server as server  # noqa: PLC0414 - expose the shared pytest fixture


def stripes():
    image = np.full((120, 200, 3), 50, np.uint8)
    image[30:40, 20:80] = 240
    image[60:70, 20:80] = [240, 220, 0]
    return image


def weak_stripes():
    image = np.full((160, 240, 3), 60, np.uint8)
    image[35:47, 30:210] = 220
    image[95:107, 30:210] = 74
    return image


def test_known_geometry_coverage_and_source_coordinates():
    rgb = stripes()
    original = rgb.copy()
    report, images = survey(rgb, [0, 0, 1, 1], np.diag([2, 2, 1]))
    assert report['summary']['fragments'] == 2
    assert report['summary']['candidate_area_px'] == 1200
    assert report['summary']['coverage_percent'] == 5
    assert report['summary']['contrast_probe_mask_iou'] == 1
    assert report['method'] == 'classical_marking_survey_v5'
    assert report['fragments'][0]['source_centre_x'] == 99
    assert report['fragments'][0]['long_side_px'] == 59
    assert report['fragments'][0]['orientation_deg_clockwise_from_right'] == 0
    assert not report['fragments'][0]['touches_roi_boundary']
    assert set(np.unique(images['cleaned-mask'])) == {0, 255}
    assert np.array_equal(original, rgb)


def test_crop_excludes_second_fragment_and_reports_truncation():
    report, _ = survey(stripes(), [.2, .1, .6, .4])
    assert report['summary']['fragments'] == 1
    assert report['fragments'][0]['touches_roi_boundary']
    assert report['fragments'][0]['bbox_x'] == 40


def test_blank_and_threshold_sensitive_input():
    empty, _ = survey(np.zeros((60, 80, 3), np.uint8), [0, 0, 1, 1])
    assert empty['summary']['fragments'] == 0
    assert empty['summary']['contrast_probe_mask_iou'] is None
    report, images = survey(weak_stripes(), [0, 0, 1, 1])
    assert report['summary']['contrast_probe_mask_iou'] == .5
    assert report['sensitivity']['base_threshold'] == 12
    assert report['sensitivity']['low']['threshold'] == 9
    assert report['sensitivity']['high']['threshold'] == 15
    assert not np.array_equal(images['contrast-075x'], images['contrast-125x'])
    assert any('substantially' in s for s in report['limitations'])


def test_uniform_bright_surface_is_withheld():
    report, _ = survey(np.full((60, 80, 3), 240, np.uint8), [0, 0, 1, 1])
    assert report['summary']['candidate_area_px'] == 0
    assert any('withheld' in s for s in report['limitations'])


@pytest.mark.parametrize('roi', [None, [0, 0, 1], [0, 0, float('nan'), 1],
                               [0, 0, True, 1], [0, 0, 2, 1], [.8, .8, .2, .2], [0, 0, .01, .01]])
def test_invalid_selection(roi):
    with pytest.raises(ValueError):
        survey(stripes(), roi)


def test_survey_api_exports_original_measurements_without_enhancement(server):
    buffer = io.BytesIO()
    Image.fromarray(stripes()).save(buffer, 'PNG')
    media = request(server, '/api/upload?name=survey.png', raw=buffer.getvalue())
    run = request(server, '/api/survey', {'media': media['id'], 'rectangle': [0, 0, 1, 1],
                                       'settings': {'gamma': 4, 'saturation': 0}, 'mode': 'manual'})
    assert run['report']['summary']['coverage_percent'] == 5
    assert run['report']['uses_ai'] is False
    assert run['report']['settings']['gamma'] == 1
    assert run['status']['state'] == 'complete'
    url = next(d['url'] for d in run['downloads'] if d['name'] == 'survey-evidence.zip')
    with urllib.request.urlopen(url) as response, zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        assert len(archive.namelist()) == 14
        assert {'local-background.png', 'local-paint-contrast.png', 'rejected-seed.png'} <= set(archive.namelist())
        assert json.loads(archive.read('report.json'))['summary']['fragments'] == 2
        assert len(archive.read('fragments.csv').decode().splitlines()) == 3
        assert Image.open(archive.open('annotated-survey.png')).size == (200, 120)
        assert {'contrast-075x.png', 'contrast-125x.png'} <= set(archive.namelist())
        assert not {'threshold-minus-20.png', 'threshold-plus-20.png'} & set(archive.namelist())


def test_survey_contrast_probe_export_matches_recorded_thresholds_masks_and_source(server):
    buffer = io.BytesIO()
    Image.fromarray(weak_stripes()).save(buffer, 'PNG')
    media = request(server, '/api/upload?name=weak-paint.png', raw=buffer.getvalue())
    run = request(server, '/api/survey', {'media': media['id'], 'rectangle': [0, 0, 1, 1]})
    report = run['report']
    assert report['method'] == 'classical_marking_survey_v5'
    assert report['parameters']['method'] == 'local_paint_support_v5'
    assert report['source_binding']['media_id'] == media['id']
    assert report['source_binding']['source_revision'] == media['source_revision']
    assert report['eligible_for_geometry'] is False and report['eligible_for_rules'] is False
    link = next(d['url'] for d in run['downloads'] if d['name'] == 'survey-evidence.zip')
    with urllib.request.urlopen(link) as response, zipfile.ZipFile(io.BytesIO(response.read())) as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read('report.json')) == report
        probes = report['sensitivity']
        masks = [np.array(Image.open(archive.open(probes[key]['mask_file']))) for key in ['low', 'high']]
        assert not np.array_equal(*masks)
        assert probes['basis'] == 'minimum_local_contrast'
        assert [probes[key]['threshold'] for key in ['low', 'high']] == [9, 15]
        for key, mask in zip(['low', 'high'], masks):
            assert probes[key]['coverage_percent'] == np.mean(mask > 0)*100
        assert probes['coverage_range_percent'] == sorted(probes[key]['coverage_percent'] for key in ['low', 'high'])
        rows = list(csv.DictReader(io.StringIO(archive.read('fragments.csv').decode())))
        assert len(rows) == report['summary']['fragments'] == 2
        assert all(r['status'] == 'review_only' and r['eligible_for_rules'] == 'False' for r in rows)


def test_sparse_survey_records_flat_rescue_and_meaningful_effective_probe_thresholds():
    image = np.full((256, 384, 3), 60, np.uint8)
    image[124:132, 144:240] = 82  # 22 Value levels above background; less than1% of the ROI.
    report, images = survey(image, [0, 0, 1, 1])
    assert report['method'] == 'classical_marking_survey_v5'
    assert report['parameters']['flat_minority_rescued'] is True
    assert report['parameters']['flat_input_withheld'] is False
    assert report['parameters']['flat_input_status'] == 'minority_support_retained'
    assert report['sensitivity']['base_threshold'] == 20
    assert report['sensitivity']['low']['threshold'] == 15
    assert report['sensitivity']['high']['threshold'] == 25
    assert report['sensitivity']['low']['flat_input_status'] == 'minority_support_retained'
    assert report['sensitivity']['high']['flat_input_status'] == 'withheld_no_supported_minority'
    assert report['sensitivity']['probe_mask_iou'] == 0
    assert np.any(images['cleaned-mask']) and np.any(images['contrast-075x'])
    assert not np.any(images['contrast-125x'])
    assert any('minority appearance was retained' in text for text in report['limitations'])
    assert all('Include both road and markings' not in text for text in report['limitations'])
    assert report['eligible_for_geometry'] is False and report['eligible_for_rules'] is False
    assert report['fragments'] and all(r['eligible_for_geometry'] is False for r in report['fragments'])
