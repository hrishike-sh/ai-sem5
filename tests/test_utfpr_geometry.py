"""Synthetic arithmetic contracts, not validation on UTFPR traffic/ground truth."""
import numpy as np
import pytest

from traffic_ai.utfpr_geometry import AuthorPlanarCalibration, PUBLISHED_FPS, author_time, timing_gate


def calibration(overlap=False):
    regions = {identity: [[x, 0], [x+500, 0], [x+500, 500], [x, 500]]
               for identity, x in [(1, 0), (2, 600), (3, 1200)]}
    if overlap:regions[2] = regions[1]
    # Synthetic identity H measured in60rectified px/m, not an author-camera fit.
    return AuthorPlanarCalibration(np.repeat(np.eye(3)[None, :, :], 3, axis=0), regions, regions)


def test_published_timebase_and_first200_span_are_not_decoder_guesses():
    assert float(PUBLISHED_FPS) == 30.15
    assert author_time(199) == pytest.approx(6.60033167495854)
    assert float(200/PUBLISHED_FPS) == pytest.approx(6.63349917081260)
    assert 25/float(PUBLISHED_FPS) == pytest.approx(.8291873963515755)


@pytest.mark.parametrize('index', [-1, 200, True, 1.5, '55'])
def test_original_frame_domain_rejects_invalid_or_out_of_prefix_positions(index):
    with pytest.raises(ValueError):author_time(index)


def test_half_second_and_eight_observations_are_both_required():
    assert timing_gate(list(range(8)))['state'] == 'withheld'
    assert timing_gate(list(range(16)))['state'] == 'withheld'
    assert timing_gate(list(range(17)))['state'] == 'eligible_timing'
    assert timing_gate([0, 3, 5, 7, 9, 12, 14, 16])['state'] == 'eligible_timing'
    assert timing_gate([0, 3, 5, 7, 9, 12, 16])['state'] == 'withheld'
    assert timing_gate([0, 1, 2, 3, 4, 5, 6, 25])['state'] == 'withheld'
    assert timing_gate([0, 1, 2, 3, 4, 5, 6, 6])['state'] == 'withheld'


def test_resize_pixel_centres_return_to_full_resolution_before_homogeneous_divide():
    author = calibration()
    full = author.project_road_contact([300, 240], feature_kind='independently_observed_road_contact')
    half = author.project_road_contact([149.75, 119.75], feature_kind='independently_observed_road_contact', analysis_size=(960, 540))
    assert full['rectified_point_m'] == pytest.approx([5, 4])
    assert half['rectified_point_m'] == pytest.approx(full['rectified_point_m'])
    assert full['lane_from_geometry'] == 1 and full['plate_height_factor_applied'] is False


@pytest.mark.parametrize('kind', ['bbox_bottom', 'plate_feature', 'vehicle_body', None])
def test_non_ground_anchors_cannot_receive_physical_projection(kind):
    assert calibration().project_road_contact([300, 240], feature_kind=kind)['state'] == 'withheld'


def test_ambiguous_outside_measurement_and_frame_points_are_withheld():
    assert calibration(overlap=True).project_road_contact([300, 240], feature_kind='independently_observed_road_contact')['state'] == 'withheld'
    assert calibration().project_road_contact([300, 800], feature_kind='independently_observed_road_contact')['state'] == 'withheld'
    assert calibration().project_road_contact([-1, 240], feature_kind='independently_observed_road_contact')['state'] == 'withheld'


def test_vehicle_truth_xml_is_not_a_calibration_input(tmp_path):
    matrix = tmp_path/'matrix.txt'
    np.savetxt(matrix, np.tile(np.eye(3), (3, 1)))
    truth = tmp_path/'vehicles.xml'
    truth.write_text('<GroundTruth><vehicle lane="2" speed="999"/></GroundTruth>')
    with pytest.raises(ValueError, match='not vehicle truth'):
        AuthorPlanarCalibration.from_files(matrix, truth)


def test_projective_division_occurs_before_rectified_pixels_to_metres():
    base = calibration()
    matrix = np.array([[60., 0, 0], [0, 60., 0], [.01, 0, 1]])
    projective = AuthorPlanarCalibration(np.repeat(matrix[None, :, :], 3, axis=0), base.lanes, base.measures)
    value = projective.project_road_contact([100, 100], feature_kind='independently_observed_road_contact')
    assert value['rectified_point_m'] == pytest.approx([50, 50])


def test_lane_is_chosen_from_current_geometry_and_not_supplied_by_vehicle_labels():
    base = calibration()
    matrices = base.matrices.copy()
    matrices[1, 0, 2] = 120
    value = AuthorPlanarCalibration(matrices, base.lanes, base.measures).project_road_contact(
        [800, 240], feature_kind='independently_observed_road_contact')
    assert value['lane_from_geometry'] == 2
    assert value['rectified_point_m'] == pytest.approx([(800+120)/60, 4])
