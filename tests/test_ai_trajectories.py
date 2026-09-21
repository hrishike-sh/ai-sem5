import pytest

from traffic_ai.contracts import Detection
from traffic_ai.trajectories import Trajectories


def detection(x, y, identity=1):
    return Detection("car", .9, (x - 5, y - 10, x + 5, y), identity)


def turn_path(tracker, transform=None):
    result = None
    for index in range(40):
        x, y = (100, 100 - index * 2) if index < 20 else (100 - (index - 20) * 2, 60)
        result = tracker.observe(detection(x, y), index * .1, True, transform)
    return result


def test_projected_turn_not_mistaken_for_legal_left_or_right():
    result = turn_path(Trajectories())
    assert result.projected_motion == "clockwise"
    assert result.angle_degrees == pytest.approx(-90)
    assert result.rule_turn is None


def test_calibrated_road_plane_resolves_left_turn():
    result = turn_path(Trajectories(), [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    assert result.rule_turn == "left"
    assert result.angle_degrees == pytest.approx(90)


def test_camera_motion_or_track_gap_discards_old_path():
    tracker = Trajectories()
    turn_path(tracker)
    assert tracker.observe(detection(60, 60), 4., False).angle_degrees is None
    assert tracker.observe(detection(58, 60), 4.1, True).angle_degrees is None
    turn_path(Trajectories())
    assert tracker.observe(detection(20, 60), 6., True).angle_degrees is None


def test_stationary_jitter_is_not_turn():
    tracker = Trajectories()
    for index in range(30):
        result = tracker.observe(detection(100 + index % 2, 100), index * .1, True)
    assert result.angle_degrees is None


def test_history_is_bounded_and_time_must_increase():
    tracker = Trajectories(max_samples=20)
    for index in range(80):
        tracker.observe(detection(index, 100), index * .1, True)
    assert len(tracker.tracks[1]) == 20
    with pytest.raises(ValueError, match="increase"):
        tracker.observe(detection(80, 100), 7.9, True)


def test_calibration_change_discards_previous_heading():
    tracker = Trajectories()
    transform = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    assert turn_path(tracker, transform).rule_turn == "left"
    result = tracker.observe(detection(58, 60), 4., True, None)
    assert result.angle_degrees is None
    assert len(tracker.tracks[1]) == 1


def test_equivalent_scaled_calibration_keeps_history():
    tracker = Trajectories()
    turn_path(tracker, [[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    result = tracker.observe(detection(58, 60), 4., True, [[-2, 0, 0], [0, 2, 0], [0, 0, -2]])
    assert result.rule_turn == "left"
    assert len(tracker.tracks[1]) == 41


def test_crossing_projective_horizon_between_samples_is_rejected():
    result = turn_path(Trajectories(), [[1, 0, 0], [0, -1, 0], [1, 0, -81]])
    assert result.rule_turn is None
    assert result.angle_degrees is None
    assert "horizon" in result.reason


def test_invalid_calibration_is_rejected_before_history_accumulates():
    with pytest.raises(ValueError, match="invertible"):
        Trajectories().observe(detection(100, 100), 0., True, [[0, 0, 0]] * 3)
