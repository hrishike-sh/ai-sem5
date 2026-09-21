import cv2
import numpy as np

from traffic_ai.turn_signs import turn_sign_observation


def sign():
    rgb = np.full((100, 100, 3), 255, np.uint8)
    cv2.circle(rgb, (50, 50), 40, (255, 0, 0), 10)
    return rgb


def test_explicit_prohibition_maps_without_inventing_approach():
    for label, direction in [("do_not_turn_l", "left"), ("do_not_turn_r", "right"), ("do_not_u_turn", "u_turn")]:
        result = turn_sign_observation(sign(), (0, 0, 100, 100), label, .95)
        assert result["prohibited_turns"] == [direction]
        assert result["associated_approach"] is None


def test_class_specific_restrictions_and_signal_labels_are_not_generalised():
    for label in ["No U-Turn for Cars", "No Motobike Left Turn", "red_light", "u_turn", "left_right_lane"]:
        result = turn_sign_observation(sign(), (0, 0, 100, 100), label, .99)
        assert result["status"] == "cannot_assess"
        assert not result["prohibited_turns"]


def test_weak_or_nonsign_crop_does_not_become_turn_constraint():
    assert not turn_sign_observation(sign(), (0, 0, 100, 100), "do_not_turn_r", .5)["prohibited_turns"]
    assert not turn_sign_observation(np.zeros((100, 100, 3), np.uint8), (0, 0, 100, 100), "do_not_turn_r", .99)["prohibited_turns"]
