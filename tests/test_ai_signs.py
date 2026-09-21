import cv2
import numpy as np

from traffic_ai.signs import colour_shape_support, sign_meaning, sign_candidates


def test_only_supported_sign_meanings_produce_rule_constraints():
    assert sign_meaning(17)["restriction"] == "no_entry"
    assert sign_meaning(15)["restriction"] == "no_vehicles"
    assert sign_meaning(16)["restriction"] is None
    assert sign_meaning(43)["restriction"] is None
    assert sign_meaning(35)["prohibited_turns"] == ["left", "right", "u_turn"]
    assert not sign_meaning(38)["prohibited_turns"]


def test_sign_colour_must_agree_with_meaning():
    red = np.zeros((100, 100, 3), np.uint8)
    cv2.circle(red, (50, 50), 42, (255, 0, 0), -1)
    cv2.rectangle(red, (20, 43), (80, 57), (255, 255, 255), -1)
    assert colour_shape_support(red, 17)
    assert not colour_shape_support(red, 35)
    assert not colour_shape_support(red * 0, 17)


def test_small_red_patch_on_billboard_is_not_sufficient():
    rgb = np.full((100, 100, 3), 150, np.uint8)
    rgb[5:15, 5:15] = (255, 0, 0)
    assert not colour_shape_support(rgb, 17)


def test_colour_proposal_finds_round_sign_without_semantic_detector():
    rgb = np.zeros((120, 160, 3), np.uint8)
    cv2.circle(rgb, (80, 60), 35, (255, 0, 0), -1)
    cv2.rectangle(rgb, (56, 55), (104, 65), (255, 255, 255), -1)
    proposals = sign_candidates(rgb)
    assert len(proposals) == 1
    assert proposals[0].model == "ivp_colour_shape"
    assert not sign_candidates(rgb * 0)
    assert len(sign_candidates(rgb, proposals)) == 1


def test_dark_red_border_remains_a_candidate_without_accepting_purple():
    for hue, expected in [(164, True), (172, True), (145, False)]:
        hsv = np.zeros((140, 140, 3), np.uint8)
        cv2.circle(hsv, (70, 70), 48, (hue, 150, 90), 12)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        assert bool(sign_candidates(rgb)) == expected
        assert colour_shape_support(rgb, 17) == expected
