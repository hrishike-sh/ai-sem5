import cv2
import numpy as np

from traffic_ai.contracts import Detection
from traffic_ai.markings import grounded_markings
from traffic_ai.signals import signal_observation


def test_signal_colour_requires_local_lamp_support_and_no_legal_assumption():
    rgb = np.zeros((120, 120, 3), np.uint8)
    rgb[0:30, 0:30] = [255, 0, 0]
    light = Detection("traffic_light", .9, (40, 10, 65, 100))
    assert signal_observation(rgb, light, 0)["state"] == "unknown"
    cv2.circle(rgb, (52, 25), 7, (255, 0, 0), -1)
    observation = signal_observation(rgb, light, .1)
    assert observation["state"] == "red"
    assert observation["associated_approach"] is None
    cv2.circle(rgb, (52, 83), 7, (0, 255, 0), -1)
    assert signal_observation(rgb, light, .2)["state"] == "unknown"


def test_tiny_signal_unassessable():
    rgb = np.full((100, 100, 3), [255, 0, 0], np.uint8)
    assert signal_observation(rgb, Detection("traffic_light", .99, (0, 0, 4, 12)), 0)["state"] == "unknown"


def test_crossing_requires_semantic_proposal_and_road_support():
    rgb = np.zeros((120, 160, 3), np.uint8)
    for y in [35, 55, 75]:
        rgb[y:y + 8, 30:130] = 220
    road = np.full((120, 160), 255, np.uint8)
    proposal = Detection("a zebra crossing", .7, (20, 25, 140, 100))
    assert not grounded_markings(rgb, [], road)[0]["crossings"]
    assert not grounded_markings(rgb, [proposal], road * 0)[0]["crossings"]
    result, white = grounded_markings(rgb, [proposal], road)
    assert result["crossings"][0]["stripe_count"] == 3
    assert white[38, 50] == 255
