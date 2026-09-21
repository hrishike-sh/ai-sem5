import cv2
import numpy as np

from traffic_ai.arrows import road_arrows
from traffic_ai.contracts import Detection


def test_arrow_direction_comes_from_shape_not_traffic_flow():
    rgb = np.zeros((100, 100, 3), np.uint8)
    shape = np.int32([[50, 5], [80, 45], [60, 45], [60, 90], [40, 90], [40, 45], [20, 45]])
    cv2.fillPoly(rgb, [shape], (255, 255, 255))
    proposal = Detection("a road arrow", .7, (10, 1, 90, 98))
    mask = np.full((100, 100), 255, np.uint8)
    arrows = road_arrows(rgb, [proposal], mask)
    assert len(arrows) == 1
    assert arrows[0]["direction"][1] < -.9
    assert not road_arrows(rgb, [], mask)
    assert not road_arrows(rgb, [proposal], mask * 0)


def test_rectangular_road_stripe_is_not_arrow():
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[5:95, 30:70] = 255
    assert not road_arrows(rgb, [Detection("a road arrow", .9, (10, 1, 90, 99))], np.full((100, 100), 255, np.uint8))
