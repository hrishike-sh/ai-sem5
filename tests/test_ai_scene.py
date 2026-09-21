import cv2
import numpy as np

from traffic_ai.segmentation import CameraStability, mask_polygons


def test_mask_polygon_excludes_other_components():
    mask = np.zeros((100, 100), np.uint8)
    mask[20:80, 10:40] = 255
    mask[5:8, 5:8] = 255
    polygons = mask_polygons(mask)
    assert len(polygons) == 1
    assert all(10 <= x <= 39 and 20 <= y <= 79 for x, y in polygons[0])


def test_camera_requires_evidence_and_rejects_motion():
    camera = CameraStability()
    blank = np.zeros((240, 320, 3), np.uint8)
    for timestamp in [0, 1, 2]:
        assert not camera.observe(blank, timestamp)[0]
    rng = np.random.default_rng(22)
    textured = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    camera.reset()
    assert not camera.observe(textured, 0)[0]
    assert not camera.observe(textured, .1)[0]
    assert camera.observe(textured, 1.2)[0]
    moved = cv2.warpAffine(textured, np.float32([[1, 0, 12], [0, 1, 0]]), (320, 240))
    assert not camera.observe(moved, 1.3)[0]
