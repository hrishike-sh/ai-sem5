import cv2
import numpy as np
from dataclasses import replace

from ivp.marking_policy import geometry_is_configured, review_only_policy
from .geometry import point_in_polygon


def road_arrows(rgb, proposals, road_mask):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    observations = []
    for proposal in proposals:
        if "arrow" not in proposal.label or proposal.confidence < .3:
            continue
        x1, y1, x2, y2 = map(int, proposal.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
        crop = gray[y1:y2, x1:x2]
        if min(crop.shape, default=0) < 12:
            continue
        _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_and(binary, road_mask[y1:y2, x1:x2])
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 50:
            continue
        polygon = cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
        if not 6 <= len(polygon) <= 9:
            continue
        points = polygon[:, 0, :].astype(float)
        centre = points.mean(axis=0)
        _, vectors = np.linalg.eigh(np.cov((points - centre).T))
        axis = vectors[:, -1]
        side = np.array([-axis[1], axis[0]])
        along, across = (points - centre) @ axis, (points - centre) @ side
        length, width = np.ptp(along), np.ptp(across)
        if length < width * 1.2:
            continue
        end_indices = [int(np.argmin(along)), int(np.argmax(along))]
        tips = []
        for index in end_indices:
            current = points[index]
            a, b = points[index - 1] - current, points[(index + 1) % len(points)] - current
            cosine = np.dot(a, b) / max(1e-9, np.linalg.norm(a) * np.linalg.norm(b))
            angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            if angle < 80:
                tips.append(index)
        if len(tips) != 1:
            continue
        tip = points[tips[0]]
        direction = axis * (1 if np.dot(tip - centre, axis) > 0 else -1)
        side = np.array([-direction[1], direction[0]])
        centre += np.array([x1, y1])
        region = [centre + a * direction * length + b * side * width for a, b in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
        observations.append({**review_only_policy(), "direction": direction.tolist(), "polygon": np.asarray(region).tolist(), "model_score": proposal.confidence,
                             "arrow_polygon": (points + np.array([x1, y1])).tolist(),
                             "basis": "Grounded road-arrow proposal with one geometric tip; applies only near this marking"})
    return observations


def apply_configured_direction(regions, vehicle, scene):
    """Explicit geometry may supply direction; automatic/legacy arrows may not."""
    previous = scene.source_evidence.get('direction_marking')
    if previous is not None and not geometry_is_configured(previous):
        evidence = {k:v for k,v in scene.source_evidence.items() if k != 'direction_marking'}
        scene = replace(scene, permitted_direction=None, direction_polygon=[], source_evidence=evidence)
    applicable = [region for region in regions if geometry_is_configured(region)
                  and point_in_polygon(vehicle.ground_point, region['polygon'])]
    if len(applicable) == 1 and point_in_polygon(vehicle.ground_point, scene.approach_polygon):
        marking = applicable[0]
        return replace(scene, permitted_direction=tuple(marking['direction']), direction_polygon=marking['polygon'],
                       source_evidence={**scene.source_evidence, 'direction_marking': marking})
    return scene
