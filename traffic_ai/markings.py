import cv2
import numpy as np
from ivp.marking_policy import review_only_policy


def grounded_markings(rgb, detections, road_mask):
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    road = cv2.dilate(road_mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    values = gray[road > 0]
    otsu = cv2.threshold(values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if values.size else 150
    white = cv2.inRange(hsv, (0, 0, max(150, int(otsu) + 1)), (179, 85, 255))
    white = cv2.bitwise_and(white, road)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    result = {"crossings": [], "stop_lines": [], **review_only_policy(),
              'proposal_origin': 'model-assisted appearance; not configured semantic geometry'}
    for detection in detections:
        if detection.confidence < .25:
            continue
        label = detection.label.lower()
        if "crossing" not in label and "crosswalk" not in label and "stop line" not in label:
            continue
        x1, y1, x2, y2 = detection.box
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
        if x2 - x1 < 12 or y2 - y1 < 4:
            continue
        crop = white[y1:y2, x1:x2]
        contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stripes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            rectangle = cv2.minAreaRect(contour)
            sides = rectangle[1]
            if min(sides) < 2 or max(sides) / min(sides) < 1.8 or area < max(25, crop.size * .002):
                continue
            if area / max(1, sides[0] * sides[1]) < .45:
                continue
            stripes.append(contour + np.array([[[x1, y1]]]))
        if ("crossing" in label or "crosswalk" in label) and len(stripes) >= 3:
            hull = cv2.convexHull(np.concatenate(stripes))[:, 0, :].tolist()
            result["crossings"].append({**review_only_policy(), "polygon": hull, "stripe_count": len(stripes), "model_score": detection.confidence,
                                        "basis": "Grounded crossing proposal supported by multiple elongated white road regions"})
        if "stop line" in label:
            lines = cv2.HoughLinesP(cv2.Canny(crop, 50, 150), 1, np.pi / 180, 15,
                                    minLineLength=max(10, (x2 - x1) * .5), maxLineGap=10)
            if lines is not None:
                line = max(lines[:, 0, :], key=lambda l: np.hypot(l[2] - l[0], l[3] - l[1]))
                a, b, c, d = map(int, line)
                result["stop_lines"].append({**review_only_policy(), "line": [[a + x1, b + y1], [c + x1, d + y1]], "model_score": detection.confidence,
                                            "basis": "Grounded stop-line proposal supported by a white line segment; travel orientation unresolved"})
    distinct = []
    for candidate in sorted(result["crossings"], key=lambda c: cv2.contourArea(np.float32(c["polygon"])), reverse=True):
        polygon = np.float32(candidate["polygon"])
        duplicate = False
        for accepted in distinct:
            other = np.float32(accepted["polygon"])
            overlap, _ = cv2.intersectConvexConvex(polygon, other)
            if overlap / max(1, min(cv2.contourArea(polygon), cv2.contourArea(other))) > .75:
                duplicate = True
                break
        if not duplicate:
            distinct.append(candidate)
    result["crossings"] = distinct
    return result, white
