import cv2

from .contracts import Verdict


def annotate_detections(rgb, detections):
    result = rgb.copy()
    height, width = result.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        x1, y1 = max(0, round(x1)), max(0, round(y1))
        x2, y2 = min(width - 1, round(x2)), min(height - 1, round(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        colour = (0, 200, 255)
        label = f"{detection.label} {detection.confidence:.2f}"
        if detection.track_id is not None:
            label += f" #{detection.track_id}"
        cv2.rectangle(result, (x1, y1), (x2, y2), colour, 2)
        position = (x1, min(height - 3, max(14, y1 - 5)))
        cv2.putText(result, label, position, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, label, position, cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
    return result


def annotate(rgb, decisions):
    result = rgb.copy()
    grouped = {}
    for decision in decisions:
        if decision.verdict == Verdict.VIOLATION and decision.box is not None:
            grouped.setdefault(tuple(decision.box), set()).add(decision.rule.value.replace("_", " "))
    height, width = result.shape[:2]
    for box, labels in grouped.items():
        x1, y1, x2, y2 = map(round, box)
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for index, label in enumerate(sorted(labels)):
            position = (x1, max(16, y1 - 8) + index * 20)
            cv2.putText(result, label, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(result, label, position, cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1, cv2.LINE_AA)
    return result
