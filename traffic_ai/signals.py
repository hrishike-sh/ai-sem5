import cv2
import numpy as np


def signal_observation(rgb, detection, timestamp):
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = detection.box
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2))
    observation = {"box": detection.box, "timestamp": timestamp, "state": "unknown", "model_score": detection.confidence,
                   "associated_approach": None, "reason": "Signal appearance is too small or ambiguous"}
    if detection.confidence < .5 or x2 - x1 < 8 or y2 - y1 < 16:
        return observation
    crop = rgb[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    bright = (saturation >= 100) & (value >= 150)
    masks = {"red": bright & ((hue < 12) | (hue > 168)), "amber": bright & (hue >= 12) & (hue <= 35),
             "green": bright & (hue >= 38) & (hue <= 95)}
    strengths = {}
    for name, mask in masks.items():
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        strength = 0
        for index in range(1, count):
            left, top, w, h, area = stats[index]
            if area < 3 or min(w, h) / max(w, h) < .35:
                continue
            cy = centres[index][1] / crop.shape[0]
            if crop.shape[0] > crop.shape[1] * 1.3:
                expected = {"red": (0, .48), "amber": (.2, .8), "green": (.48, 1)}[name]
                if not expected[0] <= cy <= expected[1]:
                    continue
            strength = max(strength, int(area))
        strengths[name] = strength
    ordered = sorted(strengths, key=strengths.get, reverse=True)
    best, runner_up = ordered[:2]
    if strengths[best] >= max(3, crop.shape[0] * crop.shape[1] * .015) and strengths[best] > 3 * strengths[runner_up]:
        observation.update(state=best, reason="Dominant bright lamp colour inside a detected signal; approach association still required")
    observation["lamp_pixel_support"] = strengths
    return observation
