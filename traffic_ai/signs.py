import cv2
import numpy as np
from PIL import Image

from .runtime import cached_pretrained, configure_cache
from .contracts import Detection
from .geometry import area, intersection


def sign_candidates(rgb, existing=()):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    candidates = list(existing)
    for colour in [((hue < 12) | (hue > 160)), ((hue > 90) & (hue < 135))]:
        mask = (colour & (saturation > 50) & (value > 40)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:24]:
            hull = cv2.convexHull(contour)
            if len(hull) < 6:
                continue
            x, y, w, h = cv2.boundingRect(hull)
            perimeter = cv2.arcLength(hull, True)
            if min(w, h) < 24 or min(w, h) / max(w, h) < .65 or not perimeter:
                continue
            circularity = 4 * np.pi * cv2.contourArea(hull) / perimeter ** 2
            if circularity < .75 or np.count_nonzero(mask[y:y+h, x:x+w]) / (w * h) < .08:
                continue
            padding = max(2, round(max(w, h) * .08))
            box = (max(0, x - padding), max(0, y - padding), min(rgb.shape[1], x + w + padding), min(rgb.shape[0], y + h + padding))
            if any(intersection(box, d.box) / max(1, min(area(box), area(d.box))) > .75 for d in candidates):
                continue
            candidates.append(Detection("traffic_sign_colour_candidate", min(1., circularity), box, model="ivp_colour_shape"))
    return candidates[:16]


def sign_meaning(class_id):
    if class_id in {15, 17}:
        return {"restriction": "no_vehicles" if class_id == 15 else "no_entry", "prohibited_turns": []}
    prohibited = {33: ["left", "straight", "u_turn"], 34: ["right", "straight", "u_turn"],
                  35: ["left", "right", "u_turn"], 36: ["left", "u_turn"], 37: ["right", "u_turn"]}
    return {"restriction": None, "prohibited_turns": prohibited.get(class_id, [])}


def colour_shape_support(rgb, class_id):
    if class_id not in {15, 17, 33, 34, 35, 36, 37}:
        return False
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    colour = (((h < 12) | (h > 160)) if class_id in {15, 17} else ((h > 90) & (h < 135))) & (s > 50) & (v > 40)
    if colour.mean() < .08:
        return False
    size = max(3, min(7, round(min(rgb.shape[:2]) * .05) | 1))
    cleaned = cv2.morphologyEx(colour.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        contour = cv2.convexHull(contour)
        if len(contour) < 6:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area < rgb.shape[0] * rgb.shape[1] * .2 or perimeter == 0:
            continue
        _, _, w, h = cv2.boundingRect(contour)
        if min(w, h) / max(w, h) > .65 and 4 * np.pi * area / perimeter ** 2 > .75:
            return True
    return False


class SignClassifier:
    MODEL = "bazyl/gtsrb-model"
    REVISION = "95f6c95c765e1705d7f92161a5a42a40a5bd8338"

    def __init__(self):
        configure_cache()
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.processor = cached_pretrained(AutoImageProcessor, self.MODEL, self.REVISION, use_fast=False)
        self.model = cached_pretrained(AutoModelForImageClassification, self.MODEL, self.REVISION, use_safetensors=False).eval()

    def classify(self, rgb, detection):
        import torch

        x1, y1, x2, y2 = map(int, detection.box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        crop = rgb[y1:y2, x1:x2]
        result = {"box": detection.box, "status": "cannot_assess", "class_id": None, "label": None,
                  "proposal_source": detection.model or "provided_sign_crop",
                  "restriction": None, "prohibited_turns": [], "associated_approach": None,
                  "reason": "Sign crop is too small, ambiguous, or outside supported classes"}
        if min(crop.shape[:2], default=0) < 24 or detection.confidence < .3:
            return result
        inputs = self.processor(images=Image.fromarray(crop), return_tensors="pt")
        with torch.inference_mode():
            scores = self.model(**inputs).logits[0].softmax(dim=-1)
        probabilities, indices = scores.topk(3)
        top = int(indices[0])
        label = self.model.config.id2label.get(top)
        result.update(class_id=top, label=label, alternatives=[{"label": self.model.config.id2label.get(int(i)), "score": float(p)} for i, p in zip(indices, probabilities)])
        meaning = sign_meaning(top)
        if float(probabilities[0]) >= .9 and float(probabilities[0] - probabilities[1]) >= .6 and colour_shape_support(crop, top):
            result.update(**meaning, status="recognised_candidate", reason="Classifier and sign colour/shape agree; approach applicability and exception signs still require assessment")
        return result
