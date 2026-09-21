import cv2
import numpy as np
from PIL import Image

from .runtime import cached_pretrained, configure_cache


def mask_polygons(mask, minimum_area=200):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < minimum_area:
            continue
        simplified = cv2.approxPolyDP(contour, .004 * cv2.arcLength(contour, True), True)
        if len(simplified) >= 3:
            polygons.append(simplified[:, 0, :].tolist())
    return polygons


class RoadSegmenter:
    MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    REVISION = "21b3847fae21ddee674abd31129307b6a1235bd9"

    def __init__(self, device="cpu"):
        configure_cache()
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        self.processor = cached_pretrained(AutoImageProcessor, self.MODEL, self.REVISION, use_fast=False)
        self.model = cached_pretrained(SegformerForSemanticSegmentation, self.MODEL, self.REVISION).to(device).eval()
        self.device = device

    def segment(self, rgb):
        import torch

        inputs = self.processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
            probabilities = logits.softmax(dim=1)
            confidence, labels = probabilities.max(dim=1)
        confidence, labels = confidence[0].cpu().numpy(), labels[0].cpu().numpy()
        masks, scores = {}, {}
        for index, label in self.model.config.id2label.items():
            if label in {"road", "sidewalk"}:
                mask = (labels == int(index)) & (confidence >= .8)
                masks[label] = mask.astype(np.uint8) * 255
                scores[label] = float(confidence[mask].mean()) if mask.any() else 0.
        return masks, scores


class CameraStability:
    def __init__(self):
        self.reference = None
        self.stable_since = None

    def reset(self):
        self.reference = None
        self.stable_since = None

    def observe(self, rgb, timestamp, dynamic_boxes=()):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        width = min(640, gray.shape[1])
        scale = width / gray.shape[1]
        gray = cv2.resize(gray, (width, max(2, round(gray.shape[0] * scale))))
        mask = np.full(gray.shape, 255, np.uint8)
        for box in dynamic_boxes:
            x1, y1, x2, y2 = [round(value * scale) for value in box]
            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        orb = cv2.ORB_create(nfeatures=700)
        keypoints, descriptors = orb.detectAndCompute(gray, mask)
        current = (keypoints, descriptors)
        if self.reference is None:
            self.reference = current
            return False, None
        old_points, old_descriptors = self.reference
        if descriptors is None or old_descriptors is None:
            self.stable_since = None
            self.reference = current
            return False, None
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(old_descriptors, descriptors)
        matches = [m for m in matches if m.distance < 40]
        if len(matches) < 20:
            self.stable_since = None
            self.reference = current
            return False, None
        before = np.float32([old_points[m.queryIdx].pt for m in matches])
        after = np.float32([keypoints[m.trainIdx].pt for m in matches])
        transform, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=2)
        if transform is None or inliers.mean() < .6:
            self.stable_since = None
            self.reference = current
            return False, None
        displacement = float(np.median(np.linalg.norm(after[inliers.ravel() > 0] - before[inliers.ravel() > 0], axis=1)))
        rotation_or_scale = float(np.max(np.abs(transform[:, :2] - np.eye(2))))
        if displacement > 1.5 or rotation_or_scale > .01:
            self.stable_since = None
            self.reference = current
            return False, displacement
        if self.stable_since is None:
            self.stable_since = timestamp
        return timestamp - self.stable_since >= 1., displacement
