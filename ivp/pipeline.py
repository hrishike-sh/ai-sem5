from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .paint import paint_support
from .marking_policy import review_only_policy


@dataclass(frozen=True)
class Settings:
    max_width: int = 0
    denoise: str = "Off"
    kernel: int = 3
    gamma: float = 1.0
    clahe: bool = False
    clip_limit: float = 2.0
    sharpen: float = 0.0
    canny_low: int = 60
    canny_high: int = 160
    threshold_mode: str = "Otsu"
    threshold_value: int = 170
    morphology: str = "Open + close"
    morph_kernel: int = 3
    hue_center: int = 0
    hue_tolerance: int = 12
    saturation_min: int = 90
    value_min: int = 70
    line_length: int = 50
    scene_threshold: float = 0.45
    exposure: float = 0.0
    white_balance: float = 0.0
    stretch: float = 0.0
    saturation: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in (self.exposure, self.white_balance, self.stretch, self.saturation)):
            raise ValueError("Colour adjustments must be finite numbers.")
        if not -3 <= self.exposure <= 3 or not 0 <= self.white_balance <= 1:
            raise ValueError("Exposure must be -3 to 3 stops; white balance strength must be 0 to 1.")
        if not 0 <= self.stretch <= 10 or not 0 <= self.saturation <= 2:
            raise ValueError("Contrast clipping must be 0 to 10 percent; saturation must be 0 to 2.")
        if self.max_width < 0 or self.kernel not in (3, 5, 7, 9):
            raise ValueError("Resize width must be non-negative; filter size must be 3, 5, 7, or 9.")
        if self.denoise not in ("Off", "Gaussian", "Median", "Bilateral"):
            raise ValueError("Unknown denoising method.")
        if not 0.2 <= self.gamma <= 4 or not 0 <= self.sharpen <= 2:
            raise ValueError("Gamma or sharpening strength is out of range.")
        if not 1 <= self.clip_limit <= 8:
            raise ValueError("CLAHE clip limit must be between 1 and 8.")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("The upper Canny threshold must exceed the lower threshold.")
        if self.threshold_mode not in ("Otsu", "Adaptive", "Fixed") or not 0 <= self.threshold_value <= 255:
            raise ValueError("Invalid threshold configuration.")
        if self.morphology not in ("Off", "Open", "Close", "Open + close") or self.morph_kernel not in (3, 5, 7, 9):
            raise ValueError("Invalid morphology configuration.")
        if not 0 <= self.hue_center <= 179 or not 1 <= self.hue_tolerance <= 45:
            raise ValueError("Hue uses OpenCV's 0–179 scale.")
        if not 0 <= self.saturation_min <= 255 or not 0 <= self.value_min <= 255:
            raise ValueError("Saturation and value must be between 0 and 255.")
        if self.line_length < 5 or not 0.05 <= self.scene_threshold <= 1:
            raise ValueError("Invalid line length or scene threshold.")

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "Original": Settings(),
    "Low light": Settings(gamma=1.5, clahe=True, denoise="Bilateral"),
    "Noisy footage": Settings(denoise="Median", kernel=3),
    "Road features": Settings(denoise="Gaussian", clahe=True, canny_low=45, canny_high=130),
    "Colour cast": Settings(white_balance=0.65),
    "Flat contrast": Settings(stretch=1.0),
}


@dataclass
class FrameResult:
    color_rgb: np.ndarray
    timestamp: float
    source_size: tuple[int, int]
    transform_to_source: np.ndarray
    measurements: dict
    stages: dict[str, np.ndarray] = field(default_factory=dict)
    features: dict[str, np.ndarray] = field(default_factory=dict)
    candidates: dict = field(default_factory=dict)


def quality(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    highlights = float(np.mean(gray >= 245))
    shadows = float(np.mean(gray <= 10))
    probabilities = np.bincount(gray.ravel(), minlength=256) / gray.size
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)))
    flags = []
    if brightness < 55:
        flags.append("Dark frame")
    if brightness > 205 or highlights > 0.35:
        flags.append("Bright / clipped highlights")
    if contrast < 28:
        flags.append("Low contrast")
    if blur < 70:
        flags.append("Low edge detail / possible blur")
    return {"brightness": brightness, "contrast": contrast, "laplacian_variance": blur,
            "highlight_fraction": highlights, "shadow_fraction": shadows, "entropy_bits": entropy, "flags": flags}


def histogram(rgb):
    return np.bincount(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).ravel(), minlength=256).tolist()


def scene_signature(rgb):
    small = cv2.resize(rgb, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def scene_distance(previous, current):
    return 0.0 if previous is None else float(cv2.compareHist(previous, current, cv2.HISTCMP_BHATTACHARYYA))


def cleanup(mask, settings):
    kernel = np.ones((settings.morph_kernel, settings.morph_kernel), np.uint8)
    if settings.morphology in ("Open", "Open + close"):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if settings.morphology in ("Close", "Open + close"):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def extract_features(rgb, original_rgb, settings):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    delta = np.abs(h.astype(np.int16) - settings.hue_center)
    hue_match = np.minimum(delta, 180 - delta) <= settings.hue_tolerance
    color = (hue_match & (s >= settings.saturation_min) & (v >= settings.value_min)).astype(np.uint8) * 255
    white = cv2.inRange(hsv, (0, 0, 160), (179, 75, 255))
    if settings.threshold_mode == "Adaptive":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    elif settings.threshold_mode == "Fixed":
        _, binary = cv2.threshold(gray, settings.threshold_value, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, settings.canny_low, settings.canny_high)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient = cv2.normalize(cv2.magnitude(gx, gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    morph_kernel = np.ones((settings.morph_kernel, settings.morph_kernel), np.uint8)
    cleaned = cleanup(binary, settings)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_view = rgb.copy()
    contours = [c for c in contours if cv2.contourArea(c) > 25]
    cv2.drawContours(contour_view, contours, -1, (56, 189, 248), 1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=settings.line_length, maxLineGap=12)
    line_view = rgb.copy()
    line_list = [] if lines is None else lines[:, 0].tolist()
    for x1, y1, x2, y2 in line_list:
        cv2.line(line_view, (x1, y1), (x2, y2), (56, 189, 248), 2)
    road = rgb.copy()
    paint = paint_support(original_rgb)
    stripe_mask = np.zeros_like(gray)
    paint_contours, _ = cv2.findContours(paint.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_boxes = []
    for contour in paint_contours:
        area = cv2.contourArea(contour)
        x, y, w, ht = cv2.boundingRect(contour)
        if (area >= max(40, gray.size * 0.00025) and max(w, ht) >= 12
                and y + ht > gray.shape[0] * 0.45):
            stripe_boxes.append([x, y, w, ht])
            stripe_mask[y:y+ht, x:x+w] |= paint.mask[y:y+ht, x:x+w]
            cv2.rectangle(road, (x, y), (x + w, y + ht), (255, 185, 65), 2)
    road_edges = cv2.Canny(stripe_mask, settings.canny_low, settings.canny_high)
    road_lines = cv2.HoughLinesP(road_edges, 1, np.pi / 180, 40,
                                minLineLength=max(settings.line_length, round(gray.shape[1] * 0.12)), maxLineGap=15)
    candidates = [] if road_lines is None else road_lines[:, 0].tolist()
    candidates.sort(key=lambda line: -abs(line[2] - line[0]))
    horizontal = []
    for line in candidates:
        x1, y1, x2, y2 = line
        duplicate = any(abs((y1 + y2 - old[1] - old[3]) / 2) < 6
                        and min(max(x1, x2), max(old[0], old[2])) - max(min(x1, x2), min(old[0], old[2]))
                        > 0.5 * min(abs(x2 - x1), abs(old[2] - old[0])) for old in horizontal)
        if (not duplicate and len(horizontal) < 40 and abs(y2 - y1) <= 0.25 * max(abs(x2 - x1), 1)
                and min(y1, y2) > gray.shape[0] * 0.45):
            cv2.line(road, (x1, y1), (x2, y2), (56, 189, 248), 2)
            horizontal.append(line)
    return {
        "Grayscale": gray, "Hue": (h.astype(np.float32) * 255 / 179).astype(np.uint8),
        "Saturation": s, "Value": v, "Colour mask": cleanup(color, settings),
        "White marking mask": cleanup(white, settings), "Edges": edges,
        "Threshold": binary, "Morphology": cleaned, "Contours": contour_view,
        "Lines": line_view, "Road candidates": road, "Stripe mask": stripe_mask,
        "Paint background": paint.background, "Paint local contrast": paint.contrast,
        "Paint candidate mask": paint.mask,
        "Paint colour seed": paint.raw, "Paint shape proposals": paint.shape_mask,
        "Paint unbracketed proposals": paint.uncertain,
        "Sobel magnitude": gradient,
        "Laplacian": cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_32F)),
        "Morphological gradient": cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel),
        "Top hat": cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, morph_kernel),
        "Black hat": cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, morph_kernel),
    }, {**review_only_policy(), "stripe_boxes": stripe_boxes, "horizontal_segments": horizontal,
        "paint_support": {"parameters": paint.parameters, "counts": paint.counts},
        "line_count": len(line_list), "contour_count": len(contours),
        "warning": "Appearance candidates only. No crossing, stop-line, signal, or violation classification."}


def process_frame(rgb, settings=None, timestamp=0.0, include_features=True, include_stages=True):
    settings = settings or Settings()
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("process_frame expects a uint8 RGB image with three channels.")
    if min(rgb.shape[:2]) < 2:
        raise ValueError("Images must be at least 2 pixels wide and high.")
    source_h, source_w = rgb.shape[:2]
    out = rgb.copy()
    if settings.max_width and source_w > settings.max_width:
        target_h = max(2, round(source_h * settings.max_width / source_w))
        out = cv2.resize(out, (settings.max_width, target_h), interpolation=cv2.INTER_AREA)
    original = out.copy()
    stages = {"Prepared": out.copy()} if include_stages else {}
    if settings.denoise == "Gaussian":
        out = cv2.GaussianBlur(out, (settings.kernel, settings.kernel), 0)
    elif settings.denoise == "Median":
        out = cv2.medianBlur(out, settings.kernel)
    elif settings.denoise == "Bilateral":
        out = cv2.bilateralFilter(out, settings.kernel, 45, 45)
    if include_stages:
        stages["Denoised"] = out.copy()
    if settings.white_balance:
        means = out.mean(axis=(0, 1))
        gains = np.clip(means.mean() / np.maximum(means, 1), 0.5, 2)
        gains = 1 + settings.white_balance * (gains - 1)
        out = np.clip(np.round(out.astype(np.float32) * gains), 0, 255).astype(np.uint8)
    if settings.exposure:
        out = np.clip(np.round(out.astype(np.float32) * 2 ** settings.exposure), 0, 255).astype(np.uint8)
    if settings.stretch:
        luminance = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        low, high = np.percentile(luminance, [settings.stretch, 100 - settings.stretch])
        if high - low > 1:
            out = np.clip(np.round((out.astype(np.float32) - low) * 255 / (high - low)), 0, 255).astype(np.uint8)
    if include_stages:
        stages["Colour + exposure"] = out.copy()
    if settings.gamma != 1:
        lut = np.round((np.arange(256) / 255) ** (1 / settings.gamma) * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)
    if settings.clahe:
        lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(settings.clip_limit, (8, 8)).apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    if include_stages:
        stages["Brightness + contrast"] = out.copy()
    if settings.sharpen:
        soft = cv2.GaussianBlur(out, (0, 0), 1.2)
        out = cv2.addWeighted(out, 1 + settings.sharpen, soft, -settings.sharpen, 0)
    if settings.saturation != 1:
        hsv_out = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        hsv_out[:, :, 1] = np.clip(hsv_out[:, :, 1].astype(np.float32) * settings.saturation, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(hsv_out, cv2.COLOR_HSV2RGB)
    if include_stages:
        stages["Final colour"] = out.copy()
    features, candidates = extract_features(out, original, settings) if include_features else ({}, {})
    dest_h, dest_w = out.shape[:2]
    sx, sy = source_w / dest_w, source_h / dest_h
    mapping = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1]], dtype=float)
    before = quality(original)
    # Identity processing is common for tracking. Reuse identical measurements
    # without sharing the mutable flags list between the two report entries.
    after = ({**before, "flags": before["flags"].copy()} if np.array_equal(original, out) else quality(out))
    return FrameResult(out, timestamp, (source_w, source_h), mapping,
                       {"original": before, "processed": after}, stages, features, candidates)
