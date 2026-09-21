"""Local photometric and shape support for paint-like appearance, without labels."""
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .paint_profiles import bracketed_components


@dataclass
class PaintSupport:
    raw: np.ndarray
    mask: np.ndarray
    background: np.ndarray
    contrast: np.ndarray
    rejected: np.ndarray
    parameters: dict
    counts: dict
    shape_mask: np.ndarray
    uncertain: np.ndarray
    profiles: list
    trace: dict = field(default_factory=dict)


def paint_support(rgb, threshold_offset=0, *, contrast_multiplier=1.0, capture_stages=False):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Paint analysis requires an RGB uint8 image.')
    height, width = rgb.shape[:2]
    if min(height, width) < 2 or not np.isfinite(threshold_offset):
        raise ValueError('Paint analysis needs at least 2 pixels per dimension and a finite threshold offset.')
    if (isinstance(contrast_multiplier, bool) or not isinstance(contrast_multiplier, (int, float))
            or not math.isfinite(contrast_multiplier) or contrast_multiplier <= 0):
        raise ValueError('Local-contrast multiplier must be a positive finite number.')
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    otsu = cv2.threshold(value, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    percentile1, percentile99 = np.percentile(value, [1, 99])
    flat = float(percentile99 - percentile1) < 20
    baseline = min(max(100, min(230, int(otsu) + 1)), max(40, int(percentile99) - 5))
    threshold = max(0, min(255, baseline + threshold_offset))
    size = min(511, max(31, int(min(height, width) * .7)))
    size += 1 - size % 2
    size = max(3, min(size, min(height, width) // 2 * 2 - 1))
    reduction = math.ceil(size / 255)
    small_value = (cv2.resize(value, (max(2, round(width/reduction)), max(2, round(height/reduction))),
                              interpolation=cv2.INTER_AREA) if reduction > 1 else value)
    median_size = min(255, max(3, round(size/reduction) // 2 * 2 + 1))
    background = cv2.medianBlur(small_value, median_size)
    if reduction > 1:
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
    contrast = cv2.subtract(value, background)
    background_level = max(20., float(np.median(background)))
    local_floor = np.clip(threshold * np.minimum(1., background.astype(np.float32) / background_level), 40, 245)
    palette = ((saturation <= 110) | ((hue >= 15) & (hue <= 38) & (saturation >= 70)))
    otsu_seed = palette & (value >= local_floor)
    median = cv2.medianBlur(value, 3)
    noise = float(np.median(np.abs(value.astype(np.int16) - median.astype(np.int16))))
    base_minimum_contrast = max(12., noise * 3., 20. if flat else 0.)
    minimum_contrast = base_minimum_contrast * contrast_multiplier
    if not math.isfinite(minimum_contrast):
        raise ValueError('Local-contrast multiplier produces a non-finite threshold.')
    contrast_backed_floor = np.clip(background.astype(np.float32) + minimum_contrast, 40, 245)
    local_floor = np.minimum(local_floor, contrast_backed_floor)
    colour_seed = palette & (value >= local_floor)
    raw = (colour_seed * 255).astype(np.uint8)
    supported = (colour_seed & (contrast >= minimum_contrast)).astype(np.uint8) * 255
    small = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(supported, cv2.MORPH_OPEN, small)
    opened = cleaned if capture_stages else None
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, small)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    minimum_area = max(20, math.ceil(value.size * .0002))
    mask = np.zeros_like(cleaned)
    rejected_reasons = {'small_or_thin': 0, 'broad_region': 0, 'nonelongated': 0, 'sparse_shape': 0, 'unsupported_frame_edge': 0}
    split_components = 0
    split_pixels_removed = 0
    bridge_attempts = 0

    def supported_shape(points, area, box):
        x, y, w, h = box
        _, (a, b), _ = cv2.minAreaRect(points)
        if min(a, b) < 2 or max(a, b) / max(min(a, b), 1) < 1.7:
            return 'nonelongated'
        if area / max(1, (a+1)*(b+1)) < .45:
            return 'sparse_shape'
        if sum([x == 0, y == 0, x+w == width, y+h == height]) >= 3:
            return 'unsupported_frame_edge'
        return None

    for i in range(1, count):
        x, y, w, h, area = map(int, stats[i])
        if area < minimum_area:
            rejected_reasons['small_or_thin'] += 1
            continue
        if area > value.size * .45:
            rejected_reasons['broad_region'] += 1
            continue
        yy, xx = np.nonzero(labels[y:y+h, x:x+w] == i)
        points = np.column_stack((xx+x, yy+y)).astype(np.float32)
        reason = supported_shape(points, area, (x, y, w, h))
        if reason is None:
            mask[y+yy, x+xx] = 255
            continue
        _, (a, b), _ = cv2.minAreaRect(points)
        if area / max(1, (a+1)*(b+1)) < .45 and min(w, h) >= 7:
            bridge_attempts += 1
            component = (labels[y:y+h, x:x+w] == i).astype(np.uint8) * 255
            padded = np.pad(component, 3)
            refined = cv2.morphologyEx(padded, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))[3:-3, 3:-3]
            n, sublabels, substats, _ = cv2.connectedComponentsWithStats(refined)
            accepted = False
            for j in range(1, n):
                sx, sy, sw, sh, sa = map(int, substats[j])
                if sa < minimum_area:
                    continue
                syy, sxx = np.nonzero(sublabels == j)
                subpoints = np.column_stack((sxx+x, syy+y)).astype(np.float32)
                if supported_shape(subpoints, sa, (x+sx, y+sy, sw, sh)) is None:
                    mask[y+syy, x+sxx] = 255
                    accepted = True
            if accepted:
                split_components += 1
                split_pixels_removed += area - int(np.count_nonzero(mask[y:y+h, x:x+w] & component))
                continue
        rejected_reasons[reason] += 1
    shape_mask = mask.copy()
    mask, profiles = bracketed_components(rgb, shape_mask)
    uncertain = cv2.subtract(shape_mask, mask)
    rejected = np.where((raw > 0) & (mask == 0), 255, 0).astype(np.uint8)
    minority_retained = flat and bool(np.any(mask))
    parameters = {'method': 'local_paint_support_v5', 'brightness_threshold': threshold,
                  'otsu_value': float(otsu), 'brightness_baseline': baseline,
                  'threshold_rule': 'Bounded Otsu/local illumination seed capped by background plus the measured local-contrast threshold; absolute floor40 remains. Experimental appearance only.',
                  'contrast_backed_seed_multiplier': 1.0,
                  'background_method': 'median on original HSV Value; area-reduce before large kernels and bilinear-expand background to avoid 8-bit median histogram overflow',
                  'background_kernel_px': median_size, 'background_footprint_requested_px': size,
                  'background_reduction': reduction, 'background_level': background_level,
                  'base_minimum_local_contrast': base_minimum_contrast,
                  'contrast_multiplier': contrast_multiplier,
                  'minimum_local_contrast': minimum_contrast,
                  'contrast_units': 'HSV Value levels (0-255)', 'white_saturation_max': 110,
                  'yellow_hue_range': [15, 38], 'yellow_saturation_min': 70,
                  'minimum_component_area_px': minimum_area, 'minimum_rotated_aspect': 1.7,
                  'maximum_component_roi_fraction': .45, 'minimum_rotated_fill': .45,
                  'morphology': '3x3 opening then closing; sparse joined components get one 7x7 bridge-opening pass, then each child must pass the same shape gates; reject components touching3+ analysis boundaries',
                  'sparse_bridge_opening_px': 7,
                  'two_flank_support': {'longitudinal_sections': 5, 'minimum_sections': 3,
                     'minimum_bracketed_fraction': .6, 'minimum_value_contrast_each_side': 12,
                     'minimum_observed_exterior_pixels_per_flank': 3,
                     'flank_sample_proposals': 5, 'flank_pixels_deduplicated': True,
                     'minimum_flank_extent_px': 5,
                     'meaning': 'Observed bright band with darker flanks; objects and reflections can still pass. Not semantic paint recognition.'},
                  'flat_input_detected': flat,
                  'flat_input_withheld': flat and not minority_retained,
                  'flat_minority_rescued': minority_retained,
                  'flat_input_status': ('minority_support_retained' if minority_retained else 'withheld_no_supported_minority') if flat else 'sufficient_percentile_range',
                  'flat_input_rule': 'Low p1-p99 range uses at least20 Value levels of base local contrast, then unchanged morphology, area, shape and two-flank gates; no percentile-only veto.',
                  'flat_contrast_floor': 20.,
                  'flat_range': float(percentile99-percentile1)}
    counts = {'colour_seed_pixels': int(np.count_nonzero(raw)),
              'contrast_rescued_seed_pixels': int(np.count_nonzero((raw > 0) & ~otsu_seed)),
              'local_contrast_pixels': int(np.count_nonzero(supported)),
              'supported_pixels': int(np.count_nonzero(mask)),
              'rejected_seed_pixels': int(np.count_nonzero(rejected)),
              'retained_components': int(cv2.connectedComponents(mask)[0]-1),
              'shape_proposal_pixels': int(np.count_nonzero(shape_mask)),
              'unbracketed_proposal_pixels': int(np.count_nonzero(uncertain)),
              'two_flank_retained_components': sum(row['retained'] for row in profiles),
              'two_flank_rejected_components': sum(not row['retained'] for row in profiles),
              'bridge_split_components': split_components, 'bridge_pixels_removed': split_pixels_removed,
              'component_rejections': rejected_reasons}
    trace = {'value': value, 'supported': supported, 'opened': opened, 'closed': cleaned,
             'bridge_attempts': bridge_attempts} if capture_stages else {}
    return PaintSupport(raw, mask, background, contrast, rejected, parameters, counts,
                        shape_mask, uncertain, profiles, trace)
