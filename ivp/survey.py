"""Reproducible image-space marking survey; no learned models or semantic labels."""
import math

import cv2
import numpy as np

from .marking_policy import review_only_policy
from .paint import paint_support


def survey(rgb, rectangle, transform=None, *, capture_stages=False):
    if (not isinstance(rectangle, list) or len(rectangle) != 4 or
        any(isinstance(v, bool) or not isinstance(v, (int, float)) or
            not math.isfinite(v) or not 0 <= v <= 1 for v in rectangle)):
        raise ValueError('Select a road region using four normalized coordinates.')
    x1, y1, x2, y2 = rectangle
    h, w = rgb.shape[:2]
    left, top, right, bottom = int(x1*w), int(y1*h), math.ceil(x2*w), math.ceil(y2*h)
    if right-left < 16 or bottom-top < 16:
        raise ValueError('Select a road region at least 16 pixels wide and high.')
    crop = rgb[top:bottom, left:right].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    support = paint_support(crop, capture_stages=True) if capture_stages else paint_support(crop)
    raw, mask = support.raw, support.mask
    low_support = paint_support(crop, contrast_multiplier=.75)
    high_support = paint_support(crop, contrast_multiplier=1.25)
    low, high = low_support.mask, high_support.mask
    low_coverage, high_coverage = float(np.mean(low > 0)*100), float(np.mean(high > 0)*100)
    union = (low > 0) | (high > 0)
    agreement = float(np.count_nonzero((low > 0) & (high > 0))/union.sum()) if union.any() else None
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    overlay = crop.copy()
    overlay[mask > 0] = (crop[mask > 0]*.35 + np.array([20, 180, 210])*.65).astype(np.uint8)
    transform = np.eye(3) if transform is None else np.asarray(transform)
    rows = []
    for i in range(1, count):
        x, y, width, height, area = map(int, stats[i])
        yy, xx = np.nonzero(labels[y:y+height, x:x+width] == i)
        points = np.column_stack((xx+x, yy+y)).astype(np.float32)
        _, (side_a, side_b), _ = cv2.minAreaRect(points)
        covariance = np.cov(points, rowvar=False)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, -1]
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180)
        orientation = angle if values[-1] > 1.2*max(values[0], 1e-9) else None
        cx, cy = map(float, centres[i])
        source = transform @ [cx+left, cy+top, 1]
        row = {'id': i, **review_only_policy(), 'area_px': area, 'bbox_x': x+left, 'bbox_y': y+top,
               'bbox_width_px': width, 'bbox_height_px': height,
               'long_side_px': float(max(side_a, side_b)), 'short_side_px': float(min(side_a, side_b)),
               'orientation_deg_clockwise_from_right': orientation,
               'touches_roi_boundary': bool(x == 0 or y == 0 or x+width == crop.shape[1] or y+height == crop.shape[0]),
               'source_centre_x': float(source[0]/source[2]), 'source_centre_y': float(source[1]/source[2])}
        rows.append(row)
        cv2.rectangle(overlay, (x, y), (x+width-1, y+height-1), (255, 180, 20), 1)
        cv2.putText(overlay, str(i), (x, max(12, y+12)), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 70, 20), 1)
    warnings = ['White vehicles, reflections and other bright surfaces can enter the mask. Select road only and review the overlay.',
                'Coverage describes visible candidate pixels, not paint wear, road occupancy or physical area. Perspective and occlusion affect results.']
    if not rows:
        warnings.append('No retained bright white/yellow fragments. This does not prove markings are absent; dark or faded markings can be missed.')
    if support.parameters['flat_input_withheld']:
        warnings.append('Low overall brightness range: support was withheld because no fragment passed the stronger local-contrast, shape and flank checks. Small or faint markings can still be missed.')
    elif support.parameters['flat_minority_rescued']:
        warnings.append('Low overall percentile range, but minority appearance was retained through stronger local contrast and the existing shape/flank checks. This does not establish paint semantics.')
    if agreement is not None and agreement < .75:
        warnings.append('Segmentation changes substantially between the 0.75x and 1.25x local-contrast probes. Inspect both sensitivity masks.')
    if any(r['touches_roi_boundary'] for r in rows):
        warnings.append('Some fragments touch the selection boundary and may be truncated.')
    warnings.append('Local brightness contrast and elongated shape reject broad pavement; joined sparse fragments are split before measurement. Faint, narrow or irregular paint can still be missed.')
    warnings.append('Two darker flanks reject unsupported illumination bands. A yellow vehicle roof can still pass. Every fragment is REVIEW ONLY and is ineligible for configured geometry or incident rules.')
    warnings.append('Probe agreement measures processing sensitivity only. Identical masks are not certainty or detection accuracy; component filtering can make coverage non-monotonic as the threshold changes.')
    report = {'method': 'classical_marking_survey_v5', 'uses_ai': False,
              **review_only_policy(),
              'roi_normalized': rectangle, 'roi_xyxy_exclusive': [left, top, right, bottom],
              'analysis_size': [w, h], 'transform_to_source_pixel_centres': transform.tolist(),
              'parameters': support.parameters, 'support': support.counts,
              'component_profile_evidence': support.profiles,
              'sensitivity': {'method': 'local_contrast_probes_v1', 'basis': 'minimum_local_contrast',
                              'units': support.parameters['contrast_units'],
                              'base_threshold': support.parameters['minimum_local_contrast'],
                              'low': {'multiplier': .75, 'threshold': low_support.parameters['minimum_local_contrast'],
                                      'coverage_percent': low_coverage, 'mask_file': 'contrast-075x.png',
                                      'flat_input_status': low_support.parameters['flat_input_status']},
                              'high': {'multiplier': 1.25, 'threshold': high_support.parameters['minimum_local_contrast'],
                                       'coverage_percent': high_coverage, 'mask_file': 'contrast-125x.png',
                                       'flat_input_status': high_support.parameters['flat_input_status']},
                              'probe_mask_iou': agreement,
                              'coverage_range_percent': [min(low_coverage, high_coverage), max(low_coverage, high_coverage)]},
              'summary': {'roi_area_px': int(gray.size), 'candidate_area_px': int(np.count_nonzero(mask)),
                          'coverage_percent': float(np.mean(mask > 0)*100), 'fragments': len(rows),
                          'contrast_low_coverage_percent': low_coverage,
                          'contrast_high_coverage_percent': high_coverage,
                          'contrast_probe_mask_iou': agreement},
              'fragments': rows, 'limitations': warnings}
    images = {'selected-road': crop, 'raw-segmentation': raw, 'cleaned-mask': mask,
                    'local-background': support.background, 'local-paint-contrast': support.contrast,
                    'colour-seed': support.raw, 'shape-proposals': support.shape_mask,
                    'unbracketed-proposals': support.uncertain,
                    'rejected-seed': support.rejected,
                    'contrast-075x': low, 'contrast-125x': high, 'annotated-survey': overlay}
    if capture_stages:
        from .presentation import road_presentation
        report['ivp_demo'], assets = road_presentation(report, images, support.trace)
        images.update({'executed-' + key: pixels for key, pixels in assets.items()})
    return report, images
