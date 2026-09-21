"""Region measurements in processed-image coordinates; never a traffic decision."""
import math

from .pipeline import histogram, quality


def inspect_region(result, rectangle):
    if not isinstance(rectangle, list) or len(rectangle) != 4:
        raise ValueError("Select a rectangle using four normalized coordinates.")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in rectangle):
        raise ValueError("Region coordinates must be between 0 and 1.")
    x1, y1, x2, y2 = rectangle
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Region must have positive width and height.")
    height, width = result.color_rgb.shape[:2]
    left, top = int(x1 * width), int(y1 * height)
    right, bottom = min(width, math.ceil(x2 * width)), min(height, math.ceil(y2 * height))
    before = result.stages["Prepared"][top:bottom, left:right]
    after = result.color_rgb[top:bottom, left:right]
    if min(after.shape[:2]) < 2:
        raise ValueError("Select a region at least two pixels wide and high.")
    corners = []
    for x, y in ((left, top), (right - 1, bottom - 1)):
        point = result.transform_to_source @ [x, y, 1]
        corners.append(point[:2].tolist())
    return {"bounds_processed_xyxy_exclusive": [left, top, right, bottom],
            "source_corner_pixel_centres": corners, "original": quality(before),
            "processed": quality(after), "histogram_original": histogram(before),
            "histogram_processed": histogram(after)}
