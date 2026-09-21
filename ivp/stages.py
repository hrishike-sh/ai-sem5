"""An explicit classical image-processing graph for inspection and teaching."""
import cv2
import numpy as np
from .marking_policy import review_only_policy


def build_stages(result, settings):
    rgb = result.stages['Prepared']
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if settings.denoise == 'Median':
        smooth = cv2.medianBlur(gray, settings.kernel)
    elif settings.denoise == 'Bilateral':
        smooth = cv2.bilateralFilter(gray, settings.kernel, 45, 45)
    elif settings.denoise == 'Gaussian':
        smooth = cv2.GaussianBlur(gray, (settings.kernel, settings.kernel), 0)
    else:
        smooth = gray.copy()
    # The analysis branch uses local contrast to examine structure, independently
    # of the conservative colour-export branch.
    contrast = cv2.createCLAHE(settings.clip_limit, (8, 8)).apply(smooth)
    otsu_value, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    if settings.threshold_mode == 'Adaptive':
        selected = adaptive
    elif settings.threshold_mode == 'Fixed':
        _, selected = cv2.threshold(contrast, settings.threshold_value, 255, cv2.THRESH_BINARY)
    else:
        selected = binary
    kernel = np.ones((settings.morph_kernel, settings.morph_kernel), np.uint8)
    opened = cv2.morphologyEx(selected, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    gx = cv2.Sobel(contrast, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(contrast, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    edges = cv2.Canny(contrast, settings.canny_low, settings.canny_high)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 25]
    contour_view = rgb.copy()
    cv2.drawContours(contour_view, contours, -1, (0, 180, 230), 1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    palette = np.zeros((count, 3), np.uint8)
    retained = 0
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            palette[i] = ((i * 71) % 200 + 55, (i * 113) % 200 + 55, (i * 151) % 200 + 55)
            retained += 1
    components = palette[labels]
    line_view = rgb.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35, minLineLength=settings.line_length, maxLineGap=12)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_view, (x1, y1), (x2, y2), (0, 180, 230), 2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    colour_masks = {}
    for name, low, high in [('Red', 0, 10), ('Amber', 11, 35), ('Green', 36, 90)]:
        mask = cv2.inRange(hsv, (low, 90, 70), (high, 255, 255))
        if name == 'Red':
            mask |= cv2.inRange(hsv, (170, 90, 70), (179, 255, 255))
        colour_masks[name] = mask
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    normalize = lambda arr: cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages = []
    def add(key, title, parent, branch, image, explanation, parameters=None):
        stages.append({'id': key, 'title': title, 'parent': parent, 'branch': branch,
                       'image': image, 'explanation': explanation, 'parameters': parameters or {},
                       **(review_only_policy() if branch == 'Markings' else {})})
    add('original', 'Original colour', None, 'Preparation', rgb, 'Display-oriented RGB, resized only when needed. The uploaded file is unchanged.')
    add('gray', 'Black & white · grayscale', 'original', 'Preparation', gray, 'Weighted RGB luminance removes colour while retaining 256 intensity levels. This is not yet a binary image.')
    add('smooth', 'Noise reduction', 'gray', 'Preparation', smooth, 'Reduce small intensity fluctuations before extracting structures. Fine detail can also be smoothed.', {'filter': settings.denoise, 'kernel': settings.kernel})
    add('contrast', 'Local contrast · CLAHE', 'smooth', 'Preparation', contrast, 'Equalize local grayscale histograms with limited amplification. This analysis branch is separate from the colour export.', {'clip_limit': settings.clip_limit, 'tiles': [8, 8]})
    add('sobel_x', 'Horizontal derivative · Sobel X', 'contrast', 'Edges', normalize(np.abs(gx)), 'Intensity change along X reveals vertical boundaries. Absolute magnitude is normalized for display.')
    add('sobel_y', 'Vertical derivative · Sobel Y', 'contrast', 'Edges', normalize(np.abs(gy)), 'Intensity change along Y reveals horizontal boundaries. This is a parallel branch from CLAHE.')
    add('sobel', 'Combined gradient magnitude', 'contrast', 'Edges', normalize(magnitude), 'Combine horizontal and vertical derivatives. Brightness is normalized per image, not a calibrated strength.')
    add('canny', 'Thin edges · Canny', 'contrast', 'Edges', edges, 'Non-maximum suppression and hysteresis link intensity boundaries. Edges are not object detections.', {'low': settings.canny_low, 'high': settings.canny_high})
    add('otsu', 'Black / white binary · Otsu', 'contrast', 'Segmentation', binary, 'Choose a global threshold by separating the intensity histogram into two classes.', {'threshold': float(otsu_value)})
    add('adaptive', 'Local adaptive threshold', 'contrast', 'Segmentation', adaptive, 'Use neighbourhood-dependent thresholds where illumination varies. This is an alternative to Otsu.', {'block_size': 31, 'constant': 7})
    add('selected', 'Selected binary mask', 'contrast', 'Segmentation', selected, 'The chosen threshold method feeds the morphology branch.', {'method': settings.threshold_mode})
    add('opening', 'Opening · remove small specks', 'selected', 'Morphology', opened, 'Erosion followed by dilation removes small bright structures.', {'kernel': settings.morph_kernel})
    add('closing', 'Closing · fill small gaps', 'opening', 'Morphology', closed, 'Dilation followed by erosion fills small dark gaps. Larger kernels may merge separate shapes.', {'kernel': settings.morph_kernel})
    add('components', 'Connected regions', 'closing', 'Shapes', components, 'Colour each retained connected foreground region. These regions have no vehicle or traffic-rule labels.', {'regions': retained, 'min_area': 25})
    add('contours', 'Contour boundaries', 'closing', 'Shapes', contour_view, 'Draw external boundaries of retained binary shapes over the original frame.', {'contours': len(contours)})
    add('lines', 'Line segments · Hough', 'canny', 'Shapes', line_view, 'Vote for line segments in the Canny map. Segments can come from buildings, vehicles or road paint; they are not confirmed lanes.', {'segments': 0 if lines is None else len(lines)})
    for name, mask in colour_masks.items():
        add(name.lower(), name + ' colour isolation', 'original', 'Colour', mask, 'HSV thresholds on original colour. A coloured patch does not identify a traffic signal or its meaning.')
    add('white', 'White road-marking mask', 'original', 'Colour', result.features['White marking mask'], 'Bright low-saturation regions may include paint, vehicles, glare and buildings.')
    add('paint_seed', 'White / yellow appearance seed', 'original', 'Markings', result.features['Paint colour seed'], 'Original-colour seed before geometric rejection. Raw proposals are preserved for inspection. REVIEW ONLY; vehicles, pavement and glare can be selected.')
    add('paint_background', 'Local road brightness background', 'original', 'Markings', result.features['Paint background'], 'Large-scale median of original HSV Value estimates the slowly changing illumination. It can also absorb wide paint; this is an estimate, not a clean road reconstruction.', result.candidates['paint_support']['parameters'])
    add('paint_contrast', 'Paint contrast above local background', 'paint_background', 'Markings', result.features['Paint local contrast'], 'Subtract the median background from original HSV Value with negative differences clipped to zero. Warm bright pavement alone does not satisfy this local contrast test.', {'inputs': ['original', 'paint_background']})
    add('paint_shapes', 'Elongated shape proposals', 'paint_contrast', 'Markings', result.features['Paint shape proposals'], 'Opening, closing, sparse-bridge splitting and rotated geometry propose elongated fragments before the two-flank check. REVIEW ONLY; an illumination band can have the same shape.')
    add('paint_uncertain', 'Unbracketed proposals · withheld', 'paint_shapes', 'Markings', result.features['Paint unbracketed proposals'], 'These shape proposals lack enough observed darker pixels on both sides. They are preserved here, excluded from the supported mask, and ineligible for geometry or rules.')
    add('paint_mask', 'Two-sided bright bands · review only', 'paint_shapes', 'Markings', result.features['Paint candidate mask'], 'Sample five sections along each proposed band. At least three sections are needed and 60% must have darker observed flanks on both sides. This reduces broad illumination bands, but a yellow vehicle roof still passes. REVIEW ONLY; never configures road geometry or incident rules.', result.candidates['paint_support']['counts'])
    add('road', 'Road-marking candidates', 'paint_mask', 'Markings', result.features['Road candidates'], 'Lower-frame paint-like fragments and near-horizontal Hough segments produce appearance candidates, not confirmed crossings, stop lines or legal-rule applicability.')
    add('frequency', 'Frequency spectrum · FFT', 'gray', 'Frequency', normalize(spectrum), 'Log magnitude of the centred 2D Fourier transform. The centre represents low frequencies; texture and edges contribute higher frequencies.')
    return stages


def manifest(stages):
    return [{key: value for key, value in stage.items() if key != 'image'} for stage in stages]
