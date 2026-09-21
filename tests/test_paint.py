import cv2
import numpy as np
import pytest

from ivp.paint import paint_support


def iou(mask, truth):
    return np.count_nonzero((mask > 0) & truth) / np.count_nonzero((mask > 0) | truth)


@pytest.mark.parametrize('paint', [[240, 240, 240], [240, 220, 0], [230, 210, 170], [175, 205, 230], [78, 78, 78]])
def test_paint_colours_and_dark_input_retain_geometry(paint):
    image = np.full((160, 240, 3), 35, np.uint8)
    truth = np.zeros(image.shape[:2], bool)
    truth[40:52, 35:180] = True
    truth[90:102, 35:180] = True
    image[truth] = paint
    before = image.copy()
    result = paint_support(image)
    assert iou(result.mask, truth) >= .95
    assert np.array_equal(image, before)


def test_shadowed_paint_retention_and_smooth_pavement_rejection():
    road = np.tile(np.linspace(35, 170, 320, dtype=np.uint8), (200, 1))
    image = np.repeat(road[:, :, None], 3, axis=2)
    truth = np.zeros(road.shape, bool)
    truth[35:47, 25:290] = True
    truth[110:122, 25:290] = True
    image[truth] = np.minimum(image[truth].astype(int) + 65, 255).astype(np.uint8)
    assert iou(paint_support(image).mask, truth) >= .8
    assert np.mean(paint_support(np.repeat(road[:, :, None], 3, axis=2)).mask > 0) <= .005


@pytest.mark.parametrize('paint', [[228, 226, 218], [235, 196, 26]])
def test_local_contrast_rescues_shadowed_seed_before_unchanged_shape_checks(paint):
    # Reproduce the frozen procedural regression family. A uniform road at
    # 110 already passes the old adaptive seed and cannot test this defect.
    rng = np.random.default_rng(20261020)
    y, x = np.mgrid[:256, :384]
    road = 57+13*x/383+3*np.sin(y/13)+rng.normal(0, 2, x.shape)
    image = np.repeat(np.clip(road, 0, 255).astype(np.uint8)[:, :, None], 3, axis=2)
    truth = np.zeros(image.shape[:2], bool)
    truth[121:136, 74:311] = True
    image[truth] = paint
    multiplier = np.ones(image.shape[:2], np.float32)
    multiplier[:, 160:256] = .45
    multiplier = cv2.GaussianBlur(multiplier, (13, 13), 3)
    image = np.round(image.astype(float)*multiplier[:, :, None]).astype(np.uint8)
    result = paint_support(image)
    shadow_paint = np.zeros(image.shape[:2], bool)
    shadow_paint[123:134, 170:246] = True
    assert np.mean(result.mask[shadow_paint] > 0) >= .95
    assert result.counts['contrast_rescued_seed_pixels'] > 0
    assert result.parameters['method'] == 'local_paint_support_v5'
    assert result.parameters['minimum_local_contrast'] == 12
    assert result.parameters['two_flank_support']['minimum_value_contrast_each_side'] == 12


def test_contrast_probe_changes_faint_band_support_and_records_actual_thresholds():
    image = np.full((160, 240, 3), 60, np.uint8)
    image[35:47, 30:210] = 220
    image[95:107, 30:210] = 74
    low = paint_support(image, contrast_multiplier=.75)
    default = paint_support(image)
    high = paint_support(image, contrast_multiplier=1.25)
    for result, threshold in [(low, 9), (default, 12), (high, 15)]:
        assert result.parameters['base_minimum_local_contrast'] == 12
        assert result.parameters['minimum_local_contrast'] == threshold
        assert np.mean(result.mask[37:45, 35:205] > 0) >= .95
    assert np.mean(low.mask[97:105, 35:205] > 0) >= .95
    assert np.mean(default.mask[97:105, 35:205] > 0) >= .95
    assert np.count_nonzero(high.mask[97:105, 35:205]) == 0
    assert not np.array_equal(low.mask, high.mask)


@pytest.mark.parametrize('multiplier', [0, -1, True, None, '1', float('nan'), float('inf'), 1e308])
def test_invalid_contrast_multiplier_rejected(multiplier):
    with pytest.raises(ValueError, match='contrast'):
        paint_support(np.full((40, 60, 3), 60, np.uint8), contrast_multiplier=multiplier)


def test_oblique_stripes_are_not_restricted_to_horizontal_lines():
    image = np.full((200, 280, 3), 60, np.uint8)
    truth = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(truth, np.array([[40, 100], [205, 35], [212, 51], [47, 116]]), 255)
    image[truth > 0] = [240, 225, 30]
    assert iou(paint_support(image).mask, truth > 0) >= .85


def test_joined_stripes_survive_without_keeping_narrow_connecting_bridge():
    image = np.full((180, 280, 3), 50, np.uint8)
    truth = np.zeros(image.shape[:2], bool)
    for y in [35, 70, 105]:
        truth[y:y+12, 30:250] = True
    image[truth] = 210
    image[35:117, 248:251] = 210
    result = paint_support(image)
    assert iou(result.mask, truth) >= .95
    assert result.counts['bridge_split_components'] > 0
    assert result.counts['retained_components'] == 3


@pytest.mark.parametrize('distractor', ['square', 'red', 'blue', 'uniform', 'noise'])
def test_nonpaint_controls_cannot_pass_by_brightness_alone(distractor):
    image = np.full((180, 280, 3), 70, np.uint8)
    if distractor == 'square':
        image[40:100, 80:140] = 250
    elif distractor in ('red', 'blue'):
        image[40:53, 30:250] = [255, 0, 0] if distractor == 'red' else [0, 0, 255]
    elif distractor == 'uniform':
        image[:] = [225, 205, 120]
    else:
        rng = np.random.default_rng(623)
        image = np.clip(image.astype(int) + rng.integers(-12, 13, image.shape), 0, 255).astype(np.uint8)
    assert np.mean(paint_support(image).mask > 0) <= .005


@pytest.mark.parametrize('image', [np.zeros((20, 20), np.uint8), np.zeros((20, 20, 4), np.uint8),
                                 np.zeros((20, 20, 3), float), np.zeros((1, 20, 3), np.uint8)])
def test_invalid_rgb_rejected(image):
    with pytest.raises(ValueError):
        paint_support(image)


def test_large_flat_regions_do_not_overflow_opencv_median_histogram():
    image = np.full((720, 1280, 3), 50, np.uint8)
    truth = np.zeros(image.shape[:2], bool)
    truth[400:420, 250:1050] = True
    truth[520:540, 250:1050] = True
    image[truth] = 240
    result = paint_support(image)
    assert result.parameters['background_kernel_px'] <= 255
    assert iou(result.mask, truth) >= .95


def test_two_flank_check_withholds_one_sided_band_without_calling_it_absent_paint():
    from ivp.paint_profiles import bracketed_components
    image = np.full((180, 300, 3), 60, np.uint8)
    image[:35, 40:260] = 190
    proposed = np.zeros(image.shape[:2], np.uint8)
    proposed[:35, 40:260] = 255
    mask, profiles = bracketed_components(image, proposed)
    assert np.count_nonzero(proposed) > 0 and np.count_nonzero(mask) == 0
    assert profiles and all(not p['retained'] for p in profiles)


def test_profile_evidence_counts_distinct_observed_pixels_and_reports_actual_values():
    image = np.full((160, 240, 3), 35, np.uint8)
    image[70:75, 30:200] = 235
    result = paint_support(image)
    assert np.count_nonzero(result.mask) > 0
    value = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 2]
    for profile in result.profiles:
        for sample in profile['samples']:
            for coordinates, count, median in zip(sample['flank_observed_xy'], sample['flank_observed_unique_pixels'], sample['flank_values']):
                assert len({tuple(point) for point in coordinates}) == count
                if median is not None:
                    assert count >= 3
                    assert median == float(np.median([value[y, x] for x, y in coordinates]))
    assert np.array_equal(result.shape_mask, cv2.bitwise_or(result.mask, result.uncertain))


def test_roof_like_yellow_rectangle_demonstrates_remaining_semantic_ambiguity():
    image = np.full((180, 300, 3), 50, np.uint8)
    image[60:95, 75:210] = [230, 205, 25]
    result = paint_support(image)
    # A still-image colour/shape test cannot say whether this rectangle is a
    # vehicle roof or road paint. Do not encode a yellow/coordinate exception.
    assert np.mean(result.mask[65:90, 85:200] > 0) > .95


@pytest.mark.parametrize('shape,width,height', [((256, 384), 96, 8), ((512, 768), 160, 10)])
@pytest.mark.parametrize('colour', [[228, 226, 218], [235, 196, 26]])
def test_sparse_high_contrast_band_survives_low_global_percentile_range(shape, width, height, colour):
    rng = np.random.default_rng(2026102003)
    background = np.clip(60+rng.normal(0, .6, shape), 0, 255).astype(np.uint8)
    image = np.repeat(background[:, :, None], 3, axis=2)
    x, y = shape[1]//2-width//2, shape[0]//2-height//2
    image[y:y+height, x:x+width] = colour
    original = image.copy()
    result = paint_support(image)
    assert width*height/np.prod(shape) < .01
    assert result.parameters['flat_input_detected'] is True
    assert result.parameters['flat_range'] < 20
    assert result.parameters['flat_input_withheld'] is False
    assert result.parameters['flat_minority_rescued'] is True
    assert result.parameters['flat_input_status'] == 'minority_support_retained'
    assert result.parameters['base_minimum_local_contrast'] == 20
    assert np.mean(result.mask[y+1:y+height-1, x+1:x+width-1] > 0) >= .7
    assert np.array_equal(image, original)


@pytest.mark.parametrize('control', ['uniform', 'noise', 'salt', 'tiny', 'square', 'thin', 'edge', 'weak', 'dark'])
def test_flat_input_rescue_retains_rejection_controls(control):
    rng = np.random.default_rng(2026102008)
    image = np.full((256, 384, 3), 60, np.uint8)
    if control == 'noise':
        gray = np.clip(60+rng.normal(0, 4, image.shape[:2]), 0, 255).astype(np.uint8)
        image = np.repeat(gray[:, :, None], 3, axis=2)
    elif control == 'salt':
        image[rng.random(image.shape[:2]) < .005] = 240
    elif control == 'tiny':
        image[100:103, 100:106] = 240
    elif control == 'square':
        image[100:112, 100:112] = 240
    elif control == 'thin':
        image[100:101, 100:196] = 240
    elif control == 'edge':
        image[:6, 100:196] = 240
    elif control == 'weak':
        image[100:108, 100:196] = 74
    elif control == 'dark':
        image[:] = 4
        image[100:108, 100:196] = 38
    result = paint_support(image)
    assert not np.any(result.mask)
    assert result.parameters['flat_input_detected'] is True
    assert result.parameters['flat_input_withheld'] is True
    assert result.parameters['flat_minority_rescued'] is False
    assert result.parameters['flat_input_status'] == 'withheld_no_supported_minority'
