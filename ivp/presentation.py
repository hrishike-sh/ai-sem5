"""Describe captured IVP arrays; never recompute a stage or infer accuracy."""
import numpy as np


def pixel_change(before, after):
    def rgb(value):
        return np.repeat(value[:, :, None], 3, axis=2) if value.ndim == 2 else value
    left, right = rgb(before), rgb(after)
    if left.shape != right.shape:
        raise ValueError('Demonstration comparisons require matching pixel coordinates.')
    changed = np.any(left != right, axis=2)
    return {'changed_pixels': int(changed.sum()), 'total_pixels': int(changed.size),
            'changed_percent': float(changed.mean() * 100),
            'mean_absolute_channel_change': float(np.abs(left.astype(float) - right.astype(float)).mean())}


class Capture:
    def __init__(self, goal, title, scope):
        self.document = {'schema_version': 1, 'goal': goal, 'title': title, 'scope': scope,
                         'steps': [], 'omitted_steps': [], 'asset_keys': [],
                         'metric_notice': 'Pixel change measures an observed effect, not accuracy or proof of better quality.'}
        self.images = {}

    def asset(self, key, pixels):
        self.images[key] = pixels
        return key

    def step(self, key, title, before, after, inputs, parameters, purpose):
        metrics = pixel_change(self.images[before], self.images[after])
        record = {'id': key, 'title': title, 'before': before, 'after': after,
                  'inputs': inputs, 'parameters': parameters, 'purpose': purpose, 'effect': metrics}
        if metrics['changed_pixels']:
            self.document['steps'].append(record)
        else:
            self.document['omitted_steps'].append({'id': key, 'reason': 'Executed output has no changed pixels.', **metrics})

    def finish(self, original, final, summary):
        used = {original, final}
        for step in self.document['steps']:
            used.update((step['before'], step['after']))
        self.document.update(original=original, final=final, summary=summary, asset_keys=sorted(used))
        return self.document, {key: self.images[key] for key in sorted(used)}


def image_presentation(result, settings, graph, goal):
    if goal not in ('colour', 'regions', 'edges'):
        raise ValueError('Choose a colour image, region map or edge map.')
    titles = {'colour': 'Colour image', 'regions': 'Connected region map', 'edges': 'Edge map'}
    cap = Capture(goal, titles[goal], 'Classical IVP only. Regions and edges have no vehicle or traffic-rule labels.')
    cap.asset('original', result.stages['Prepared'])
    if goal == 'colour':
        colour = {k: v for k, v in {'white_balance': settings.white_balance, 'exposure_stops': settings.exposure, 'percentile_stretch': settings.stretch}.items() if v}
        brightness = ({'gamma': settings.gamma} if settings.gamma != 1 else {})
        if settings.clahe:
            brightness.update(CLAHE_clip_limit=settings.clip_limit, CLAHE_tiles=[8, 8])
        detail = ({'unsharp_strength': settings.sharpen} if settings.sharpen else {})
        if settings.saturation != 1:
            detail['saturation'] = settings.saturation
        light_title = ' + '.join((['Gamma correction'] if settings.gamma != 1 else []) + (['Luminance CLAHE'] if settings.clahe else []))
        colour_title = ' + '.join(name for key, name in [('white_balance','White balance'), ('exposure_stops','Exposure'), ('percentile_stretch','Percentile stretch')] if key in colour)
        detail_title = ' + '.join((['Unsharp masking'] if settings.sharpen else []) + (['HSV saturation'] if settings.saturation != 1 else []))
        stages = [('denoised', 'Denoised', settings.denoise + ' filtering', {'method': settings.denoise, 'kernel': settings.kernel}, 'Applied the selected spatial filter to the colour image.'),
                  ('colour', 'Colour + exposure', colour_title, colour, 'Applied the recorded colour or exposure operations to this image.'),
                  ('brightness', 'Brightness + contrast', light_title, brightness, 'Changed the colour image using the recorded intensity transform.'),
                  ('final', 'Final colour', detail_title, detail, 'Applied the recorded detail or saturation operations to the final colour output.')]
        before = 'original'
        for key, captured, title, params, purpose in stages:
            cap.asset(key, result.stages[captured])
            cap.step(key, title, before, key, [captured + ' input'], params, purpose)
            before = key
        summary = {'output': 'Colour output', 'changed_pixels_from_original': pixel_change(result.stages['Prepared'], result.color_rgb)['changed_pixels'],
                   'colour_metrics': result.measurements}
        return cap.finish('original', 'final', summary)
    by_id = {stage['id']: stage for stage in graph}
    chain = ['gray', 'smooth', 'contrast'] + (['selected', 'opening', 'closing', 'components'] if goal == 'regions' else ['canny'])
    before = 'original'
    for key in chain:
        stage = by_id[key]
        cap.asset(key, stage['image'])
        params = dict(stage['parameters'])
        if key == 'selected':
            params.update(by_id['otsu']['parameters'] if settings.threshold_mode == 'Otsu' else
                          by_id['adaptive']['parameters'] if settings.threshold_mode == 'Adaptive' else {'threshold': settings.threshold_value})
        purpose = {'gray': 'Removed colour to form the intensity input used by this analysis path.',
                   'smooth': 'Filtered the grayscale input used by local contrast and the selected result.',
                   'contrast': 'Applied local grayscale histogram equalisation before the selected analysis result.',
                   'selected': 'Applied the selected threshold to produce the binary mask used by morphology.',
                   'opening': 'Removed or changed small foreground structures before closing.',
                   'closing': 'Filled or changed gaps before connected-component labelling.',
                   'components': f"Labelled {params.get('regions', 0)} connected regions meeting the recorded minimum area.",
                   'canny': f"Produced {int(np.count_nonzero(stage['image']))} edge pixels with the recorded hysteresis thresholds."}[key]
        cap.step(key, stage['title'], before, key, [by_id[before]['title'] if before != 'original' else 'Prepared original colour'], params, purpose)
        before = key
    summary = ({'regions': int(by_id['components']['parameters']['regions']), 'minimum_region_area_px': 25}
               if goal == 'regions' else {'edge_pixels': int(np.count_nonzero(by_id['canny']['image']))})
    return cap.finish('original', before, summary)


def road_presentation(report, images, trace):
    cap = Capture('road_appearance', 'Road appearance measurements',
                  'REVIEW ONLY. Candidate pixels are not verified paint, crossing or stop-line geometry and cannot configure incident rules.')
    values = {'original': images['selected-road'], 'value': trace['value'], 'background': images['local-background'],
              'contrast': images['local-paint-contrast'], 'seed': images['colour-seed'], 'supported': trace['supported'],
              'opened': trace['opened'], 'closed': trace['closed'], 'shapes': images['shape-proposals'],
              'mask': images['cleaned-mask'], 'overlay': images['annotated-survey']}
    for key, pixels in values.items():
        cap.asset(key, pixels)
    p, counts = report['parameters'], report['support']
    shape_params = {key: p[key] for key in ['minimum_component_area_px', 'minimum_rotated_aspect', 'minimum_rotated_fill', 'maximum_component_roi_fraction']}
    if trace['bridge_attempts']:
        shape_params.update(bridge_opening_kernel=[7, 7], bridge_attempted_components=trace['bridge_attempts'], accepted_splits=counts['bridge_split_components'])
    ops = [
        ('value', 'HSV brightness channel', 'original', 'value', ['Original RGB crop'], {'channel': 'HSV Value (maximum RGB channel)'}, 'Extracted the original brightness channel used by the local background and segmentation.'),
        ('background', 'Local median background', 'value', 'background', ['HSV Value'], {'kernel': p['background_kernel_px'], 'area_reduction': p['background_reduction']}, 'Estimated the slowly varying brightness background used by the local thresholds and subtraction.'),
        ('contrast', 'Positive local contrast', 'value', 'contrast', ['HSV Value', 'Local median background'], {'operation': 'max(Value - background, 0)'}, 'Subtracted the local background to measure positive brightness differences.'),
        ('seed', 'White and yellow appearance seed', 'value', 'seed', ['Original HSV', 'Local median background'], {'brightness_baseline': p['brightness_baseline'], 'white_saturation_max': p['white_saturation_max'], 'yellow_hue_range': p['yellow_hue_range'], 'absolute_value_floor': 40}, f"Retained {counts['colour_seed_pixels']} colour-and-brightness candidate pixels."),
        ('supported', 'Require local contrast', 'seed', 'supported', ['Appearance seed', 'Positive local contrast'], {'minimum_local_contrast': p['minimum_local_contrast'], 'units': p['contrast_units']}, f"Retained {counts['local_contrast_pixels']} pixels meeting the measured local-contrast threshold."),
        ('opened', 'Morphological opening', 'supported', 'opened', ['Contrast-supported seed'], {'kernel': [3, 3], 'operation': 'erosion then dilation'}, 'Changed small foreground structures before the closing operation.'),
        ('closed', 'Morphological closing', 'opened', 'closed', ['Opened mask'], {'kernel': [3, 3], 'operation': 'dilation then erosion'}, 'Changed small gaps before connected-component shape filtering.'),
        ('shapes', 'Connected shape filtering', 'closed', 'shapes', ['Closed mask'], shape_params, f"Retained {counts['shape_proposal_pixels']} pixels in elongated shape proposals."),
        ('mask', 'Observed darker flanks', 'shapes', 'mask', ['Shape proposals', 'Original RGB crop'], p['two_flank_support'], f"Retained {counts['supported_pixels']} candidate pixels after the two-sided flank check."),
        ('overlay', 'Measure and display retained fragments', 'original', 'overlay', ['Final candidate mask', 'Original RGB crop'], {'coordinates': 'analysis pixel centres', 'physical_calibration': False}, f"Measured {report['summary']['fragments']} fragments and {report['summary']['coverage_percent']:.3f}% candidate coverage in the selected crop.")]
    for args in ops:
        cap.step(*args)
    return cap.finish('original', 'overlay', dict(report['summary']))
