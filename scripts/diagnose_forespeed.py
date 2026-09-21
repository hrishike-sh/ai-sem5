"""Audit acquired originals and record why physical speed is currently withheld."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ivp.media import orient_video, probe_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.is_relative_to(dataset):
        raise ValueError('Diagnostic output must be outside the original dataset.')
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = dataset / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    records = []
    for clip in manifest['clips']:
        source = Path(clip['path']).resolve()
        if not source.is_relative_to(dataset) or hashlib.sha256(source.read_bytes()).hexdigest() != clip['sha256']:
            raise ValueError('Original source hash/path mismatch: ' + clip['id'])
        info = probe_video(source)
        dest = output / clip['id']
        dest.mkdir()
        stamps, intervals, first_absolute = [], [], None
        selected_at = iter(np.linspace(0, max(0, info['duration']-.5), 6))
        target, previews = next(selected_at), []
        with av.open(str(source)) as container, (dest / 'native-timestamps.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'relative_pts_seconds', 'absolute_pts_seconds', 'pts_ticks', 'time_base'])
            stream = container.streams[info['stream_index']]
            for index, frame in enumerate(container.decode(stream)):
                if frame.pts is None:
                    raise ValueError('Missing native frame PTS.')
                absolute = float(frame.pts * frame.time_base)
                if first_absolute is None:
                    first_absolute = absolute
                stamp = absolute-first_absolute
                if stamps:
                    interval = stamp-stamps[-1]
                    if interval <= 0:
                        raise ValueError('Non-increasing native PTS.')
                    intervals.append(interval)
                stamps.append(stamp)
                writer.writerow([index, stamp, absolute, frame.pts, str(frame.time_base)])
                if target is not None and stamp >= target:
                    rgb = orient_video(frame.to_ndarray(format='rgb24'), info['rotation'], info.get('reflected', False))
                    image = Image.fromarray(rgb)
                    image.thumbnail((640, 420))
                    image.save(dest / f'preview-{len(previews):02}.png')
                    previews.append({'frame': index, 'timestamp_seconds': stamp, 'file': f'preview-{len(previews):02}.png'})
                    target = next(selected_at, None)
        if not stamps:
            raise ValueError('No decoded frames.')
        rectangle = manifest['reference_rectangles_m'][clip['perspective']]
        mps = clip['reference_speed_kmh']/3.6
        max_chord_seconds = math.hypot(*rectangle)/mps
        long_side_seconds = max(rectangle)/mps
        record = {'id': clip['id'], 'source': str(source), 'source_sha256': clip['sha256'],
                  'source_url': clip['source_url'], 'repository_commit': manifest['repository_commit'],
                  'license': manifest['license'], 'attribution': str(dataset/'ATTRIBUTION.txt'),
                  'media': info, 'decoded_frames': len(stamps), 'first_absolute_pts_seconds': first_absolute,
                  'last_relative_pts_seconds': stamps[-1],
                  'native_intervals_seconds': {'min': min(intervals), 'median': float(np.median(intervals)),
                                               'max': max(intervals), 'distinct_rounded_6dp': len(set(round(t, 6) for t in intervals))},
                  'native_pts_preserved': True, 'all_frames_decoded': True, 'previews': previews,
                  'published_reference': {'speed_mph': clip['reference_speed_mph'], 'speed_kmh': clip['reference_speed_kmh'],
                                          'meaning': manifest['reference_speed_scope'], 'used_to_fit_prediction': False},
                  'measured_reference_rectangle_m': rectangle,
                  'camera_specific_pixel_corners': None, 'corner_status': 'not yet independently registered to this camera view',
                  'lens_status': 'camera-specific distortion parameters and rectification unverified',
                  'calibration': None, 'physical_speed_kmh': None, 'physical_speed_status': 'withheld',
                  'reasons': ['Camera-specific measured-patch corners unresolved.', 'Camera lens correction unverified.',
                              'No extrapolation outside a measured patch is certified.'],
                  'eligibility_diagnostic': {'reference_speed_assumed_constant_only_for_this_bound': True,
                                             'maximum_patch_chord_seconds': max_chord_seconds,
                                             'long_side_pass_seconds': long_side_seconds,
                                             'minimum_observation_seconds': .5, 'minimum_samples': 8,
                                             'diagonal_cannot_support_duration_gate': max_chord_seconds < .5,
                                             'meaning': 'Geometric upper bound from published rectangle and reference speed, not an observed track or speed estimate.'},
                  'posted_speed_limit': None, 'legal_rule_applicability': 'unresolved',
                  'source_unchanged': hashlib.sha256(source.read_bytes()).hexdigest() == clip['sha256']}
        if max_chord_seconds < .5:
            record['reasons'].append('Even the longest possible path inside this measured rectangle is too short for the existing 0.5 second speed gate at the published reference speed.')
        (dest/'diagnostic.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        records.append(record)
        print(json.dumps({'id': clip['id'], 'frames': len(stamps), 'size': [info['width'], info['height']],
                          'duration': info['duration'], 'physical_speed_status': 'withheld'}), flush=True)
    result = {'kind': 'real_foreSpeed_native_decode_and_calibration_diagnostics',
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'clips': records, 'all_originals_unchanged': all(r['source_unchanged'] for r in records),
              'no_detector_inference': True, 'no_physical_speed_predictions': True}
    (output/'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
