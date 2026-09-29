"""Prepare and validate small local real-video demos without changing originals.

Explicit source paths are required. No acquisition, model download, hosted model,
accuracy estimate or ground-truth claim is performed. Media and local manifests
belong only in ignored uploads/demo-*, runs/demo-* and validation/real-demo/.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAX_DURATION = 8.
MAX_FPS = 10
MAX_SIDE = 1280
NOTICE = ('A small manually inspectable demonstration, not a labeled benchmark or '
          'detector accuracy evaluation. Missing legal or visual context requires abstention.')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sample_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', value):
        raise ValueError('Sample identifiers must contain 1–64 lowercase letters, digits or hyphens.')
    return value


def owned_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError('Demo outputs must remain inside the selected project.')
    return path


def source_info(source):
    from ivp.media import probe_video, source_revision

    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError('Provide an existing explicitly permitted local video file.')
    revision = source_revision(source)
    info = probe_video(source)
    if not math.isfinite(info['duration']) or info['duration'] <= 0:
        raise ValueError('Source video must have a finite positive duration.')
    digest = sha256(source)
    if source_revision(source) != revision:
        raise ValueError('Source changed while inspecting it.')
    return source, info, {'path': str(source), 'filename': source.name, 'bytes': source.stat().st_size,
                          'sha256': digest, 'revision': revision, 'video_info': info}


def validate_transform(info, offset, duration, fps, max_side):
    for value in (offset, duration, fps, max_side):
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Clip parameters must be finite numbers.')
    if not 0 <= offset < info['duration'] or not 0 < duration <= MAX_DURATION or offset + duration > info['duration']:
        raise ValueError('Clip must stay within the source and last at most eight seconds.')
    if type(fps) is not int or not 1 <= fps <= MAX_FPS or type(max_side) is not int or not 128 <= max_side <= MAX_SIDE:
        raise ValueError('Use an integer frame rate of 1–10 and maximum dimension of 128–1280 pixels.')
    ratio = min(1., max_side / max(info['width'], info['height']))
    width = max(2, int(info['width'] * ratio) // 2 * 2)
    height = max(2, int(info['height'] * ratio) // 2 * 2)
    return width, height


def extract_inspection(source, root, sample_id, timestamps):
    from ivp.media import png_bytes, preview_frame, source_revision, write_json
    import cv2

    sample_id = sample_name(sample_id)
    source, info, origin = source_info(source)
    if not 1 <= len(timestamps) <= 6 or any(type(t) not in (int, float) or not math.isfinite(t)
                                            or not 0 <= t < info['duration'] for t in timestamps):
        raise ValueError('Request one to six finite timestamps inside the source.')
    folder = owned_path(root, f'validation/real-demo/{sample_id}')
    manifest_path = folder / 'inspection.json'
    if manifest_path.exists():
        raise ValueError('Inspection already exists; choose a new sample identifier.')
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for index, target in enumerate(timestamps):
        rgb, actual = preview_frame(source, info, target)
        ratio = min(1., 960 / max(rgb.shape[:2]))
        preview = cv2.resize(rgb, (round(rgb.shape[1] * ratio), round(rgb.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
        path = owned_path(root, f'validation/real-demo/{sample_id}/inspection-{index:02d}.png')
        if path.exists():
            raise ValueError('An inspection image already exists; choose a new sample identifier.')
        path.write_bytes(png_bytes(preview))
        records.append({'id': f'{sample_id}-inspection-{index:02d}', 'requested_source_seconds': target,
                        'actual_source_seconds': actual, 'image': path.relative_to(Path(root).resolve()).as_posix(),
                        'image_sha256': sha256(path),
                        'display_size': [preview.shape[1], preview.shape[0]], 'source_size': [rgb.shape[1], rgb.shape[0]]})
    if source_revision(source) != origin['revision'] or sha256(source) != origin['sha256']:
        raise ValueError('Original source changed during preview extraction.')
    result = {'schema_version': 1, 'sample_id': sample_id, 'kind': 'real_local_source_inspection',
              'source': origin, 'frames': records, 'notice': NOTICE, 'annotations': []}
    write_json(manifest_path, result)
    return result


def prepare_clip(source, root, sample_id, *, offset=0., duration=4., fps=5, max_side=960):
    from ivp.media import hidden_process_options, preview_frame, probe_video, source_revision, tool_path, write_json

    root = Path(root).resolve()
    sample_id = sample_name(sample_id)
    source, info, origin = source_info(source)
    width, height = validate_transform(info, offset, duration, fps, max_side)
    clip = owned_path(root, f'uploads/demo-{sample_id}.mp4')
    folder = owned_path(root, f'validation/real-demo/{sample_id}')
    manifest_path = folder / 'manifest.json'
    if clip.exists() or manifest_path.exists():
        raise ValueError('Derived clip already exists; choose a new sample identifier.')
    _, first_actual = preview_frame(source, info, offset)
    clip.parent.mkdir(parents=True, exist_ok=True)
    folder.mkdir(parents=True, exist_ok=True)
    command = [tool_path('ffmpeg'), '-nostdin', '-hide_banner', '-n', '-ss', str(offset), '-i', str(source),
               '-t', str(duration), '-map', '0:v:0', '-an', '-vf', f'scale={width}:{height}:flags=lanczos,fps={fps}:round=near',
               '-map_metadata', '-1', '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-threads', '1', '-movflags', '+faststart', str(clip)]
    started = time.monotonic()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=180, **hidden_process_options())
    (folder / 'ffmpeg-extraction.log').write_text(completed.stderr, encoding='utf-8')
    if completed.returncode:
        raise RuntimeError('Bounded local clip extraction failed; inspect its ignored FFmpeg log.')
    if source_revision(source) != origin['revision'] or sha256(source) != origin['sha256']:
        raise ValueError('Original source changed during clip extraction.')
    clip_info = probe_video(clip)
    clip_info['source_revision'] = source_revision(clip)
    if clip_info['duration'] > MAX_DURATION + 1 / fps or max(clip_info['width'], clip_info['height']) > MAX_SIDE:
        raise ValueError('Derived clip exceeds the bounded validation limits.')
    result = {'schema_version': 1, 'sample_id': sample_id, 'kind': 'real_local_video_derivative',
              'synthetic': False, 'source': origin, 'clip': {'path': str(clip), 'sha256': sha256(clip), 'info': clip_info},
              'transform': {'source_offset_seconds': offset, 'requested_duration_seconds': duration,
                            'first_decoded_source_frame_at_or_after_offset': first_actual,
                            'output_fps': fps, 'output_size': [width, height], 'audio': 'removed',
                            'resampling': 'FFmpeg fps filter with nearest rounding; no speed change requested.',
                            'timestamp_mapping': 'Approximate original-source time = requested offset + derived clip time. '
                                                 'Resampling and source frame timing can shift the selected original frame. '
                                                 'Evidence frame timestamps refer exactly to the derived clip, not to inferred original PTS.',
                            'command': command, 'elapsed_seconds': time.monotonic() - started},
              'created_at': datetime.now(timezone.utc).isoformat(), 'notice': NOTICE,
              'split_policy': 'Each original source has its own sample ID; these are demonstration samples, not train/test splits.'}
    write_json(manifest_path, result)
    return result


def read_manifest(root, sample_id):
    root = Path(root).resolve()
    sample_id = sample_name(sample_id)
    path = owned_path(root, f'validation/real-demo/{sample_id}/manifest.json')
    if path.stat().st_size > 128 * 1024:
        raise ValueError('Demo manifest exceeds the size limit.')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    expected = owned_path(root, f'uploads/demo-{sample_id}.mp4')
    if (manifest.get('sample_id') != sample_id or manifest.get('kind') != 'real_local_video_derivative'
            or Path(manifest['clip']['path']).resolve() != expected.resolve()):
        raise ValueError('Manifest does not bind this source-separated owned demo clip.')
    if sha256(expected) != manifest['clip']['sha256']:
        raise ValueError('Derived clip changed after preparation.')
    return manifest


def analyze_clip(root, sample_id):
    from ivp.media import source_revision, write_json
    from ivp.pipeline import Settings
    from traffic_ai.runtime import configure_cache, runtime_config

    root = Path(root).resolve()
    manifest = read_manifest(root, sample_id)
    config = runtime_config()
    if config.get('offline') is not True or config.get('isolate_caches') is not True:
        raise ValueError('Real demo validation requires explicit offline and isolate_caches runtime configuration.')
    configure_cache()  # Must precede imports that may initialize model-library caches.
    from traffic_ai.worker import process_run

    clip = Path(manifest['clip']['path'])
    info = manifest['clip']['info']
    if source_revision(clip) != info.get('source_revision'):
        raise ValueError('Derived clip revision changed after preparation.')
    run = owned_path(root, f'runs/demo-{sample_name(sample_id)}')
    if run.exists():
        raise ValueError('Demo run already exists; completed evidence is never overwritten.')
    run.mkdir(parents=True)
    request = {'source': str(clip), 'info': info, 'settings': Settings().to_dict(), 'selected_masks': [], 'ai': True,
               'synthetic': False, 'provenance': {'kind': manifest['kind'], 'sample_id': sample_id,
                                               'origin_sha256': manifest['source']['sha256'],
                                               'source_offset_seconds': manifest['transform']['source_offset_seconds'],
                                               'resampling': manifest['transform']['resampling'], 'notice': NOTICE},
               'created_at': datetime.now(timezone.utc).isoformat()}
    write_json(run / 'request.json', request)
    write_json(run / 'status.json', {'state': 'queued', 'progress': 0, 'message': 'Bounded real local demonstration.'})
    started = time.monotonic()
    process_run(run)  # The existing complete Analyzer/video export pipeline, not a replacement evaluator.
    state = json.loads((run / 'status.json').read_text())
    if state.get('state') != 'complete':
        raise RuntimeError('The real Analyzer did not complete this demo run.')
    if sha256(clip) != manifest['clip']['sha256'] or source_revision(clip) != info['source_revision']:
        raise ValueError('Derived source changed during actual analysis.')
    records = [json.loads(line) for line in (run / 'detections.jsonl').read_text().splitlines() if line.strip()]
    if not records or len(records) > MAX_DURATION * MAX_FPS + 1:
        raise ValueError('Actual analysis must contain a nonempty bounded sequence of frame records.')
    labels = Counter(d['label'] for frame in records for d in frame.get('detections', []))
    decisions = Counter(d['rule'] + '/' + d['verdict'] for frame in records for d in frame.get('decisions', []))
    result = {'schema_version': 1, 'sample_id': sample_id, 'kind': 'actual_local_analyzer_demo', 'synthetic': False,
              'run_id': run.name, 'elapsed_seconds': time.monotonic() - started, 'frames': len(records),
              'detection_observations_by_label': dict(labels), 'decision_observations': dict(decisions),
              'counts_notice': 'Repeated per-frame algorithm observations, not distinct people, vehicles or verified violations.',
              'source_revision': info['source_revision'], 'clip_sha256': manifest['clip']['sha256'],
              'export_hashes': {name: sha256(run / name) for name in ('request.json', 'detections.jsonl', 'events.jsonl', 'report.json')},
              'geometry_recorded_on_every_frame': all(all(key in frame for key in
                   ('source_size', 'processed_size', 'transform_to_source_pixel_centres')) for frame in records),
              'notice': NOTICE}
    write_json(owned_path(root, f'validation/real-demo/{sample_id}/analysis.json'), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inspect', 'prepare', 'analyze'))
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--times', type=float, nargs='+', default=[0.])
    parser.add_argument('--offset', type=float, default=0.)
    parser.add_argument('--duration', type=float, default=4.)
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--max-side', type=int, default=960)
    args = parser.parse_args()
    if args.action != 'analyze' and args.source is None:
        parser.error('--source is required for explicitly permitted inspection or clip preparation.')
    if args.action == 'inspect':
        result = extract_inspection(args.source, args.root, args.sample_id, args.times)
    elif args.action == 'prepare':
        result = prepare_clip(args.source, args.root, args.sample_id, offset=args.offset,
                              duration=args.duration, fps=args.fps, max_side=args.max_side)
    else:
        result = analyze_clip(args.root, args.sample_id)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
