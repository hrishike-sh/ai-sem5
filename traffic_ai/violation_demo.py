"""Local, visibly synthetic demonstrations of the real geometric rule engine.

Tracks are generated from known rectangle coordinates, never detector output.
No model, account, network request, external media or legal ground truth is used.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

import cv2
import numpy as np

from ivp.marking_policy import configured_geometry_policy
from ivp.media import hidden_process_options, probe_video, source_revision, tool_path, write_json
from .contracts import Detection, Rule
from .rules import RuleEngine, SceneEvidence


SCENARIOS = {
    'opposing': 'Opposing the configured direction',
    'following': 'Following the configured direction',
    'unknown': 'Direction context unavailable',
}
NOTICE = ('SYNTHETIC GEOMETRIC DEMO: generated rectangles and known preset tracks. '
          'The existing rule engine evaluates these tracks; no detector or tracking model runs. '
          'Configured-rule observations are not legal violations or real-world accuracy evidence.')
WIDTH, HEIGHT, FPS, FRAME_COUNT = 640, 360, 10, 30
REGION = [[30, 130], [610, 130], [610, 290], [30, 290]]


def records_and_frames(scenario):
    """Yield the frame and independently computed rule record for each timestamp."""
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ValueError('Choose opposing, following or unknown for the synthetic demo.')
    configured = scenario != 'unknown'
    direction = [1, 0] if configured else None
    marking = {**configured_geometry_policy(), 'direction': direction, 'polygon': REGION,
               'synthetic': True, 'basis': 'Explicit synthetic demo preset, not a real road direction.'}
    scene = SceneEvidence(stationary_camera=True, confidence=1.0,
                          approach_polygon=REGION, direction_polygon=REGION,
                          permitted_direction=direction,
                          source_evidence={'direction_marking': marking} if configured else {})
    engine = RuleEngine()
    for index in range(FRAME_COUNT):
        stamp = index / FPS
        center = 140 + 80 * stamp if scenario == 'following' else 500 - 80 * stamp
        box = (center - 28, 185., center + 28, 225.)
        known_track = Detection('car', 1.0, box, track_id=1,
                                model='synthetic-known-rectangle-coordinate; no detector')
        decisions = [d.to_dict() for d in engine.evaluate(known_track, stamp, scene, video=True)
                     if d.rule == Rule.WRONG_WAY]
        for decision in decisions:
            decision['evidence']['synthetic'] = True
        record = {'timestamp': stamp, 'scene_id': 0, 'camera_stable': True, 'synthetic': True,
                  'source_size': [WIDTH, HEIGHT], 'processed_size': [WIDTH, HEIGHT],
                  'transform_to_source_pixel_centres': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                  'synthetic_track': {'id': 1, 'box': list(box), 'basis': 'Generated rectangle coordinates'},
                  'decisions': decisions}
        frame = np.full((HEIGHT, WIDTH, 3), (24, 31, 44), dtype=np.uint8)
        cv2.rectangle(frame, (30, 130), (610, 290), (76, 92, 112), 2)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (245, 187, 75), -1)
        if configured:
            cv2.arrowedLine(frame, (240, 155), (400, 155), (100, 214, 168), 3, tipLength=.12)
        lines = [('SYNTHETIC GEOMETRIC DEMO', 28, .72, (245, 187, 75)),
                 (SCENARIOS[scenario], 62, .58, (235, 240, 248)),
                 ('Preset direction: RIGHT' if configured else 'Preset direction: UNKNOWN', 96, .55, (180, 197, 215)),
                 (f'Known rectangle track 1 | t = {stamp:.1f}s', 320, .55, (235, 240, 248)),
                 ('No detector inference. No legal conclusion.', 347, .50, (180, 197, 215))]
        for text, y, size, colour in lines:
            cv2.putText(frame, text, (26, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 1, cv2.LINE_AA)
        yield frame, record


def _encode(path, frames):
    command = [tool_path('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH}x{HEIGHT}', '-r', str(FPS),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-n', str(path)]
    try:
        result = subprocess.run(command, input=b''.join(frame.tobytes() for frame in frames),
                                capture_output=True, timeout=30, **hidden_process_options())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError('Synthetic demo encoding failed. Check that FFmpeg is installed and retry.') from exc
    if result.returncode:
        raise ValueError('Synthetic demo encoding failed. Check FFmpeg H.264 support and retry.')


def create_demo(root, data):
    """Create one new immutable demo run; never overwrite an existing review."""
    if not isinstance(data, dict) or set(data) != {'scenario'}:
        raise ValueError('Select one synthetic demo scenario.')
    scenario = data['scenario']
    if not isinstance(scenario, str) or scenario not in SCENARIOS:
        raise ValueError('Choose opposing, following or unknown for the synthetic demo.')
    root = Path(root).resolve()
    folders = [(root / name).resolve() for name in ('runs', 'uploads')]
    if any(folder.parent != root for folder in folders):
        raise ValueError('Synthetic demo files must stay inside this app.')
    for folder in folders:
        folder.mkdir(exist_ok=True)
    identity = f'synthetic-{scenario}-{uuid.uuid4().hex[:16]}'
    run, source = folders[0] / identity, folders[1] / f'{identity}.mp4'
    run.mkdir()
    created_at = datetime.now(timezone.utc).isoformat()
    write_json(run / 'status.json', {'state': 'running', 'updated_at': created_at,
                                    'progress': 0, 'message': 'Generating a local synthetic geometry demo.'})
    try:
        generated = list(records_and_frames(scenario))
        _encode(source, [frame for frame, _ in generated])
        records = [record for _, record in generated]
        info = {**probe_video(source), 'source_revision': source_revision(source), 'synthetic': True}
        metadata = {'scenario': scenario, 'label': SCENARIOS[scenario], 'notice': NOTICE,
                    'generated_at': created_at, 'track_source': 'Known generated rectangle coordinates',
                    'detector_executed': False, 'rule_engine_executed': True,
                    'rule': 'wrong_way', 'assessment_scope': ['wrong_way'], 'evaluated_frames': FRAME_COUNT,
                    'confidence_notice': 'Confidence 1.0 describes generated input, not model probability.',
                    'configured_direction': [1, 0] if scenario != 'unknown' else None,
                    'configured_region': REGION,
                    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'recorded_labels': {label: sum(d['verdict'] == label for r in records for d in r['decisions'])
                                        for label in ('violation', 'no_violation_observed', 'cannot_assess')}}
        request = {'source': str(source), 'info': info, 'synthetic': True, 'synthetic_demo': metadata}
        write_json(run / 'request.json', request)
        write_json(run / 'report.json', {**request, 'ai': {'synthetic': True, 'demonstration': metadata}})
        (run / 'detections.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        # This is a rule-observation demo, not the event aggregator or a model benchmark.
        (run / 'events.jsonl').write_bytes(b'')
        write_json(run / 'status.json', {'state': 'complete', 'progress': 1,
                   'updated_at': datetime.now(timezone.utc).isoformat(),
                   'message': 'Synthetic geometry demo ready. No detector inference or legal conclusion.'})
    except Exception:
        write_json(run / 'status.json', {'state': 'failed', 'progress': 0,
                   'updated_at': datetime.now(timezone.utc).isoformat(),
                   'message': 'Synthetic demo did not complete. Check local media dependencies and retry.'})
        raise
    return {'id': identity, 'synthetic': True, 'scenario': scenario, 'label': SCENARIOS[scenario], 'notice': NOTICE}
