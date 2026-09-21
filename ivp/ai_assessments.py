"""Bounded read-only pages of recorded AI decisions; no media/model dependency."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

PAGE_SIZE = 20
MAX_STREAM_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_DECISION_BYTES = 64 * 1024
MAX_DECISIONS = 1_000_000
VERDICTS = ('all', 'cannot_assess', 'no_violation_observed', 'violation')
RULES = ('all', 'helmet_absence', 'triple_riding', 'red_light_crossing',
         'stop_line_encroachment', 'crossing_obstruction_during_red',
         'wrong_way', 'footpath_riding', 'restricted_entry', 'prohibited_turn')


def _invalid_constant(value):
    raise ValueError('Saved assessments contain a nonfinite number.')


def _json(raw):
    try:
        return json.loads(raw, parse_constant=_invalid_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Saved assessment files contain unreadable JSON. Use the original downloads for diagnosis.') from exc


def _bytes(path, maximum):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Saved assessment file exceeds the inspector limit. Use the original downloads.')
    return raw


def _digest(path):
    digest = hashlib.sha256()
    size = 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_STREAM_BYTES:
                raise ValueError('AI frame decisions exceed the 64 MiB inspector limit. Use the original download.')
            digest.update(chunk)
    return digest.hexdigest()


def _cursor(value):
    if value is None:
        return None, 0
    try:
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,1024}', value):
            raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(value + '=' * (-len(value) % 4)))
        offset = data['offset']
        if (not isinstance(data, dict) or type(offset) is not int or offset <= 0
                or offset % PAGE_SIZE or offset >= MAX_DECISIONS):
            raise ValueError()
        return data, offset
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise ValueError('Invalid saved-assessment page. Open the first page again.') from exc


def saved_ai_page(root, name, cursor=None, verdict='cannot_assess', rule='all'):
    """Stream one source record at a time; return at most twenty exact decisions."""
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', name):
        raise ValueError('Invalid saved run identifier.')
    if verdict not in VERDICTS or rule not in RULES:
        raise ValueError('Choose a supported assessment verdict and rule.')
    decoded, offset = _cursor(cursor)
    runs = (Path(root) / 'runs').resolve()
    run = (runs / name).resolve()
    if run.parent != runs:
        raise ValueError('Saved assessments must stay inside the run folder.')
    paths = {name: run / name for name in ('status.json', 'report.json', 'detections.jsonl')}
    if any(p.resolve().parent != run or not p.is_file() for p in paths.values()):
        raise ValueError('Saved AI frame decisions are unavailable. Keep the original run files together.')
    status_raw = _bytes(paths['status.json'], 64 * 1024)
    report_raw = _bytes(paths['report.json'], 8 * 1024 * 1024)
    status, report = _json(status_raw), _json(report_raw)
    if not isinstance(status, dict) or status.get('state') != 'complete':
        raise ValueError('Inspect assessments after the saved AI run finishes successfully.')
    if not isinstance(report, dict) or not isinstance(report.get('ai'), dict):
        raise ValueError('This saved run has no AI assessment report.')
    if paths['detections.jsonl'].stat().st_size > MAX_STREAM_BYTES:
        raise ValueError('AI frame decisions exceed the 64 MiB inspector limit. Use the original download.')
    total = filtered = frames = size = 0
    counts = Counter()
    cards = []
    digest = hashlib.sha256()
    with paths['detections.jsonl'].open('rb') as stream:
        for line_number in range(1, MAX_DECISIONS + 1):
            raw = stream.readline(MAX_LINE_BYTES + 1)
            if not raw:
                break
            size += len(raw)
            if size > MAX_STREAM_BYTES or len(raw) > MAX_LINE_BYTES:
                raise ValueError('AI frame decisions exceed inspector limits. Use the original download.')
            digest.update(raw)
            if not raw.strip():
                continue
            record = _json(raw)
            if not isinstance(record, dict) or not isinstance(record.get('decisions'), list):
                raise ValueError(f'Invalid AI decision record at source line {line_number}.')
            frames += 1
            for index, decision in enumerate(record['decisions']):
                if (not isinstance(decision, dict) or not isinstance(decision.get('rule'), str)
                        or not isinstance(decision.get('verdict'), str)
                        or len(decision['rule']) > 128 or len(decision['verdict']) > 128):
                    raise ValueError(f'Invalid AI assessment at source line {line_number}.')
                total += 1
                if total > MAX_DECISIONS:
                    raise ValueError('Too many assessments for the inspector. Use the original download.')
                counts[f"{decision['rule']}/{decision['verdict']}"] += 1
                if verdict != 'all' and decision['verdict'] != verdict or rule != 'all' and decision['rule'] != rule:
                    continue
                filtered += 1
                if offset < filtered <= offset + PAGE_SIZE:
                    card = {'source_position': total, 'decision': decision,
                                  'frame_context': {key: record.get(key) for key in
                                                    ('frame', 'timestamp', 'scene_id', 'scene_revision', 'camera_stable',
                                                     'synthetic', 'provenance', 'event_confirmation',
                                                     'event_vehicle_indices', 'event_confirmation_indexing')},
                                  'source': {'file': 'detections.jsonl', 'line': line_number,
                                             'pointer': f'/decisions/{index}',
                                             'line_sha256': hashlib.sha256(raw).hexdigest()}}
                    if len(json.dumps(card, allow_nan=False).encode()) > MAX_DECISION_BYTES:
                        raise ValueError('An assessment exceeds the inspector evidence limit. Use the original download.')
                    cards.append(card)
        else:
            raise ValueError('Too many source lines for the inspector. Use the original download.')
    if offset and offset >= filtered:
        raise ValueError('This saved-assessment page is outside the selected results. Open the first page.')
    expected = report['ai'].get('assessment_counts')
    if expected is not None and (not isinstance(expected, dict) or expected != dict(counts)
                                 or any(type(v) is not int or v < 0 for v in expected.values())):
        raise ValueError('Saved AI decision counts do not match the report. Original verdicts have not been changed; inspect the downloads.')
    fingerprints = {'detections.jsonl': digest.hexdigest(),
                    'report.json': hashlib.sha256(report_raw).hexdigest(),
                    'status.json': hashlib.sha256(status_raw).hexdigest()}
    # Verify the actual parsed bytes against current files before exposing cards.
    if any(_digest(paths[p]) != value for p, value in fingerprints.items()):
        raise ValueError('Saved AI files changed while reading. Reopen the first page.')
    binding = {'run': name, 'verdict': verdict, 'rule': rule,
               'sha256': hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()}
    if decoded is not None and decoded != {**binding, 'offset': offset}:
        raise ValueError('Saved-assessment page is outdated or belongs to another filter. Open the first page.')

    def token(position):
        if position == 0:
            return None
        return base64.urlsafe_b64encode(json.dumps({**binding, 'offset': position}, separators=(',', ':')).encode()).decode().rstrip('=')

    return {'run': name, 'decisions': cards, 'counts': dict(counts), 'source_records': frames,
            'fingerprints': fingerprints, 'integrity': 'Current saved files fingerprinted; original export authenticity is not established.',
            'report_counts': 'match' if expected is not None else 'not_recorded',
            'provenance_notice': 'Only recorded provenance is shown. Missing synthetic/provenance fields are unrecorded, not proof of real footage.',
            'limits': 'Read-only recorded assessments; no new inference or verdict changes. Source-file and decision order are preserved, including reversed timestamps.',
            'pagination': {'size': PAGE_SIZE, 'total': total, 'filtered_total': filtered,
                           'start': offset + 1 if filtered else 0, 'end': min(offset + PAGE_SIZE, filtered),
                           'verdict': verdict, 'rule': rule, 'has_previous': offset > 0,
                           'previous_cursor': token(max(0, offset - PAGE_SIZE)),
                           'next_cursor': token(offset + PAGE_SIZE) if offset + PAGE_SIZE < filtered else None}}
