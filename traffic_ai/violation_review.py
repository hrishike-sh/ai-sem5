"""Evidence-first, local review of immutable saved algorithm output.

Supported means the recorded evidence passes this review contract, never a legal
finding. Reviewer choices live separately and do not change the rule engine.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import threading
from datetime import datetime, timezone

from ivp.marking_policy import geometry_is_configured
from ivp.media import load_image, png_bytes, preview_frame, source_revision, write_json


MAX_FILE = 64 * 1024 * 1024
MAX_LINE = 2 * 1024 * 1024
MAX_JSON = 8 * 1024 * 1024
MAX_DECISION = 64 * 1024
MAX_RECORDS = 100_000
MAX_DECISIONS = 1_000_000
MAX_CARDS = 200
RULES = ('helmet_absence', 'wrong_way', 'red_light_crossing',
         'stop_line_encroachment', 'crossing_obstruction_during_red',
         'footpath_riding', 'restricted_entry', 'prohibited_turn')
NOTICE = ('Recorded algorithm output for human review; supported evidence is not '
          'a legal finding, determination of guilt, or basis for a fine. '
          'Reviewer decisions do not change the original detections or events.')


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, length):
    return isinstance(value, (list, tuple)) and len(value) == length and all(_finite(x) for x in value)


def _box(value):
    return _vector(value, 4) and value[2] > value[0] and value[3] > value[1]


def _json(raw):
    def invalid(value):
        raise ValueError('Nonfinite numbers are not valid review evidence.')
    def check(value, depth=0):
        if depth > 40:
            raise ValueError('Review evidence nesting exceeds the limit.')
        if isinstance(value, float) and not math.isfinite(value):
            invalid(value)
        if isinstance(value, dict):
            for item in value.values():
                check(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                check(item, depth + 1)
    try:
        value = json.loads(raw, parse_constant=invalid)
        check(value)
        return value
    except (UnicodeError, RecursionError, ValueError) as exc:
        raise ValueError('Saved review evidence contains invalid JSON or nonfinite values.') from exc


def _read(path, limit=MAX_JSON):
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Saved review evidence exceeds the file size limit.')
    return raw


def _hash(path):
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE:
                raise ValueError('Saved review evidence exceeds the file size limit.')
            digest.update(chunk)
    return digest.hexdigest()


def _identity(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, allow_nan=False).encode()).hexdigest()[:24]


class ViolationReview:
    def __init__(self, root, files):
        self.root = Path(root).resolve()
        self.files = files
        self.lock = threading.RLock()

    def _run(self, run_id):
        if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,180}', run_id):
            raise ValueError('Invalid saved run identifier.')
        runs = (self.root / 'runs').resolve()
        run = (runs / run_id).resolve()
        if not runs.is_relative_to(self.root) or run.parent != runs or not run.is_dir():
            raise ValueError('Review must use an existing run inside this project.')
        return run

    @staticmethod
    def _path(run, name):
        path = run / name
        if path.resolve().parent != run:
            raise ValueError('Review files must remain inside their saved run.')
        return path

    def _sources(self, run):
        result = {}
        for name in ('status.json', 'request.json', 'report.json', 'detections.jsonl', 'events.jsonl'):
            path = self._path(run, name)
            if name == 'events.jsonl' and not path.exists():
                result[name] = None
            else:
                if not path.is_file():
                    raise ValueError('A complete saved AI run with request, report and detections is required.')
                result[name] = _hash(path)
        return result

    def _snapshot(self, run_id):
        run = self._run(run_id)
        hashes = self._sources(run)
        status = _json(_read(self._path(run, 'status.json'), 64 * 1024))
        request = _json(_read(self._path(run, 'request.json')))
        report = _json(_read(self._path(run, 'report.json')))
        if not isinstance(status, dict) or status.get('state') != 'complete':
            raise ValueError('Review is available after the run completes successfully.')
        if not isinstance(request, dict) or not isinstance(request.get('info'), dict):
            raise ValueError('Saved source metadata is unavailable.')
        if request['info'].get('kind') not in ('image', 'video'):
            raise ValueError('Saved source must be an image or video.')
        if not isinstance(report, dict) or not isinstance(report.get('ai'), dict):
            raise ValueError('This run has no saved AI assessment report.')
        fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        groups, total, skipped, deferred, size = {}, 0, 0, 0, 0
        nonapplicable = 0
        synthetic = (request.get('synthetic') is True or request['info'].get('synthetic') is True
                     or report.get('synthetic') is True or report['ai'].get('synthetic') is True)
        with self._path(run, 'detections.jsonl').open('rb') as stream:
            for line_no in range(1, MAX_RECORDS + 1):
                raw = stream.readline(MAX_LINE + 1)
                if not raw:
                    break
                size += len(raw)
                if len(raw) > MAX_LINE or size > MAX_FILE:
                    raise ValueError('Saved detections exceed the review line or file limit.')
                if not raw.strip():
                    continue
                record = _json(raw)
                if not isinstance(record, dict) or not isinstance(record.get('decisions'), list):
                    raise ValueError('Saved detections contain an invalid frame record.')
                stamp, scene = record.get('timestamp'), record.get('scene_id', 0)
                if not _finite(stamp) or stamp < 0 or type(scene) is not int or scene < 0:
                    raise ValueError('Frame timestamps and scene identifiers must be valid finite values.')
                synthetic = synthetic or record.get('synthetic') is True
                for index, decision in enumerate(record['decisions']):
                    total += 1
                    if total > MAX_DECISIONS:
                        raise ValueError('Saved detections exceed the review decision limit.')
                    self._validate_decision(decision, stamp)
                    synthetic = synthetic or decision.get('synthetic') is True or decision.get('evidence', {}).get('synthetic') is True
                    rule = decision['rule']
                    if rule == 'triple_riding':
                        deferred += 1
                        continue
                    if rule not in RULES:
                        skipped += 1
                        continue
                    if (rule == 'helmet_absence' and decision['verdict'] == 'cannot_assess'
                            and decision['reason'] == 'This rider rule applies to motorcycles'):
                        nonapplicable += 1
                        continue
                    key = (rule, decision.get('track_id'), scene)
                    if key not in groups and len(groups) >= MAX_CARDS:
                        # Retain the priority rules even if lower-priority
                        # unknown observations filled the bounded card set first.
                        lowest = max(groups, key=lambda existing: RULES.index(existing[0]))
                        if RULES.index(rule) < RULES.index(lowest[0]):
                            skipped += groups.pop(lowest)['observation_count']
                        else:
                            skipped += 1
                            continue
                    card = self._card(decision, record, request)
                    card.update(scene_id=scene, source={'file': 'detections.jsonl', 'line': line_no,
                                'pointer': f'/decisions/{index}', 'line_sha256': hashlib.sha256(raw).hexdigest()})
                    rank = ({'insufficient_evidence': 0, 'needs_review': 1, 'supported': 2}[card['status']],
                            decision['verdict'] == 'violation')
                    if key in groups:
                        previous = groups[key]
                        count = previous['observation_count'] + 1
                        first = min(previous['first_timestamp'], stamp)
                        last = max(previous['last_timestamp'], stamp)
                        if rank <= previous['_rank']:
                            previous.update(observation_count=count, first_timestamp=first, last_timestamp=last)
                            continue
                    else:
                        count, first, last = 1, stamp, stamp
                    card.update(_rank=rank, observation_count=count, first_timestamp=first, last_timestamp=last)
                    groups[key] = card
            else:
                if stream.read(1):
                    raise ValueError('Saved detections exceed the review record limit.')
        if hashes != self._sources(run):
            raise ValueError('Saved evidence changed while reading. Refresh the review.')
        cards = []
        for key, card in sorted(groups.items(), key=lambda pair: (RULES.index(pair[0][0]), pair[1]['timestamp'])):
            card.pop('_rank')
            card['id'] = _identity(run_id, fingerprint, key)
            roles = ('before', 'during', 'after') if request['info']['kind'] == 'video' else ('during',)
            card['frame_ids'] = [_identity(card['id'], role) for role in roles]
            cards.append(card)
        state = self._state(run, fingerprint)
        for card in cards:
            card['review'] = state.get(card['id'], {'decision': None, 'note': ''})
            card['synthetic'] = True if synthetic else None
        result = {'run_id': run_id, 'fingerprint': fingerprint, 'cards': cards, 'synthetic': True if synthetic else None,
                  'source_name': Path(request.get('source', '')).name,
                  'omitted_nonapplicable_count': nonapplicable,
                  'provenance_notice': 'Synthetic recorded fixture or demonstration; not live inference or evaluation data.' if synthetic else
                      'Recorded algorithm output; synthetic provenance is unrecorded.',
                  'limits': {'max_cards': MAX_CARDS, 'truncated': skipped > 0,
                             'omitted_observations': skipped, 'deferred_triple_riding_observations': deferred,
                             'nonapplicable_observations': nonapplicable,
                             'recorded_decisions': total, 'grouping': 'rule + track + scene',
                             'notice': NOTICE},
                  'provenance': {'source_hashes': hashes, 'synthetic': True if synthetic else None,
                                 'synthetic_notice': 'Synthetic fixture or demonstration; not evaluation data.' if synthetic else
                                     'Synthetic provenance is unrecorded; this is not proof of real footage.',
                                 'source_revision': request['info'].get('source_revision'),
                                 'source_revision_kind': 'Local path, file size and modification-time fingerprint; not a footage content hash.',
                                 'media_kind': request['info']['kind'],
                                 'integrity': 'Current saved-file hashes bind this review; original authenticity is not established.',
                                 'notice': NOTICE}}
        result['explanations'] = self._explanations(run, fingerprint, cards)
        return run, request, result

    @staticmethod
    def _normalise_explanation(result, card, *, cached=False):
        from .violation_llm import VISUAL_OBSERVATIONS, VISUAL_NOTICE, _fact_text, card_fingerprint, validate_selection
        from .violation_images import image_fingerprint, PRIVACY_NOTICE
        if (not isinstance(result, dict) or result.get('state') != 'explained'
                or result.get('provider') != 'chatgpt_subscription' or result.get('card_id') != card['id']
                or result.get('rule') != card['rule'] or result.get('status') != card['status']
                or not isinstance(result.get('model'), str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,150}', result['model'])):
            raise ValueError('Saved explanation is not bound to this evidence card.')
        current_fingerprint = card_fingerprint(card)
        evidence_fingerprint = card_fingerprint({key: value for key, value in card.items() if key != 'review'})
        if cached:
            if result.get('evidence_fingerprint') != evidence_fingerprint:
                raise ValueError('Saved explanation evidence fingerprint is invalid.')
        elif result.get('card_fingerprint') != current_fingerprint:
            raise ValueError('Explanation card changed before saving.')
        selection = {'card_id': card['id'], 'card_fingerprint': current_fingerprint,
                     **{key: result.get(key) for key in ('fact_ids', 'missing_fact_ids', 'frame_ids')}}
        validate_selection(selection, card)
        chosen = [{f['id']: f for f in card['facts']}[key] for key in selection['fact_ids']]
        gaps = [{f['id']: f for f in card['missing_facts']}[key] for key in selection['missing_fact_ids']]
        mode, images, visual = result.get('vision_mode', 'facts_only'), [], []
        if mode == 'selected_frames':
            descriptors = result.get('images_sent')
            if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= 3:
                raise ValueError('Saved explanation image references are invalid.')
            for image in descriptors:
                if (not isinstance(image, dict) or set(image) != {'id','timestamp','role','sha256','width','height','byte_length','privacy'}
                        or image.get('id') not in card['frame_ids']
                        or not isinstance(image.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', image['sha256'])
                        or not _finite(image.get('timestamp')) or image['timestamp'] < 0
                        or image.get('role') not in ('context', 'subject_crop')
                        or any(type(image.get(key)) is not int or not 1 <= image[key] <= 768 for key in ('width','height'))
                        or type(image.get('byte_length')) is not int or not 1 <= image['byte_length'] <= 256*1024):
                    raise ValueError('Saved explanation image references are invalid.')
                privacy = image.get('privacy')
                if (not isinstance(privacy, dict) or set(privacy) != {'metadata_stripped','blurred_regions','notice'}
                        or privacy['metadata_stripped'] is not True or privacy['notice'] != PRIVACY_NOTICE
                        or type(privacy['blurred_regions']) is not int or not 0 <= privacy['blurred_regions'] <= 100):
                    raise ValueError('Saved explanation image privacy metadata is invalid.')
                images.append(image)
            if len({image['id'] for image in images}) != len(images) or result.get('image_fingerprint') != image_fingerprint(images):
                raise ValueError('Saved explanation image fingerprint is invalid.')
            expected = {(image['id'], category) for image in images for category in VISUAL_OBSERVATIONS}
            observations = result.get('visual_observations')
            if not isinstance(observations, list) or len(observations) != len(expected):
                raise ValueError('Saved visual observations are incomplete.')
            for observation in observations:
                if not isinstance(observation, dict):
                    raise ValueError('Saved visual observations are invalid.')
                identity, category, value = (observation.get(key) for key in ('frame_id', 'category', 'observation'))
                if not isinstance(category, str) or not isinstance(value, str) or (identity, category) not in expected or value not in VISUAL_OBSERVATIONS[category]:
                    raise ValueError('Saved visual observations are invalid.')
                expected.remove((identity, category))
                visual.append({'frame_id': identity, 'category': category, 'observation': value,
                               'text': 'Model observation: ' + VISUAL_OBSERVATIONS[category][value]})
        elif mode != 'facts_only' or result.get('visual_observations') or result.get('images_sent'):
            raise ValueError('Saved explanation mode is invalid.')
        return {'state': 'explained', 'provider': 'chatgpt_subscription', 'model': result['model'],
                'card_id': card['id'], 'card_fingerprint': current_fingerprint,
                'evidence_fingerprint': evidence_fingerprint,
                'rule': card['rule'], 'status': card['status'], 'reason': card['reason'],
                'explanation': '\n\n'.join([card['reason']] + [_fact_text(item) for item in chosen]
                    + ['Missing evidence: ' + item['text'] for item in gaps]),
                'facts': chosen, 'missing_facts': gaps, **{key: selection[key] for key in ('fact_ids','missing_fact_ids','frame_ids')},
                'mode': 'images_and_facts' if images else 'facts_only', 'vision_mode': mode,
                'images_sent': images, 'image_fingerprint': result.get('image_fingerprint') if images else None,
                'visual_observations': visual, 'visual_notice': VISUAL_NOTICE if images else 'Facts only: no images were sent or inspected.'}

    def _explanations(self, run, fingerprint, cards):
        path = self._path(run, 'violation-explanations.json')
        if not path.exists():
            return {}
        try:
            saved = _json(_read(path))
            if not isinstance(saved, dict) or saved.get('fingerprint') != fingerprint or not isinstance(saved.get('results'), dict):
                return {}
            result = {}
            for card in cards:
                item = saved['results'].get(card['id'])
                if item is not None:
                    try:
                        stamp = item.get('saved_at') if isinstance(item, dict) else None
                        if not isinstance(stamp, str) or len(stamp) > 40:
                            continue
                        parsed_stamp = datetime.fromisoformat(stamp)
                        if parsed_stamp.tzinfo is None:
                            continue
                        result[card['id']] = {**self._normalise_explanation(item, card, cached=True), 'saved': True,
                                             'saved_at': parsed_stamp.isoformat()}
                    except (ValueError, TypeError, KeyError):
                        continue
            return result
        except (ValueError, OSError, TypeError):
            return {}

    def save_explanation(self, data, result):
        with self.lock:
            run, _, view, card = self._bound(data)
            normalized = self._normalise_explanation(result, card)
            normalized['saved_at'] = datetime.now(timezone.utc).isoformat()
            entries = view['explanations']
            entries[card['id']] = normalized
            if self._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Evidence changed before saving the explanation.')
            payload = {'schema_version': 1, 'fingerprint': view['fingerprint'], 'results': entries}
            encoded = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
            if len(encoded) > MAX_JSON:
                raise ValueError('Saved explanation storage limit reached; existing explanations were preserved.')
            self._write_bytes(self._path(run, 'violation-explanations.json'), encoded)
            return {**normalized, 'saved': True}

    @staticmethod
    def _validate_decision(decision, stamp):
        if (not isinstance(decision, dict) or not isinstance(decision.get('rule'), str)
                or len(decision['rule']) > 128 or decision.get('verdict') not in
                ('violation', 'cannot_assess', 'no_violation_observed')
                or not isinstance(decision.get('reason'), str) or len(decision['reason']) > 4000
                or not isinstance(decision.get('evidence', {}), dict)
                or not _finite(decision.get('timestamp')) or decision['timestamp'] != stamp
                or (decision.get('track_id') is not None and (type(decision['track_id']) is not int or decision['track_id'] < 0))):
            raise ValueError('Saved detections contain an invalid decision or frame binding.')
        if len(json.dumps(decision, allow_nan=False).encode()) > MAX_DECISION:
            raise ValueError('A saved decision exceeds the review evidence limit.')

    @staticmethod
    def _card(decision, record, request):
        rule, verdict, evidence = decision['rule'], decision['verdict'], decision.get('evidence', {})
        facts = [{'id': 'recorded_verdict', 'text': 'Recorded algorithm verdict', 'value': verdict},
                 {'id': 'recorded_reason', 'text': 'Recorded algorithm explanation', 'value': decision['reason']}]
        missing = []
        def gate(identity, text, value, present):
            if present:
                facts.append({'id': identity, 'text': text, 'value': value})
            else:
                missing.append({'id': identity, 'text': text})
        positive = verdict == 'violation'
        if rule == 'helmet_absence':
            heads = evidence.get('heads')
            gate('bare_heads', 'Explicit visible bare-head evidence; absent helmet detections are not enough', heads,
                 isinstance(heads, list) and bool(heads) and all(_box(head) for head in heads))
            association = evidence.get('association')
            gate('rider_association', 'Bare head associated with a motorcycle rider', association,
                 isinstance(association, str) and bool(association.strip()))
        elif rule == 'wrong_way':
            motion, direction, marking = evidence.get('observed_motion'), evidence.get('permitted_direction'), evidence.get('marking')
            gate('observed_motion', 'Recorded movement vector from tracked video', motion,
                 request['info']['kind'] == 'video' and decision.get('track_id') is not None
                 and _vector(motion, 2) and math.hypot(*motion) >= 15)
            gate('direction_authority', 'Independently configured direction authority (appearance or majority traffic is insufficient)', marking,
                 geometry_is_configured(marking) and _vector(marking.get('direction'), 2)
                 and _vector(direction, 2) and math.hypot(*direction) > 0
                 and all(abs(a - b) < 1e-6 for a, b in zip(marking['direction'], direction)))
            gate('camera_stable', 'Recorded stationary-camera gate', record.get('camera_stable'), record.get('camera_stable') is True)
            cosine = evidence.get('direction_cosine')
            actual_cosine = ((sum(a*b for a, b in zip(motion, direction)) / (math.hypot(*motion)*math.hypot(*direction)))
                             if _vector(motion, 2) and _vector(direction, 2) and math.hypot(*motion) > 0 and math.hypot(*direction) > 0 else None)
            gate('opposing_direction', 'Observed movement opposes the established direction', cosine,
                 _finite(cosine) and -1 <= cosine < -.7 and actual_cosine is not None and abs(actual_cosine-cosine) < .01)
        elif rule in ('red_light_crossing', 'stop_line_encroachment', 'crossing_obstruction_during_red'):
            signal = evidence.get('signal_association')
            relevant = (isinstance(signal, dict) and signal.get('state') == 'red'
                        and signal.get('associated_approach') is not None
                        and signal.get('associated_approach') == evidence.get('approach_id')
                        and _finite(signal.get('timestamp')) and 0 <= record['timestamp'] - signal['timestamp'] <= .5
                        and _finite(signal.get('model_score')) and signal['model_score'] >= .7)
            gate('associated_red_signal', 'Observed red signal associated with this approach; stopped traffic does not establish red', signal,
                 evidence.get('signal') == 'red' and relevant)
            gate('temporal_context', 'Tracked video transition or sustained stopped occupancy', evidence.get('observed_red_entry'),
                 request['info']['kind'] == 'video' and decision.get('track_id') is not None
                 and isinstance(evidence.get('observed_red_entry'), dict))
            if rule == 'crossing_obstruction_during_red':
                gate('stopped_crossing_context', 'Sustained stopped occupancy and applicable crossing context; zebra appearance alone is insufficient',
                     None, False)
            else:
                line = evidence.get('stop_line')
                gate('stop_line', 'Established approach stop line', line,
                     isinstance(line, list) and len(line) == 2 and all(_vector(p, 2) for p in line))
        else:
            gate('context_review', 'Independent applicable scene context and temporal evidence need review', None, False)
        if not positive:
            missing.append({'id': 'positive_observation', 'text': 'No positive candidate was recorded for this observation.'})
        status = 'supported' if positive and not missing and rule in ('helmet_absence', 'wrong_way') else 'insufficient_evidence'
        if positive and not missing:
            status = 'supported' if rule in ('helmet_absence', 'wrong_way') else 'needs_review'
        # Missing direction authority, red association and explicit bare heads are
        # hard abstention gates; a reviewer acceptance cannot promote the status.
        reason = ('Recorded evidence supports this candidate for human review; no legal conclusion.' if status == 'supported'
                  else 'Recorded evidence needs contextual human review.' if status == 'needs_review'
                  else 'Required evidence is missing: ' + '; '.join(item['text'] for item in missing))
        return {'rule': rule, 'status': status, 'timestamp': decision['timestamp'],
                'track_id': decision.get('track_id'), 'reason': reason, 'facts': facts,
                'missing_facts': missing, 'recorded_verdict': verdict}

    def _state(self, run, fingerprint):
        path = self._path(run, 'violation-review-state.json')
        if not path.exists():
            return {}
        saved = _json(_read(path))
        if not isinstance(saved, dict) or saved.get('fingerprint') != fingerprint:
            return {}
        reviews = saved.get('reviews')
        if not isinstance(reviews, dict) or len(reviews) > MAX_CARDS:
            raise ValueError('Saved reviewer state is invalid.')
        for key, value in reviews.items():
            if (not isinstance(value, dict) or value.get('decision') not in ('accept', 'reject')
                    or not isinstance(value.get('note'), str) or len(value['note']) > 4000):
                raise ValueError('Saved reviewer state is invalid.')
        return reviews

    def view(self, run_id):
        with self.lock:
            return self._snapshot(run_id)[2]

    def _bound(self, data, card_required=True):
        if not isinstance(data, dict):
            raise ValueError('Review request must be an object.')
        run, request, view = self._snapshot(data.get('id'))
        if data.get('fingerprint') != view['fingerprint']:
            raise ValueError('Saved evidence changed or the review is stale. Refresh before continuing.')
        card = next((card for card in view['cards'] if card['id'] == data.get('card_id')), None)
        if card_required and card is None:
            raise ValueError('Select a card from this exact saved review.')
        return run, request, view, card

    def decide(self, data):
        with self.lock:
            run, _, view, card = self._bound(data)
            action, note = data.get('decision'), data.get('note', '')
            if action not in ('accept', 'reject', 'reset') or not isinstance(note, str) or len(note) > 4000:
                raise ValueError('Choose accept, reject or reset with a note of at most 4000 characters.')
            reviews = self._state(run, view['fingerprint'])
            if action == 'reset':
                reviews.pop(card['id'], None)
            else:
                reviews[card['id']] = {'decision': action, 'note': note}
            if self._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Saved evidence changed. Refresh before saving the review.')
            write_json(self._path(run, 'violation-review-state.json'),
                       {'schema_version': 1, 'fingerprint': view['fingerprint'], 'reviews': reviews})
            return self._snapshot(data['id'])[2]

    def _source(self, request):
        source = request.get('source')
        if not isinstance(source, str) or not source:
            raise ValueError('Original source path is unrecorded.')
        path = Path(source).resolve()
        allowed = [self.root / folder for folder in ('uploads', 'demo')]
        if not any(folder.resolve().is_relative_to(self.root) and path.is_relative_to(folder.resolve()) for folder in allowed):
            raise ValueError('Frames can only be read from this project\'s uploads or demo folder.')
        revision = request['info'].get('source_revision')
        if not isinstance(revision, str) or not revision:
            raise ValueError('Original source revision is unrecorded; frames cannot be bound to this run.')
        if source_revision(path) != revision:
            raise ValueError('Original source changed since analysis; frames are unavailable.')
        return path, revision

    def frames(self, data):
        with self.lock:
            run, request, view, card = self._bound(data)
            frames = []
            source_error = None
            try:
                source, revision = self._source(request)
            except (ValueError, OSError) as exc:
                source_error = str(exc)
            for role, offset in (('before', -.5), ('during', 0), ('after', .5)):
                item = {'position': role, 'role': role, 'id': _identity(card['id'], role),
                        'timestamp': None, 'url': None, 'available': False}
                frames.append(item)
                if source_error:
                    item['reason'] = source_error
                    continue
                if request['info']['kind'] == 'image' and role != 'during':
                    item['reason'] = 'A still image has no before or after frames.'
                    continue
                target = card['timestamp'] + offset
                item['requested_timestamp'] = target
                if target < 0:
                    item['reason'] = 'No earlier source frame exists before the start of the clip.'
                    continue
                try:
                    if source_revision(source) != revision:
                        raise ValueError('Original source changed before frame extraction.')
                    if request['info']['kind'] == 'image':
                        rgb, actual = load_image(source), 0.
                    else:
                        rgb, actual = preview_frame(source, request['info'], target)
                    if source_revision(source) != revision:
                        raise ValueError('Original source changed during frame extraction.')
                    if not _finite(actual) or actual < 0:
                        raise ValueError('Decoder returned an invalid actual frame timestamp.')
                    if role == 'before' and actual >= card['timestamp'] or role == 'after' and actual <= card['timestamp']:
                        raise ValueError('No distinct source frame exists on this side of the observation.')
                    if request['info']['kind'] == 'image' and card['timestamp'] != 0:
                        raise ValueError('The saved image observation is not bound to timestamp zero.')
                    if role == 'during' and request['info']['kind'] == 'video' and abs(actual - target) > .25:
                        raise ValueError('No source frame close enough to the recorded observation is available.')
                    path = self._path(run, 'violation-frame-' + item['id'] + '.png')
                    pixels = png_bytes(rgb)
                    self._write_bytes(path, pixels)
                    item.update(timestamp=actual, available=True, url=self.files.url(path),
                                source_revision=revision, sha256=hashlib.sha256(pixels).hexdigest(),
                                reason='Actual decoded source frame; no inference rerun.')
                except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
                    item['reason'] = str(exc)
            if self._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Saved evidence changed during frame extraction. Refresh the review.')
            if not source_error and source_revision(source) != revision:
                raise ValueError('Original source changed during frame extraction. No frames may be used.')
            return {'run_id': data['id'], 'card_id': card['id'], 'fingerprint': view['fingerprint'], 'frames': frames}

    @staticmethod
    def _write_bytes(path, payload):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def export(self, data):
        with self.lock:
            run, _, view, _ = self._bound(data, card_required=False)
            report = {'schema_version': 1, 'kind': 'violationlens_local_review', **view,
                      'notice': NOTICE, 'external_reporting': False,
                      'supported_rules': ['helmet_absence', 'wrong_way'], 'deferred_rules': ['triple_riding']}
            payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False).encode('utf-8')
            name = 'violation-review-' + hashlib.sha256(payload).hexdigest()[:24] + '.json'
            path = self._path(run, name)
            if self._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Saved evidence changed before export. Refresh the review.')
            self._write_bytes(path, payload)
            if self._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Saved evidence changed during export. Refresh the review.')
            return {'run_id': data['id'], 'fingerprint': view['fingerprint'],
                    'url': self.files.url(path, download=True), 'filename': name}
