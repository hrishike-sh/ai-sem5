"""Local, source-bound image review storage and optional local box proposals."""
import base64
import hashlib
import json
import math
import io
from pathlib import Path
import re
import threading
from datetime import datetime, timezone
from uuid import uuid4
from PIL import Image

from ivp.media import load_image, preview_frame, source_revision, write_json
from .violation_images import image_descriptors, image_fingerprint, prepare_evidence_image, validate_evidence_images


MAX_JSON = 512 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
NOTICE = ('Direct review of the selected image by a fallible model; no detector inference, '
          'legal finding, fine, or external reporting. Reviewer decisions do not establish facts.')


def _encoded(value):
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('Review data must be bounded finite JSON.') from exc
    if len(raw) > MAX_JSON:
        raise ValueError('Review data exceeds the size limit.')
    return raw


def _read(path, maximum=MAX_JSON):
    with path.open('rb') as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError('Saved review exceeds its size limit.')
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError('Saved review is invalid.') from exc
    _encoded(value)
    return value


def _sha(path):
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Source exceeds the two-gigabyte review limit.')
    digest, size = hashlib.sha256(), 0
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            size += len(chunk)
            if size > MAX_SOURCE_BYTES:
                raise ValueError('Source exceeds the review size limit.')
            digest.update(chunk)
    return digest.hexdigest()


class DirectReview:
    def __init__(self, root, files):
        self.root = Path(root).resolve()
        self.files = files
        self.lock = threading.RLock()

    def _folder(self, review_id):
        if not isinstance(review_id, str) or not re.fullmatch(r'direct-[a-f0-9]{32}', review_id):
            raise ValueError('Invalid direct review ID.')
        folder = self.root / 'direct-reviews' / review_id
        if folder.resolve() != folder or not folder.resolve().is_relative_to(self.root):
            raise ValueError('Direct reviews must remain inside this project.')
        return folder

    @staticmethod
    def _path(folder, name):
        path = folder / name
        if path.resolve().parent != folder:
            raise ValueError('Review files must remain inside their review folder.')
        return path

    def _source(self, source):
        if not isinstance(source, (str, Path)):
            raise ValueError('A registered local source is required.')
        path = Path(source).resolve()
        allowed = [(self.root / name).resolve() for name in ('uploads', 'demo')]
        if not any(base.is_relative_to(self.root) and path.is_relative_to(base) for base in allowed) or not path.is_file():
            raise ValueError('Source must be an existing upload or demo inside this project.')
        return path

    def _check_source(self, manifest):
        source = manifest['source']
        path = self._source(source['path'])
        if source_revision(path) != source['revision'] or _sha(path) != source['sha256']:
            raise ValueError('Original source changed after preview. Prepare a new review.')
        if source_revision(path) != source['revision']:
            raise ValueError('Original source changed during verification.')
        return path

    def _manifest(self, review_id):
        folder = self._folder(review_id)
        manifest = _read(self._path(folder, 'manifest.json'))
        if not isinstance(manifest, dict) or manifest.get('review_id') != review_id or manifest.get('version') != 1:
            raise ValueError('Saved direct review identity is invalid.')
        fingerprint = hashlib.sha256(_encoded(manifest)).hexdigest()
        return folder, manifest, fingerprint

    def _bound(self, data):
        if not isinstance(data, dict):
            raise ValueError('A direct review request is required.')
        folder, manifest, fingerprint = self._manifest(data.get('review_id'))
        if data.get('fingerprint') != fingerprint:
            raise ValueError('Direct review changed. Reload or prepare it again.')
        return folder, manifest, fingerprint

    def _state(self, folder, fingerprint):
        path = self._path(folder, 'state.json')
        state = _read(path) if path.exists() else {}
        if not isinstance(state, dict) or state.get('fingerprint') != fingerprint:
            return {'fingerprint': fingerprint, 'model_result': None, 'review': {'decision': None, 'note': ''}}
        review = state.get('review')
        if (not isinstance(review, dict) or review.get('decision') not in (None, 'accept', 'reject')
                or not isinstance(review.get('note'), str) or len(review['note']) > 4000):
            raise ValueError('Saved reviewer disposition is invalid.')
        return state

    def prepare(self, source, info, timestamp=0., synthetic=None):
        with self.lock:
            source = self._source(source)
            if not isinstance(info, dict) or info.get('kind') not in ('image', 'video'):
                raise ValueError('Registered image or video metadata is required.')
            if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError('Select a finite nonnegative timestamp.')
            if info['kind'] == 'image' and timestamp != 0:
                raise ValueError('A still image has only timestamp zero.')
            duration = info.get('duration')
            if info['kind'] == 'video' and (type(duration) not in (int, float) or not math.isfinite(duration)
                                          or not 0 <= timestamp < duration):
                raise ValueError('Select a timestamp inside this video.')
            revision = source_revision(source)
            if info.get('source_revision') and info['source_revision'] != revision:
                raise ValueError('Registered source changed. Reopen it before preparation.')
            digest = _sha(source)
            rgb, actual = (load_image(source), 0.) if info['kind'] == 'image' else preview_frame(source, info, timestamp)
            if not isinstance(actual, (int, float)) or not math.isfinite(actual) or actual < 0:
                raise ValueError('Decoder returned an invalid frame timestamp.')
            if info['kind'] == 'video' and actual >= duration:
                raise ValueError('Decoded frame timestamp is outside this video.')
            review_id = 'direct-' + uuid4().hex
            image = prepare_evidence_image(rgb, frame_id=review_id + '-frame-0', timestamp=actual)
            manifest = {'version': 1, 'review_id': review_id,
                        'created_at': datetime.now(timezone.utc).isoformat(),
                        'source': {'path': str(source), 'name': source.name, 'kind': info['kind'],
                                   'revision': revision, 'sha256': digest},
                        'requested_timestamp': timestamp, 'synthetic': True if synthetic is True else None,
                        'frames': [{key: value for key, value in image.items() if key != 'data_url'}]}
            self._check_source(manifest)
            folder = self._folder(review_id)
            folder.mkdir(parents=True, exist_ok=False)
            payload = base64.b64decode(image['data_url'].split(',', 1)[1], validate=True)
            self._path(folder, 'frame-0.jpg').write_bytes(payload)
            write_json(self._path(folder, 'manifest.json'), manifest)
            return self.view(review_id)

    def _original_images(self, folder, manifest):
        frames = manifest.get('frames')
        if not isinstance(frames, list) or len(frames) != 1:
            raise ValueError('Prepared frame manifest is invalid.')
        image = frames[0]
        if not isinstance(image, dict) or image.get('id') != manifest['review_id'] + '-frame-0':
            raise ValueError('Prepared frame identity is invalid.')
        with self._path(folder, 'frame-0.jpg').open('rb') as stream:
            payload = stream.read(256 * 1024 + 1)
        if len(payload) > 256 * 1024 or hashlib.sha256(payload).hexdigest() != image.get('sha256'):
            raise ValueError('Prepared frame bytes changed. Prepare a new review.')
        images = [{**image, 'data_url': 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')}]
        validate_evidence_images(images, [image['id']])
        return images

    def _images(self, folder, manifest):
        original = self._original_images(folder, manifest)
        annotations = manifest.get('annotations')
        if annotations is None:
            return original
        from .subject_annotations import correct_subjects, render_annotations
        try:
            subjects = annotations['subjects']
            if (annotations['image_width'], annotations['image_height']) != (original[0]['width'], original[0]['height']):
                raise ValueError('Annotation dimensions changed.')
            crop_source = annotations['crop_source']
            if (not isinstance(crop_source, dict) or set(crop_source) != {'width', 'height', 'timestamp', 'sha256'}
                    or any(type(crop_source[key]) is not int or crop_source[key] < 1 for key in ('width', 'height'))
                    or crop_source['width'] * crop_source['height'] > 40_000_000
                    or crop_source['timestamp'] != original[0]['timestamp']
                    or crop_source['sha256'] != manifest['source']['sha256']):
                raise ValueError('Annotation crop source binding is invalid.')
            corrected, _ = correct_subjects(subjects, original[0]['width'], original[0]['height'], subjects,
                                             manifest['subject_counters'])
            raw = base64.b64decode(original[0]['data_url'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as source:
                expected_subjects, expected = render_annotations(source.convert('RGB'), corrected,
                                                                  manifest['review_id'], original[0]['timestamp'],
                                                                  compact_labels=annotations.get('compact_labels', False))
            descriptors = annotations['images']
            if (subjects != expected_subjects or not isinstance(descriptors, list) or len(descriptors) != len(expected)
                    or descriptors[0] != image_descriptors(expected)[0]):
                raise ValueError('Annotation evidence does not match its source boxes.')
            images = []
            for item, reference in zip(descriptors, expected):
                if (item.get('id'), item.get('role'), item.get('timestamp')) != (reference['id'], reference['role'], reference['timestamp']):
                    raise ValueError('Annotation crop identity changed.')
                if not isinstance(item.get('sha256'), str) or not re.fullmatch(r'[a-f0-9]{64}', item['sha256']):
                    raise ValueError('Annotation image hash is invalid.')
                path = self._path(folder, 'annotation-' + item['sha256'] + '.jpg')
                with path.open('rb') as stream:
                    payload = stream.read(256 * 1024 + 1)
                if len(payload) > 256 * 1024 or hashlib.sha256(payload).hexdigest() != item['sha256']:
                    raise ValueError('Annotation image bytes changed.')
                image = {**item, 'data_url': 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')}
                validate_evidence_images([image], [reference['id']])
                images.append(image)
            return images
        except (KeyError, TypeError) as exc:
            raise ValueError('Saved annotations are invalid.') from exc

    def annotate(self, data, subjects=None):
        """Produce local proposals or apply explicit corrections to the original saved JPEG."""
        from .subject_annotations import correct_subjects, detect_subjects, render_annotations, NOTICE as BOX_NOTICE
        with self.lock:
            folder, manifest, _ = self._bound(data)
            self._check_source(manifest)
            self._images(folder, manifest)
            original = self._original_images(folder, manifest)[0]
            raw = base64.b64decode(original['data_url'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert('RGB')
            prior = manifest.get('annotations', {})
            if subjects is None:
                if prior:
                    raise ValueError('Detector proposals already exist; edit the boxes or prepare a new review.')
                try:
                    rows, detector = detect_subjects(image)
                except (RuntimeError, OSError, ImportError):
                    rows, detector = [], {'name': 'yolo11s', 'local_only': True, 'state': 'unavailable',
                        'reason': 'Subject detector unavailable. Add boxes manually; other local specialists can still inspect this image.'}
            else:
                rows, detector = subjects, prior.get('detector', {'name': None, 'local_only': True})
            corrected, counters = correct_subjects(rows, image.width, image.height,
                                                    prior.get('subjects', []), manifest.get('subject_counters'),
                                                    detected=subjects is None)
            source_path = self._check_source(manifest)
            if manifest['source']['kind'] == 'image':
                full_rgb = load_image(source_path)
            else:
                # Use the decoder's selected timestamp and freshly probed source metadata.
                from ivp.media import probe_video
                full_rgb, actual = preview_frame(source_path, probe_video(source_path), original['timestamp'])
                if abs(actual - original['timestamp']) > .001:
                    raise ValueError('Source decoder no longer returns the selected frame.')
            full_image = Image.fromarray(full_rgb) if not isinstance(full_rgb, Image.Image) else full_rgb
            source_check = prepare_evidence_image(full_image.convert('RGB'), frame_id=original['id'], timestamp=original['timestamp'])
            if source_check['sha256'] != original['sha256']:
                raise ValueError('Decoded original frame no longer matches the prepared image.')
            bound, images = render_annotations(image, corrected, manifest['review_id'], original['timestamp'], full_image.convert('RGB'), compact_labels=True)
            manifest['annotations'] = {'subjects': bound, 'image_width': image.width, 'image_height': image.height,
                                       'compact_labels': True,
                                       'detector': detector, 'notice': BOX_NOTICE, 'images': image_descriptors(images),
                                       'crop_source': {'width': full_image.width, 'height': full_image.height,
                                                       'timestamp': original['timestamp'], 'sha256': manifest['source']['sha256']}}
            manifest['subject_counters'] = counters
            # A revision changes even for an identical correction, invalidating in-flight requests.
            manifest['annotation_revision'] = manifest.get('annotation_revision', 0) + 1
            self._check_source(manifest)
            for item in images:
                self._path(folder, 'annotation-' + item['sha256'] + '.jpg').write_bytes(
                    base64.b64decode(item['data_url'].split(',', 1)[1]))
            write_json(self._path(folder, 'manifest.json'), manifest)
            fingerprint = hashlib.sha256(_encoded(manifest)).hexdigest()
            write_json(self._path(folder, 'state.json'), {'fingerprint': fingerprint, 'model_result': None,
                                                         'review': {'decision': None, 'note': ''}})
            retained = {'annotation-' + item['sha256'] + '.jpg' for item in images}
            for path in folder.glob('annotation-*.jpg'):
                if path.name not in retained and path.resolve().parent == folder:
                    path.unlink()
            return self.view(manifest['review_id'])

    def run_models(self, data):
        """Run the pinned specialist against the unchanged original selected frame."""
        from .subject_models import build_evidence, validate_local_evidence
        return self._run_specialist(data, 'local_evidence', build_evidence, validate_local_evidence)

    def run_vehicle_types(self, data):
        from .vehicle_types import build_evidence, validate_vehicle_evidence
        return self._run_specialist(data, 'vehicle_type_evidence', build_evidence, validate_vehicle_evidence)

    def run_plates(self, data):
        from .plate_models import build_evidence, validate_plate_evidence
        return self._run_specialist(data, 'plate_evidence', build_evidence, validate_plate_evidence)

    def correct_plate(self, data):
        """Keep raw OCR immutable and invalidate downstream review after a correction."""
        from .plate_models import correct_plate_evidence
        with self.lock:
            folder, manifest, _ = self._bound(data)
            self._check_source(manifest)
            self._images(folder, manifest)
            annotations = manifest.get('annotations')
            if not annotations or annotations.get('plate_evidence') is None:
                raise ValueError('Run the local plate check before correcting its reading.')
            annotations['plate_evidence'] = correct_plate_evidence(
                annotations['plate_evidence'], annotations['subjects'],
                data.get('vehicle_id'), data.get('text'), plate_id=data.get('plate_id'))
            manifest['local_evidence_revision'] = manifest.get('local_evidence_revision', 0) + 1
            self._check_source(manifest)
            _encoded(manifest)
            write_json(self._path(folder, 'manifest.json'), manifest)
            fingerprint = hashlib.sha256(_encoded(manifest)).hexdigest()
            write_json(self._path(folder, 'state.json'), {'fingerprint': fingerprint, 'model_result': None,
                                                         'review': {'decision': None, 'note': ''}})
            return self.view(manifest['review_id'])

    def _run_specialist(self, data, field, builder, validator):
        with self.lock:
            folder, manifest, _ = self._bound(data)
            source_path = self._check_source(manifest)
            self._images(folder, manifest)
            annotations = manifest.get('annotations')
            if annotations is None:
                raise ValueError('Prepare numbered subject boxes before running local models.')
            original = manifest['frames'][0]
            if manifest['source']['kind'] == 'image':
                rgb = load_image(source_path)
            else:
                from ivp.media import probe_video
                rgb, actual = preview_frame(source_path, probe_video(source_path), original['timestamp'])
                if abs(actual - original['timestamp']) > .001:
                    raise ValueError('Source decoder no longer returns the selected frame.')
            full_image = Image.fromarray(rgb) if not isinstance(rgb, Image.Image) else rgb
            full_image = full_image.convert('RGB')
            check = prepare_evidence_image(full_image, frame_id=original['id'], timestamp=original['timestamp'])
            if (check['sha256'] != original['sha256'] or
                    (full_image.width, full_image.height) != (annotations['crop_source']['width'], annotations['crop_source']['height'])):
                raise ValueError('Decoded original frame no longer matches the prepared image.')
            import numpy as np
            evidence = builder(np.asarray(full_image), annotations['subjects'],
                                      annotations['image_width'], annotations['image_height'],
                                      truncated=annotations['detector'].get('truncated', False))
            validator(evidence, annotations['subjects'])
            self._check_source(manifest)
            self._images(folder, manifest)
            annotations[field] = evidence
            manifest['local_evidence_revision'] = manifest.get('local_evidence_revision', 0) + 1
            # Commit only after all inference and source checks succeed; invalidate downstream decisions.
            _encoded(manifest)
            write_json(self._path(folder, 'manifest.json'), manifest)
            fingerprint = hashlib.sha256(_encoded(manifest)).hexdigest()
            write_json(self._path(folder, 'state.json'), {'fingerprint': fingerprint, 'model_result': None,
                                                         'review': {'decision': None, 'note': ''}})
            return self.view(manifest['review_id'])

    def view(self, review_id):
        with self.lock:
            folder, manifest, fingerprint = self._manifest(review_id)
            images = self._images(folder, manifest)
            try:
                self._check_source(manifest)
                readiness = {'available': True, 'reason': 'Source content and metadata match the prepared review.'}
            except (OSError, ValueError):
                readiness = {'available': False, 'reason': 'Original media changed or is unavailable. Historical review remains available; prepare new evidence before sending.'}
            state = self._state(folder, fingerprint)
            model_result = (self._normalise_result(state['model_result'], manifest, fingerprint, images)
                            if state.get('model_result') is not None else None)
            annotations = manifest.get('annotations')
            local_evidence = annotations.get('local_evidence') if annotations else None
            vehicle_type_evidence = annotations.get('vehicle_type_evidence') if annotations else None
            if local_evidence is not None:
                from .subject_models import validate_local_evidence
                validate_local_evidence(local_evidence, annotations['subjects'])
            original = manifest['frames'][0]
            from .feature_report import build_feature_report
            feature_report = build_feature_report(annotations)
            from .evidence_table import build_evidence_table
            evidence_table = build_evidence_table(annotations, model_result, state['review'])
            return {'review_id': review_id, 'fingerprint': fingerprint,
                    'source': {key: manifest['source'][key] for key in ('name', 'kind')},
                    'frames': [{**{k: v for k, v in image.items() if k != 'data_url'}, 'position': 'selected',
                                'subject_ids': [row['id'] for row in annotations['subjects'] if image['id'] in row['evidence_ids']] if annotations else [],
                                'available': True, 'url': self.files.url(self._path(folder, 'annotation-' + image['sha256'] + '.jpg' if annotations else 'frame-0.jpg'))} for image in images],
                    'original_frame': {**original, 'url': self.files.url(self._path(folder, 'frame-0.jpg'))},
                    'annotations': annotations, 'local_evidence': local_evidence,
                    'vehicle_type_evidence': vehicle_type_evidence, 'feature_report': feature_report,
                    'plate_evidence': annotations.get('plate_evidence') if annotations else None,
                    'evidence_table': evidence_table,
                    'synthetic': manifest.get('synthetic'), 'model_result': model_result,
                    'review': state['review'], 'source_readiness': readiness,
                    'notice': annotations['notice'] if annotations else NOTICE}

    def load(self, data):
        with self.lock:
            folder, manifest, _ = self._bound(data)
            self._check_source(manifest)
            images = self._images(folder, manifest)
            self._check_source(manifest)
            return {**self.view(manifest['review_id']), 'images': images,
                    'subjects': manifest.get('annotations', {}).get('subjects', [])}

    @staticmethod
    def _normalise_result(result, manifest, fingerprint, images):
        if manifest.get('annotations') is not None:
            from .identified_vision import canonicalize_result
            if not isinstance(result, dict) or result.get('review_id') != manifest['review_id'] or result.get('fingerprint') != fingerprint:
                raise ValueError('Model result does not match this exact annotated review.')
            _encoded(result)
            canonical = canonicalize_result({key: value for key, value in result.items() if key not in ('review_id', 'fingerprint')},
                                            images, manifest['annotations']['subjects'], manifest['annotations'].get('local_evidence'),
                                            manifest['annotations'].get('vehicle_type_evidence'))
            return {**canonical, 'review_id': manifest['review_id'], 'fingerprint': fingerprint}
        from .direct_vision import MODEL, PROVIDER, LIMITATIONS, OBSERVATIONS, ISSUES, UNCERTAINTIES, validate_direct_selection
        if (not isinstance(result, dict) or result.get('review_id') != manifest['review_id']
                or result.get('fingerprint') != fingerprint or result.get('state') != 'reviewed'
                or result.get('provider') != PROVIDER or result.get('model') != MODEL
                or result.get('mode') != 'direct_images' or result.get('evidence') != image_descriptors(images)
                or result.get('image_fingerprint') != image_fingerprint(images)):
            raise ValueError('Model result does not match this exact prepared evidence.')
        _encoded(result)
        try:
            selection = {'image_fingerprint': result['image_fingerprint'], **{
                field: [{key: row[key] for key in ('kind', 'evidence_ids', 'uncertainties')} for row in result[field]]
                for field in ('observations', 'suspected_issues')}}
        except (KeyError, TypeError) as exc:
            raise ValueError('Model result selection is invalid.') from exc
        selected = validate_direct_selection(selection, images)
        def render(rows, labels):
            return [{**row, 'text': labels[row['kind']],
                     'uncertainty_text': [UNCERTAINTIES[key] for key in row['uncertainties']]} for row in rows]
        return {'review_id': manifest['review_id'], 'fingerprint': fingerprint, 'state': 'reviewed',
                'provider': PROVIDER, 'model': MODEL, 'mode': 'direct_images',
                'image_fingerprint': image_fingerprint(images), 'evidence': image_descriptors(images),
                'observations': render(selected['observations'], OBSERVATIONS),
                'suspected_issues': render(selected['suspected_issues'], ISSUES), 'limitations': list(LIMITATIONS)}

    def save_result(self, data, result):
        """Caller validates the model schema; storage enforces evidence identity."""
        with self.lock:
            folder, manifest, fingerprint = self._bound(data)
            loaded = self.load(data)
            result = self._normalise_result(result, manifest, fingerprint, loaded['images'])
            self._check_source(manifest)
            self._images(folder, manifest)
            state = self._state(folder, fingerprint)
            state.update(model_result=result, review={'decision': None, 'note': ''})
            write_json(self._path(folder, 'state.json'), state)
            return self.view(manifest['review_id'])

    def decide(self, data):
        with self.lock:
            folder, manifest, fingerprint = self._bound(data)
            action, note = data.get('decision'), data.get('note', '')
            if action not in ('accept', 'reject', 'reset') or not isinstance(note, str) or len(note) > 4000:
                raise ValueError('Choose accept, reject or reset with a note of at most 4000 characters.')
            state = self._state(folder, fingerprint)
            if state.get('model_result') is None:
                raise ValueError('No model result is available to review.')
            state['review'] = {'decision': None, 'note': ''} if action == 'reset' else {'decision': action, 'note': note}
            write_json(self._path(folder, 'state.json'), state)
            return self.view(manifest['review_id'])

    def export(self, data):
        with self.lock:
            folder, manifest, _ = self._bound(data)
            report = {'kind': 'direct_image_review', 'external_reporting': False, **self.view(manifest['review_id'])}
            name = 'direct-review-' + hashlib.sha256(_encoded(report)).hexdigest()[:24] + '.json'
            path = self._path(folder, name)
            write_json(path, report)
            return {'review_id': manifest['review_id'], 'filename': name, 'url': self.files.url(path, download=True)}

    def recent(self, limit=30):
        if type(limit) is not int or not 1 <= limit <= 30:
            raise ValueError('Direct review limit must be between 1 and 30.')
        folder = self.root / 'direct-reviews'
        if not folder.exists():
            return []
        if folder.resolve() != folder:
            raise ValueError('Direct reviews must remain inside this project.')
        result = []
        paths = [path for path in folder.glob('direct-*') if path.resolve() == path and path.is_dir()]
        for path in sorted(paths, key=lambda path: path.stat().st_mtime_ns, reverse=True):
            try:
                _, manifest, fingerprint = self._manifest(path.name)
                result.append({'review_id': path.name, 'fingerprint': fingerprint,
                               'source': {key: manifest['source'][key] for key in ('name', 'kind')},
                               'synthetic': manifest.get('synthetic')})
            except (OSError, ValueError, KeyError):
                continue
            if len(result) >= limit:
                break
        return result
