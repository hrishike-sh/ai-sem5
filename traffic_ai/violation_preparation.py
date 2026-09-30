"""Preview and bind the exact local images consented for hosted explanation."""
import base64
import hashlib
import io
import json
import math
import re

import numpy as np
from PIL import Image

from ivp.media import source_revision
from .violation_review import MAX_FILE, MAX_LINE, MAX_RECORDS, _json, _read, _box


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _preparation_id(manifest):
    return hashlib.sha256(_encoded(manifest)).hexdigest()[:24]


class EvidencePreparations:
    def __init__(self, review):
        self.review = review

    def _crop(self, run, card, stamp, shape):
        """Crop only with a nearby recorded subject and explicit source mapping."""
        if card['rule'] != 'helmet_absence' or card.get('track_id') is None:
            return None
        candidate, distance = None, .251
        with self.review._path(run, 'detections.jsonl').open('rb') as stream:
            for _ in range(MAX_RECORDS):
                line = stream.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE:
                    raise ValueError('Recorded evidence exceeds the frame-crop limit.')
                if not line.strip():
                    continue
                record = _json(line)
                current = record.get('timestamp')
                if type(current) not in (int, float) or not math.isfinite(current) or abs(current-stamp) >= distance:
                    continue
                candidate, distance = record, abs(current-stamp)
        if candidate is None or candidate.get('scene_id', 0) != card.get('scene_id', 0):
            return None
        record = candidate
        decision = next((d for d in record.get('decisions', []) if d.get('track_id') == card['track_id']
                         and d.get('rule') == card['rule']), None)
        if decision is None:
            return None
        height, width = shape[:2]
        if record.get('source_size') != [width, height]:
            return None
        transform = np.asarray(record.get('transform_to_source_pixel_centres'), dtype=float)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            return None
        heads = decision.get('evidence', {}).get('heads', [])
        boxes = [decision.get('box')] + (heads if isinstance(heads, list) else [])
        boxes = [box for box in boxes if _box(box)]
        if not boxes:
            return None
        points = np.array([[x, y, 1.] for box in boxes for x in (box[0], box[2]) for y in (box[1], box[3])])
        mapped = points @ transform.T
        if np.any(np.abs(mapped[:, 2]) < 1e-8):
            return None
        mapped = mapped[:, :2] / mapped[:, 2:]
        left, top = mapped.min(axis=0)
        right, bottom = mapped.max(axis=0)
        padding_x, padding_y = max(12., (right-left)*.3), max(12., (bottom-top)*.3)
        crop = (max(0, math.floor(left-padding_x)), max(0, math.floor(top-padding_y)),
                min(width, math.ceil(right+padding_x)), min(height, math.ceil(bottom+padding_y)))
        return crop if crop[2]-crop[0] >= 16 and crop[3]-crop[1] >= 16 else None

    def prepare(self, data):
        from .violation_images import prepare_evidence_image
        review = self.review
        with review.lock:
            run, request, view, card = review._bound(data)
            extracted = review.frames(data)
            manifest = {'version': 1, 'run_id': view['run_id'], 'card_id': card['id'],
                        'fingerprint': view['fingerprint'], 'source_revision': request['info'].get('source_revision'),
                        'images': [], 'unavailable': []}
            for frame in extracted['frames']:
                if not frame['available']:
                    manifest['unavailable'].append({key: frame.get(key) for key in ('id', 'role', 'timestamp', 'reason')})
                    continue
                path = review._path(run, 'violation-frame-' + frame['id'] + '.png')
                pixels = _read(path, MAX_FILE)
                if hashlib.sha256(pixels).hexdigest() != frame.get('sha256'):
                    raise ValueError('Decoded frame changed before preparation. Preview again.')
                with Image.open(io.BytesIO(pixels)) as decoded:
                    rgb = np.asarray(decoded.convert('RGB'))
                crop = self._crop(run, card, frame['timestamp'], rgb.shape)
                image = prepare_evidence_image(rgb, frame_id=frame['id'], timestamp=frame['timestamp'],
                                               role='subject_crop' if crop else 'context', crop_box=crop)
                payload = base64.b64decode(image['data_url'].split(',', 1)[1], validate=True)
                filename = 'violation-image-' + image['sha256'] + '.jpg'
                review._write_bytes(review._path(run, filename), payload)
                manifest['images'].append({'image': {k: v for k, v in image.items() if k != 'data_url'},
                                           'position': frame['role'], 'file': filename})
            if review._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Evidence changed while preparing images. Reload the review.')
            if manifest['images']:
                review._source(request)
            identity = _preparation_id(manifest)
            review._write_bytes(review._path(run, 'violation-preparation-' + identity + '.json'), _encoded(manifest))
            return self._public(run, manifest, identity, card)

    def _public(self, run, manifest, identity, card):
        frames = []
        for entry in manifest['images']:
            image = entry['image']
            frames.append({**image, 'image_role': image['role'], 'role': entry['position'],
                           'position': entry['position'], 'available': True,
                           'url': self.review.files.url(self.review._path(run, entry['file'])),
                           'reason': 'Exact prepared JPEG for the selected-frame request.'})
        frames.extend({**item, 'position': item['role'], 'available': False, 'url': None}
                      for item in manifest['unavailable'])
        return {'preparation_id': identity, 'run_id': manifest['run_id'], 'card_id': card['id'],
                'fingerprint': manifest['fingerprint'], 'mode': 'images_and_facts' if manifest['images'] else 'facts_only',
                'frames': frames, 'facts': card['facts'], 'missing_facts': card['missing_facts'],
                'limitations': ['These exact resized JPEGs and structured facts will be sent only after approval.',
                    'Subject crops are used only when recorded source coordinates are available; context frames may contain identities.',
                    'No guaranteed anonymization. Review every preview before sending.',
                    'Before/after frames provide temporal context; the subject may differ across a scene cut.',
                    'Visual observations are fallible and cannot establish speed, legal guilt or missing road authority.']}

    def load(self, data):
        """Recheck immutable preview bytes and source binding immediately before use."""
        review = self.review
        with review.lock:
            run, request, view, card = review._bound(data)
            identity = data.get('preparation_id')
            if not isinstance(identity, str) or not re.fullmatch(r'[a-f0-9]{24}', identity):
                raise ValueError('Preview the selected evidence images before sending them.')
            manifest = _json(_read(review._path(run, 'violation-preparation-' + identity + '.json'), 128*1024))
            if not isinstance(manifest, dict) or _preparation_id(manifest) != identity:
                raise ValueError('Prepared evidence changed. Preview it again.')
            if (manifest.get('run_id') != view['run_id'] or manifest.get('card_id') != card['id']
                    or manifest.get('fingerprint') != view['fingerprint']):
                raise ValueError('Prepared evidence belongs to another card or revision. Preview it again.')
            source, revision = review._source(request)
            if manifest.get('source_revision') != revision:
                raise ValueError('The original footage changed after preview.')
            entries = manifest.get('images')
            if not isinstance(entries, list) or not 1 <= len(entries) <= 3:
                raise ValueError('No prepared evidence images are available. Choose facts-only explicitly.')
            images = []
            for entry in entries:
                image = entry.get('image', {})
                digest = image.get('sha256')
                if (not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest)
                        or entry.get('file') != 'violation-image-' + digest + '.jpg'
                        or image.get('id') not in card['frame_ids']):
                    raise ValueError('Prepared image references are invalid. Preview again.')
                payload = _read(review._path(run, entry['file']), 256*1024)
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError('Prepared image bytes changed. Preview again.')
                images.append({**image, 'data_url': 'data:image/jpeg;base64,' + base64.b64encode(payload).decode('ascii')})
            if source_revision(source) != revision or review._sources(run) != view['provenance']['source_hashes']:
                raise ValueError('Source evidence changed before the request. Preview again.')
            return images
