"""Local image annotations. IDs describe boxes, never identities or rider roles."""
import math
import re

import numpy as np
from PIL import ImageDraw, ImageFont

from .violation_images import prepare_evidence_image

KINDS = {'person', 'motorcycle', 'bicycle', 'car', 'bus', 'truck'}
MAX_SUBJECTS = 12
NOTICE = ('Boxes and within-image IDs are editable proposals. Person/vehicle overlap does not establish '
          'a rider relationship. Context is a prepared JPEG; padded crops use the bound original frame '
          'and are limited to 768 pixels per edge. No identity, legal finding, fine, or external reporting is established.')


def detect_subjects(image):
    from .perception import Perception
    detector = Perception(helmet=False)
    width, height = image.size
    rows = []
    for detection in detector.detect(np.asarray(image), track=False):
        if detection.label not in KINDS:
            continue
        x1, y1, x2, y2 = detection.box
        box = [max(0, math.floor(x1)), max(0, math.floor(y1)),
               min(width, math.ceil(x2)), min(height, math.ceil(y2))]
        if box[0] < box[2] and box[1] < box[3]:
            rows.append({'kind': detection.label, 'bbox': box, 'origin': 'detector',
                         'confidence': detection.confidence})
    rows.sort(key=lambda row: (-row['confidence'], row['bbox'][1], row['bbox'][0]))
    return rows[:MAX_SUBJECTS], {'name': 'yolo11s', 'local_only': True,
                                'weights': detector.manifest['road_users'],
                                'truncated': len(rows) > MAX_SUBJECTS}


def correct_subjects(rows, width, height, previous=(), counters=None, *, detected=False):
    if not isinstance(rows, list) or len(rows) > MAX_SUBJECTS:
        raise ValueError('Supply at most twelve subject boxes.')
    counters = dict(counters or {'P': 0, 'V': 0})
    if set(counters) != {'P', 'V'} or any(type(v) is not int or not 0 <= v <= 1000000 for v in counters.values()):
        raise ValueError('Subject ID counters are invalid.')
    old = {row['id']: row for row in previous}
    if len(old) != len(previous):
        raise ValueError('Saved subject IDs are duplicated.')
    for identifier, row in old.items():
        if (not isinstance(identifier, str) or not re.fullmatch(r'[PV][1-9][0-9]*', identifier)
                or int(identifier[1:]) > counters[identifier[0]] or row.get('origin') not in ('detector', 'manual')):
            raise ValueError('Saved subject identity or provenance is invalid.')
        if 'confidence' in row and (type(row['confidence']) not in (int, float)
                                   or not math.isfinite(row['confidence']) or not 0 <= row['confidence'] <= 1):
            raise ValueError('Saved detector confidence is invalid.')
    result, seen = [], set()
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) - {'id', 'kind', 'bbox', 'origin', 'confidence', 'evidence_ids'}:
            raise ValueError('Subject contains unsupported fields.')
        kind, box, identifier = raw.get('kind'), raw.get('bbox'), raw.get('id')
        if kind not in KINDS or not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Subject requires a supported kind and four integer pixel coordinates.')
        if not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height:
            raise ValueError('Subject box lies outside the prepared image.')
        prefix = 'P' if kind == 'person' else 'V'
        if identifier is not None:
            if not isinstance(identifier, str) or not re.fullmatch(prefix + r'[1-9][0-9]*', identifier) or identifier not in old or identifier in seen:
                raise ValueError('Existing subject IDs must be unique and retain their person/vehicle category.')
            seen.add(identifier)
        row = {'kind': kind, 'bbox': list(box), 'origin': 'detector' if detected else 'manual'}
        # Unchanged detector proposals keep their provenance; supplied confidence cannot invent certainty.
        prior = old.get(identifier)
        if prior and prior['kind'] == kind and prior['bbox'] == list(box):
            row['origin'] = prior['origin']
            if 'confidence' in prior:
                row['confidence'] = prior['confidence']
        elif detected and 'confidence' in raw:
            confidence = raw['confidence']
            if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Detector confidence must be finite and between zero and one.')
            row['confidence'] = confidence
        row['id'] = identifier
        result.append(row)
    if not previous:
        result.sort(key=lambda row: (row['bbox'][1], row['bbox'][0], row['kind'], row['bbox']))
    for row in result:
        if row['id'] is None:
            prefix = 'P' if row['kind'] == 'person' else 'V'
            counters[prefix] += 1
            row['id'] = prefix + str(counters[prefix])
    return result, counters


def _font(size=16):
    for name in ('DejaVuSans.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _label(draw, text, left, top, bottom, width, height, color, font):
    bounds = draw.textbbox((0, 0), text, font=font)
    label_width, label_height = bounds[2] - bounds[0] + 6, bounds[3] - bounds[1] + 6
    x = max(0, min(left, width - label_width))
    y = top - label_height if top >= label_height else (bottom if bottom + label_height <= height else max(0, bottom - label_height))
    draw.rectangle((x, y, min(width - 1, x + label_width), min(height - 1, y + label_height)), fill='#101827')
    draw.text((x + 3 - bounds[0], y + 3 - bounds[1]), text, fill=color, font=font)


def render_annotations(image, subjects, review_id, timestamp, full_image=None, compact_labels=False):
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for subject in subjects:
        left, top, right, bottom = subject['bbox']
        color = '#ffcf40' if subject['kind'] == 'person' else '#40e5ff'
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=3)
        label = subject['id'] if compact_labels else subject['id'] + ' ' + subject['kind']
        _label(draw, label, left, top, bottom, image.width, image.height, color, _font())
    context_id = review_id + '-context'
    images = [prepare_evidence_image(canvas, frame_id=context_id, timestamp=timestamp)]
    bound = []
    for subject in subjects:
        left, top, right, bottom = subject['bbox']
        pad = max(8, round(max(right - left, bottom - top) * .2))
        crop = [max(0, left - pad), max(0, top - pad), min(image.width, right + pad), min(image.height, bottom + pad)]
        crop_id = review_id + '-crop-' + subject['id']
        crop_source = canvas
        if full_image is not None:
            sx, sy = full_image.width / image.width, full_image.height / image.height
            crop = [math.floor(crop[0] * sx), math.floor(crop[1] * sy),
                    min(full_image.width, math.ceil(crop[2] * sx)), min(full_image.height, math.ceil(crop[3] * sy))]
            crop_source = full_image.copy()
            detail = ImageDraw.Draw(crop_source)
            detail.rectangle((math.floor(left*sx), math.floor(top*sy), math.ceil(right*sx)-1,
                              math.ceil(bottom*sy)-1), outline='#ffcf40', width=max(2, round(sx*2)))
            font = _font(max(16, math.ceil(16 * max(crop[2]-crop[0], crop[3]-crop[1]) / 768)))
            # Draw inside the padded crop margin, scaled to stay legible after encoding.
            bounds = detail.textbbox((0, 0), subject['id'], font=font)
            label_height = bounds[3] - bounds[1] + 6
            detail.rectangle((crop[0], crop[3]-label_height, min(crop[2], crop[0]+bounds[2]+6), crop[3]-1), fill='#101827')
            detail.text((crop[0]+3, crop[3]-label_height+3-bounds[1]), subject['id'], fill='#ffcf40', font=font)
        images.append(prepare_evidence_image(crop_source, frame_id=crop_id, timestamp=timestamp,
                                             role='subject_crop', crop_box=crop))
        bound.append({**subject, 'evidence_ids': [context_id, crop_id]})
    return bound, images
