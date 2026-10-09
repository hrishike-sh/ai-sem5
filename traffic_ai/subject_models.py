"""Pinned, local-only helmet evidence; geometric matches are never rider identities."""
import math
import threading
import numpy as np
from .local_models import HELMET_REPO, HELMET_REVISION, PINS, verified_local_models

SCORE_NOTICE = 'Raw detector scores are not calibrated probabilities or measured accuracy.'
THRESHOLDS = {'inference_floor': .1, 'candidate_score': .5, 'head_upper_fraction': .5,
              'head_containment': .8, 'notice': 'Heuristic thresholds; not validated accuracy or legal criteria.'}
LIMITATIONS = ['Single selected image only; no tracking or temporal vehicle count.',
              'Missing head detections mean not observed, never no helmet.',
              'Geometric motorcycle candidates do not establish rider roles or violations.',
              'Full-context inference resized to 960 pixels can miss small or occluded heads.']
_CLASSES = {'0': 'With Helmet', '1': 'Without Helmet'}
_LOCK = threading.RLock()
_CACHE = {}


def _model_info():
    return {'name': HELMET_REPO, 'revision': HELMET_REVISION, 'sha256': PINS['helmet'][0],
            'classes': dict(_CLASSES), 'preprocessing': 'Bound original RGB converted to contiguous BGR; CPU, imgsz=960, max_det=100; boxes in original-source pixels.'}


def predict_heads(rgb):
    """Verify pinned files each run; cache only the specialist, serialize its predictor."""
    from .runtime import configure_cache
    with _LOCK:
        configure_cache()
        paths, _ = verified_local_models()
        key = (str(paths['helmet']), PINS['helmet'][0])
        model = _CACHE.get(key)
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(paths['helmet']))
            names = {str(k): str(v) for k, v in model.names.items()}
            if names != _CLASSES:
                raise RuntimeError('Pinned helmet model class mapping is unsupported.')
            _CACHE.clear()
            _CACHE[key] = model
        result = model.predict(source=np.ascontiguousarray(rgb[:, :, ::-1]), device='cpu',
                               imgsz=960, conf=.1, max_det=100, verbose=False)[0]
        if result.boxes is None:
            return []
        return [{'label': {0: 'helmet', 1: 'no_helmet'}[int(cls)], 'model_score': float(score),
                 'box': list(map(float, box))}
                for cls, score, box in zip(result.boxes.cls.cpu().tolist(),
                                           result.boxes.conf.cpu().tolist(), result.boxes.xyxy.cpu().tolist())]


def _status(detections):
    if not detections:
        return 'not_observed', 'No unambiguous head detection matched this person.'
    if _head_cluster_count(detections) > 1:
        return 'unclear', 'Multiple distinct head candidates within one person box; correct/split boxes.'
    labels = {d['label'] for d in detections}
    if len(labels) > 1:
        return 'unclear', 'Conflicting helmet and no-helmet detections; inspect the image.'
    if max(d['model_score'] for d in detections) < THRESHOLDS['candidate_score']:
        return 'low_confidence', 'Matched detections remain below the heuristic candidate threshold.'
    return next(iter(labels)), 'Detector candidate above the heuristic threshold; requires visual review.'


def _intersection(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def _head_cluster_count(detections):
    # Heuristic duplicate grouping: IoU >= .3 OR >= .8 of the smaller box.
    # Complete-link grouping prevents a broad/bridging prediction from joining
    # two spatially distinct heads. Scores order anchors deterministically.
    def same_head(a, b):
        overlap = _intersection(a, b)
        area_a = (a[2]-a[0])*(a[3]-a[1])
        area_b = (b[2]-b[0])*(b[3]-b[1])
        return overlap/(area_a+area_b-overlap) >= .3 or overlap/min(area_a,area_b) >= .8
    clusters = []
    for detection in sorted(detections, key=lambda d: (-d['model_score'], d['box'])):
        box = detection['box']
        for cluster in clusters:
            if all(same_head(box, other) for other in cluster):
                cluster.append(box)
                break
        else:
            clusters.append([box])
    return len(clusters)


def vehicle_counts(subjects, truncated=False):
    counts = {kind: {'total': 0, 'detector': 0, 'manual': 0}
              for kind in ('motorcycle', 'bicycle', 'car', 'bus', 'truck')}
    for row in subjects:
        if row['kind'] in counts:
            counts[row['kind']]['total'] += 1
            counts[row['kind']][row['origin']] += 1
    return {'by_kind': counts, 'total': sum(row['total'] for row in counts.values()),
            'scope': 'current_image_numbered_boxes', 'subject_cap': 12, 'truncated': bool(truncated)}


def _candidates(subject, subjects):
    if subject['kind'] != 'person':
        return []
    p = subject['bbox']
    return [{'id': v['id'], 'relationship': 'geometric_unconfirmed'} for v in subjects
            if v['kind'] == 'motorcycle' and _intersection(p, v['bbox']) > 0
            and (v['bbox'][1]+v['bbox'][3])/2 > p[1]+.25*(p[3]-p[1])]


def build_evidence(rgb, subjects, image_width, image_height, *, truncated=False, detections=None):
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Helmet evidence requires an original uint8 RGB frame.')
    result = {'version': 1, 'state': 'complete', 'reason': 'Pinned local helmet model completed.',
              'model': _model_info(), 'score_notice': SCORE_NOTICE, 'thresholds': dict(THRESHOLDS),
              'subjects': [], 'unassigned_detections': [], 'vehicle_counts': vehicle_counts(subjects, truncated),
              'limitations': list(LIMITATIONS)}
    for s in subjects:
        result['subjects'].append({'id': s['id'], 'helmet': {'status': 'not_observed' if s['kind']=='person' else 'not_applicable',
                                  'detections': [], 'reason': 'Helmet evidence applies only to person boxes.'},
                                  'vehicle_candidates': _candidates(s, subjects)})
    if not any(s['kind'] == 'person' for s in subjects):
        result['reason'] = 'No numbered person boxes; vehicle counts only. Helmet inference was not run.'
        return result
    try:
        heads = predict_heads(rgb) if detections is None else detections
    except (RuntimeError, OSError, ImportError, ValueError):
        result['state'] = 'unavailable'
        result['reason'] = 'Pinned local helmet inference is unavailable. Check local weights and installed runtime; no download was attempted.'
        for row, s in zip(result['subjects'], subjects):
            if s['kind'] == 'person':
                row['helmet'].update(status='unavailable', reason=result['reason'])
        return result
    sx, sy = rgb.shape[1]/image_width, rgb.shape[0]/image_height
    people = [(i, [s['bbox'][0]*sx, s['bbox'][1]*sy, s['bbox'][2]*sx, s['bbox'][3]*sy])
              for i, s in enumerate(subjects) if s['kind']=='person']
    for raw in heads[:100]:
        label, score, box = raw['label'], raw['model_score'], raw['box']
        if label not in ('helmet', 'no_helmet') or type(score) not in (int,float) or not math.isfinite(score) or not .1 <= score <= 1:
            continue
        if len(box)!=4 or any(type(x) not in (int,float) or not math.isfinite(x) for x in box):
            continue
        if not (0 <= box[0] < box[2] <= rgb.shape[1] and 0 <= box[1] < box[3] <= rgb.shape[0]):
            continue
        detection = {'label': label, 'model_score': float(score), 'box': list(map(float,box))}
        area = (box[2]-box[0])*(box[3]-box[1])
        matches = []
        for i, p in people:
            upper = [p[0], p[1], p[2], p[1]+.5*(p[3]-p[1])]
            if (_intersection(box, upper)/area >= .8 and box[2]-box[0] <= .8*(p[2]-p[0])
                    and box[3]-box[1] <= .5*(p[3]-p[1])):
                matches.append(i)
        if len(matches) == 1:
            i = matches[0]
            detection['evidence_id'] = subjects[i]['evidence_ids'][-1]
            result['subjects'][i]['helmet']['detections'].append(detection)
        else:
            detection['reason'] = 'ambiguous_person_overlap' if matches else 'no_plausible_person_head_match'
            result['unassigned_detections'].append(detection)
    for row, s in zip(result['subjects'], subjects):
        if s['kind'] == 'person':
            status, reason = _status(row['helmet']['detections'])
            row['helmet'].update(status=status, reason=reason)
    return result


def validate_local_evidence(evidence, subjects):
    """Validate saved evidence without loading weights or re-running inference."""
    def require(test):
        if not test:
            raise ValueError('Saved local model evidence is invalid or does not match its subjects.')
    require(isinstance(evidence, dict) and set(evidence)=={'version','state','reason','model','score_notice','thresholds','subjects','unassigned_detections','vehicle_counts','limitations'})
    require(evidence['version']==1 and evidence['state'] in ('complete','unavailable'))
    require(evidence['model']==_model_info() and evidence['score_notice']==SCORE_NOTICE and evidence['thresholds']==THRESHOLDS and evidence['limitations']==LIMITATIONS)
    require(evidence['reason'] in ('Pinned local helmet model completed.', 'No numbered person boxes; vehicle counts only. Helmet inference was not run.', 'Pinned local helmet inference is unavailable. Check local weights and installed runtime; no download was attempted.'))
    require((evidence['state']=='unavailable') == (evidence['reason']=='Pinned local helmet inference is unavailable. Check local weights and installed runtime; no download was attempted.'))
    require(isinstance(evidence['subjects'],list) and len(evidence['subjects'])==len(subjects))
    counts=evidence['vehicle_counts']
    require(isinstance(counts,dict) and type(counts.get('truncated')) is bool and counts==vehicle_counts(subjects,counts['truncated']))
    total=0
    def check_detection(d, assigned, subject=None):
        nonlocal total
        total+=1
        require(isinstance(d,dict) and set(d)=={'label','model_score','box', 'evidence_id' if assigned else 'reason'})
        require(d['label'] in ('helmet','no_helmet') and type(d['model_score']) in (int,float) and math.isfinite(d['model_score']) and .1<=d['model_score']<=1)
        b=d['box']
        require(isinstance(b,list) and len(b)==4 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100000 for v in b) and b[0]<b[2] and b[1]<b[3])
        require(d['evidence_id']==subject['evidence_ids'][-1] if assigned else d['reason'] in ('ambiguous_person_overlap','no_plausible_person_head_match'))
    for row,s in zip(evidence['subjects'],subjects):
        require(isinstance(row,dict) and set(row)=={'id','helmet','vehicle_candidates'} and row['id']==s['id'] and row['vehicle_candidates']==_candidates(s,subjects))
        h=row['helmet']
        require(isinstance(h,dict) and set(h)=={'status','detections','reason'} and isinstance(h['detections'],list))
        for d in h['detections']:
            check_detection(d,True,s)
        if s['kind']!='person':
            require(h=={'status':'not_applicable','detections':[],'reason':'Helmet evidence applies only to person boxes.'})
        elif evidence['state']=='unavailable':
            require(h=={'status':'unavailable','detections':[],'reason':evidence['reason']})
        else:
            status,reason=_status(h['detections'])
            require(h['status']==status and h['reason']==reason)
    require(isinstance(evidence['unassigned_detections'],list))
    for d in evidence['unassigned_detections']:
        check_detection(d,False)
    require(total<=100 and (evidence['state']!='unavailable' or total==0))
    return evidence
