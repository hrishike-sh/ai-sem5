"""One row per source subject; candidate evidence never becomes a legal finding."""
from copy import deepcopy

from .subject_models import SCORE_NOTICE, validate_local_evidence


def build_evidence_table(annotations, model_result=None, review=None):
    if annotations is None:
        return None
    subjects = annotations['subjects']
    local = annotations.get('local_evidence')
    vehicle = annotations.get('vehicle_type_evidence')
    plates = annotations.get('plate_evidence')
    if local is not None:
        validate_local_evidence(local, subjects)
    if vehicle is not None:
        from .vehicle_types import validate_vehicle_evidence
        validate_vehicle_evidence(vehicle, subjects)
    if plates is not None:
        from .plate_models import validate_plate_evidence
        validate_plate_evidence(plates, subjects)
    by_local = {s['id']: s for s in (local or {}).get('subjects', [])}
    by_vehicle = {s['id']: s for s in (vehicle or {}).get('subjects', [])}
    by_plate = {s['id']: s for s in (plates or {}).get('subjects', [])}
    # The storage layer supplies a canonical, source-validated LLM result.
    results = {s['id']: s for s in (model_result or {}).get('subject_results', [])}
    source = {s['id']: s for s in subjects}
    decision = (review or {}).get('decision')
    review_status = {'accept': 'accepted', 'reject': 'rejected'}.get(decision, 'pending')
    rows, seen_issues = [], set()
    for subject in subjects:
        sid = subject['id']
        result = results.get(sid)
        head = by_local.get(sid, {}).get('helmet')
        typed = by_vehicle.get(sid)
        vehicle_id = sid if subject['kind'] != 'person' else (result or {}).get('associated_vehicle_id')
        if vehicle_id not in source or source[vehicle_id]['kind'] == 'person':
            vehicle_id = None
        scores = []
        if 'confidence' in subject:
            scores.append({'source': 'localization', 'label': subject['kind'], 'model_score': subject['confidence'], 'evidence_id': subject['evidence_ids'][-1]})
        for provider, detections in (('helmet', (head or {}).get('detections', [])), ('vehicle_type', (typed or {}).get('candidates', []))):
            scores.extend({'source': provider, **{key: d[key] for key in ('label', 'model_score', 'evidence_id')}} for d in detections)
        observations = [o['text'] for o in (result or {}).get('observations', [])]
        if head and subject['kind'] == 'person':
            label = {'unclear': 'needs review', 'low_confidence': 'low-confidence detection', 'not_observed': 'no matched head detection'}.get(head['status'], head['status'].replace('_', ' '))
            observations.append('Local helmet evidence: ' + label + '. ' + head['reason'])
        if typed and subject['kind'] != 'person':
            observations.append('Local vehicle type: ' + ('needs review' if typed['status'] == 'unclear' else typed['status'].replace('_', ' ')) + '.')
        plate = None
        p = by_plate.get(vehicle_id)
        if p and p['status'] != 'not_applicable':
            candidates = deepcopy(p['candidates'])
            correction = deepcopy(p.get('manual_correction'))
            original = candidates[0]['text'] if p['status'] == 'candidate' and len(candidates) == 1 else None
            if correction:
                original = next(d['text'] for d in candidates if d['plate_id'] == correction['plate_id'])
            plate = {'status': p['status'], 'text': correction['text'] if correction else original,
                     'original_text': original, 'corrected': correction is not None,
                     'candidates': candidates, 'manual_correction': correction}
            if sid == vehicle_id:
                for d in candidates:
                    scores.append({'source': 'plate_detector', 'label': 'plate', 'model_score': d['detector_score'], 'evidence_id': d['evidence_id']})
                    if d['ocr_score'] is not None:
                        scores.append({'source': 'plate_ocr', 'label': d['text'], 'model_score': d['ocr_score'], 'evidence_id': d['evidence_id']})
        issues = []
        for issue in (result or {}).get('suspected_issues', []):
            kind = issue['kind']
            if kind not in ('helmet_possible', 'triple_riding_possible'):
                continue
            if kind == 'helmet_possible' and (head and head['status'] != 'no_helmet'):
                continue
            if kind == 'triple_riding_possible' and subject['kind'] != 'motorcycle':
                observations.append('Shared rider-group suspicion is listed once on vehicle ' + str(vehicle_id) + '.')
                continue
            key = (kind, tuple(sorted(issue['related_subject_ids'])))
            if key not in seen_issues:
                seen_issues.add(key)
                issues.append(deepcopy(issue))
        assessable = result is not None and result['status'] != 'unclear'
        if head and head['status'] in ('unclear', 'not_observed', 'low_confidence', 'unavailable'):
            assessable = False
        rows.append({'row_id': sid, 'subject_id': sid, 'kind': subject['kind'],
                     'person_ids': [sid] if subject['kind'] == 'person' else sorted(r['id'] for r in results.values() if r.get('associated_vehicle_id') == sid),
                     'vehicle_id': vehicle_id, 'plate': plate, 'observations': observations,
                     'suspected_issues': issues, 'assessment': 'suspected_issue' if issues else 'observed' if assessable else 'cannot_assess',
                     'raw_model_scores': scores, 'evidence_ids': deepcopy(subject['evidence_ids']),
                     'review_status': review_status if result else 'pending',
                     'association': 'apparent_llm_association' if subject['kind'] == 'person' and vehicle_id else 'unconfirmed' if subject['kind'] == 'person' else 'vehicle_box',
                     'geometric_vehicle_candidates': deepcopy(by_local.get(sid, {}).get('vehicle_candidates', []))})
    return {'version': 1, 'rows': rows, 'score_notice': SCORE_NOTICE,
            'summary': {'subject_count': len(rows), 'suspected_issue_count': len(seen_issues),
                        'cannot_assess_count': sum(r['assessment'] == 'cannot_assess' for r in rows)},
            'limitations': ['IDs label boxes in this selected image, not people or vehicle identities.',
                            'Plate text is an OCR candidate or a reviewer correction; it cannot establish an offense or identify an owner.',
                            'Geometric overlap does not establish a rider relationship. LLM associations remain apparent.',
                            'One shared rider-group suspicion is counted once on its motorcycle row.',
                            'A single image cannot establish speed, red-light crossing, wrong-way travel, or parking legality.',
                            'Reviewer acceptance is a disposition, not a finding of fact.']}
