"""Derived feature results for one prepared image; never a global legal verdict."""
from copy import deepcopy

from .subject_models import SCORE_NOTICE, validate_local_evidence, vehicle_counts


def _feature(state, summary, data, *, errors=(), limitations=()):
    return {'state': state, 'summary': summary, 'data': data,
            'errors': list(errors), 'limitations': list(limitations)}


def build_feature_report(annotations, vehicle_type_evidence=None):
    """Use already validated annotations; do not infer, load weights, or mutate storage."""
    if annotations is None:
        return None
    subjects = annotations['subjects']
    evidence = annotations.get('local_evidence')
    if vehicle_type_evidence is None:
        vehicle_type_evidence = annotations.get('vehicle_type_evidence')
    if evidence is not None:
        validate_local_evidence(evidence, subjects)
    truncated = annotations.get('detector', {}).get('truncated', False)
    counts = vehicle_counts(subjects, truncated)
    detector_count = sum(s['origin'] == 'detector' for s in subjects)
    localization = _feature(
        'complete' if subjects else 'needs_input',
        'Numbered subject boxes are available for this image.' if subjects else 'No numbered subjects are available; inspect the image and add missing boxes.',
        {'subjects': [{key: deepcopy(row[key]) for key in ('id', 'kind', 'bbox', 'origin', 'confidence') if key in row}
                      for row in subjects], 'total': len(subjects),
         'detector_count': detector_count, 'manual_count': len(subjects) - detector_count},
        limitations=['Boxes are editable proposals, not confirmed identities or rider relationships.',
                     'At most twelve numbered boxes are included; missing or occluded subjects may be omitted.', SCORE_NOTICE])
    localization_unavailable = not subjects and annotations.get('detector', {}).get('state') == 'unavailable'
    if localization_unavailable:
        localization.update(state='unavailable', summary='Local subject detection is unavailable; add subject boxes manually.',
                            errors=['Local subject detection is unavailable.'])
    helmet_status_counts = {}
    if evidence is None:
        helmet = _feature('pending', 'The local helmet check has not run for these boxes.', {'evidence': None},
                          limitations=['No helmet conclusion is available until the specialist check completes.'])
    else:
        for row in evidence['subjects']:
            status = row['helmet']['status']
            helmet_status_counts[status] = helmet_status_counts.get(status, 0) + 1
        helmet = _feature(evidence['state'], evidence['reason'], {'evidence': deepcopy(evidence)},
                          errors=[evidence['reason']] if evidence['state'] == 'unavailable' else [],
                          limitations=[SCORE_NOTICE, *evidence['limitations']])
    features = {'localization': localization, 'helmet': helmet,
                'vehicle_counts': _feature('complete',
                    'Counts describe the current numbered vehicle boxes in this image.', counts,
                    limitations=['Counts include detector proposals and manually added boxes, listed separately.',
                                 'These are not tracked traffic volumes or a count of all vehicles in the scene.',
                                 'The twelve-box limit can omit vehicles.'])}
    if localization_unavailable:
        features['vehicle_counts'].update(state='needs_input',
            summary='Vehicle counts require numbered boxes; subject detection was unavailable.',
            errors=['No numbered vehicle evidence is available.'])
    if vehicle_type_evidence is None:
        features['vehicle_types'] = _feature('pending', 'The local vehicle-type check has not run for these boxes.',
                                             {'evidence': None}, limitations=['No specialized vehicle-type prediction is available yet.'])
    else:
        from .vehicle_types import validate_vehicle_evidence
        validate_vehicle_evidence(vehicle_type_evidence, subjects)
        features['vehicle_types'] = _feature(vehicle_type_evidence['state'], vehicle_type_evidence['reason'],
                                             {'evidence': deepcopy(vehicle_type_evidence)},
                                             errors=[vehicle_type_evidence['reason']] if vehicle_type_evidence['state'] == 'unavailable' else [],
                                             limitations=[vehicle_type_evidence['score_notice'], *vehicle_type_evidence['limitations']])
    return {'version': 1, 'scope': 'selected_image', 'features': features,
            'summary': {'subject_count': len(subjects), 'vehicle_count': counts['total'],
                        'helmet_status_counts': helmet_status_counts,
                        **{name + '_features': [key for key, value in features.items() if value['state'] == state]
                           for name, state in (('completed', 'complete'), ('unavailable', 'unavailable'),
                                               ('pending', 'pending'), ('needs_input', 'needs_input'))}},
            'limitations': ['Feature states are independent; an unavailable feature does not invalidate completed features.',
                            'This report contains image evidence and model outputs, not a legal verdict.',
                            'A single image cannot establish speed, wrong-way travel, signal crossing, or parking legality.']}
