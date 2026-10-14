"""Canonical table uses synthetic evidence only."""
from copy import deepcopy

from traffic_ai.evidence_table import build_evidence_table
from tests.test_subject_models import subject, evidence


def test_local_no_helmet_is_not_an_allegation_and_scores_are_raw():
    subjects = [subject(), subject('V1', 'motorcycle')]
    local = evidence(subjects, [{'label': 'no_helmet', 'model_score': .8, 'box': [15, 15, 30, 30]}])
    annotations = {'subjects': subjects, 'local_evidence': local}
    before = deepcopy(annotations)
    table = build_evidence_table(annotations)
    assert table['summary']['suspected_issue_count'] == 0
    assert table['rows'][0]['assessment'] == 'cannot_assess'
    assert table['rows'][0]['vehicle_id'] is None
    assert any(s['model_score'] == .8 for s in table['rows'][0]['raw_model_scores'])
    assert annotations == before


def test_shared_group_is_counted_once_on_vehicle_and_review_separate():
    subjects = [subject('P1'), subject('P2'), subject('P3'), subject('V1', 'motorcycle')]
    group = ['P1', 'P2', 'P3', 'V1']
    results = []
    for s in subjects:
        results.append({'id': s['id'], 'associated_vehicle_id': 'V1' if s['kind'] == 'person' else None,
                        'status': 'suspected_issue', 'observations': [],
                        'suspected_issues': [{'kind': 'triple_riding_possible', 'text': 'Possible shared rider group',
                                              'related_subject_ids': group, 'evidence_ids': s['evidence_ids']}]})
    table = build_evidence_table({'subjects': subjects}, {'subject_results': results}, {'decision': 'accept'})
    assert table['summary']['suspected_issue_count'] == 1
    assert [r['subject_id'] for r in table['rows'] if r['suspected_issues']] == ['V1']
    assert all(r['review_status'] == 'accepted' for r in table['rows'])
    assert table['rows'][-1]['person_ids'] == ['P1', 'P2', 'P3']


def test_conflicting_helmet_evidence_never_promoted():
    subjects = [subject(), subject('V1', 'motorcycle')]
    local = evidence(subjects, [{'label': label, 'model_score': .9, 'box': [15, 15, 30, 30]}
                                for label in ('helmet', 'no_helmet')])
    model = {'subject_results': [{'id': 'P1', 'associated_vehicle_id': 'V1', 'status': 'suspected_issue',
                                  'observations': [], 'suspected_issues': [{'kind': 'helmet_possible',
                                  'text': 'unsupported', 'related_subject_ids': ['P1', 'V1'], 'evidence_ids': subjects[0]['evidence_ids']}]}]}
    table = build_evidence_table({'subjects': subjects, 'local_evidence': local}, model)
    assert table['summary']['suspected_issue_count'] == 0
    assert not table['rows'][0]['suspected_issues']


def test_legacy_without_annotations_has_no_invented_table():
    assert build_evidence_table(None) is None
    assert build_evidence_table({'subjects': []})['rows'] == []


def test_correction_original_text_tracks_selected_ambiguous_candidate():
    import numpy as np
    from traffic_ai.plate_models import build_evidence, correct_plate_evidence
    subjects = [subject('V1', 'car')]
    plates = build_evidence(np.zeros((200, 200, 3), dtype=np.uint8), subjects, 200, 200,
                            detections=[{'box': [20., 120., 40., 130.], 'detector_score': .9,
                                         'ocr_score': .9, 'text': 'FIRST123'},
                                        {'box': [50., 120., 70., 130.], 'detector_score': .8,
                                         'ocr_score': .4, 'text': 'SECOND12'}])
    assert plates['subjects'][0]['status'] == 'unclear'
    before = build_evidence_table({'subjects': subjects, 'plate_evidence': plates})
    assert before['rows'][0]['plate']['text'] is None
    selected = plates['subjects'][0]['candidates'][1]['plate_id']
    corrected = correct_plate_evidence(plates, subjects, 'V1', 'SECOND13', plate_id=selected)
    row = build_evidence_table({'subjects': subjects, 'plate_evidence': corrected})['rows'][0]
    assert row['plate']['original_text'] == 'SECOND12'
    assert row['plate']['text'] == 'SECOND13'
    assert row['plate']['status'] == 'unclear'
    assert row['plate']['candidates'] == plates['subjects'][0]['candidates']
    assert row['suspected_issues'] == []
