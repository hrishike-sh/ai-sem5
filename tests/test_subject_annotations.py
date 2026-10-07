from PIL import Image
import pytest
from traffic_ai.subject_annotations import correct_subjects, render_annotations, detect_subjects
from traffic_ai.contracts import Detection


def test_initial_spatial_ids_and_no_overlap_rider_assumption(monkeypatch):
    class Fake:
        def __init__(self, helmet):
            assert helmet is False
            self.manifest = {'road_users': {'sha256': 'pinned'}}
        def detect(self, pixels, track):
            assert track is False
            return [Detection('motorcycle', .8, (1, 2, 20, 30)), Detection('person', .9, (2, 1, 15, 25)),
                    Detection('traffic_light', .8, (1, 1, 5, 8))]
    monkeypatch.setattr('traffic_ai.perception.Perception', Fake)
    rows, metadata = detect_subjects(Image.new('RGB', (40, 40)))
    subjects, counters = correct_subjects(rows, 40, 40, detected=True)
    assert [(s['id'], s['kind']) for s in subjects] == [('P1', 'person'), ('V1', 'motorcycle')]
    assert all('role' not in s and 'associated_vehicle_id' not in s for s in subjects)
    assert metadata['local_only'] is True and counters == {'P': 1, 'V': 1}


def test_correction_retains_ids_and_deleted_ids_never_reused():
    subjects, counters = correct_subjects([{'kind':'person','bbox':[1,1,10,20]},
                                           {'kind':'person','bbox':[12,1,20,20]}], 40, 40)
    revised, counters = correct_subjects([{'id':'P2','kind':'person','bbox':[11,2,21,21]}], 40, 40, subjects, counters)
    revised, counters = correct_subjects(revised + [{'kind':'person','bbox':[1,1,5,5]}], 40, 40, revised, counters)
    assert [r['id'] for r in revised] == ['P2', 'P3']
    with pytest.raises(ValueError):
        correct_subjects([{'id':'P1','kind':'person','bbox':[1,1,5,5]}], 40, 40, revised, counters)


def test_manual_changes_drop_detector_confidence_and_origin():
    subjects, counters = correct_subjects([{'kind':'person','bbox':[1,1,10,20],'confidence':.9}], 40, 40, detected=True)
    corrected, _ = correct_subjects([{**subjects[0], 'bbox':[2,1,10,20]}], 40, 40, subjects, counters)
    assert corrected[0]['origin'] == 'manual' and 'confidence' not in corrected[0]


def test_render_bounded_context_and_linked_crops():
    subjects, _ = correct_subjects([{'kind':'car','bbox':[x,0,x+3,10]} for x in range(12)], 40, 40)
    bound, images = render_annotations(Image.new('RGB',(40,40)), subjects, 'test', 0.)
    assert len(images) == 13 and images[0]['role'] == 'context'
    assert all(row['evidence_ids'] == ['test-context','test-crop-'+row['id']] for row in bound)
    assert all('subject_ids' not in image for image in images)
    with pytest.raises(ValueError):
        correct_subjects([{'kind':'car','bbox':[0,0,3,10]}] * 13, 40, 40)


@pytest.mark.parametrize('box', [[0,0,41,10], [1,1,1,5], [0.,0,5,5], [True,0,5,5]])
def test_invalid_box(box):
    with pytest.raises(ValueError):
        correct_subjects([{'kind':'person','bbox':box}],40,40)
