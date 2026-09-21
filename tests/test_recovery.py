"""Adverse association cases for repeated, observed weak-box recovery."""
from types import SimpleNamespace
import numpy as np
import pytest
from traffic_ai.runtime import configure_cache

configure_cache()
pytest.importorskip('ultralytics')
from ultralytics.engine.results import Boxes
from traffic_ai.recovery import RepeatedWeakRecovery
from traffic_ai.robust_recovery import RobustObservedRecovery


@pytest.fixture(params=[RepeatedWeakRecovery,RobustObservedRecovery],ids=['current','robust-history-trial'])
def engine(request):
    return request.param(SimpleNamespace(track_high_thresh=.35,track_low_thresh=.1,
        new_track_thresh=.4,track_buffer=24,match_thresh=.75,fuse_score=True))


def update(engine, rows):
    return engine.update(Boxes(np.asarray(rows,np.float32).reshape(-1,6),(300,400)))


def warm(engine, boxes=None, count=36):
    boxes=boxes or [[20,30,60,90]]
    for _ in range(count):rows=update(engine,[box+[.8,2] for box in boxes])
    return rows


def test_two_observed_weak_boxes_recover_recent_id_without_inflating_confidence(engine):
    key=int(warm(engine)[0,4])
    for _ in range(2):assert update(engine,[]).size==0
    assert update(engine,[[20,30,60,90,.23,2]]).size==0
    rows=update(engine,[[21,30,61,90,.24,2]])
    assert len(rows)==1 and int(rows[0,4])==key
    assert rows[0,5]==pytest.approx(.24) and int(rows[0,7])==0
    assert len(engine.recovery_events)==1


def test_single_weak_observation_or_nonconsecutive_observations_cannot_recover(engine):
    warm(engine);update(engine,[])
    assert update(engine,[[20,30,60,90,.25,2]]).size==0
    assert update(engine,[]).size==0
    assert update(engine,[[20,30,60,90,.25,2]]).size==0
    assert not engine.recovery_events


@pytest.mark.parametrize('kind',['two_lost','one_active'])
def test_nearby_competing_identity_blocks_weak_recovery(engine,kind):
    warm(engine,[[20,30,60,90],[26,30,66,90]])
    active=[[26,30,66,90,.8,2]] if kind=='one_active' else []
    for _ in range(2):update(engine,active)
    for _ in range(3):
        rows=update(engine,active+[[23,30,63,90,.25,2]])
        assert len(rows)<=len(active)
    assert not engine.recovery_events


@pytest.mark.parametrize('scenario',['stale','newborn','distant','too_weak'])
def test_unsupported_weak_recovery_is_rejected(engine,scenario):
    warm(engine,count=5 if scenario=='newborn' else 36)
    for _ in range(14 if scenario=='stale' else 2):update(engine,[])
    box=[200,30,240,90] if scenario=='distant' else [20,30,60,90]
    score=.15 if scenario=='too_weak' else .25
    for _ in range(3):assert update(engine,[box+[score,2]]).size==0
    assert not engine.recovery_events


def test_weak_detections_never_create_new_ids_and_reset_clears_pending_recovery(engine):
    for _ in range(10):assert update(engine,[[20,30,60,90,.25,2]]).size==0
    warm(engine);update(engine,[]);update(engine,[[20,30,60,90,.25,2]])
    assert engine.pending_recovery
    engine.reset()
    assert not engine.pending_recovery
    assert update(engine,[[20,30,60,90,.25,2]]).size==0
