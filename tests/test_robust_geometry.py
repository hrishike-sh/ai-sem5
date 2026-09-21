"""Observed-history fallback trial: partial boxes must not invent motion."""
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_recovery import update,warm
from traffic_ai.robust_recovery import RobustObservedRecovery


def tracker():
    return RobustObservedRecovery(SimpleNamespace(track_high_thresh=.35,track_low_thresh=.1,
        new_track_thresh=.4,track_buffer=24,match_thresh=.75,fuse_score=True))


@pytest.mark.parametrize('velocity',[(0,0),(2,1)])
def test_truncated_bottom_does_not_force_new_identity_when_full_box_returns(velocity):
    engine=tracker();vx,vy=velocity
    def box(i,height=60):return [20+i*vx,30+i*vy,60+i*vx,30+i*vy+height]
    for i in range(30):rows=update(engine,[box(i)+[.8,2]])
    key=int(rows[0,4])
    for i in [30,31]:update(engine,[box(i,44)+[.8,2]])
    assert update(engine,[]).size==0
    assert update(engine,[box(33)+[.25,2]]).size==0
    assert update(engine,[box(34)+[.25,2]]).size==0
    rows=update(engine,[box(35)+[.25,2]])
    assert len(rows)==1 and int(rows[0,4])==key
    assert rows[0,5]==pytest.approx(.25)
    assert engine.recovery_events[0]['geometry']=='robust observed history'


def test_history_is_bounded_and_low_rate_old_positions_are_not_extrapolated():
    engine=tracker();warm(engine,count=100)
    assert len(engine.observations)==1
    assert len(next(iter(engine.observations.values())))==12
    engine.max_frames_lost=4  # 5 fps: eight samples do not fit a short window.
    track=engine.tracked_stracks[0]
    assert np.array_equal(engine.robust_box(track),track.xyxy)
    engine.reset()
    assert not engine.observations and not engine.robust_pending


def test_uneven_frame_times_use_real_elapsed_time_for_recovery_geometry():
    engine=tracker()
    def observe(stamp,height=60,score=.8,missing=False):
        engine.observation_timestamp=stamp
        box=[20+100*stamp,30,60+100*stamp,30+height]
        return update(engine,[] if missing else [box+[score,2]])
    for i in range(30):rows=observe(i/30)
    key=int(rows[0,4])
    observe(1.,44);observe(31/30,44)
    engine.max_frames_lost=8  # New observations now arrive 0.1 seconds apart.
    observe(31/30+.1,missing=True)
    assert observe(31/30+.2,score=.25).size==0
    assert observe(31/30+.3,score=.25).size==0
    rows=observe(31/30+.4,score=.25)
    assert len(rows)==1 and int(rows[0,4])==key
    assert engine.recovery_events[0]['geometry']=='robust observed history'
