"""Association-only trajectories: no detector, video fixture, or annotation seeds."""
from types import SimpleNamespace
import numpy as np
import pytest

from traffic_ai.runtime import configure_cache
configure_cache()
pytest.importorskip('ultralytics')
from ultralytics.engine.results import Boxes
from traffic_ai.association import GroundAnchorTracker


def tracker():
    return GroundAnchorTracker(SimpleNamespace(track_high_thresh=.35, track_low_thresh=.1,
        new_track_thresh=.4, track_buffer=24, match_thresh=.75, fuse_score=True))


def update(engine, rows):
    return engine.update(Boxes(np.asarray(rows,np.float32).reshape(-1,6),(400,600)))


def nearest_id(rows, box):
    from scripts.check_city_identities import iou
    score, key = max((iou(row[:4],np.array(box)), int(row[4])) for row in rows)
    assert score>.5
    return key


@pytest.mark.parametrize('reverse_order', [False,True])
def test_close_opposing_vehicles_keep_ids_with_brief_occlusion(reverse_order):
    engine=tracker();ids=None
    for i in range(60):
        a=[40+3*i,100,80+3*i,160]
        b=[250-3*i,112,290-3*i,172]
        detections=[a+[.8,2],b+[.75,2]]
        if 31<=i<34:detections=detections[1:]
        if reverse_order or i%2:detections=detections[::-1]
        rows=update(engine,detections)
        if i==10:ids=(nearest_id(rows,a),nearest_id(rows,b))
        if i==59:assert (nearest_id(rows,a),nearest_id(rows,b))==ids


def test_top_occlusion_changes_height_without_stealing_lower_vehicle_id():
    engine=tracker();identity=None
    for i in range(50):
        y=250-i*2
        rear=[230,y,290,y+90]
        ahead=[215,y-62,270,y+10]
        rows=update(engine,[rear+[.8,2],ahead+[.8,2]])
        if i==10:identity=nearest_id(rows,rear)
    # A merged proposal and the two individual boxes coexist; the lower box
    # has moderate confidence but the same road-contact point as its track.
    for i in range(50,70):
        y=250-i*2
        rear=[230,y,290,y+90];ahead=[215,y-62,270,y+10]
        grouped=[212,y-62,292,y+90]
        rows=update(engine,[grouped+[.46,2],rear+[.39,2],ahead+[.7,2]])
        assert nearest_id(rows,rear)==identity


def test_no_overlap_does_not_reconnect_distant_vehicle():
    engine=tracker()
    for _ in range(12):rows=update(engine,[[20,30,60,90,.8,2]])
    first=int(rows[0,4])
    for _ in range(5):update(engine,[])
    for _ in range(10):rows=update(engine,[[240,30,280,90,.8,2]])
    assert int(rows[0,4])!=first


def test_reset_drops_previous_scene_tracks():
    engine=tracker()
    for _ in range(10):update(engine,[[20,30,60,90,.8,2]])
    engine.reset()
    assert not engine.tracked_stracks and not engine.lost_stracks
    assert update(engine,[]).size==0
