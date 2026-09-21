import cv2
import numpy as np
import pytest
from ivp.auto_track import AutomaticTracker
from ivp.speed import SpeedEstimator


def fixture():
    rng=np.random.default_rng(52)
    a=rng.integers(45,230,(28,36,3),np.uint8)
    b=rng.integers(45,230,(28,36,3),np.uint8)
    a[:,:,0]=240; b[:,:,2]=240
    return a,b


def scene(patches, locations):
    rgb=np.full((180,320,3),25,np.uint8); mask=np.zeros((180,320),np.uint8)
    boxes=[]
    for patch,(x,y) in zip(patches,locations):
        h,w=patch.shape[:2];rgb[y:y+h,x:x+w]=patch;mask[y:y+h,x:x+w]=255
        boxes.append((x,y,w,h))
    return rgb,mask,boxes


def update(tracker,rgb,mask,boxes,t):
    return tracker.update(boxes,t,mask.shape,cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),mask,rgb)


def test_automatic_discovery_survives_merged_and_fragmented_foreground():
    tracker=AutomaticTracker('vertical',.5); patches=fixture();speed=SpeedEstimator()
    first_ids=None; events=[]
    for i in range(55):
        rgb,mask,boxes=scene(patches,[(20+3*i,40),(20+3*i,105)])
        # Simulate segmentation merging separate objects, then breaking into bits.
        if 15<=i<30:boxes=[(20+3*i,40,36,93)]
        elif 30<=i<40:boxes=[(20+3*i,40,14,28),(40+3*i,40,16,28),(20+3*i,105,36,28)]
        tracks,new=update(tracker,rgb,mask,boxes,i/30);events+=new
        for _,t in tracks:speed.update(t,i/30,mask.shape)
        if i==12:first_ids=[k for k,_ in tracks]
        if i>12:assert [k for k,_ in tracks]==first_ids
    assert len(first_ids)==2 and tracker.next_id==3
    assert tracker.merge_frames>=15
    assert len(events)==2
    assert all(t['speed']==pytest.approx(90,abs=2) for _,t in tracks)


def test_occlusion_recovery_retains_id_but_withholds_gap_speed():
    tracker=AutomaticTracker(); patches=fixture()[:1];speed=SpeedEstimator()
    identity=None
    for i in range(60):
        rgb,mask,boxes=scene(patches,[(20+2*i,70)])
        if 30<=i<35:
            rgb[:]=25;mask[:]=0;boxes=[]
        tracks,_=update(tracker,rgb,mask,boxes,i/30)
        for _,t in tracks:speed.update(t,i/30,mask.shape)
        if i==29:
            identity=tracks[0][0]; assert tracks[0][1]['speed'] is not None
        if 30<=i<35:assert not tracks
        if i==35:
            assert tracks[0][0]==identity
            assert tracks[0][1]['speed'] is None
    assert tracker.recovered==1
    assert tracks[0][0]==identity and tracks[0][1]['speed']==pytest.approx(60,abs=2)


def test_static_foreground_ghost_never_confirms_and_memory_is_bounded():
    tracker=AutomaticTracker();patches=fixture()
    for i in range(80):
        rgb,mask,boxes=scene(patches,[(30,40),(120,90)])
        visible,events=update(tracker,rgb,mask,boxes,i/30)
        assert not visible and not events
    assert tracker.confirmed==0 and len(tracker.tracks)<=2


def test_timestamp_discontinuity_does_not_bridge_identity():
    tracker=AutomaticTracker();patches=fixture()[:1]
    for i in range(20):
        rgb,mask,boxes=scene(patches,[(20+i,50)])
        tracks,_=update(tracker,rgb,mask,boxes,i/30)
    old=tracks[0][0]
    tracks,events=update(tracker,rgb,mask,boxes,4.)
    assert not tracks and not events and old not in tracker.tracks


def test_five_fps_feature_track_retains_identity_and_measures_speed():
    tracker=AutomaticTracker(frame_interval=.2);speed=SpeedEstimator(frame_interval=.2)
    patch=fixture()[:1];identity=None
    for i in range(35):
        rgb,mask,boxes=scene(patch,[(20+6*i,70)])
        tracks,_=update(tracker,rgb,mask,boxes,i/5)
        for key,track in tracks:
            speed.update(track,i/5,mask.shape)
            if identity is None:identity=key
            assert key==identity
    assert identity is not None and tracker.next_id==2
    assert tracks[0][1]['speed']==pytest.approx(30,abs=1)
