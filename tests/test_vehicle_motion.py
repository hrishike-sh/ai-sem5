from types import SimpleNamespace
import numpy as np
from traffic_ai.vehicle_motion import VehicleTracker,ambiguous_overlaps
from ivp.speed import SpeedEstimator
from ivp.timing import FrameTiming


class Association:
    def __init__(self):self.rows=[];self.resets=0
    def update(self,*args):return self.rows
    def reset(self):self.resets+=1


def tracker():
    t=VehicleTracker('vertical',.5)
    t.model=SimpleNamespace(names={2:'car'},predict=lambda *a,**k:[SimpleNamespace(boxes=SimpleNamespace(cpu=lambda:SimpleNamespace(numpy=lambda:None)))])
    t.association=Association()
    t.device='cpu'
    return t


def step(t,i,rows):
    t.association.rows=rows
    rgb=np.zeros((200,300,3),np.uint8)
    return t.update([],i/30,(200,300),rgb[:,:,0],rgb[:,:,0],rgb)


def test_vehicle_ids_speed_and_count_survive_short_detection_gap():
    t=tracker();estimator=SpeedEstimator();events=[]
    for i in range(70):
        rows=[] if 31<=i<=34 else [[20+i*2,60,50+i*2,85,1,.8,2]]
        visible,new=step(t,i,rows);events+=new
        for _,track in visible:estimator.update(track,i/30,(200,300))
        if 31<=i<=34:assert not visible
        if i==35:assert visible[0][0]==1 and visible[0][1]['speed'] is None
    assert t.confirmed==1 and t.recovered==1 and t.association.resets==0
    assert visible[0][1]['speed']==60
    assert len(events)==1


def test_vehicle_roi_and_clipped_frame_withhold_speed():
    t=tracker();t.rectangle=[.1,.2,.9,.9];estimator=SpeedEstimator()
    for i in range(30):
        visible,_=step(t,i,[[0,50,35,100,1,.8,2],[80,50,120,100,2,.8,2]])
    assert [key for key,_ in visible]==[2]
    t.rectangle=[0,0,1,1]
    for i in range(30,65):
        visible,_=step(t,i,[[0,50,35,100,1,.8,2]])
        for _,track in visible:estimator.update(track,i/30,(200,300))
    assert visible[0][1]['speed'] is None
    assert visible[0][1]['speed_status']=='vehicle clipped by frame'


def test_long_gap_resets_identity_namespace_and_never_reuses_id():
    t=tracker()
    for i in range(15):visible,_=step(t,i,[[40,50,70,80,1,.8,2]])
    assert visible[0][0]==1
    for i in range(60,75):visible,_=step(t,i,[[40,50,70,80,1,.8,2]])
    assert visible[0][0]==1000001 and t.association.resets==1


def test_explicit_scene_reset_clears_underlying_association():
    t=tracker()
    for i in range(15):step(t,i,[[40,50,70,80,1,.8,2]])
    t.reset()
    assert not t.tracks and t.association.resets==1
    for i in range(15,30):visible,_=step(t,i,[[40,50,70,80,1,.8,2]])
    assert visible[0][0]==1000001


def test_camera_warp_border_does_not_publish_speed_for_partial_vehicle():
    t=tracker();estimator=SpeedEstimator()
    t.valid_mask=np.full((200,300),255,np.uint8);t.valid_mask[:,:41]=0
    for i in range(25):
        visible,_=step(t,i,[[42,50,70,80,1,.8,2]])
        for _,track in visible:estimator.update(track,i/30,(200,300))
    assert visible[0][1]['speed_status']=='vehicle touches a camera-warp border'
    assert visible[0][1]['speed'] is None


def test_only_nested_boxes_with_shared_contact_are_ambiguous():
    rows=[[50,30,90,120,1,.7,2],[55,75,85,120,2,.5,2],[120,80,150,120,3,.8,2]]
    assert ambiguous_overlaps(rows)=={0,1}
    rows[1][3]=100
    assert ambiguous_overlaps(rows)==set()


def test_ambiguous_hypotheses_withhold_speed_and_do_not_bridge_crossing():
    t=tracker();estimator=SpeedEstimator()
    events=[]
    for i in range(55):
        x=20+i*3
        rows=[[x,60,x+30,100,1,.8,2]]
        if i>=30:rows.append([x-3,30,x+33,100,2,.7,2])
        visible,new=step(t,i,rows);events+=new
        for _,track in visible:estimator.update(track,i/30,(200,300))
        if i==29:assert visible[0][1]['speed'] is not None
        if i>=30:
            assert all(track['speed'] is None and track['tracking_state']=='ambiguous overlap' for _,track in visible)
    assert events==[] and t.ambiguous_frames==25


def test_five_fps_does_not_look_like_repeated_tracking_gaps():
    t=tracker();t.timing=FrameTiming(.2);estimator=SpeedEstimator(frame_interval=.2)
    rgb=np.zeros((200,300,3),np.uint8)
    for i in range(30):
        x=20+6*i;t.association.rows=[[x,60,x+30,85,1,.8,2]]
        visible,_=t.update([],i/5,(200,300),rgb[:,:,0],rgb[:,:,0],rgb)
        for _,track in visible:estimator.update(track,i/5,(200,300))
    assert t.association.resets==0 and t.recovered==0
    assert visible[0][0]==1 and visible[0][1]['speed']==30


def test_weak_recovery_always_restarts_speed_and_does_not_infer_crossing():
    t=tracker();estimator=SpeedEstimator()
    for i in range(30):
        visible,_=step(t,i,[[80+i,60,100+i,85,1,.8,2]])
        for _,track in visible:estimator.update(track,i/30,(200,300))
    assert visible[0][1]['speed']==30
    step(t,30,[])
    t.association.recovery_events=[{'track_id':1}]
    visible,events=step(t,31,[[170,60,190,85,1,.24,2]])
    estimator.update(visible[0][1],31/30,(200,300))
    assert visible[0][0]==1 and not events
    assert visible[0][1]['speed'] is None
    assert visible[0][1]['tracking_state']=='recovered from weak observations'
    assert t.recovered==1
