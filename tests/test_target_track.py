import cv2
import numpy as np
import pytest
from ivp.target_track import SelectedTarget
from ivp.speed import SpeedEstimator


def test_selected_texture_tracks_known_motion_and_speed_without_detector():
    rng=np.random.default_rng(3)
    patch=rng.integers(40,240,(50,50,3),np.uint8)
    tracker=SelectedTarget([.2,.25,.4,.5]);speed=SpeedEstimator()
    for i in range(31):
        frame=np.full((200,250,3),20,np.uint8)
        frame[50:100,50+i*2:100+i*2]=patch
        track=tracker.step(frame,i/30)
        assert track is not None
        speed.update(track,i/30,frame.shape[:2])
    assert track['box'][0]==pytest.approx(110,abs=.5)
    assert track['speed']==pytest.approx(60,abs=1)
    assert tracker.summary()['tracked_frames']==31


def test_target_stops_on_occlusion_instead_of_reacquiring_background():
    rng=np.random.default_rng(8);frame=rng.integers(0,255,(200,250,3),np.uint8)
    tracker=SelectedTarget([.2,.25,.4,.5]);assert tracker.step(frame,0)
    assert tracker.step(np.zeros_like(frame),.04) is None
    assert tracker.step(frame,.08) is None


@pytest.mark.parametrize('rect',[[0,0,0,0],[0,0,1],[-1,0,1,1],[0,0,float('nan'),1]])
def test_invalid_selected_target(rect):
    with pytest.raises(ValueError):SelectedTarget(rect)
