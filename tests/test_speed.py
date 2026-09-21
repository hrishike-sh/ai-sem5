import cv2
import numpy as np
import pytest

from ivp.motion import RegionTracker
from ivp.speed import RoadCalibration, SpeedEstimator

CONFIG = {'points':[[.3,.2],[.7,.2],[.95,.9],[.05,.9]], 'width_m':10, 'length_m':40}


@pytest.mark.parametrize('calibrated', [False, True])
@pytest.mark.parametrize('reason', ['blocked', 'gap', 'duplicate', 'backward', 'size', 'position'])
def test_withheld_speed_clears_the_previous_estimate_window(calibrated, reason):
    config = {'points': [[0,0],[1,0],[1,1],[0,1]], 'width_m': 10, 'length_m': 10}
    estimator = SpeedEstimator(config if calibrated else None)
    track = {}
    for index in range(35):
        track['box'] = (20+index,20,10,10)
        estimator.update(track,index/30,(100,100))
    assert track['speed'] == pytest.approx(10.8 if calibrated else 30)
    assert track['speed_window_seconds'] > .5
    stamp = 35/30
    track['box'] = (55,20,10,10)
    if reason == 'blocked': track['speed_blocked'] = 'overlapping vehicle hypotheses'
    elif reason == 'gap': stamp = 3
    elif reason == 'duplicate': stamp = 34/30
    elif reason == 'backward': stamp = 33/30
    elif reason == 'size': track['box'] = (55,20,30,30)
    elif reason == 'position': track['box'] = (75,20,10,10)
    estimator.update(track,stamp,(100,100))
    assert track['speed'] is None
    assert track['speed_window_seconds'] is None


@pytest.mark.parametrize('reason', ['outside', 'invalidated'])
def test_calibration_withholding_clears_the_previous_estimate_window(reason):
    config = {'points': [[0,0],[1,0],[1,1],[0,1]], 'width_m': 10, 'length_m': 10}
    estimator = SpeedEstimator(config)
    track = {}
    for index in range(35):
        track['box'] = (20+index,20,10,10)
        estimator.update(track,index/30,(100,100))
    assert track['speed'] is not None and track['speed_window_seconds'] > .5
    if reason == 'invalidated': estimator.invalidate()
    track['box'] = (110 if reason == 'outside' else 55,20,10,10)
    estimator.update(track,35/30,(100,100))
    assert track['speed'] is None
    assert track['speed_window_seconds'] is None


def test_recovered_speed_records_only_its_new_observation_window():
    estimator = SpeedEstimator()
    track = {}
    for index in range(35):
        track['box'] = (20+index,20,10,10)
        estimator.update(track,index/30,(200,300))
    track['speed_blocked'] = 'overlapping vehicle hypotheses'
    estimator.update(track,35/30,(200,300))
    assert track['speed_window_seconds'] is None
    track['speed_blocked'] = None
    for index in range(36,58):
        track['box'] = (20+index,20,10,10)
        estimator.update(track,index/30,(200,300))
    assert track['speed'] == pytest.approx(30)
    assert track['speed_window_seconds'] == .7
    assert estimator.summary()['status_counts']['overlapping vehicle hypotheses'] == 1


def test_speed_outcomes_account_for_every_observed_sample_and_preserve_withheld_reasons():
    from collections import Counter
    config={'points':[[0,0],[1,0],[1,1],[0,1]],'width_m':10,'length_m':10}
    estimator=SpeedEstimator(config)
    track={}
    outcomes=[]
    accepted=[]
    def observe(stamp, x, blocked=None):
        track.update(box=(x,20,10,10),speed_blocked=blocked)
        estimator.update(track,stamp,(100,100))
        outcomes.append(track['speed_status'])
        if track['speed'] is not None:
            accepted.append(track['speed'])
    for i in range(30):
        observe(i/30,i)
    observe(1,50)  # One discontinuous position, not a trustworthy speed.
    observe(1.1,51,'overlapping vehicle hypotheses')
    observe(1.2,101)  # Outside the calibrated patch.
    estimator.invalidate()
    observe(1.3,52)
    summary=estimator.summary()
    assert summary['status_counts']==dict(Counter(outcomes))
    assert summary['observed_track_samples']==len(outcomes)
    assert summary['accepted_track_samples']==len(accepted)
    assert summary['withheld_track_samples']==len(outcomes)-len(accepted)
    assert summary['status_counts']['estimated']==len(accepted)
    for reason in ['collecting stable track','abrupt position change; collecting a new speed window',
                   'overlapping vehicle hypotheses','outside calibrated road',
                   'camera/scene changed; calibration no longer valid']:
        assert summary['status_counts'][reason]>0
    assert summary['sample_mean']==pytest.approx(10.8,abs=.01)


@pytest.mark.parametrize('fps',[5,30])
def test_outcome_accounting_includes_gap_warmup_without_carrying_an_accepted_speed(fps):
    estimator=SpeedEstimator(frame_interval=1/fps)
    track={}
    for i in range(2*fps+1):
        stamp=i/fps
        track['box']=(20+30*stamp,50,20,20)
        estimator.update(track,stamp,(300,600))
    previous=estimator.summary()
    track['box']=(20+30*2.4,50,20,20)
    estimator.update(track,2.4,(300,600))
    summary=estimator.summary()
    assert track['speed'] is None
    assert summary['accepted_track_samples']==previous['accepted_track_samples']
    assert summary['observed_track_samples']==previous['observed_track_samples']+1
    assert summary['withheld_track_samples']==previous['withheld_track_samples']+1
    assert summary['status_counts']['collecting stable track']==previous['status_counts']['collecting stable track']+1
    assert SpeedEstimator().summary()['observed_track_samples']==0


@pytest.mark.parametrize('calibrated',[False,True])
def test_same_size_position_jump_withholds_speed_then_recovers(calibrated):
    config={'points':[[0,0],[1,0],[1,1],[0,1]],'width_m':10,'length_m':10}
    estimator=SpeedEstimator(config if calibrated else None);track={}
    def observe(i,offset=0):
        track['box']=(i+offset,20,10,10)
        estimator.update(track,i/30,(100,100))
    for i in range(30):observe(i)
    assert track['speed']==pytest.approx(10.8 if calibrated else 30)
    accepted=estimator.accepted
    observe(30,20)
    assert track['speed'] is None and track['speed_status'].startswith('abrupt position')
    assert estimator.accepted==accepted and estimator.summary()['rejected_position_jumps']==1
    for i in range(31,47):observe(i,20)
    assert track['speed']==pytest.approx(10.8 if calibrated else 30)


@pytest.mark.parametrize('fps',[5,10,30])
@pytest.mark.parametrize('motion',['acceleration','turn','jitter'])
def test_smooth_motion_and_small_box_noise_do_not_trigger_position_gate(fps,motion):
    estimator=SpeedEstimator(frame_interval=1/fps);track={};rng=np.random.default_rng(40)
    accepted=0
    for i in range(fps*4):
        stamp=i/fps
        if motion=='acceleration':x,y=10+5*stamp+3*stamp**2,20
        elif motion=='turn':x,y=200+100*np.sin(.4*stamp),200+100*np.cos(.4*stamp)
        else:x,y=20+30*stamp+rng.uniform(-1,1),50+rng.uniform(-1,1)
        track['box']=(x,y,20,20);estimator.update(track,stamp,(500,600))
        accepted+=track['speed'] is not None
    assert estimator.position_jumps==0 and accepted>fps


@pytest.mark.parametrize('fps',[5,10,30])
def test_speed_uses_actual_time_on_low_frame_rate_footage(fps):
    estimator=SpeedEstimator(frame_interval=1/fps);track={}
    for i in range(2*fps+1):
        stamp=i/fps;track['box']=(20+30*stamp,50,20,20)
        estimator.update(track,stamp,(200,300))
    assert track['speed']==pytest.approx(30)
    calibrated=SpeedEstimator(CONFIG,frame_interval=1/fps)
    inverse=np.linalg.inv(calibrated.calibration.matrix)
    for i in range(2*fps+1):
        stamp=i/fps
        image=cv2.perspectiveTransform(np.float32([[[5,5+10*stamp]]]),inverse)[0,0]*[1000,1000]
        track['box']=(image[0]-10,image[1]-20,20,20)
        calibrated.update(track,stamp,(1000,1000))
    assert track['speed']==pytest.approx(36,abs=.01)


def test_missing_low_rate_observation_still_requires_fresh_speed_window():
    estimator=SpeedEstimator(frame_interval=.2);track={}
    def observe(stamp):
        track['box']=(20+30*stamp,50,20,20)
        estimator.update(track,stamp,(300,500))
    for i in range(20):observe(i/5)
    assert track['speed']==30
    observe(4.2)  # One missing observation at 4.0 s; no speed across the gap.
    assert track['speed'] is None
    for i in range(22,28):observe(i/5)
    assert track['speed'] is None
    observe(5.6)
    assert track['speed']==30


@pytest.mark.parametrize('shape',[(1000,1000),(500,500),(720,1280)])
def test_projective_speed_known_36_kmh_irregular_timestamps(shape):
    estimator=SpeedEstimator(CONFIG)
    inverse=np.linalg.inv(estimator.calibration.matrix)
    track={}
    for stamp in np.r_[np.linspace(0,.3,13),np.linspace(.34,1.1,30)]:
        ground=np.float32([[[5,5+10*stamp]]])
        image=cv2.perspectiveTransform(ground,inverse)[0,0]*[shape[1],shape[0]]
        track['box']=(image[0]-10,image[1]-20,20,20)
        estimator.update(track,float(stamp),shape)
    assert track['speed']==pytest.approx(36,abs=.01)
    assert track['speed_unit']=='km/h'


def test_pixel_speed_and_gap_rejection():
    e=SpeedEstimator();t={}
    for i in range(35):
        t['box']=(i,30,10,10);e.update(t,i/30,(200,300))
    assert t['speed']==pytest.approx(30)
    e.update(t,3,(200,300))
    assert t['speed'] is None


def test_outside_calibration_and_size_jump_withhold_speed():
    e=SpeedEstimator(CONFIG);t={'box':(0,0,5,5)}
    e.update(t,0,(100,100));assert t['speed_status']=='outside calibrated road'
    e=SpeedEstimator()
    for i in range(30):
        t['box']=(i,20,10,10);e.update(t,i/30,(100,100))
    t['box']=(30,20,50,50);e.update(t,1,(100,100))
    assert t['speed'] is None


@pytest.mark.parametrize('config',[{}, {'points':[[0,0],[1,1],[1,0],[0,1]],'width_m':5,'length_m':20},
    {**CONFIG,'length_m':float('nan')},{**CONFIG,'width_m':0},
    {**CONFIG,'points':[[0,0],[.001,0],[.001,.001],[0,.001]]}])
def test_invalid_calibration(config):
    with pytest.raises(ValueError):RoadCalibration(config)


def test_merged_regions_do_not_steal_identity_or_speed_history():
    tracker=RegionTracker()
    for i in range(5):tracker.update([(10+i,20,10,10),(40+i,20,10,10)],i*.04,(100,100))
    for t in tracker.tracks.values():t['speed_history']=[1]
    tracks,events=tracker.update([(14,20,40,10)],.2,(100,100))
    assert tracks==[] and events==[] and tracker.ambiguous_boxes==1
    assert all('speed_history' not in t for t in tracker.tracks.values())


def test_calibration_invalidated_after_camera_change():
    e=SpeedEstimator(CONFIG);e.invalidate();t={'box':(40,40,10,10)}
    for i in range(40):e.update(t,i/30,(100,100))
    assert t['speed'] is None and 'camera/scene' in t['speed_status']


def test_small_camera_jitter_is_corrected_using_static_context():
    from ivp.stabilization import TranslationStabilizer
    rng=np.random.default_rng(6)
    frame=rng.integers(0,255,(200,300,3),np.uint8)
    stabilizer=TranslationStabilizer([0,.5,1,1])
    stabilizer.step(frame)
    moved=cv2.warpAffine(frame,np.float32([[1,0,2],[0,1,1]]),(300,200),borderMode=cv2.BORDER_REFLECT_101)
    corrected,ok=stabilizer.step(moved)
    assert ok and stabilizer.offset==pytest.approx([-2,-1],abs=.1)
    assert np.mean(abs(corrected[10:-10,10:-10].astype(float)-frame[10:-10,10:-10]))<3


def test_stabilizer_keyframe_refresh_retains_original_coordinates():
    from ivp.stabilization import TranslationStabilizer
    frame=np.random.default_rng(19).integers(0,255,(200,300,3),np.uint8)
    s=TranslationStabilizer([0,.5,1,1])
    for i in range(65):
        shift=i*.15
        moved=cv2.warpAffine(frame,np.float32([[1,0,shift],[0,1,shift/2]]),(300,200))
        _,ok=s.step(moved)
    assert ok and s.rejected_frames==0
    assert s.offset==pytest.approx([-9.6,-4.8],abs=.7)


def test_measured_camera_pan_with_large_remaining_overlap_is_not_rejected():
    from ivp.stabilization import TranslationStabilizer
    frame=np.random.default_rng(20).integers(0,255,(200,300,3),np.uint8)
    s=TranslationStabilizer([0,.5,1,1])
    for i in range(41):
        moved=cv2.warpAffine(frame,np.float32([[1,0,i*2],[0,1,0]]),(300,200))
        _,ok=s.step(moved)
    assert ok and s.rejected_frames==0
    assert s.offset==pytest.approx([-80,0],abs=.8)


def test_calibrated_recovery_requires_a_new_observation_window():
    e=SpeedEstimator(CONFIG);inverse=np.linalg.inv(e.calibration.matrix);track={}
    def observe(t):
        ground=np.float32([[[5,5+10*t]]])
        image=cv2.perspectiveTransform(ground,inverse)[0,0]*[1000,1000]
        track['box']=(image[0]-10,image[1]-20,20,20)
        e.update(track,t,(1000,1000))
    for t in np.linspace(0,.7,22):observe(float(t))
    assert track['speed']==pytest.approx(36,abs=.01)
    observe(1.1)  # The object was unseen for 0.4 s: no inferred speed in the gap.
    assert track['speed'] is None
    for t in np.linspace(1.13,1.5,12):observe(float(t))
    assert track['speed'] is None
    for t in np.linspace(1.53,1.8,10):observe(float(t))
    assert track['speed']==pytest.approx(36,abs=.01)
    track['box']=(0,0,5,5);e.update(track,1.83,(1000,1000))
    assert track['speed'] is None and track['speed_status']=='outside calibrated road'
