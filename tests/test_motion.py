import numpy as np
import pytest

from ivp.motion import MotionAnalyzer, RegionTracker


def test_directional_crossing_with_hysteresis_and_no_duplicate():
    tracker=RegionTracker('horizontal',.5)
    events=[]
    for i,y in enumerate([10,15,20,30,40,48,52,58,65,70,60,48,40]):
        _,new=tracker.update([(40,y,10,10)],i*.04,(100,100))
        events.extend(new)
    assert len(events)==1 and events[0]['direction']=='positive'
    assert tracker.counts=={'positive':1,'negative':0}


def test_reverse_crossing_and_expired_identity():
    tracker=RegionTracker('vertical',.5)
    for i,x in enumerate([80,75,70,65,55,45,35,30]):
        tracker.update([(x,40,10,10)],i*.04,(100,100))
    assert tracker.counts['negative']==1
    tracks,_=tracker.update([(30,40,10,10)],3,(100,100))
    assert not tracks  # new, unconfirmed region
    assert tracker.next_id==3


def test_static_frames_have_no_activity_or_crossings():
    engine=MotionAnalyzer()
    rgb=np.random.default_rng(5).integers(0,255,(120,160,3),dtype=np.uint8)
    for i in range(50):
        _,m,events,_=engine.step(rgb,i*.04)
    assert m['state']=='measured' and m['moving_area_percent']==0
    assert engine.summary()['crossings']=={'positive':0,'negative':0}
    assert engine.heat.max()==0


def test_moving_shape_builds_heat_and_scene_reset_suppresses():
    engine=MotionAnalyzer(position=.5)
    background=np.full((160,240,3),35,np.uint8)
    for i in range(25):
        engine.step(background,i*.04)
    for i in range(35):
        rgb=background.copy()
        rgb[10+i*3:30+i*3,100:125]=[230,80,30]
        mosaic,m,events,_=engine.step(rgb,1+i*.04)
    assert mosaic.shape==background.shape
    assert engine.heat.max()>0
    assert engine.summary()['crossings']['positive']==1
    _,m,events,_=engine.step(rgb,2.5,scene_change=True)
    assert m['state']=='camera/scene change' and not events


@pytest.mark.parametrize('kwargs',[{'axis':'bad'},{'position':float('nan')},{'position':True},
                                  {'tracking_mode':'unknown'},{'output_view':'unknown'},
                                  {'rectangle':[0,0,1]},{'rectangle':[0,0,1,.4],'position':.6}])
def test_invalid_motion_configuration(kwargs):
    with pytest.raises(ValueError):
        MotionAnalyzer(**kwargs)


def test_camera_translation_is_excluded():
    import cv2
    rng=np.random.default_rng(1)
    original=rng.integers(0,255,(180,240,3),dtype=np.uint8)
    engine=MotionAnalyzer()
    for i in range(25):
        engine.step(original,i*.04)
    moved=cv2.warpAffine(original,np.float32([[1,0,6],[0,1,0]]),(240,180))
    _,m,events,_=engine.step(moved,1)
    assert m['state']=='camera/scene change'
    assert m['moving_area_percent'] is None and not events


def test_verified_static_context_prevents_crowd_flow_camera_false_alarm():
    import cv2
    original=np.random.default_rng(42).integers(0,255,(240,320,3),dtype=np.uint8)
    engine=MotionAnalyzer(rectangle=[0,.2,1,1])
    for i in range(25):engine.step(original,i*.04)
    moved=original.copy()
    moved[48:]=cv2.warpAffine(original[48:],np.float32([[1,0,6],[0,1,0]]),(320,192),borderMode=cv2.BORDER_REFLECT_101)
    _,metrics,_,_=engine.step(moved,1)
    assert metrics['global_flow_px']>2.5
    assert metrics['state']!='camera/scene change'
    assert not engine.stabilizer.failed


def test_motion_worker_preserves_timing_audio_and_exports(source,tmp_path):
    import json
    import zipfile
    from ivp.motion_worker import run_motion
    from ivp.media import probe_video,write_json
    from tests.test_video import request_run,stamps
    run=request_run(tmp_path,source)
    request=json.loads((run/'request.json').read_text())
    request['motion']={'axis':'horizontal','position':.65,'rectangle':None}
    write_json(run/'request.json',request)
    run_motion(run)
    report=json.loads((run/'report.json').read_text())
    assert report['motion']['uses_ai'] is False and 'ai' not in report
    assert probe_video(run/'processed.mp4')['has_audio']
    assert stamps(run/'processed.mp4')==pytest.approx(stamps(source),abs=2/90000)
    with zipfile.ZipFile(run/'motion-evidence.zip') as z:
        assert {'motion-activity.csv','motion-crossings.csv','motion-seconds.npy','coordinates.json'} <= set(z.namelist())


def test_motion_cancel_publishes_no_completed_video(source,tmp_path):
    import json
    from ivp.motion_worker import run_motion
    from ivp.media import write_json
    from tests.test_video import request_run
    run=request_run(tmp_path,source)
    request=json.loads((run/'request.json').read_text());request['motion']={}
    write_json(run/'request.json',request)
    (run/'cancel.request').touch()
    run_motion(run)
    assert json.loads((run/'status.json').read_text())['state']=='cancelled'
    assert not (run/'processed.mp4').exists() and not (run/'report.json').exists()


def test_activity_timeline_uses_decoded_endpoint_when_metadata_duration_is_wrong(source,tmp_path):
    import json
    from ivp.motion_worker import run_motion
    from ivp.media import write_json
    from tests.test_video import request_run
    run=request_run(tmp_path,source)
    request=json.loads((run/'request.json').read_text());request['motion']={}
    request['info']['duration']=.4
    write_json(run/'request.json',request);run_motion(run)
    report=json.loads((run/'report.json').read_text());motion=report['motion']
    assert motion['timing']['source_metadata_duration_seconds']==.4
    assert motion['timeline'][-1]['end']==pytest.approx(report['duration_seconds'])
    assert sum(row['observed_frames'] for row in motion['timeline'])==report['frames']
    assert sum(row['observed_frames'] for row in motion['timeline'] if row['start']>.4)>0


from tests.test_video import source


def test_midstream_cancellation_then_clean_retry(source, tmp_path, monkeypatch):
    import json
    from ivp.motion_worker import run_motion
    from ivp.media import write_json
    from tests.test_video import request_run
    original_step = MotionAnalyzer.step
    run = request_run(tmp_path, source)
    request = json.loads((run/'request.json').read_text()); request['motion'] = {}
    write_json(run/'request.json', request)
    calls = []
    def cancel_after_frames(self, *args, **kwargs):
        result = original_step(self, *args, **kwargs)
        calls.append(1)
        if len(calls) == 3:
            (run/'cancel.request').touch()
        return result
    monkeypatch.setattr(MotionAnalyzer, 'step', cancel_after_frames)
    run_motion(run)
    assert len(calls) == 3
    assert json.loads((run/'status.json').read_text())['state'] == 'cancelled'
    assert not any((run/name).exists() for name in ('processed.mp4','report.json','motion-evidence.zip'))
    monkeypatch.setattr(MotionAnalyzer, 'step', original_step)
    retry = request_run(tmp_path, source)
    write_json(retry/'request.json', request)
    run_motion(retry)
    assert json.loads((retry/'status.json').read_text())['state'] == 'complete'
    assert (retry/'processed.mp4').is_file() and (retry/'motion-evidence.zip').is_file()


def test_calibrated_export_keeps_source_and_units(source, tmp_path):
    import json
    import zipfile
    from ivp.motion_worker import run_motion
    from ivp.media import source_revision, write_json
    from tests.test_video import request_run
    run = request_run(tmp_path, source)
    request = json.loads((run/'request.json').read_text())
    calibration = {'points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]], 'width_m':8, 'length_m':20,
                   'media_id':'test-fixture', 'source_revision':source_revision(source), 'reference_frame':0}
    request['info']['source_revision'] = calibration['source_revision']
    request['motion'] = {'tracking_mode':'ivp', 'calibration':calibration}
    write_json(run/'request.json', request)
    run_motion(run)
    report = json.loads((run/'report.json').read_text())
    assert report['source'] == source.name
    assert report['source_info']['source_revision'] == calibration['source_revision']
    assert report['motion']['speed']['unit'] == 'km/h'
    assert report['motion']['speed']['calibration'] == calibration
    assert not report['motion']['uses_ai']
    with zipfile.ZipFile(run/'motion-evidence.zip') as archive:
        saved = json.loads(archive.read('request.json'))
        assert saved['motion'] == request['motion'] and saved['source'] == str(source)


def test_display_modes_preserve_measurements_and_peak_evidence():
    import cv2
    base=np.random.default_rng(19).integers(0,80,(120,160,3),dtype=np.uint8)
    results=[]
    for mode in ('tracks','stages'):
        cv2.setRNGSeed(0)
        engine=MotionAnalyzer(rectangle=[0,.3,1,1],output_view=mode)
        metrics=[]
        for i in range(45):
            frame=base.copy();x=10+i*2
            frame[65:85,x:x+15]=230
            output,measurements,events,tracks=engine.step(frame,i/30)
            assert output.shape==frame.shape
            metrics.append(measurements)
        results.append((engine,metrics))
    first,second=results[0][0],results[1][0]
    assert results[0][1]==results[1][1]
    assert first.peak is not None and second.peak is not None
    for name in ('mask','tracks','mosaic','flow','original'):
        assert np.array_equal(first.peak[name],second.peak[name])
    assert np.array_equal(first.heatmap(),second.heatmap())


def test_integer_heatmap_blend_matches_previous_pixels():
    import cv2
    engine=MotionAnalyzer()
    rng=np.random.default_rng(31)
    engine.heat=rng.integers(0,100,(40,50)).astype(np.float32)
    engine.last_rgb=rng.integers(0,256,(40,50,3),dtype=np.uint8)
    scale=(engine.heat/engine.heat.max()*255).astype(np.uint8)
    color=cv2.cvtColor(cv2.applyColorMap(scale,cv2.COLORMAP_TURBO),cv2.COLOR_BGR2RGB)
    previous=engine.last_rgb.copy();active=scale>0
    previous[active]=(previous[active]*.25+color[active]*.75).astype(np.uint8)
    assert np.array_equal(engine.heatmap(),previous)
