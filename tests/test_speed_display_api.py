from pathlib import Path
import pytest
import web_server
from ivp.media import source_revision


@pytest.fixture
def captured_api(tmp_path, monkeypatch):
    path=tmp_path/'owned-source.mp4';path.write_bytes(b'No decoding in this route contract fixture.')
    info={'kind':'video','source_revision':source_revision(path),'width':1280,'height':720}
    lab=web_server.Lab.__new__(web_server.Lab);lab.media={'m':path}
    lab._source=lambda request:(path,info)
    lab._check_source=lambda *args:None
    launched=[]
    def capture(path,run_info,settings,masks,motion):
        launched.append({'motion':motion,'info':run_info,'max_width':settings.max_width});return Path('capture-only')
    monkeypatch.setattr(web_server,'start_video',capture)
    lab.run=lambda name:launched[-1]
    return lab,info,launched


def test_route_preserves_pixel_display_and_fixed_analysis_width(captured_api):
    lab,info,launched=captured_api
    result=lab.post('/api/motion',{'media':'m','source_revision':info['source_revision'],'ui_feature':'tracking',
        'tracking_mode':'ivp','speed_display':{'enabled':True,'limit':30,'unit':'px/s'},'settings':{'max_width':1920}})
    assert result['motion']['speed_display']=={'enabled':True,'limit':30,'unit':'px/s'}
    assert result['max_width']==960 and result['motion']['calibration'] is None
    assert len(launched)==1


@pytest.mark.parametrize('display',[{'enabled':True,'limit':0,'unit':'px/s'},
    {'enabled':True,'limit':float('nan'),'unit':'px/s'},{'enabled':True,'limit':30,'unit':'km/h'},
    {'enabled':False,'limit':30,'unit':'px/s'},{'enabled':'yes','limit':30,'unit':'px/s'}])
def test_route_rejects_bad_limits_before_dispatch(captured_api,display):
    lab,info,launched=captured_api
    with pytest.raises(ValueError):lab.post('/api/motion',{'media':'m','ui_feature':'tracking','speed_display':display})
    assert not launched


def test_legacy_off_request_does_not_acquire_display_settings(captured_api):
    lab,info,_=captured_api
    result=lab.post('/api/motion',{'media':'m','ui_feature':'tracking'})
    assert 'speed_display' not in result['motion']


def test_calibrated_display_never_changes_incident_limit(captured_api,monkeypatch):
    lab,info,_=captured_api
    # Route capture only: avoid constructing any detector for this test.
    import ivp.motion
    monkeypatch.setattr(ivp.motion,'MotionAnalyzer',lambda **kwargs:None)
    calibration={'media_id':'m','source_revision':info['source_revision'],'reference_frame':0,
        'points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'width_m':16,'length_m':40}
    incidents={'enabled':True,'speed_limit_kmh':35,'condition':'observe','calibration_confirmed':True,'real_time_confirmed':True}
    result=lab.post('/api/motion',{'media':'m','ui_feature':'speed_review','tracking_mode':'vehicles','calibration':calibration,
        'incidents':incidents,'speed_display':{'enabled':True,'limit':18,'unit':'km/h'}})
    assert result['motion']['speed_display']['limit']==18
    assert result['motion']['incidents']['speed_limit_kmh']==35
