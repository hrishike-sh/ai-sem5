from pathlib import Path
import pytest
import web_server
from tests.test_configured_direction import config


def lab(monkeypatch,kind='video'):
    instance=web_server.Lab.__new__(web_server.Lab);instance.media={'m':None}
    instance._source=lambda data:(Path('offline'),{'kind':kind,'source_revision':'r','duration':10.})
    instance._check_source=lambda *a:None;instance.run=lambda name:{'id':name};calls=[]
    def launch(*args,**kw):calls.append((args,kw));return Path('fixture')
    monkeypatch.setattr(web_server,'start_video',launch)
    return instance,calls


def test_helmet_api_exact_timestamp_and_source_handoff(monkeypatch):
    instance,calls=lab(monkeypatch)
    assert instance.post('/api/helmet',{'media':'m','timestamp':2.4})=={'id':'fixture'}
    assert calls[0][1]=={'helmet':{'timestamp':2.4,'media_id':'m'}}


@pytest.mark.parametrize('stamp',[None,'0',True,-1.,float('nan'),10.])
def test_bad_helmet_timestamp_cannot_launch(monkeypatch,stamp):
    instance,calls=lab(monkeypatch)
    with pytest.raises(ValueError):instance.post('/api/helmet',{'media':'m','timestamp':stamp})
    assert not calls


def test_direction_api_requires_current_review_and_vehicle_mode(monkeypatch):
    instance,calls=lab(monkeypatch);d={k:v for k,v in config().items() if k!='config_id'}
    monkeypatch.setattr(web_server,'source_revision',lambda path:'r')
    value={'media':'m','direction':d,'tracking_mode':'vehicles'}
    assert instance.post('/api/motion',value)=={'id':'fixture'}
    motion=calls[0][1]['motion'];assert motion['direction']['source_revision']=='r' and motion['calibration'] is None
    value['tracking_mode']='ivp'
    with pytest.raises(ValueError,match='vehicle tracking'):instance.post('/api/motion',value)
    value['tracking_mode']='vehicles';d['source_revision']='old'
    with pytest.raises(ValueError,match='different footage'):instance.post('/api/motion',value)
    assert len(calls)==1


@pytest.mark.parametrize('state',['failed','cancelled','running'])
def test_incomplete_legacy_observation_reports_expose_no_positive_summaries(monkeypatch,tmp_path,state):
    import json
    root=tmp_path/'runs'/'legacy';root.mkdir(parents=True)
    report={'source':'fixture.png','helmet':{'counts':{'bare_head':1}},'zebra':{'counts':{'observed':1}},
            'motion':{'direction':{'episodes':1},'zebra':{'episodes':1}}}
    raw=json.dumps(report);(root/'report.json').write_text(raw);(root/'status.json').write_text('{}')
    (root/'processed.png').write_bytes(b'partial fixture')
    monkeypatch.setattr(web_server,'ROOT',tmp_path)
    monkeypatch.setattr(web_server,'refreshed_status',lambda run:{'state':state})
    instance=web_server.Lab.__new__(web_server.Lab)
    from types import SimpleNamespace
    instance.files=SimpleNamespace(url=lambda *a,**kw:'unpublished-partial')
    result=instance.run('legacy')
    assert result['downloads']==[] and result['preview'] is None
    assert 'helmet' not in result['report'] and 'zebra' not in result['report']
    assert 'direction' not in result['report']['motion'] and 'zebra' not in result['report']['motion']
    assert (root/'report.json').read_text()==raw
