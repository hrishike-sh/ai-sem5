"""Source lifecycle contracts through real Lab methods, without any server."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
from PIL import Image

import web_server


@pytest.fixture
def source_lab(tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, 'ROOT', tmp_path)
    monkeypatch.setattr(web_server, 'FileServer', lambda: pytest.fail('No test server needed'))
    monkeypatch.setattr(web_server, 'start_video', lambda *a, **k: pytest.fail('Changed source must not launch work'))
    created = []
    def new_run():
        folder = tmp_path/'runs'/f'export-{len(created)}'
        folder.mkdir(parents=True)
        created.append(folder)
        return folder
    monkeypatch.setattr(web_server, 'new_run', new_run)
    path = tmp_path/'source.png'
    Image.fromarray(np.full((48,64,3),20,np.uint8)).save(path)
    lab = web_server.Lab.__new__(web_server.Lab)
    lab.media, lab.auto_profiles, lab.cache = {}, {}, None
    lab.files = SimpleNamespace(url=lambda file, *args: '/offline/'+Path(file).name)
    media = lab.register(path)
    payload = {'media': media['id'], 'source_revision': media['source_revision'], 'mode': 'manual'}
    def replace():
        previous = path.stat().st_mtime_ns
        Image.fromarray(np.full((72,96,3),230,np.uint8)).save(path)
        os.utime(path, ns=(previous+2_000_000_000,previous+2_000_000_000))
    return SimpleNamespace(lab=lab,path=path,media=media,payload=payload,replace=replace,created=created)


@pytest.mark.parametrize('operation',['warm','cold','frame','profile','export','auto_export','region','job','motion'])
def test_replaced_registered_source_is_rejected_before_reuse_or_launch(source_lab, operation):
    c = source_lab
    if operation == 'warm': c.lab.processed(c.payload)
    if operation in ('profile','auto_export'):
        c.lab.resolve_settings({**c.payload,'mode':'automatic'})
    if operation in ('job','motion'):
        c.lab.media[c.media['id']][1].update(kind='video',duration=1,fps=30)
    c.replace()
    actions = {
        'warm': lambda: c.lab.processed(c.payload), 'cold': lambda: c.lab.processed(c.payload),
        'frame': lambda: c.lab.frame(c.payload),
        'profile': lambda: c.lab.resolve_settings({**c.payload,'mode':'automatic'}),
        'export': lambda: c.lab.export(c.payload),
        'auto_export': lambda: c.lab.export({**c.payload,'mode':'automatic'}),
        'region': lambda: c.lab.post('/api/region',{**c.payload,'rectangle':[0,0,1,1]}),
        'job': lambda: c.lab.post('/api/job',c.payload),
        'motion': lambda: c.lab.post('/api/motion',c.payload),
    }
    with pytest.raises(ValueError,match='[Ss]ource.*changed'):
        actions[operation]()
    assert c.created == []


@pytest.mark.parametrize('stage',['decode','processing','automatic'])
def test_replacement_during_preparation_is_rejected_without_caching(source_lab, monkeypatch, stage):
    c = source_lab
    name = {'decode':'load_image','processing':'process_frame','automatic':'choose_settings'}[stage]
    original = getattr(web_server,name)
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        c.replace()
        return result
    monkeypatch.setattr(web_server,name,changed)
    with pytest.raises(ValueError,match='[Ss]ource.*changed'):
        if stage == 'decode': c.lab.frame(c.payload)
        elif stage == 'automatic': c.lab.resolve_settings({**c.payload,'mode':'automatic'})
        else: c.lab.processed(c.payload)
    assert c.lab.cache is None and c.lab.auto_profiles == {} and c.created == []


def test_replacement_during_registration_does_not_rebind_metadata(source_lab, monkeypatch):
    c = source_lab
    original = web_server.load_image
    prior = c.lab.media[c.media['id']]
    def changed(path):
        pixels = original(path)
        c.replace()
        return pixels
    monkeypatch.setattr(web_server,'load_image',changed)
    with pytest.raises(ValueError,match='[Ss]ource.*changed'):
        c.lab.register(c.path)
    assert c.lab.media[c.media['id']] is prior


@pytest.mark.parametrize('stage',['stages','report'])
def test_replacement_during_export_cannot_publish_completed_output(source_lab, monkeypatch, stage):
    c = source_lab
    original = getattr(web_server,'build_stages' if stage == 'stages' else 'write_json')
    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        if stage == 'stages' or Path(args[0]).name == 'report.json': c.replace()
        return result
    monkeypatch.setattr(web_server,'build_stages' if stage == 'stages' else 'write_json',changed)
    with pytest.raises(ValueError,match='[Ss]ource.*changed'):
        c.lab.export(c.payload)
    assert len(c.created) == 1
    status = json.loads((c.created[0]/'status.json').read_text())
    assert status['state'] == 'failed'
    saved = c.lab.run(c.created[0].name)
    assert saved['downloads'] == [] and saved['preview'] is None


@pytest.mark.parametrize('operation',['frame','processed','profile','export','job','motion'])
def test_old_client_revision_is_rejected_after_another_registration(source_lab, operation):
    c = source_lab
    c.replace()
    c.lab.register(c.path)
    if operation in ('job','motion'):
        c.lab.media[c.media['id']][1].update(kind='video',duration=1,fps=30)
    with pytest.raises(ValueError,match='[Ss]ource.*changed'):
        if operation == 'profile': c.lab.resolve_settings({**c.payload,'mode':'automatic'})
        elif operation in ('job','motion'): c.lab.post('/api/'+operation,c.payload)
        else: getattr(c.lab,operation)(c.payload)
    assert c.created == []


def test_reopen_refreshes_profile_binding_and_preserves_historical_export(source_lab):
    c = source_lab
    data = {**c.payload,'mode':'automatic'}
    old = c.lab.export(data)
    pins = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in c.created[0].iterdir()}
    assert old['report']['processing_policy']['measurements']['brightness'] == 20
    c.replace()
    with pytest.raises(ValueError): c.lab.export(data)
    fresh = c.lab.register(c.path)
    current = c.lab.export({**data,'source_revision':fresh['source_revision']})
    report = current['report']
    assert report['processed_size'] == [96,72]
    assert report['processing_policy']['measurements']['brightness'] == 230
    assert report['settings']['gamma'] == .85
    binding = report['source_binding']
    assert binding['source_revision'] == fresh['source_revision'] != c.media['source_revision']
    assert binding['timestamp_seconds'] == report['timestamp_seconds'] == 0
    assert binding['transform_to_source_pixel_centres'] == report['transform_to_source_pixel_centres']
    with zipfile.ZipFile(c.created[-1]/'masks.zip') as archive:
        assert json.loads(archive.read('manifest.json'))['source_binding'] == binding
    with zipfile.ZipFile(c.created[-1]/'ivp-stages.zip') as archive:
        assert json.loads(archive.read('processing.json'))['source_binding'] == binding
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == value for p,value in pins.items())


def test_unchanged_registration_reuses_profile_and_legacy_payload(source_lab, monkeypatch):
    c = source_lab
    calls = []
    original = web_server.choose_settings
    def measured(pixels):
        calls.append(1)
        return original(pixels)
    monkeypatch.setattr(web_server,'choose_settings',measured)
    data = {'media':c.media['id'],'mode':'automatic'}  # Older clients omit the optional revision.
    first = c.lab.resolve_settings(data)
    c.lab.register(c.path)
    second = c.lab.resolve_settings(data)
    assert first is second and calls == [1]
    assert c.lab.export(data)['status']['state'] == 'complete'


def test_missing_source_is_actionable_and_cannot_create_an_export(source_lab):
    c = source_lab
    c.path.unlink()
    with pytest.raises(ValueError,match='[Ss]ource.*[Rr]eopen'):
        c.lab.export(c.payload)
    assert c.created == []


def test_old_calibration_is_not_rebound_to_reopened_source(source_lab):
    c = source_lab
    c.replace()
    fresh = c.lab.register(c.path)
    c.lab.media[fresh['id']][1].update(kind='video',duration=1,fps=30)
    calibration = {'media_id':fresh['id'],'source_revision':c.media['source_revision'],'reference_frame':0}
    with pytest.raises(ValueError,match='outdated'):
        c.lab.post('/api/motion',{'media':fresh['id'],'source_revision':fresh['source_revision'],'calibration':calibration})
    assert c.created == []


def test_legacy_direct_frame_provider_without_revision_keeps_fixed_profile(source_lab, monkeypatch):
    c = source_lab
    c.lab.media['clip'] = (None,{'kind':'video'})
    calls = []
    def frame(data):
        calls.append(data['timestamp'])
        return None,{'kind':'video'},np.full((60,80,3),20,np.uint8),0
    monkeypatch.setattr(c.lab,'frame',frame)
    first = c.lab.resolve_settings({'media':'clip','mode':'automatic'})
    assert c.lab.resolve_settings({'media':'clip','mode':'automatic'}) is first
    assert calls == [0]
