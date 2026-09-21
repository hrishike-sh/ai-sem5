import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

import web_server
from tests.test_video import source


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(web_server, 'ROOT', tmp_path)
    def new_run():
        run = tmp_path / 'runs' / 'test-run'
        run.mkdir(parents=True)
        return run
    monkeypatch.setattr(web_server, 'new_run', new_run)
    instance = web_server.create_server(0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    instance.lab.close()


def request(server, route, data=None, raw=None, token=True, host=None):
    headers = {'X-Traffic-Token': server.lab.token} if token else {}
    if host:
        headers['Host'] = host
    body = raw if raw is not None else json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(f'http://127.0.0.1:{server.server_port}'+route, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def test_real_upload_preview_region_and_export(server):
    buffer = io.BytesIO()
    Image.new('RGB', (60, 40), (80, 40, 20)).save(buffer, 'PNG')
    media = request(server, '/api/upload?name=traffic.png', raw=buffer.getvalue())
    server.lab.media.clear()
    restored = request(server, '/api/media?id='+media['id'])
    assert restored['id'] == media['id'] and restored['name'] == 'traffic.png'
    payload = {'media':media['id'], 'settings':{}, 'timestamp':0}
    preview = request(server, '/api/preview', payload)
    assert preview['size'] == [60, 40]
    assert preview['mapping'] == np.eye(3).tolist()
    with urllib.request.urlopen(preview['processed']) as response:
        pixels = np.array(Image.open(io.BytesIO(response.read())))
    assert np.array_equal(pixels[0,0], [80,40,20])
    region = request(server, '/api/region', {**payload,'rectangle':[0,0,.5,1]})
    assert sum(region['histogram_processed']) == 30*40
    exported = request(server, '/api/export', {**payload,'masks':['Sobel magnitude','Top hat']})
    assert exported['status']['state'] == 'complete'
    assert exported['report']['selected_masks'] == ['Sobel magnitude','Top hat']
    assert len(exported['downloads']) == 4
    with urllib.request.urlopen(exported['preview']['url']) as response:
        assert np.array_equal(np.array(Image.open(io.BytesIO(response.read()))), pixels)


def test_request_boundaries_and_corrupt_file(server):
    assert request(server, '/api/health')['app'] == 'traffic-lab-web'
    for route, kwargs, code in [
        ('/api/preview', {'data':{},'token':False}, 403),
        ('/api/config', {'host':'attacker.example'}, 403),
        ('/api/upload?name=bad.png', {'raw':b'broken'}, 400),
        ('/api/upload?name=script.exe', {'raw':b'broken'}, 400),
        ('/api/validate-settings', {'data':{'settings':{'gamma':float('inf')}}}, 400),
        ('/api/run?id=..', {}, 400),
    ]:
        with pytest.raises(urllib.error.HTTPError) as error:
            try:
                request(server, route, **kwargs)
            except ConnectionAbortedError as exc:
                raise AssertionError(f"Connection aborted for {route}, {kwargs}") from exc
        assert error.value.code == code
    server.lab.lock.acquire()
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, '/api/preview', {})
        assert error.value.code == 409
    finally:
        server.lab.lock.release()


def test_motion_calibration_scoped_to_source(server, source, monkeypatch):
    from ivp.media import write_json
    media = server.lab.register(source)
    calibration = {'points': [[.1,.1],[.9,.1],[.9,.9],[.1,.9]],
                   'width_m': 8, 'length_m': 20, 'media_id': media['id'],
                   'source_revision': media['source_revision'], 'reference_frame': 0}
    launched = []
    def start(path, info, settings, masks, motion):
        launched.append(motion)
        run = web_server.ROOT / 'runs' / str(len(launched))
        run.mkdir(parents=True)
        write_json(run/'request.json', {'source': str(path), 'motion': motion})
        write_json(run/'status.json', {'state': 'cancelled', 'message': 'Test request captured'})
        return run
    monkeypatch.setattr(web_server, 'start_video', start)
    for patch in ({'media_id':'another-source'}, {'source_revision':'old'},
                  {'reference_frame':10}, {'points':[[.1,.1]]}, {'width_m':0}):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, '/api/motion', {'media':media['id'], 'calibration':{**calibration, **patch}})
        assert error.value.code == 400
    assert not launched
    # Normalized calibration remains valid across tracking mode and ROI changes.
    for mode in ('ivp', 'vehicles'):
        result = request(server, '/api/motion', {'media':media['id'], 'calibration':calibration,
                         'tracking_mode':mode, 'rectangle':[0,.2,1,1]})
        assert result['source'] == source.name
        assert result['report']['motion']['calibration'] == calibration
        assert result['report']['motion']['tracking_mode'] == mode
        assert result['downloads'] == [] and result['preview'] is None
    source.touch()
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, '/api/motion', {'media':media['id'], 'calibration':calibration})
    assert error.value.code == 400


def test_reopening_replaced_source_recomputes_automatic_ivp_profile(server, tmp_path):
    path = tmp_path/'replaced.png'
    Image.fromarray(np.full((48, 64, 3), 20, np.uint8)).save(path)
    original = server.lab.register(path)
    old_settings, old_profile = server.lab.resolve_settings({'media': original['id'], 'mode': 'automatic'})
    assert old_settings.gamma == 1.5 and old_profile['measurements']['brightness'] == 20

    Image.fromarray(np.full((72, 96, 3), 230, np.uint8)).save(path)
    replacement = server.lab.register(path)
    assert replacement['id'] == original['id']
    assert replacement['source_revision'] != original['source_revision']
    new_settings, new_profile = server.lab.resolve_settings({'media': replacement['id'], 'mode': 'automatic'})
    assert new_settings.gamma == .85
    assert new_profile['measurements']['brightness'] == 230
    assert old_settings.gamma == 1.5  # Historical objects remain inspectable.
    assert old_profile['measurements']['brightness'] == 20


def test_unchanged_reopened_source_retains_fixed_automatic_profile(server, tmp_path, monkeypatch):
    path = tmp_path/'unchanged.png'
    Image.fromarray(np.full((48, 64, 3), 20, np.uint8)).save(path)
    calls=[]
    choose=web_server.choose_settings
    def measured(rgb):
        calls.append(1)
        return choose(rgb)
    monkeypatch.setattr(web_server,'choose_settings',measured)
    media=server.lab.register(path)
    first=server.lab.resolve_settings({'media':media['id'],'mode':'automatic'})
    reopened=server.lab.register(path)
    second=server.lab.resolve_settings({'media':reopened['id'],'mode':'automatic'})
    assert calls == [1]
    assert first == second and media['source_revision'] == reopened['source_revision']


def test_replaced_source_preview_and_export_use_new_profile_keep_old_result(server, tmp_path, monkeypatch):
    import hashlib
    serial=[]
    def fresh_run():
        serial.append(1)
        run=tmp_path/'runs'/f'reload-{len(serial)}'
        run.mkdir(parents=True)
        return run
    monkeypatch.setattr(web_server,'new_run',fresh_run)
    path=tmp_path/'reload.png'
    Image.fromarray(np.full((48,64,3),20,np.uint8)).save(path)
    media=server.lab.register(path)
    payload={'media':media['id'],'mode':'automatic'}
    before=request(server,'/api/preview',payload)
    assert before['settings']['gamma']==1.5
    old=request(server,'/api/export',payload)
    original={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'runs'/old['id']).iterdir() if p.is_file()}
    Image.fromarray(np.full((72,96,3),230,np.uint8)).save(path)
    server.lab.register(path)
    after=request(server,'/api/preview',payload)
    exported=request(server,'/api/export',payload)
    assert after['settings']['gamma']==exported['report']['settings']['gamma']==.85
    assert after['size']==exported['report']['processed_size']==[96,72]
    assert exported['report']['processing_policy']['measurements']['brightness']==230
    assert exported['report']['measurements']['original']['brightness']==230
    assert np.array_equal(exported['report']['transform_to_source_pixel_centres'],np.eye(3))
    old_again=request(server,'/api/run?id='+old['id'])
    assert old_again['status']['state']=='complete' and old_again['preview']
    assert old_again['report']['settings']['gamma']==1.5
    assert original=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (tmp_path/'runs'/old['id']).iterdir() if p.is_file()}


def test_incident_api_validation_and_optional_default(server, source, monkeypatch):
    from ivp.media import write_json
    media=server.lab.register(source)
    captured=[]
    def start(path,info,settings,masks,motion):
        captured.append(motion)
        run=web_server.ROOT/'runs'/str(len(captured));run.mkdir(parents=True)
        write_json(run/'request.json',{'source':str(path),'motion':motion})
        write_json(run/'status.json',{'state':'cancelled'})
        return run
    monkeypatch.setattr(web_server,'start_video',start)
    base={'media':media['id'],'tracking_mode':'vehicles'}
    config={'enabled':True,'line':[[.15,.65],[.85,.65]],'condition':'red','synthetic':True}
    result=request(server,'/api/motion',{**base,'incidents':config})
    assert result['report']['motion']['incidents']['synthetic'] is True
    assert result['report']['motion']['incidents']['zone_role']=='observe'
    assert result['downloads']==[]
    for payload in ({**base,'incidents':{**config,'speed_limit_kmh':0}},
                    {**base,'tracking_mode':'ivp','incidents':config}):
        with pytest.raises(urllib.error.HTTPError) as error:request(server,'/api/motion',payload)
        assert error.value.code==400
    request(server,'/api/motion',base)
    assert 'incidents' not in captured[-1]


def test_incident_thumbnails_are_scoped_to_completed_run(server,tmp_path):
    from ivp.media import write_json
    run=tmp_path/'runs'/'incident-result';run.mkdir(parents=True)
    (run/'incident-000001.png').write_bytes(b'fixture')
    events=[{'thumbnail':'incident-000001.png'},{'thumbnail':'../private.png'}]
    write_json(run/'report.json',{'source':'fixture.mp4','motion':{'incidents':{'events':events}}})
    write_json(run/'status.json',{'state':'cancelled'})
    assert 'thumbnail_url' not in server.lab.run(run.name)['report']['motion']['incidents']['events'][0]
    write_json(run/'status.json',{'state':'complete'})
    result=server.lab.run(run.name)['report']['motion']['incidents']['events']
    assert result[0]['thumbnail_url'] and 'thumbnail_url' not in result[1]


def test_early_rejections_deliver_json_without_resetting_windows_socket(server):
    for _ in range(20):
        for route, kwargs, code in [
            ('/api/preview', {'data':{}, 'token':False}, 403),
            ('/api/upload?name=script.exe', {'raw':b'broken'}, 400),
        ]:
            with pytest.raises(urllib.error.HTTPError) as error:
                request(server, route, **kwargs)
            assert error.value.code == code
            assert 'error' in json.load(error.value)


def test_rejected_partial_body_has_a_bounded_drain(server):
    import socket
    import http.client
    with socket.create_connection(('127.0.0.1', server.server_port), timeout=2) as sock:
        sock.sendall((f'POST /api/preview HTTP/1.0\r\nHost: 127.0.0.1:{server.server_port}\r\n'
                      'Content-Length: 10\r\n\r\n{}').encode())
        response = http.client.HTTPResponse(sock)
        response.begin()
        assert response.status == 403
        assert 'error' in json.loads(response.read())


def test_interrupted_and_corrupt_video_uploads_allow_a_clean_retry(server, source):
    import socket
    import http.client
    from ivp.worker import process_video
    from ivp.jobs import read_status
    from ivp.media import probe_video
    from tests.test_video import request_run, stamps

    with socket.create_connection(('127.0.0.1', server.server_port), timeout=3) as sock:
        sock.sendall((f'POST /api/upload?name=interrupted.mp4 HTTP/1.0\r\n'
                      f'Host: 127.0.0.1:{server.server_port}\r\n'
                      f'X-Traffic-Token: {server.lab.token}\r\n'
                      'Content-Length: 1000\r\n\r\n').encode()+b'partial')
        sock.shutdown(socket.SHUT_WR)
        response = http.client.HTTPResponse(sock)
        response.begin()
        assert response.status == 400
        assert 'interrupted' in json.loads(response.read())['error'].lower()
    for suffix in ('mp4', 'mov', 'avi'):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server, '/api/upload?name=corrupt.'+suffix, raw=b'not a video')
        assert error.value.code == 400
        assert 'H.264' in json.load(error.value)['error']
    assert not server.lab.media
    assert not list((web_server.ROOT/'uploads').rglob('*.*')), 'Rejected upload copies leaked'
    assert request(server, '/api/health')['app'] == 'traffic-lab-web'

    media = request(server, '/api/upload?name=retry.mp4', raw=source.read_bytes())
    payload = {'media':media['id'], 'settings':{}, 'timestamp':0}
    assert request(server, '/api/preview', payload)['size'] == [160,120]
    exported = request(server, '/api/export', {**payload,'masks':['Edges']})
    assert exported['status']['state'] == 'complete'
    with urllib.request.urlopen(exported['preview']['url']) as image:
        assert Image.open(image).size == (160,120)
    uploaded = server.lab.media[media['id']][0]
    assert uploaded.read_bytes() == source.read_bytes()
    run = request_run(web_server.ROOT, uploaded)
    process_video(run)
    assert read_status(run)['state'] == 'complete'
    assert probe_video(run/'processed.mp4')['has_audio']
    assert stamps(run/'processed.mp4') == pytest.approx(stamps(source), abs=2/90000)


def test_terminated_worker_recovers_status_and_retries_in_a_new_run(server, source, monkeypatch):
    import os
    import subprocess
    import sys
    import time
    import hashlib
    from pathlib import Path
    from datetime import datetime, timedelta
    from ivp import jobs
    from ivp.media import hidden_process_options, probe_video, write_json
    from tests.test_video import request_run, stamps

    isolated_runs = web_server.ROOT/'runs'
    isolated_runs.mkdir()
    monkeypatch.setattr(jobs, 'RUNS', isolated_runs)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stopped = request_run(isolated_runs, source)
    child_code = '''
import os,sys,time
from pathlib import Path
from ivp import jobs
from ivp.media import write_json
from ivp.worker import process_video
jobs.RUNS=Path(sys.argv[2])
run=Path(sys.argv[1])
def gate(result,rgb,index,distance):
    if index==5:
        write_json(run/'termination-ready',{'pid':os.getpid(),'checkpoint':'encoder initialized; five frames processed'})
        while True:time.sleep(.1)
    return result.color_rgb
with jobs.job_lock():
    process_video(run,frame_hook=gate if sys.argv[3]=='pause' else None)
'''
    command = [sys.executable,'-c',child_code,str(stopped),str(isolated_runs),'pause']
    worker_handle = None
    lifecycle = []
    with (stopped/'worker.log').open('wb') as log:
        child = subprocess.Popen(command, stdout=log, stderr=log, **hidden_process_options())
        try:
            deadline = time.monotonic()+20
            while not (stopped/'termination-ready').exists():
                assert child.poll() is None, (stopped/'worker.log').read_text()
                assert time.monotonic()<deadline, 'Worker did not reach encoding checkpoint'
                time.sleep(.05)
            worker_pid = json.loads((stopped/'termination-ready').read_text())['pid']
            lifecycle.append({'event':'worker_ready','at_ns':time.perf_counter_ns(),
                              'launcher_pid':child.pid,'worker_pid':worker_pid})
            if os.name == 'nt':
                import ctypes
                from ctypes import wintypes
                kernel = ctypes.WinDLL('kernel32', use_last_error=True)
                kernel.OpenProcess.argtypes = (wintypes.DWORD,wintypes.BOOL,wintypes.DWORD)
                kernel.OpenProcess.restype = wintypes.HANDLE
                kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE,wintypes.DWORD)
                kernel.WaitForSingleObject.restype = wintypes.DWORD
                kernel.TerminateProcess.argtypes = (wintypes.HANDLE,wintypes.UINT)
                kernel.TerminateProcess.restype = wintypes.BOOL
                kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
                kernel.CloseHandle.restype = wintypes.BOOL
                # Pin this owned process while it is alive, before terminating
                # its venv launcher. Waiting for the launcher alone can return
                # before the actual interpreter exits and releases its lock.
                worker_handle = kernel.OpenProcess(0x100001,False,worker_pid)
                assert worker_handle, ctypes.get_last_error()
                assert kernel.WaitForSingleObject(worker_handle,0) == 258
            with pytest.raises(RuntimeError, match='Another video job'):
                with jobs.job_lock():
                    pass
            assert jobs.read_status(stopped)['state'] == 'running'
            child.terminate()  # Only this test's owned child; no app PID lookup.
            child.wait(timeout=10)
            lifecycle.append({'event':'launcher_wait_returned','at_ns':time.perf_counter_ns()})
            if worker_handle:
                assert kernel.WaitForSingleObject(worker_handle,10000) == 0, 'Owned worker has not exited.'
            lifecycle.append({'event':'worker_exit_observed','at_ns':time.perf_counter_ns()})
            with jobs.job_lock():
                lifecycle.append({'event':'lock_reacquired','at_ns':time.perf_counter_ns()})
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
            if worker_handle:
                try:
                    if kernel.WaitForSingleObject(worker_handle,0) != 0:
                        assert kernel.TerminateProcess(worker_handle,1)
                        assert kernel.WaitForSingleObject(worker_handle,5000) == 0
                finally:
                    kernel.CloseHandle(worker_handle)
    assert child.returncode != 0
    # libx264 can buffer every packet of a short clip until encoder flush;
    # a killed process need not have created the partial MP4 yet.
    diagnostic_files = ['coordinates.json','measurements.csv','request.json']
    if (stopped/'video.partial.mp4').exists():
        diagnostic_files.append('video.partial.mp4')
    diagnostics = {name:hashlib.sha256((stopped/name).read_bytes()).hexdigest() for name in diagnostic_files}
    assert not (stopped/'processed.mp4').exists()
    assert jobs.refreshed_status(stopped)['state'] == 'running'  # Grace period.

    class AfterGrace(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz)+timedelta(seconds=40)

    with monkeypatch.context() as clock:
        clock.setattr(jobs, 'datetime', AfterGrace)
        state = request(server, '/api/run?id='+stopped.name)
    assert state['status']['state'] == 'interrupted'
    assert state['downloads'] == [] and state['preview'] is None
    assert diagnostics == {name:hashlib.sha256((stopped/name).read_bytes()).hexdigest() for name in diagnostic_files}

    retry = request_run(isolated_runs, source)
    with (retry/'worker.log').open('wb') as log:
        result = subprocess.run([sys.executable,'-c',child_code,str(retry),str(isolated_runs),'run'],
                                stdout=log,stderr=log,timeout=20,**hidden_process_options())
    assert result.returncode == 0, (retry/'worker.log').read_text()
    state = request(server, '/api/run?id='+retry.name)
    assert state['status']['state'] == 'complete' and state['preview']['video']
    assert state['downloads']
    with urllib.request.urlopen(state['preview']['url']) as response:
        served = response.read()
    assert served == (retry/'processed.mp4').read_bytes()
    assert probe_video(retry/'processed.mp4')['has_audio']
    assert stamps(retry/'processed.mp4') == pytest.approx(stamps(source), abs=2/90000)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    write_json(web_server.ROOT/'termination-validation.json', {
        'checked_at_utc':jobs.now(),'terminated_owned_pid':child.pid,'returncode':child.returncode,
        'interrupted_run':str(stopped),'retry_run':str(retry),'diagnostic_sha256':diagnostics,
        'worker_pid':worker_pid,'lifecycle':lifecycle,'actual_worker_exit_observed':True,
        'retry_served_bytes_match':True,'retry_audio_and_timestamps_preserved':True,
        'original_source_unchanged':True,'grace_period':'Clock advanced 40 seconds in parent test only; process termination and file locking are real.',
        'scope':'Isolated local test server and owned worker processes; no live demo process or run modified.'})
