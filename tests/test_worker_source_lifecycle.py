"""Video publication boundaries with trapped media/process constructors."""
import csv
import json
import os
import socket
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ivp.jobs as jobs
import ivp.worker as worker
import web_server
from ivp.media import source_revision, write_json
from ivp.pipeline import Settings


@pytest.fixture
def offline_video(tmp_path, monkeypatch):
    source = tmp_path/'SYNTHETIC-NOT-A-VIDEO.bin'
    original_bytes = b'ORIGINAL-STUB-SOURCE'
    source.write_bytes(original_bytes)
    run = tmp_path/'runs'/'offline-stub'
    run.mkdir(parents=True)
    trace = SimpleNamespace(mux_sources=[], packets=[], terminated=False, waited=False,
                            encoder_closed=False, decode_calls=0, encoder_calls=0)
    control = SimpleNamespace(phase=None, action=None, mutated=False, fail=None)
    revision = source_revision(source)
    info = {'kind':'video','fps':10,'duration':.3,'width':12,'height':8,'has_audio':True,
            'source_revision':revision}

    def request(legacy=False):
        metadata = {k:v for k,v in info.items() if k!='source_revision' or not legacy}
        write_json(run/'request.json', {'source':str(source),'info':metadata,
            'settings':Settings().to_dict(),'selected_masks':['Edges']})
        return metadata

    def mutate(phase):
        if control.mutated or phase != control.phase:
            return
        control.mutated = True
        if 'replace' in control.action or control.action == 'inplace':
            stamp = source.stat().st_mtime_ns + 2_000_000_000
            replacement_bytes = b'REPLACED-STUB-SOURCE'
            assert len(replacement_bytes)==len(original_bytes)
            if control.action == 'inplace':
                with source.open('r+b') as stream:
                    stream.write(replacement_bytes)
            else:
                pending=source.with_suffix('.replacement')
                pending.write_bytes(replacement_bytes)
                pending.replace(source)
            os.utime(source, ns=(stamp,stamp))
        if 'remove' in control.action:
            source.unlink()
        if 'cancel' in control.action:
            (run/'cancel.request').touch()

    def frames(*args):
        trace.decode_calls += 1
        for index in range(3):
            # Coherent OLD-frame model; never claim an actual decoder snapshot.
            yield np.full((8,12,3),30+index,np.uint8), index/10,.1,4+index/10
        mutate('decode-drain')

    class Frame:
        @staticmethod
        def from_ndarray(array, format):
            return SimpleNamespace(pts=None,time_base=None,duration=None)

    class Stream:
        def __init__(self):
            self.codec_context=SimpleNamespace()

        def encode(self, frame=None):
            return [] if frame is None else [SimpleNamespace(pts=frame.pts,duration=None)]

    class Encoder:
        def add_stream(self,*args,**kwargs):
            return Stream()

        def mux(self,packet):
            trace.packets.append((packet.pts,packet.duration))

        def close(self):
            trace.encoder_closed=True

    def encoder(*args,**kwargs):
        trace.encoder_calls += 1
        if control.fail=='encoder':
            raise RuntimeError('Deliberate stub encoder failure')
        (run/'video.partial.mp4').write_text('STUB: not a video')
        return Encoder()

    class Mux:
        returncode=None

        def poll(self):
            if control.phase=='mux-wait' and not control.mutated:
                mutate('mux-wait')
                return None
            mutate('mux-end')
            self.returncode=1 if control.fail=='mux' else 0
            return self.returncode

        def terminate(self):
            trace.terminated=True

        def wait(self,timeout):
            trace.waited=True
            self.returncode=-15
            return self.returncode

        def kill(self):
            pytest.fail('Stub mux should terminate normally')

    def mux(command,**kwargs):
        mutate('mux-open')
        trace.mux_sources.append(source.read_bytes())
        trace.command=command
        (run/'processed.mp4').write_text('STUB: not a video')
        return Mux()

    def hook(result,rgb,index,distance):
        if index==0:
            mutate('first-frame')
        return result.color_rgb

    def finish():
        mutate('final-hook')
        return {'stub_only':True}

    def write(path,value):
        write_json(path,value)
        if Path(path)==run/'report.json':
            mutate('report-write')

    def forbidden(*args,**kwargs):
        pytest.fail('Real process/network/job/media constructor forbidden')

    for module,attr,replacement in [
        (worker,'video_frames',frames),(worker,'tool_path',lambda name:'STUB-NOT-EXECUTED'),
        (worker.av,'open',encoder),(worker.av,'VideoFrame',Frame),
        (worker.subprocess,'Popen',mux),(worker.subprocess,'run',forbidden),
        (worker,'job_lock',forbidden),(jobs,'new_run',forbidden),(jobs,'start_video',forbidden),
        (socket,'socket',forbidden),(worker.cv2,'VideoCapture',forbidden),
        (worker,'write_json',write),(worker.time,'sleep',lambda seconds:None),
    ]:
        monkeypatch.setattr(module,attr,replacement)

    def start():
        mutate('before')
        return worker.process_video(run,frame_hook=hook,final_hook=finish,final_key='offline_stub')

    def publication():
        monkeypatch.setattr(web_server,'ROOT',tmp_path)
        monkeypatch.setattr(web_server,'FileServer',forbidden)
        lab=web_server.Lab.__new__(web_server.Lab)
        lab.files=SimpleNamespace(url=lambda file,*args:'/OFFLINE-NOT-SERVED/'+Path(file).name)
        return lab.run(run.name)

    return SimpleNamespace(run=run,source=source,original_bytes=original_bytes,revision=revision,
        trace=trace,control=control,request=request,start=start,publication=publication)


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('phase',['first-frame','decode-drain','mux-open','mux-end','final-hook','report-write'])
def test_replaced_source_cannot_publish_complete(offline_video,legacy,phase):
    c=offline_video
    c.request(legacy)
    c.control.phase,c.control.action=phase,'replace'
    with pytest.raises(ValueError,match='Source video.*[Rr]eopen'):
        c.start()
    assert json.loads((c.run/'status.json').read_text())['state']=='failed'
    view=c.publication()
    assert view['preview'] is None and view['downloads']==[]
    assert c.trace.encoder_closed
    if phase in ('first-frame','decode-drain'):
        assert c.trace.mux_sources==[]


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('phase',['first-frame','final-hook'])
def test_inplace_edit_cannot_publish_old_revision(offline_video,legacy,phase):
    c=offline_video
    c.request(legacy)
    c.control.phase,c.control.action=phase,'inplace'
    with pytest.raises(ValueError,match='Source video.*[Rr]eopen'):
        c.start()
    assert c.publication()['status']['state']=='failed'
    assert c.publication()['downloads']==[]


@pytest.mark.parametrize('legacy',[False,True])
@pytest.mark.parametrize('phase',['first-frame','mux-end','final-hook','report-write'])
def test_removed_source_has_actionable_failure_and_no_publication(offline_video,legacy,phase):
    c=offline_video
    c.request(legacy)
    c.control.phase,c.control.action=phase,'remove'
    with pytest.raises(ValueError,match='Source video.*[Rr]eopen'):
        c.start()
    view=c.publication()
    assert view['status']['state']=='failed' and view['downloads']==[] and view['preview'] is None


@pytest.mark.parametrize('legacy',[False,True])
def test_unchanged_source_retains_report_manifest_and_timing_contract(offline_video,legacy):
    c=offline_video
    metadata=c.request(legacy)
    c.start()
    report=json.loads((c.run/'report.json').read_text())
    assert report['source_info']==metadata
    assert report['frames']==3 and report['duration_seconds']==pytest.approx(.3)
    assert c.source.read_bytes()==c.original_bytes
    assert c.trace.mux_sources==[c.original_bytes]
    assert c.trace.packets==[(0,9000),(9000,9000),(18000,9000)]
    rows=list(csv.DictReader((c.run/'measurements.csv').open()))
    assert [float(r['timestamp_seconds']) for r in rows]==[0,.1,.2]
    assert all(float(r['duration_seconds'])==.1 for r in rows)
    command=c.trace.command
    assert command[command.index('-itsoffset')+1]=='-4.0'
    assert command[command.index('-t')+1]=='0.30000000000000004'
    assert command[command.index('-c:a')+1]=='aac'
    with zipfile.ZipFile(c.run/'masks.zip') as archive:
        manifest=json.loads(archive.read('manifest.json'))
        assert manifest['source_binding']['source_revision']==c.revision
        assert manifest['eligible_for_geometry'] is False and manifest['eligible_for_rules'] is False
    assert c.publication()['status']['state']=='complete'


@pytest.mark.parametrize('action',['cancel-replace','cancel-remove'])
@pytest.mark.parametrize('phase',['before','first-frame','decode-drain','mux-wait','mux-end','final-hook','report-write'])
def test_cancellation_takes_precedence_over_simultaneous_source_change(offline_video,action,phase):
    c=offline_video
    c.request()
    c.control.phase,c.control.action=phase,action
    c.start()
    view=c.publication()
    assert view['status']['state']=='cancelled' and view['preview'] is None and view['downloads']==[]
    assert not any((c.run/name).exists() for name in ('processed.mp4','video.partial.mp4','report.json','motion-evidence.zip'))
    if phase=='mux-wait':
        assert c.trace.terminated and c.trace.waited


@pytest.mark.parametrize('failure',['encoder','mux'])
def test_existing_encoder_mux_failures_remain_failed(offline_video,failure):
    c=offline_video
    c.request()
    c.control.fail=failure
    with pytest.raises(RuntimeError):
        c.start()
    view=c.publication()
    assert view['status']['state']=='failed' and view['preview'] is None and view['downloads']==[]
