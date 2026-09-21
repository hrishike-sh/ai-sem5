"""Primary API lifecycle checks; model responses are mocked unless noted."""
import hashlib
import json
import threading
import time
from types import SimpleNamespace
import urllib.error

from PIL import Image
import pytest

from ivp import jobs
from ivp.media import write_json
from ivp.pipeline import Settings
from ivp.scene_reviews import SceneReviews
from tests.test_web_server import request, server as server
from traffic_ai import bounded_vision, local_vision


@pytest.fixture
def scene(server, tmp_path):
    source=tmp_path/'source.png'
    image=Image.new('RGB',(80,60),(35,70,105))
    image.paste((90,80,70),(40,0,80,60))
    image.save(source)
    return source,server.lab.register(source)


def start(server, media):
    return request(server,'/api/review',{'media':media['id'],'timestamp':0,'settings':{}})


def finished(server, attempt):
    server.lab.scene_reviews.future.result(timeout=10)
    return request(server,'/api/review?id='+attempt['id'])


def complete(rgb, stamp, **options):
    return {'kind':'local_vision_review','state':'complete','timestamp':stamp,'total_seconds':.1,
            'observation':{'summary':'Unverified scene feature categories: car.','visible_features':['car'],
                           'sign_text_or_symbols':[],'uncertainties':['Synthetic test response; no model called.']}}


def test_scene_success_source_binding_and_failed_retry_preserve_previous_result(server,scene,monkeypatch):
    source,media=scene
    original=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(local_vision,'review_frame',complete)
    first=finished(server,start(server,media))
    assert first['state']=='complete' and first['media_id']==media['id']
    assert first['source_revision']==media['source_revision'] and first['timestamp']==0
    saved=server.lab.scene_reviews.folder(first['id'])/'review.json'
    previous=saved.read_bytes()
    def unavailable(*args,**kwargs):
        raise bounded_vision.VisionUnavailable('Synthetic timeout; no model called')
    monkeypatch.setattr(local_vision,'review_frame',unavailable)
    second=finished(server,start(server,media))
    assert second['id']!=first['id'] and second['state']=='unavailable' and 'review' not in second
    assert request(server,'/api/review?id='+first['id'])['review']==first['review']
    assert saved.read_bytes()==previous and hashlib.sha256(source.read_bytes()).hexdigest()==original
    restored=SceneReviews(server.lab.scene_reviews.root.parents[1],server.lab.executor)
    assert restored.get(first['id'])['review']==first['review']


def test_scene_cancel_and_ivp_export_remain_responsive_without_worker_lock(server,scene,tmp_path,monkeypatch):
    source,media=scene
    entered,release=threading.Event(),threading.Event()
    def waiting(rgb, stamp, *, cancel, **kwargs):
        entered.set()
        while not release.wait(.01):
            if cancel():
                error=bounded_vision.VisionCancelled('Cancelled test provider')
                error.record={'state':'cancelled','observation':{'must':'discard'}}
                raise error
        return complete(rgb,stamp)
    monkeypatch.setattr(local_vision,'review_frame',waiting)
    monkeypatch.setattr(jobs,'RUNS',tmp_path/'runs')
    attempt=start(server,media)
    try:
        assert entered.wait(3)
        # Acquiring the actual OS lock proves the scene provider does not hold it.
        with jobs.job_lock():pass
        preview=request(server,'/api/preview',{'media':media['id'],'timestamp':0,'settings':{}})
        assert preview['size']==[80,60]
        exported=request(server,'/api/export',{'media':media['id'],'timestamp':0,'settings':{},'masks':[]})
        assert exported['status']['state']=='complete'
        with pytest.raises(urllib.error.HTTPError):start(server,media)
        with pytest.raises(urllib.error.HTTPError) as busy:
            request(server,'/api/evidence-review',{'source_run':'not-started'})
        assert 'scene review' in json.load(busy.value)['error']
        monkeypatch.setattr(jobs.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=987654321))
        video=jobs.start_video(tmp_path/'placeholder.mp4',{'kind':'video'},Settings(),[])
        assert jobs.read_status(video)['state']=='queued'
        before=time.monotonic()
        result=request(server,'/api/review/cancel',{'id':attempt['id']})
        assert time.monotonic()-before<1 and result['state'] in ('cancelling','cancelled')
        final=finished(server,attempt)
        assert final['state']=='cancelled' and 'review' not in final
        record=json.loads((server.lab.scene_reviews.folder(attempt['id'])/'attempt.json').read_text(encoding='utf8'))
        assert 'observation' not in record
        assert source.is_file() and (tmp_path/'runs/test-run/status.json').is_file()
        assert jobs.read_status(video)['state']=='queued' and not (video/'cancel.request').exists()
        assert request(server,'/api/cancel',{'id':video.name})['message']=='Cancellation requested.'
        assert (video/'cancel.request').is_file()
    finally:release.set()


def test_missing_cached_model_uses_real_unavailable_path_without_launch_or_download(server,scene,tmp_path,monkeypatch):
    _,media=scene
    monkeypatch.setattr(bounded_vision,'RuntimeSpec',lambda:SimpleNamespace(
        binary=tmp_path/'absent.exe',model=tmp_path/'absent.gguf',projector=tmp_path/'absent-projector.gguf'))
    def forbidden(*args,**kwargs):raise AssertionError('Missing cache must never launch a process')
    monkeypatch.setattr(bounded_vision.subprocess,'Popen',forbidden)
    result=finished(server,start(server,media))
    assert result['state']=='unavailable' and 'missing' in result['message'] and 'review' not in result
    folder=server.lab.scene_reviews.folder(result['id'])
    record=json.loads((folder/'model/review.json').read_text(encoding='utf8'))
    assert record['state']=='unavailable' and record['runtime']['owned_server_stopped']
    assert 'observation' not in record


def test_cancelling_a_completed_older_overview_does_not_cancel_current_request(server,scene,monkeypatch):
    _,media=scene
    monkeypatch.setattr(local_vision,'review_frame',complete)
    old=finished(server,start(server,media))
    entered,release=threading.Event(),threading.Event()
    def waiting(rgb,stamp,**kwargs):entered.set();assert release.wait(5);return complete(rgb,stamp)
    monkeypatch.setattr(local_vision,'review_frame',waiting)
    current=start(server,media)
    try:
        assert entered.wait(3)
        assert request(server,'/api/review/cancel',{'id':old['id']})['state']=='complete'
        assert not server.lab.scene_reviews.cancel_event.is_set()
        assert request(server,'/api/review?id='+current['id'])['state']=='running'
    finally:release.set()
    assert finished(server,current)['state']=='complete'


def test_restart_marks_only_unfinished_overview_interrupted(server,scene,monkeypatch):
    _,media=scene
    monkeypatch.setattr(local_vision,'review_frame',complete)
    saved=finished(server,start(server,media))
    folder=server.lab.scene_reviews.folder(saved['id'])
    record=(folder/'review.json').read_bytes()
    write_json(folder/'status.json',{'state':'running'})
    restored=SceneReviews(folder.parents[2],server.lab.executor)
    assert restored.get(saved['id'])['state']=='interrupted'
    assert (folder/'review.json').read_bytes()==record


def test_changed_source_requires_reopen_and_invalid_cancel_ids_are_rejected(server,scene):
    source,media=scene
    Image.new('RGB',(81,61),(200,100,80)).save(source)
    with pytest.raises(urllib.error.HTTPError) as error:start(server,media)
    assert error.value.code==400 and 'changed' in json.load(error.value)['error']
    for identifier in ('../other','a'*24,None):
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server,'/api/review/cancel',{'id':identifier})
        assert error.value.code==400


def test_scene_shutdown_requests_cancellation_only_of_its_own_request(server,scene,monkeypatch):
    _,media=scene
    entered=threading.Event()
    def waiting(rgb,stamp,*,cancel,**kwargs):
        entered.set()
        deadline=time.monotonic()+3
        while not cancel() and time.monotonic()<deadline:time.sleep(.01)
        assert cancel()
        raise bounded_vision.VisionCancelled('Owned callback cancelled')
    monkeypatch.setattr(local_vision,'review_frame',waiting)
    attempt=start(server,media)
    assert entered.wait(2)
    server.lab.scene_reviews.close()
    assert finished(server,attempt)['state']=='cancelled'


def test_active_evidence_review_rejects_scene_queue(server,scene,monkeypatch):
    _,media=scene
    monkeypatch.setattr(server.lab.evidence_reviews,'future',SimpleNamespace(done=lambda:False))
    with pytest.raises(urllib.error.HTTPError) as error:start(server,media)
    assert error.value.code==400 and 'evidence review' in json.load(error.value)['error']
    assert server.lab.scene_reviews.future is None


@pytest.mark.parametrize('broken',[
    b'{broken', b'[]', b' '* (2*1024*1024+1), b'{}',
    json.dumps({**complete(None,0),'state':'unavailable'}).encode(),
    json.dumps({**complete(None,0),'observation':{}}).encode(),
    json.dumps({**complete(None,0),'timestamp':1}).encode(),
    json.dumps({**complete(None,0),'total_seconds':'forty'}).encode(),
    json.dumps({**complete(None,0),'total_seconds':float('nan')}).encode(),
    json.dumps({**complete(None,0),'total_seconds':10**400}).encode(),
],ids=['malformed','wrong-type','oversized','empty-object','not-complete','invalid-observation',
       'wrong-timestamp','invalid-duration','nonfinite-duration','overflow-duration'])
def test_damaged_or_oversized_overview_never_breaks_another_saved_result(server,scene,monkeypatch,broken):
    source,media=scene
    monkeypatch.setattr(local_vision,'review_frame',complete)
    good=finished(server,start(server,media))
    bad=finished(server,start(server,media))
    (server.lab.scene_reviews.folder(bad['id'])/'review.json').write_bytes(broken)
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,'/api/review?id='+bad['id'])
    assert error.value.code==400 and 'unreadable' in json.load(error.value)['error']
    assert request(server,'/api/review?id='+good['id'])['review']==good['review']
    assert source.is_file()


def test_invalid_completed_provider_record_never_replaces_previous_overview(server,scene,monkeypatch):
    source,media=scene
    monkeypatch.setattr(local_vision,'review_frame',complete)
    good=finished(server,start(server,media))
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(local_vision,'review_frame',lambda *a,**k:{**complete(None,0),'observation':{}})
    bad=finished(server,start(server,media))
    assert bad['state']=='unavailable' and 'review' not in bad
    assert not (server.lab.scene_reviews.folder(bad['id'])/'review.json').exists()
    assert request(server,'/api/review?id='+good['id'])['review']==good['review']
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash


def test_bounded_category_text_mismatch_is_rejected_without_changing_previous_result(server,scene,monkeypatch):
    _,media=scene
    selection={'visible_features':['car','road'],'uncertainty_codes':[]}
    def selected(rgb,stamp,**options):
        return {**complete(rgb,stamp),'provider':'local_llama_cpp_cpu','selection':selection,
                'observation':bounded_vision.validate_selection(selection)}
    monkeypatch.setattr(local_vision,'review_frame',selected)
    good=finished(server,start(server,media));bad=finished(server,start(server,media))
    folder=server.lab.scene_reviews.folder(bad['id']);record=json.loads((folder/'review.json').read_text(encoding='utf8'))
    record['observation'].update(summary='Unverified scene feature categories: bus.',visible_features=['bus'])
    (folder/'review.json').write_text(json.dumps(record),encoding='utf8')
    with pytest.raises(urllib.error.HTTPError) as error:request(server,'/api/review?id='+bad['id'])
    assert error.value.code==400
    assert request(server,'/api/review?id='+good['id'])['review']==good['review']


def test_inconsistent_bounded_provider_response_is_never_published(server,scene,monkeypatch):
    _,media=scene
    def inconsistent(rgb,stamp,**options):
        return {**complete(rgb,stamp),'provider':'local_llama_cpp_cpu',
                'selection':{'visible_features':['car'],'uncertainty_codes':[]}}
    monkeypatch.setattr(local_vision,'review_frame',inconsistent)
    bad=finished(server,start(server,media))
    assert bad['state']=='unavailable' and 'review' not in bad
    assert not (server.lab.scene_reviews.folder(bad['id'])/'review.json').exists()
