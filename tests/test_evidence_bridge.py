import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from ivp.media import write_json
from tests.test_web_server import server, request


@pytest.fixture
def evidence_source(server, tmp_path):
    folder=tmp_path/'runs'/'evidence-source'
    folder.mkdir(parents=True)
    event={'event_id':'incident-000001','timestamp_seconds':2.,'track_id':1,
           'type':'line_crossing','status':'withheld','reason':'Unknown condition <script>bad()</script>',
           'synthetic':True,'source_revision':'fixture-revision','calibration':None,
           'configuration':{'condition':'unknown','synthetic':True},
           'measurements':{'speed':12.,'speed_unit':'px/s','condition':'unknown','future_field':{'kept':True}},
           'quality':{'tracking_state':'observed','detector_confidence':.9,'speed_status':'estimated'}}
    (folder/'motion-incidents.jsonl').write_text(json.dumps(event)+'\n',encoding='utf8')
    write_json(folder/'motion-incidents.json',{'event_count':1,'counts':{'withheld':1},'events':[event],'omitted_from_index':0})
    write_json(folder/'status.json',{'state':'complete'})
    write_json(folder/'report.json',{'source':'fixture.mp4','kind':'video'})
    return folder,event


def fingerprints(folder):
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def finished(server, result):
    server.lab.evidence_reviews.future.result(timeout=10)
    return request(server,'/api/evidence-review?id='+result['id'])


def test_review_routes_preserve_originals_and_reopen_separate_artifacts(server,evidence_source,monkeypatch):
    source,event=evidence_source
    before=fingerprints(source)
    def no_network(*args,**kwargs):
        raise AssertionError('Offline review must not call a model or network service')
    # Exercise the server route directly while the global outbound spy is active;
    # the later HTTP calls/downloads use their normal loopback transport.
    with monkeypatch.context() as spy:
        spy.setattr(urllib.request.OpenerDirector,'open',no_network)
        result=server.lab.post('/api/evidence-review',{'source_run':source.name})
        server.lab.evidence_reviews.future.result(timeout=10)
    ready=request(server,'/api/evidence-review?id='+result['id'])
    assert ready['state']=='complete' and ready['source_integrity']['state']=='verified'
    assert ready['review']['events'][0]['event']==event
    assert ready['model_states']=={'not_requested':1}
    assert ready['review']['grounding']['state']=='source_citations_checked'
    assert ready['review']['grounding']['event_count']==1
    context=next(f for f in ready['review']['events'][0]['facts'] if f['id']=='metric_context')
    refs={r['pointer']:r for r in context['citations']}
    assert refs['/configuration/calibration_confirmed']['recorded'] is False
    assert refs['/calibration']=={'pointer':'/calibration','recorded':True,'value':None}
    assert 'calibration confirmed=unknown (not recorded)' in context['text']
    assert ready['review']['summary']['synthetic_events']==1
    assert before==fingerprints(source)
    assert ready['downloads'] and (source.parent/'evidence-reviews'/ready['id']/'offline'/'review.json').is_file()
    saved=request(server,'/api/evidence-review?source='+source.name)
    assert saved['id']==ready['id']
    link=next(d['url'] for d in ready['downloads'] if d['name']=='review.json')
    with urllib.request.urlopen(link,timeout=5) as response:
        assert 'attachment' in response.headers['Content-Disposition']
        full=json.load(response)
    assert full['events'][0]['event']==event


def test_optional_model_failure_leaves_offline_review_available_and_does_not_block_video(server,evidence_source,monkeypatch,tmp_path):
    from traffic_ai.review_llm import LocalReviewSelector
    from ivp import jobs
    from ivp.pipeline import Settings
    source,event=evidence_source
    entered,release=threading.Event(),threading.Event()
    def unavailable(self,review):
        entered.set()
        assert release.wait(5)
        raise OSError('Local model missing')
    monkeypatch.setattr(LocalReviewSelector,'select',unavailable)
    result=request(server,'/api/evidence-review',{'source_run':source.name,'use_model':True})
    try:
        assert entered.wait(5)
        interim=request(server,'/api/evidence-review?id='+result['id'])
        assert interim['state']=='running' and interim['phase']=='local_model'
        assert interim['review']['events'][0]['event']==event and interim['downloads']
        # A real processing-job launch check must ignore these nested review
        # statuses; a review never holds the processing worker's OS lock.
        monkeypatch.setattr(jobs,'ROOT',tmp_path)
        monkeypatch.setattr(jobs,'RUNS',tmp_path/'runs')
        monkeypatch.setattr(jobs.subprocess,'Popen',lambda *a,**k:SimpleNamespace(pid=987654321))
        video=jobs.start_video(tmp_path/'placeholder.mp4',{'kind':'video'},Settings(),[])
        assert json.loads((video/'status.json').read_text(encoding='utf8'))['state']=='queued'
        cancelled=request(server,'/api/cancel',{'id':video.name})
        assert cancelled['message']=='Cancellation requested.' and (video/'cancel.request').is_file()
        assert not release.is_set()  # Cancel responded while model work still waited.
        with pytest.raises(urllib.error.HTTPError) as error:
            request(server,'/api/evidence-review',{'source_run':source.name,'use_model':False})
        assert error.value.code==400
        assert 'already running' in json.load(error.value)['error']
    finally:
        release.set()
    ready=finished(server,result)
    assert ready['state']=='complete' and ready['model_states']['unavailable']==1
    assert ready['review']['events'][0]['event']==event
    assert ready['review']['events'][0]['language_model']['state']=='unavailable'


@pytest.mark.parametrize('payload',[{'source_run':'../outside'},{'source_run':'evidence-source','use_model':'yes'}])
def test_review_request_rejects_unsafe_identifiers_and_flags(server,evidence_source,payload):
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,'/api/evidence-review',payload)
    assert error.value.code==400


def test_active_or_corrupt_sources_never_publish_a_successful_review(server,evidence_source):
    source,_=evidence_source
    write_json(source/'status.json',{'state':'running'})
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server,'/api/evidence-review',{'source_run':source.name})
    assert error.value.code==400
    write_json(source/'status.json',{'state':'complete'})
    (source/'motion-incidents.jsonl').write_text('{broken\n',encoding='utf8')
    before=fingerprints(source)
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    assert ready['state']=='failed' and not ready['downloads'] and 'review' not in ready
    assert before==fingerprints(source)
    assert request(server,'/api/evidence-review?source='+source.name)['state']=='none'


def test_changed_source_warns_without_rewriting_the_saved_review(server,evidence_source):
    import os
    source,event=evidence_source
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    path=source/'report.json'
    stat=path.stat()
    old=path.read_bytes()
    path.write_bytes(old.replace(b'fixture',b'changed'))  # Same byte count.
    os.utime(path,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    changed=request(server,'/api/evidence-review?id='+ready['id'])
    assert changed['source_integrity']['state']=='changed'
    assert changed['review']['events'][0]['event']==event


def test_restart_retains_completed_offline_review_when_optional_focus_was_interrupted(server,evidence_source,tmp_path):
    from ivp.evidence_reviews import EvidenceReviews
    source,event=evidence_source
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    job=source.parent/'evidence-reviews'/ready['id']
    write_json(job/'status.json',{'state':'running','phase':'local_model'})
    restored=EvidenceReviews(tmp_path,server.lab.executor,server.lab.files).get(ready['id'])
    assert restored['state']=='interrupted' and restored['downloads']
    assert restored['review']['events'][0]['event']==event


@pytest.mark.parametrize('problem',['changed_json','missing_marker'])
def test_changed_or_partial_review_bundle_is_not_advertised_as_intact(server,evidence_source,problem):
    source,_=evidence_source
    original=fingerprints(source)
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    assert ready['artifact_integrity']['state']=='verified'
    artifact=source.parent/'evidence-reviews'/ready['id']/'offline'
    if problem=='changed_json':
        with (artifact/'review.json').open('ab') as stream:stream.write(b'\n')
    else:
        (artifact/'review-complete.json').unlink()
    changed=request(server,'/api/evidence-review?id='+ready['id'])
    assert changed['artifact_integrity']['state']=='changed'
    assert not changed['downloads'] and 'review' not in changed
    assert original==fingerprints(source)
    retried=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    assert retried['id']!=ready['id'] and retried['artifact_integrity']['state']=='verified'


def test_native_view_is_derived_from_validated_full_review_not_edited_cache(server,evidence_source):
    source,event=evidence_source
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    path=source.parent/'evidence-reviews'/ready['id']/'view.json'
    cached=json.loads(path.read_text(encoding='utf8'))
    cached['events'][0]['event']['status']='candidate'
    write_json(path,cached)
    restored=request(server,'/api/evidence-review?id='+ready['id'])
    assert restored['artifact_integrity']['state']=='verified'
    assert restored['review']['events'][0]['event']==event


def test_legacy_snapshot_remains_inspectable_without_claiming_verified_downloads(server,evidence_source):
    source,event=evidence_source
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    job=source.parent/'evidence-reviews'/ready['id']
    cached=json.loads((job/'view.json').read_text(encoding='utf8'))
    cached.pop('artifact_integrity_version')
    write_json(job/'view.json',cached)
    (job/'offline/review-complete.json').unlink()
    restored=request(server,'/api/evidence-review?id='+ready['id'])
    assert restored['artifact_integrity']['state']=='legacy' and not restored['downloads']
    assert restored['review']['events'][0]['event']==event


def test_public_stop_context_is_bounded_while_full_download_keeps_every_sample(server,evidence_source):
    source,event=evidence_source
    event['calibration']={'points':[[.1,.1],[.9,.1],[.9,.9],[.1,.9]],'width_m':16,'length_m':40,'source_revision':event['source_revision']}
    event['configuration'].update(condition='stop_required',calibration_confirmed=True,real_time_confirmed=True)
    event['measurements'].update(condition='stop_required',stopped_in_approach=True,speed_unit='km/h',stop_evidence={
        'start_seconds':.5,'end_seconds':1.5,'duration_seconds':1.,'observed_samples':8,
        'position_diameter_bound_m':.1,'maximum_position_diameter_m':2/3.6,
        'fitted_speed_kmh':.1,'maximum_fitted_speed_kmh':2.,'method':'Generated recorded-window regression'})
    (source/'motion-incidents.jsonl').write_text(json.dumps(event)+'\n',encoding='utf8')
    write_json(source/'motion-incidents.json',{'event_count':1,'counts':{'withheld':1},'events':[event],'omitted_from_index':0})
    (source/'motion-tracks.csv').write_text('timestamp_seconds,track_id,speed,speed_unit,speed_status,tracking_state\n'+
        ''.join(f'{.5+i/7},1,.1,km/h,estimated,observed\n' for i in range(8)),encoding='utf8')
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    card=ready['review']['events'][0]
    assert card['stop_review']['csv_coverage']=='count_and_states_match'
    assert 'samples' not in card['stop_context'] and card['stop_context']['sample_count']==8
    assert {f['id'] for f in card['facts']} >= {'stop_duration','stop_samples','stop_fitted_speed','stop_displacement','stop_context'}
    link=next(d['url'] for d in ready['downloads'] if d['name']=='review.json')
    with urllib.request.urlopen(link,timeout=5) as response:full=json.load(response)
    assert len(full['events'][0]['stop_context']['samples'])==8


def test_failed_report_write_can_retry_without_changing_source(server,evidence_source,monkeypatch):
    from traffic_ai import review_cli
    source,_=evidence_source
    before=fingerprints(source)
    original=review_cli.export_review
    def failed(*args,**kwargs):raise OSError('Simulated output write failure')
    monkeypatch.setattr(review_cli,'export_review',failed)
    bad=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    assert bad['state']=='failed' and not bad['downloads']
    monkeypatch.setattr(review_cli,'export_review',original)
    good=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    assert good['state']=='complete' and good['artifact_integrity']['state']=='verified'
    assert good['id']!=bad['id'] and fingerprints(source)==before


@pytest.mark.parametrize('failure',['thread_interrupted','status_write_failed'])
def test_finished_review_future_recovers_in_same_server_retaining_offline_and_retry(server,evidence_source,monkeypatch,failure):
    source,event=evidence_source
    before=fingerprints(source)
    service=server.lab.evidence_reviews
    previous=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    previous_folder=source.parent/'evidence-reviews'/previous['id']/'offline'
    previous_files=fingerprints(previous_folder)
    original=service.state
    exception=KeyboardInterrupt if failure=='thread_interrupted' else OSError
    def interrupted(folder,state,message,**extra):
        if state=='complete' or state=='failed' and failure=='status_write_failed':
            raise exception('Injected final status publication interruption')
        return original(folder,state,message,**extra)
    with monkeypatch.context() as fault:
        fault.setattr(service,'state',interrupted)
        attempt=request(server,'/api/evidence-review',{'source_run':source.name})
        with pytest.raises(exception):service.future.result(timeout=10)
    assert service.future.done()
    recovered=request(server,'/api/evidence-review?id='+attempt['id'])
    assert recovered['state']=='interrupted' and recovered['downloads']
    assert recovered['review']['events'][0]['event']==event
    assert recovered['source_integrity']['state']==recovered['artifact_integrity']['state']=='verified'
    by_source=request(server,'/api/evidence-review?source='+source.name)
    assert by_source['id']==attempt['id'] and by_source['state']=='interrupted'
    retry=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    assert retry['state']=='complete' and retry['id']!=attempt['id']
    assert retry['review']['events'][0]['event']==event
    assert fingerprints(source)==before and fingerprints(previous_folder)==previous_files


def test_review_recovery_rereads_completion_after_initial_stale_status(server,evidence_source,monkeypatch):
    source,_=evidence_source
    service=server.lab.evidence_reviews
    ready=finished(server,request(server,'/api/evidence-review',{'source_run':source.name}))
    status=source.parent/'evidence-reviews'/ready['id']/'status.json'
    before=status.read_bytes();original=service.read;once=[False]
    def stale(path,*args,**kwargs):
        value=original(path,*args,**kwargs)
        if path==status and not once[0]:
            once[0]=True
            return {**value,'state':'running','phase':'offline'}
        return value
    monkeypatch.setattr(service,'read',stale)
    actual=request(server,'/api/evidence-review?id='+ready['id'])
    assert once[0] and actual['state']=='complete' and actual['review']
    assert status.read_bytes()==before
