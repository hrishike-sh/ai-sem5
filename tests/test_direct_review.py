"""Synthetic pixels/model selections only; no detector, credentials or network."""
import base64
import json
import os
import socket

import numpy as np
from PIL import Image
import pytest

from ivp.media import source_revision
from traffic_ai import direct_review as module
from traffic_ai.direct_vision import MODEL, PROVIDER
from traffic_ai.violation_images import image_descriptors, image_fingerprint


class Files:
    def __init__(self):
        self.paths = {}

    def url(self, path, download=False):
        value = '/download/' + path.name if download else '/preview/' + path.parent.name + '/' + path.name
        self.paths[value] = path
        return value


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Direct storage cannot connect to any network service.')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    root = tmp_path / 'project'
    source = root / 'uploads/synthetic.bmp'
    source.parent.mkdir(parents=True)
    Image.fromarray(np.full((32, 48, 3), 100, np.uint8)).save(source)
    files = Files()
    service = module.DirectReview(root, files)
    info = {'kind': 'image', 'source_revision': source_revision(source)}
    view = service.prepare(source, info, synthetic=True)
    data = {'review_id': view['review_id'], 'fingerprint': view['fingerprint']}
    return service, source, info, view, data, files


def result_for(service, data):
    images = service.load(data)['images']
    return {**data, 'state': 'reviewed', 'provider': PROVIDER, 'model': MODEL, 'mode': 'direct_images',
            'image_fingerprint': image_fingerprint(images), 'evidence': image_descriptors(images),
            'observations': [{'kind': 'head_protection_unclear', 'evidence_ids': [images[0]['id']],
                              'uncertainties': ['model_may_misread'], 'text': 'UNTRUSTED PROSE MUST NOT SURVIVE'}],
            'suspected_issues': [], 'limitations': ['UNTRUSTED LIMITATION']}


def test_direct_preview_is_exact_bounded_jpeg_with_no_detector_required(prepared):
    service, source, _, view, data, files = prepared
    original = source.read_bytes()
    loaded = service.load(data)
    frame, image = view['frames'][0], loaded['images'][0]
    assert frame['available'] is True and frame['position'] == 'selected'
    assert frame['timestamp'] == 0 and frame['role'] == 'context'
    assert base64.b64decode(image['data_url'].split(',')[1]) == files.paths[frame['url']].read_bytes()
    assert view['synthetic'] is True and view['model_result'] is None
    assert view['source_readiness']['available'] is True
    assert source.read_bytes() == original
    assert not (service.root / 'runs').exists()


def test_result_and_human_disposition_persist_separately_and_export(prepared):
    service, source, _, _, data, files = prepared
    original = source.read_bytes()
    saved = service.save_result(data, result_for(service, data))
    assert 'UNTRUSTED' not in json.dumps(saved['model_result'])
    model_result = saved['model_result']
    for action in ('accept', 'reject', 'reset'):
        result = service.decide({**data, 'decision': action, 'note': '<reviewer note>'})
        assert result['model_result'] == model_result
        assert result['review']['decision'] == (None if action == 'reset' else action)
    reopened = module.DirectReview(service.root, files).view(data['review_id'])
    assert reopened['model_result'] == model_result
    exported = service.export(data)
    report = json.loads(files.paths[exported['url']].read_text())
    assert report['external_reporting'] is False and report['model_result'] == model_result
    assert report['synthetic'] is True and source.read_bytes() == original
    assert service.recent()[0]['review_id'] == data['review_id']


@pytest.mark.parametrize('change', ['source', 'same_revision_source', 'missing_source', 'jpeg', 'manifest'])
def test_changed_source_or_prepared_bytes_cannot_be_loaded_or_saved(prepared, change):
    service, source, _, view, data, files = prepared
    model_result = result_for(service, data)
    if change in ('source', 'same_revision_source'):
        stat = source.stat()
        raw = bytearray(source.read_bytes()); raw[-1] ^= 1; source.write_bytes(raw)
        if change == 'same_revision_source':
            os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif change == 'missing_source':
        source.unlink()
    elif change == 'jpeg':
        files.paths[view['frames'][0]['url']].write_bytes(b'not a JPEG')
    else:
        path = service.root / 'direct-reviews' / data['review_id'] / 'manifest.json'
        manifest = json.loads(path.read_text()); manifest['requested_timestamp'] = 2; path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        service.load(data)
    with pytest.raises((ValueError, OSError)):
        service.save_result(data, model_result)
    if change in ('source', 'same_revision_source', 'missing_source'):
        assert service.view(data['review_id'])['source_readiness']['available'] is False


@pytest.mark.parametrize('change', ['model', 'state', 'fingerprint', 'review_id', 'image_fingerprint', 'evidence', 'citation'])
def test_mismatched_model_result_is_not_saved(prepared, change):
    service, _, _, _, data, _ = prepared
    result = result_for(service, data)
    if change == 'evidence':
        result['evidence'][0]['sha256'] = '0' * 64
    elif change == 'citation':
        result['observations'][0]['evidence_ids'] = ['other-frame']
    else:
        result[change] = 'wrong'
    with pytest.raises(ValueError):
        service.save_result(data, result)
    assert service.view(data['review_id'])['model_result'] is None


def test_video_preparation_uses_actual_decoded_timestamp(prepared, monkeypatch):
    service, source, _, _, _, _ = prepared
    calls = []
    def decode(path, info, timestamp):
        calls.append((path, timestamp))
        return np.full((20, 30, 3), 40, np.uint8), 1.3
    monkeypatch.setattr(module, 'preview_frame', decode)
    view = service.prepare(source, {'kind': 'video', 'duration': 3, 'source_revision': source_revision(source)}, timestamp=1.2)
    assert calls == [(source.resolve(), 1.2)]
    assert view['frames'][0]['timestamp'] == 1.3
    assert view['synthetic'] is None


def test_source_change_during_decode_is_rejected(prepared, monkeypatch):
    service, source, info, _, _, _ = prepared
    def decode(path):
        source.write_bytes(b'changed during decoding')
        return np.zeros((10, 10, 3), np.uint8)
    monkeypatch.setattr(module, 'load_image', decode)
    with pytest.raises(ValueError, match='source changed'):
        service.prepare(source, info)


@pytest.mark.parametrize('timestamp', [-1, float('nan'), float('inf'), True, 1])
def test_still_image_rejects_invalid_timestamp(prepared, timestamp):
    service, source, info, _, _, _ = prepared
    with pytest.raises(ValueError):
        service.prepare(source, info, timestamp=timestamp)


def test_path_bounds_stale_requests_and_size_limit(prepared, tmp_path, monkeypatch):
    service, source, info, _, data, _ = prepared
    external = tmp_path / 'outside.bmp'; external.write_bytes(source.read_bytes())
    with pytest.raises(ValueError):
        service.prepare(external, info)
    with pytest.raises(ValueError):
        service.view('../uploads')
    with pytest.raises(ValueError):
        service.load({**data, 'fingerprint': 'stale'})
    with pytest.raises(ValueError):
        service.decide({**data, 'decision': 'accept'})
    monkeypatch.setattr(module, 'MAX_SOURCE_BYTES', 8)
    with pytest.raises(ValueError, match='limit'):
        service.prepare(source, info)


def test_stored_result_revalidated_and_prose_canonical_on_restore(prepared):
    service, _, _, _, data, _ = prepared
    service.save_result(data, result_for(service, data))
    path = service.root / 'direct-reviews' / data['review_id'] / 'state.json'
    state = json.loads(path.read_text())
    state['model_result']['observations'][0]['text'] = 'UNTRUSTED GUILT CLAIM'
    path.write_text(json.dumps(state))
    assert 'UNTRUSTED' not in json.dumps(service.view(data['review_id']))
    state['model_result']['observations'][0]['evidence_ids'] = ['wrong']
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        service.view(data['review_id'])


def test_manual_annotations_reopen_export_and_exact_preview(prepared):
    service, source, _, view, data, files = prepared
    before = source.read_bytes()
    result = service.annotate(data, subjects=[{'kind':'person','bbox':[2,2,12,25]},
                                             {'kind':'motorcycle','bbox':[1,10,30,30]}])
    assert result['fingerprint'] != data['fingerprint']
    request = {'review_id':result['review_id'], 'fingerprint':result['fingerprint']}
    loaded = service.load(request)
    assert len(loaded['images']) == len(result['frames']) == 3
    assert loaded['subjects'] == result['annotations']['subjects']
    assert [row['id'] for row in loaded['subjects']] == ['P1','V1']
    for frame, image in zip(result['frames'], loaded['images']):
        assert files.paths[frame['url']].read_bytes() == base64.b64decode(image['data_url'].split(',')[1])
        assert frame['subject_ids']
    assert source.read_bytes() == before
    assert files.paths[result['original_frame']['url']] == files.paths[view['frames'][0]['url']]
    assert module.DirectReview(service.root, files).view(result['review_id']) == result
    export = service.export(request)
    assert json.loads(files.paths[export['url']].read_text())['annotations'] == result['annotations']
    with pytest.raises(ValueError, match='changed'):
        service.load(data)


def test_annotations_invalidate_result_and_stale_edit(prepared):
    service, _, _, _, data, _ = prepared
    service.save_result(data, result_for(service, data))
    service.decide({**data, 'decision':'accept'})
    result = service.annotate(data, subjects=[])
    assert result['model_result'] is None and result['review']['decision'] is None
    with pytest.raises(ValueError, match='changed'):
        service.annotate(data, subjects=[])


def test_annotation_crop_tampering_rejected(prepared):
    service, _, _, _, data, files = prepared
    result = service.annotate(data, subjects=[{'kind':'person','bbox':[2,2,12,25]}])
    files.paths[result['frames'][1]['url']].write_bytes(b'tampered crop')
    with pytest.raises(ValueError, match='bytes changed'):
        service.view(result['review_id'])


def test_delete_all_then_add_uses_monotonic_ids(prepared):
    service, _, _, _, data, _ = prepared
    def change(request, subjects):
        view = service.annotate(request, subjects=subjects)
        return view, {'review_id':view['review_id'], 'fingerprint':view['fingerprint']}
    view, request = change(data, [{'kind':'person','bbox':[1,1,10,20]}])
    view, request = change(request, [])
    view, request = change(request, [{'kind':'person','bbox':[1,1,10,20]}])
    assert view['annotations']['subjects'][0]['id'] == 'P2'


def test_full_resolution_crops_and_bounded_stored_images(prepared):
    service, source, _, _, _, files = prepared
    Image.new('RGB', (1536, 1024), 'gray').save(source)
    view = service.prepare(source, {'kind':'image'}, synthetic=True)
    data = {'review_id':view['review_id'], 'fingerprint':view['fingerprint']}
    view = service.annotate(data, subjects=[{'kind':'person','bbox':[100,100,150,200]}])
    assert view['annotations']['image_width'] == 768
    assert view['annotations']['crop_source']['width'] == 1536
    assert view['frames'][1]['width'] == 180 and view['frames'][1]['height'] == 280
    data = {'review_id':view['review_id'], 'fingerprint':view['fingerprint']}
    view = service.annotate(data, subjects=[])
    folder = service.root / 'direct-reviews' / view['review_id']
    assert len(list(folder.glob('annotation-*.jpg'))) == 1


def test_identified_result_roundtrip_and_annotation_edit_invalidates_it(prepared):
    from traffic_ai.identified_vision import canonicalize_result, subjects_fingerprint
    service, _, _, _, data, files = prepared
    view = service.annotate(data, subjects=[{'kind':'person','bbox':[2,2,12,25]}])
    data = {'review_id':view['review_id'], 'fingerprint':view['fingerprint']}
    loaded = service.load(data)
    subjects, images = loaded['subjects'], loaded['images']
    raw = {'image_fingerprint':image_fingerprint(images), 'subjects_fingerprint':subjects_fingerprint(subjects),
           'subject_results':[{'id':'P1','role':'person_unclear','associated_vehicle_id':None,'status':'unclear',
                               'observations':[{'kind':'association_unclear','evidence_ids':subjects[0]['evidence_ids']}],
                               'suspected_issues':[], 'uncertainties':['association_unclear','model_may_misread']}]}
    result = {**canonicalize_result(raw, images, subjects), **data}
    saved = service.save_result(data, result)
    assert saved['model_result']['mode'] == 'identified_images'
    service.decide({**data, 'decision':'accept'})
    assert module.DirectReview(service.root, files).view(data['review_id'])['review']['decision'] == 'accept'
    changed = service.annotate(data, subjects=[{**subjects[0], 'bbox':[1,1,13,26]}])
    assert changed['model_result'] is None and changed['review']['decision'] is None
    with pytest.raises(ValueError, match='changed'):
        service.save_result(data, result)


def test_local_evidence_bound_run_reopen_export_and_correction(prepared, monkeypatch):
    from traffic_ai import subject_models
    service, source, _, _, data, files = prepared
    calls=[]
    def predict(rgb):
        calls.append(rgb.copy())
        return [{'label':'no_helmet','model_score':.73,'box':[4.,3.,9.,8.]}]
    monkeypatch.setattr(subject_models,'predict_heads',predict)
    view=service.annotate(data,subjects=[{'kind':'person','bbox':[2,2,12,25]}, {'kind':'motorcycle','bbox':[1,15,30,30]}])
    request={'review_id':view['review_id'],'fingerprint':view['fingerprint']}
    before=source.read_bytes()
    modeled=service.run_models(request)
    assert modeled['fingerprint']!=request['fingerprint']
    assert modeled['local_evidence']==modeled['annotations']['local_evidence']
    assert modeled['local_evidence']['subjects'][0]['helmet']['status']=='no_helmet'
    assert len(calls)==1 and calls[0].shape==(32,48,3)
    assert source.read_bytes()==before
    fresh={'review_id':modeled['review_id'],'fingerprint':modeled['fingerprint']}
    assert module.DirectReview(service.root,files).view(fresh['review_id'])==modeled
    assert service.load(fresh)['local_evidence']==modeled['local_evidence']
    report=service.export(fresh)
    assert json.loads(files.paths[report['url']].read_text())['local_evidence']==modeled['local_evidence']
    assert len(calls)==1
    with pytest.raises(ValueError,match='changed'):service.run_models(request)
    changed=service.annotate(fresh,subjects=modeled['annotations']['subjects'])
    assert changed['local_evidence'] is None
    assert changed['model_result'] is None and changed['review']['decision'] is None


def test_local_run_rejects_missing_annotations_changed_source_and_decode(prepared,monkeypatch):
    from traffic_ai import subject_models
    service,source,_,_,data,_=prepared
    calls=[]
    monkeypatch.setattr(subject_models,'predict_heads',lambda rgb:calls.append(1) or [])
    with pytest.raises(ValueError,match='numbered'):service.run_models(data)
    view=service.annotate(data,subjects=[{'kind':'person','bbox':[2,2,12,25]}])
    data={'review_id':view['review_id'],'fingerprint':view['fingerprint']}
    original_load=module.load_image
    monkeypatch.setattr(module,'load_image',lambda path:np.zeros((32,48,3),np.uint8))
    with pytest.raises(ValueError,match='Decoded original'):service.run_models(data)
    monkeypatch.setattr(module,'load_image',original_load)
    stat=source.stat()
    raw=bytearray(source.read_bytes());raw[-1]^=1;source.write_bytes(raw)
    os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    with pytest.raises(ValueError,match='source changed'):service.run_models(data)
    assert not calls


def test_local_run_detects_source_mutation_during_inference(prepared,monkeypatch):
    from traffic_ai import subject_models
    service,source,_,_,data,_=prepared
    view=service.annotate(data,subjects=[{'kind':'person','bbox':[2,2,12,25]}])
    data={'review_id':view['review_id'],'fingerprint':view['fingerprint']}
    def predict(rgb):
        source.write_bytes(b'changed during inference')
        return []
    monkeypatch.setattr(subject_models,'predict_heads',predict)
    with pytest.raises(ValueError,match='source changed'):service.run_models(data)
    assert service.view(data['review_id'])['local_evidence'] is None


def test_local_evidence_fingerprint_bound_model_result_and_rerun_reset(prepared,monkeypatch):
    from traffic_ai import subject_models
    from traffic_ai.identified_vision import canonicalize_result, subjects_fingerprint, local_evidence_fingerprint
    service,_,_,_,data,files=prepared
    monkeypatch.setattr(subject_models,'predict_heads',lambda rgb:[])
    view=service.annotate(data,subjects=[{'kind':'person','bbox':[2,2,12,25]}])
    view=service.run_models({'review_id':view['review_id'],'fingerprint':view['fingerprint']})
    data={'review_id':view['review_id'],'fingerprint':view['fingerprint']}
    loaded=service.load(data)
    raw={'image_fingerprint':image_fingerprint(loaded['images']),
         'subjects_fingerprint':subjects_fingerprint(loaded['subjects']),
         'local_evidence_fingerprint':local_evidence_fingerprint(loaded['local_evidence']),
         'subject_results':[{'id':'P1','role':'person_unclear','associated_vehicle_id':None,'status':'unclear',
                             'observations':[{'kind':'association_unclear','evidence_ids':loaded['subjects'][0]['evidence_ids']}],'suspected_issues':[],
                             'uncertainties':['association_unclear','model_may_misread']}]}
    result={**canonicalize_result(raw,loaded['images'],loaded['subjects'],loaded['local_evidence']),**data}
    service.save_result(data,result)
    service.decide({**data,'decision':'accept'})
    assert module.DirectReview(service.root,files).view(data['review_id'])['review']['decision']=='accept'
    rerun=service.run_models(data)
    assert rerun['fingerprint']!=data['fingerprint']
    assert rerun['review']['decision'] is None and rerun['model_result'] is None


def test_vehicle_specialist_survives_failed_subject_detector(prepared, monkeypatch):
    service, _, _, _, data, _ = prepared
    def failed_detector(image):
        raise RuntimeError('Private weight path must not appear')
    monkeypatch.setattr('traffic_ai.subject_annotations.detect_subjects', failed_detector)
    monkeypatch.setattr('traffic_ai.vehicle_types.predict_types', lambda rgb: [
        {'class_id': 7, 'label': 'Two-wheeler', 'model_score': .82, 'box': [1., 1., 20., 20.]}])
    view = service.annotate(data)
    assert view['annotations']['detector']['state'] == 'unavailable'
    assert 'Private' not in str(view)
    request = {key: view[key] for key in ('review_id', 'fingerprint')}
    result = service.run_vehicle_types(request)
    assert result['vehicle_type_evidence']['state'] == 'complete'
    assert result['vehicle_type_evidence']['candidate_counts']['total'] == 1
    assert result['local_evidence'] is None
    fresh = {key: result[key] for key in ('review_id', 'fingerprint')}
    exported = service.export(fresh)
    payload = json.loads((service.root/'direct-reviews'/view['review_id']/exported['filename']).read_text())
    assert payload['feature_report']['features']['vehicle_types']['state'] == 'complete'
    assert service.view(view['review_id'])['vehicle_type_evidence'] == result['vehicle_type_evidence']


def test_plate_correction_preserves_model_reading_and_invalidates_review(prepared, monkeypatch):
    from traffic_ai import plate_models
    from traffic_ai.identified_vision import subjects_fingerprint
    service, _, _, _, data, files = prepared
    builder = plate_models.build_evidence
    def synthetic(rgb, subjects, width, height, **kwargs):
        return builder(rgb, subjects, width, height, **kwargs, detections=[
            {'box': [10., 20., 30., 27.], 'detector_score': .8, 'ocr_score': .7, 'text': 'AB12CD3456'}])
    monkeypatch.setattr(plate_models, 'build_evidence', synthetic)
    view = service.annotate(data, subjects=[{'kind': 'car', 'bbox': [2, 2, 45, 31]}])
    bound = lambda v: {key: v[key] for key in ('review_id', 'fingerprint')}
    view = service.run_plates(bound(view))
    original = view['plate_evidence']['subjects'][0]['candidates']
    assert original[0]['text'] == 'AB12CD3456'
    loaded = service.load(bound(view))
    selection = {**bound(view), 'image_fingerprint': image_fingerprint(loaded['images']),
                 'subjects_fingerprint': subjects_fingerprint(loaded['subjects']),
                 'subject_results': [{'id': 'V1', 'role': 'vehicle', 'associated_vehicle_id': None,
                     'status': 'no_visible_violation', 'observations': [{'kind': 'no_visible_concern',
                     'evidence_ids': loaded['subjects'][0]['evidence_ids']}], 'suspected_issues': [],
                     'uncertainties': ['model_may_misread']}]}
    service.save_result(bound(view), selection)
    service.decide({**bound(view), 'decision': 'accept'})
    corrected = service.correct_plate({**bound(view), 'vehicle_id': 'V1', 'plate_id': original[0]['plate_id'], 'text': 'AB12CD3458'})
    assert corrected['fingerprint'] != view['fingerprint']
    assert corrected['model_result'] is None and corrected['review']['decision'] is None
    assert corrected['plate_evidence']['subjects'][0]['candidates'] == original
    assert corrected['evidence_table']['rows'][0]['plate']['text'] == 'AB12CD3458'
    assert corrected['evidence_table']['summary']['suspected_issue_count'] == 0
    with pytest.raises(ValueError, match='changed'):
        service.correct_plate({**bound(view), 'vehicle_id': 'V1', 'plate_id': original[0]['plate_id'], 'text': 'AB12CD3458'})
    exported = service.export(bound(corrected))
    saved = json.loads(files.paths[exported['url']].read_text())
    assert saved['plate_evidence'] == corrected['plate_evidence']
    assert saved['evidence_table'] == corrected['evidence_table']
    reopened = module.DirectReview(service.root, files).view(view['review_id'])
    assert reopened['plate_evidence'] == corrected['plate_evidence']
    rerun = service.run_plates(bound(corrected))
    assert rerun['plate_evidence']['subjects'][0]['manual_correction'] is None
    edited = service.annotate(bound(rerun), subjects=[{'id': 'V1', 'kind': 'car', 'bbox': [3, 2, 45, 31]}])
    assert edited['plate_evidence'] is None
