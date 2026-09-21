import hashlib
import json

from PIL import Image
import pytest

import web_server
from ivp.examples import examples, identify_example
from tests.test_web_server import server as server, request
from tests.test_video import source as source, request_run


def test_real_and_synthetic_examples_are_explicit_without_implicit_calibration(tmp_path):
    catalog = examples(tmp_path)
    assert len(catalog) == 8
    assert sum(item['synthetic'] for item in catalog.values()) == 4
    assert all('calibration_status' in item and 'calibration' not in item for item in catalog.values())
    assert all(item['path'].is_relative_to(tmp_path) for item in catalog.values() if item['synthetic'])


def test_external_example_resume_allows_only_the_exact_registered_file(tmp_path, monkeypatch):
    project = tmp_path/'project'
    project.mkdir()
    monkeypatch.setattr(web_server, 'ROOT', project)
    source = tmp_path/'permitted.png'
    other = tmp_path/'other.png'
    Image.new('RGB', (20, 20)).save(source)
    Image.new('RGB', (20, 20)).save(other)
    monkeypatch.setattr(web_server, 'EXAMPLES', {'known': {'path': source}})
    lab = web_server.Lab()
    try:
        media = lab.register(source)
        lab.media.clear()
        assert lab.resume(media['id'])['id'] == media['id']
        key = hashlib.sha256(str(other.resolve()).encode()).hexdigest()[:24]
        (project/'temp/webmedia'/f'{key}.json').write_text(json.dumps({'path': str(other)}))
        with pytest.raises(ValueError, match='outside'):
            lab.resume(key)
    finally:
        lab.close()


def test_known_content_is_identified_after_upload_copy_but_not_after_modification(tmp_path):
    original = tmp_path/'original.mp4'
    original.write_bytes(b'declared fixture bytes')
    uploaded = tmp_path/'uploaded.mp4'
    uploaded.write_bytes(original.read_bytes())
    catalog = {'known': {'path': original, 'sha256': hashlib.sha256(original.read_bytes()).hexdigest(), 'synthetic': True}}
    assert identify_example(uploaded, catalog)['synthetic'] is True
    uploaded.write_bytes(b'changed fixture bytes!')
    assert identify_example(uploaded, catalog) is None


def test_synthetic_provenance_is_stored_and_cannot_be_unchecked_for_incidents(server, source, monkeypatch):
    from ivp.media import write_json
    catalog = {'known': {'path': source, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                         'synthetic': True, 'calibration_status': 'Declared virtual scale',
                         'virtual_setup': {'width_m': 16, 'length_m': 40, 'points': [[.1, .1], [.9, .1], [.9, .9], [.1, .9]]}}}
    monkeypatch.setattr(web_server, 'EXAMPLES', catalog)
    media = server.lab.register(source)
    assert server.lab.media[media['id']][1]['example']['synthetic'] is True
    launched = []
    def capture(path, info, settings, masks, motion):
        launched.append({'info': info, 'motion': motion})
        run = web_server.ROOT/'runs'/str(len(launched))
        run.mkdir(parents=True)
        write_json(run/'request.json', {'source': str(path), 'info': info, 'motion': motion})
        write_json(run/'status.json', {'state': 'cancelled'})
        return run
    monkeypatch.setattr(web_server, 'start_video', capture)
    request(server, '/api/motion', {'media': media['id'], 'tracking_mode': 'ivp'})
    assert launched[0]['motion']['calibration'] is None
    payload = {'media': media['id'], 'tracking_mode': 'vehicles', 'calibration': {
        'points': [[0, 0], [1, 0], [1, 1], [0, 1]], 'width_m': 16, 'length_m': 40,
        'media_id': media['id'], 'source_revision': media['source_revision'], 'reference_frame': 0}}
    request(server, '/api/motion', payload)
    assert launched[1]['info']['example']['synthetic'] is True
    assert 'incidents' not in launched[1]['motion']
    request(server, '/api/motion', {**payload, 'incidents': {'enabled': True, 'line': [[.1, .5], [.9, .5]], 'synthetic': False}})
    assert launched[2]['motion']['incidents']['synthetic'] is True


def test_calibrated_synthetic_worker_exports_provenance_with_incident_review_off(source, tmp_path):
    import zipfile
    from ivp.media import write_json
    from ivp.motion_worker import run_motion
    run = request_run(tmp_path, source)
    config = json.loads((run/'request.json').read_text())
    config['info']['example'] = {'synthetic': True, 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    config['motion'] = {'calibration': {'points': [[0, 0], [1, 0], [1, 1], [0, 1]], 'width_m': 16, 'length_m': 40}}
    write_json(run/'request.json', config)
    run_motion(run)
    report = json.loads((run/'report.json').read_text())
    assert report['source_info']['example']['synthetic'] is True
    assert report['motion']['synthetic'] is True
    assert report['motion']['source_provenance']['synthetic'] is True
    assert report['motion']['speed']['calibrated'] is True
    assert 'incidents' not in report['motion']
    with zipfile.ZipFile(run/'motion-evidence.zip') as archive:
        assert json.loads(archive.read('motion-summary.json'))['synthetic'] is True
        assert json.loads(archive.read('request.json'))['info']['example']['synthetic'] is True
