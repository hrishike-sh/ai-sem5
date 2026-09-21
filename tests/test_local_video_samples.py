import hashlib
import json

import pytest

from ivp.examples import examples
import web_server


def local_catalog(tmp_path):
    baseline = examples(tmp_path)
    nested = tmp_path/'Downloads'/'Test Video.mp4'
    nested.mkdir(parents=True)
    paths = [tmp_path/'Downloads'/'12010437_2160_3840_30fps.mp4',
             tmp_path/'Downloads'/'16922539_2160_3840_30fps.mp4', nested/'Test Video.mp4']
    records = []
    for i, path in enumerate(paths):
        path.write_bytes(('SYNTHETIC catalog test bytes ' + str(i)).encode())
        records.append({'key':f'local-user-test-{i}.mp4','path':str(path),'label':f'Local {i} · uncalibrated',
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'probed_metadata':{'width':2160,'height':3840,'fps':29.9166666667}})
    (tmp_path/'demo').mkdir()
    (tmp_path/'demo/local-videos.json').write_text(json.dumps({'schema_version':1,'videos':records}))
    return baseline, paths, examples(tmp_path)


def test_exact_three_local_paths_preserve_all_existing_examples_and_nested_filename(tmp_path):
    baseline, paths, catalog = local_catalog(tmp_path)
    assert len(catalog) == len(baseline) + 3
    assert all(catalog[key] == value for key,value in baseline.items())
    local = [v for v in catalog.values() if v.get('local_source')]
    assert [v['path'] for v in local] == paths
    assert local[2]['path'].is_file() and local[2]['path'].parent.name == 'Test Video.mp4'
    assert all('virtual_setup' not in v and 'calibration' not in v and v.get('synthetic') is not True for v in local)
    assert all('Uncalibrated' in v['calibration_status'] for v in local)


def test_local_catalog_rejects_duplicate_keys_without_replacing_legacy_examples(tmp_path):
    local_catalog(tmp_path)
    path=tmp_path/'demo/local-videos.json';data=json.loads(path.read_text())
    data['videos'].append(data['videos'][0]);path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Invalid local video'):
        examples(tmp_path)


def test_original_playback_url_is_only_published_for_exact_matching_local_video(tmp_path,monkeypatch):
    _, paths, catalog=local_catalog(tmp_path)
    monkeypatch.setattr(web_server,'ROOT',tmp_path)
    monkeypatch.setattr(web_server,'EXAMPLES',catalog)
    monkeypatch.setattr(web_server,'probe_video',lambda path:{'kind':'video','width':16,'height':24,'duration':1.,'fps':25.,'has_audio':False})
    lab=web_server.Lab()
    try:
        media=lab.register(paths[2])
        assert media['original_video'].startswith('http://127.0.0.1:')
        assert media['example']['local_source'] and 'virtual_setup' not in media['example']
        paths[2].write_bytes(b'changed source bytes')
        assert 'original_video' not in lab.register(paths[2])
    finally:
        lab.close()
