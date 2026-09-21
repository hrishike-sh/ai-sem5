"""Explicit local demonstration files; no acquisition or calibration guessing."""
from pathlib import Path
import hashlib
import json
import re

from traffic_ai.runtime import runtime_root


def examples(project_root):
    dataset = runtime_root() / 'datasets/forespeed-20261020'
    result = {}
    for clip, title, digest in [
        ('eufy-low-t1p1', 'Eufy · low view', '6249f0949a01c29bbcc537fedd8a9022ab2e5995a993e43655cf261d7b26098a'),
        ('eufy-strong-t2p1', 'Eufy · strong perspective', '3aae03741a2ee39ed66725a0ae632513c39ed25b8bcf5498aa77351b8cafa384'),
        ('kasa-low-t1p2', 'Kasa · low view', '01be3ef58ce67f96104489a1974793a8373e2f39c0223b7ef4bdb6e709a59240'),
        ('ring-strong-t2p2', 'Ring · strong perspective', 'ebf76ecab4c7baa25f33e96d62ef5c10c7bb9ccffcb819564b7f76d5ed52e49c')]:
        result['forespeed-' + clip + '.mp4'] = {
            'path': dataset / (clip + '.mp4'), 'label': 'Real · ForeSpeed ' + title + ' · uncalibrated',
            'sha256': digest, 'synthetic': False,
            'calibration_status': 'Physical speed withheld: camera corners and lens correction unresolved.'}
    for clip, title, digest in [
        ('green_cross_18', '18 km/h', '03768baaafe97182b1ff1590d54a0b88290596e5e52a00593a0116e0dae90fc8'),
        ('speed_equal_30', '30 km/h', '6db2b88a711d4afa09048e9b18d6840c8f3cb839b88066edc59307fdeaf8023d'),
        ('speed_above_36', '36 km/h', 'e3ab6969f033d791b6e2ffeba31965c07f13eb12d588bc6c0071d146073c8854'),
        ('speed_36_recovery_early', '36 km/h · obstruction recovery', '0e6a692740af7f24830ea4ca73771f01ac44286e6e571fa12247e63281497d33')]:
        result['synthetic-speed-v1/videos/' + clip + '.mp4'] = {
            'path': Path(project_root) / 'demo/synthetic-speed-v1/videos' / (clip + '.mp4'),
            'label': 'Synthetic · ' + title + ' · declared virtual scale', 'synthetic': True, 'sha256': digest,
            'virtual_setup': {'kind': 'fixed_translation_fixture_v1',
                              'points': [[.1, .1], [.9, .1], [.9, .9], [.1, .9]],
                              'width_m': 16, 'length_m': 40, 'reference_frame': 0,
                              'threshold_is_legal_limit': False},
            'calibration_status': 'Declared virtual geometry only; select calibration explicitly before reporting virtual km/h.'}
    manifest = Path(project_root) / 'demo/local-videos.json'
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or not isinstance(data.get('videos'), list):
            raise ValueError('Unsupported local video catalog. Check demo/local-videos.json.')
        for item in data['videos']:
            key, path = item['key'], Path(item['path'])
            if (not isinstance(key, str) or not re.fullmatch(r'local-user-[a-z0-9-]+\.mp4', key)
                    or key in result or not path.is_absolute() or path.suffix.lower() != '.mp4'
                    or not re.fullmatch(r'[a-f0-9]{64}', item['sha256'])):
                raise ValueError('Invalid local video entry. Check demo/local-videos.json.')
            result[key] = {'path': path, 'label': item['label'], 'sha256': item['sha256'],
                           'local_source': True, 'probed_metadata': item['probed_metadata'],
                           'source_provenance': 'User-selected local file; content and speed truth are unverified.',
                           'calibration_status': 'Uncalibrated local video. No measured road dimensions or speed truth were supplied.'}
    return result


def identify_example(path, catalog):
    path = Path(path)
    digest = None
    for item in catalog.values():
        if not item.get('sha256') or not item['path'].is_file() or path.stat().st_size != item['path'].stat().st_size:
            continue
        if digest is None:
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest == item['sha256']:
            return {key: value for key, value in item.items() if key != 'path'}
    return None
