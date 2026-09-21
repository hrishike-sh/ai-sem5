"""Acquire four pinned public noncommercial reference clips, without re-encoding."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request


CLIPS = [
    ('eufy-low-t1p1', 'Eufy/T1P1-Eufy-62715e80826c78491bcc6a6cbe707193.mp4', 7736303, 29, 'low'),
    ('eufy-strong-t2p1', 'Eufy/T2P1-eufy-da66251744ce9b16b3b3cd3ca7c98e76.mp4', 7084462, 30, 'strong'),
    ('kasa-low-t1p2', 'Kasa/T1P2-kasa-S_09_1628986170_1628986230.mp4', 8126538, 30, 'low'),
    ('ring-strong-t2p2', 'Ring/Download/T2P2-Ring-65073258_6996445117600275776_stamp.mp4', 8246707, 30, 'strong'),
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def request(url):
    return urllib.request.Request(url, headers={'User-Agent': 'TrafficLab-noncommercial-reference-evaluation'})


def small(url, limit=2 * 1024 * 1024):
    with urllib.request.urlopen(request(url), timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Unexpectedly large metadata response: ' + url)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    out = args.destination.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / 'manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    revision = previous['repository_commit'] if previous else json.loads(small('https://api.github.com/repos/ampedsoftware/forespeed/commits/main'))['sha']
    if not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('Invalid repository revision.')
    metadata = out / 'metadata'
    metadata.mkdir(exist_ok=True)
    license_bytes = small(f'https://raw.githubusercontent.com/ampedsoftware/forespeed/{revision}/LICENSE.txt')
    if b'Attribution-NonCommercial 4.0 International' not in license_bytes:
        raise ValueError('Expected CC BY-NC4.0 licence was not found; no media acquired.')
    license_path = metadata / 'LICENSE.txt'
    if license_path.exists():
        assert license_path.read_bytes() == license_bytes, 'Preserved licence differs.'
    else:
        license_path.write_bytes(license_bytes)
    record = {'dataset': 'ForeSpeed', 'repository': 'https://github.com/ampedsoftware/forespeed',
              'paper': 'https://arxiv.org/html/2512.19364v1', 'repository_commit': revision,
              'acquired_at_utc': datetime.now(timezone.utc).isoformat(), 'license': 'CC BY-NC 4.0',
              'license_sha256': digest(license_path), 'purpose': 'Local noncommercial academic evaluation; no redistribution or endorsement implied.',
              'original_bytes_native_pts_preserved': True,
              'reference_speed_scope': 'Paper Table4 speedometer/GPS references, with measurement limitations; not exact physical truth or posted speed limits.',
              'reference_rectangles_m': {'low': [6.5024, 3.6576], 'strong': [4.2418, 3.6576]},
              'geometry_scope': 'Measured reference rectangles, NOT full road width. Camera-specific pixel corners and lens distortion must be resolved before a physical-speed claim.',
              'clips': []}
    for label, relative, expected_size, mph, perspective in CLIPS:
        pointer_url = f'https://raw.githubusercontent.com/ampedsoftware/forespeed/{revision}/{relative}'
        pointer = small(pointer_url, 2048).decode('utf8')
        oid = re.search(r'^oid sha256:([a-f0-9]{64})$', pointer, re.M)
        length = re.search(r'^size (\d+)$', pointer, re.M)
        if not oid or not length or int(length[1]) != expected_size:
            raise ValueError('Unexpected LFS metadata for ' + relative)
        target = out / (label + '.mp4')
        url = f'https://media.githubusercontent.com/media/ampedsoftware/forespeed/{revision}/{relative}'
        if target.exists():
            assert target.stat().st_size == expected_size and digest(target) == oid[1], target
            action = 'verified existing'
        else:
            part = target.with_suffix('.mp4.part')
            with part.open('xb') as stream, urllib.request.urlopen(request(url), timeout=90) as response:
                size = 0
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > expected_size:
                        raise ValueError('Media exceeds pinned byte count: ' + relative)
                    stream.write(chunk)
            assert part.stat().st_size == expected_size and digest(part) == oid[1], part
            part.rename(target)
            action = 'downloaded once'
        item = {'id': label, 'path': str(target), 'source_path': relative, 'source_url': url,
                'bytes': expected_size, 'sha256': oid[1], 'lfs_pointer_url': pointer_url,
                'reference_speed_mph': mph, 'reference_speed_kmh': mph * 1.609344,
                'perspective': perspective, 'calibration_status': 'unresolved_camera_specific_geometry'}
        record['clips'].append(item)
        manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
        print(label, action, expected_size, oid[1], flush=True)
    for name in ['measure_street1.png', 'measure_street2.png']:
        path = metadata / name
        if not path.exists():
            path.write_bytes(small('https://arxiv.org/html/2512.19364v1/figs/' + name))
        record.setdefault('paper_figures', []).append({'path': str(path), 'sha256': digest(path),
                                                      'scope': 'Paper reference figure for private geometry inspection; no inferred extension beyond marked rectangle.'})
    attribution = ('ForeSpeed reference dataset, Amped Software repository and the authors of ForeSpeed: A real-world video dataset of CCTV cameras with different settings for vehicle speed estimation.\n'
                   'Source: https://github.com/ampedsoftware/forespeed\nPaper: https://arxiv.org/html/2512.19364v1\n'
                   'Dataset licence: CC BY-NC4.0, https://creativecommons.org/licenses/by-nc/4.0/ . Full supplied licence is in metadata/LICENSE.txt.\n'
                   'Local noncommercial academic testing only. Original media bytes are unchanged. Any derived masks/annotations/exports must be identified as project processing.\n'
                   'No camera calibration, measured speed accuracy, legal limit or endorsement is established by this acquisition.\n')
    (out / 'ATTRIBUTION.txt').write_text(attribution, encoding='utf8')
    record['state'] = 'four_originals_acquired_and_lfs_hash_verified'
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf8')
    print('MANIFEST', manifest, flush=True)


if __name__ == '__main__':
    main()
