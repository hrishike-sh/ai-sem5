"""Acquire exactly two capped UTFPR research prefixes and isolated metadata.

No downloaded code is executed. Videos require exact 206/Content-Range before
any body read. No redirect, unbounded body, whole-video fetch or automatic retry.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.error
import urllib.request

BASE = 'https://pessoal.dainf.ct.utfpr.edu.br/rminetto/projects/vehicle-speed/'
PREFIX_BYTES = 16777216
EXPECTED_XML = {'Set04_video01': 73890, 'Set05_video01': 184566}
MATRIX_SHA256 = 'ed184dc3c3cf13586e1f8bd17cdc2ee7b427f2e466c5d2c69e0f334127728986'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise ValueError('Redirect refused before body consumption: '+newurl)


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fetch(url, destination, *, exact_bytes=None, cap=1024*1024, byte_range=False, opener=None):
    destination = Path(destination)
    if destination.exists() or destination.with_suffix(destination.suffix+'.part').exists():
        raise FileExistsError('Preserved download or partial already exists; verify it instead of fetching again: '+str(destination))
    request_headers = {'User-Agent': 'TrafficLab-academic-bounded-provenance/1', 'Accept-Encoding': 'identity'}
    if byte_range:
        if exact_bytes != PREFIX_BYTES or cap != PREFIX_BYTES:
            raise ValueError('Video range must be exactly the first16MiB.')
        request_headers['Range'] = 'bytes=0-16777215'
    opener = opener or urllib.request.build_opener(NoRedirect())
    started = now()
    request = urllib.request.Request(url, headers=request_headers)
    try:
        response = opener.open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        exc.close()  # Header-only failure; never consume its body.
        raise
    with response:
        status = response.status
        headers = dict(response.headers.items())
        if response.geturl() != url:
            raise ValueError('Response URL differs; no body read.')
        if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
            raise ValueError('Encoded response rejected before body read.')
        if byte_range:
            content_range = response.headers.get('Content-Range', '')
            match = re.fullmatch(r'bytes 0-16777215/(\d+)', content_range)
            if status != 206 or not match or int(match[1]) <= PREFIX_BYTES:
                raise ValueError('Expected exact206 bytes0-16777215/total before body read; got '+str(status)+' '+content_range)
        elif status != 200:
            raise ValueError('Small complete metadata requires200 before body read.')
        length = response.headers.get('Content-Length')
        if ((length is None and (byte_range or exact_bytes is not None))
                or (length is not None and (not length.isdecimal() or int(length) > cap
                    or (exact_bytes is not None and int(length) != exact_bytes)))):
            raise ValueError('Missing/unexpected Content-Length rejected before body read: '+str(length))
        expected = int(length) if length is not None else None
        maximum = expected if expected is not None else cap
        part = destination.with_suffix(destination.suffix+'.part')
        read_bytes = 0
        with part.open('xb') as handle:
            while read_bytes < maximum:
                chunk = response.read(min(256*1024, maximum-read_bytes))
                if not chunk:
                    if expected is not None and read_bytes != expected:
                        raise ValueError('Truncated body; partial preserved: '+str(part))
                    break
                if len(chunk) > maximum-read_bytes:
                    raise ValueError('Response violated bounded read; partial preserved.')
                handle.write(chunk)
                read_bytes += len(chunk)
        if expected is not None and read_bytes != expected:
            raise ValueError('Response length differs.')
        if expected is None and read_bytes == cap:
            raise ValueError('Unspecified metadata length reached cap without verified EOF; partial preserved.')
    part.rename(destination)
    return {'url': url, 'final_url': url, 'started_at_utc': started, 'completed_at_utc': now(),
            'request_headers': request_headers, 'response_status': status, 'response_headers': headers,
            'bytes_read': read_bytes, 'cap_bytes': cap, 'sha256': fingerprint(destination),
            'path': str(destination), 'body_read_after_header_validation': True, 'automatic_retries': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    roots = [p.resolve() for p in [args.media, args.truth, args.metadata]]
    if any(p.drive.lower() != 'h:' for p in roots):
        raise ValueError('All acquired data must be on H:.')
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i+1:]):
        raise ValueError('Media, truth and metadata roots must be physically separate.')
    for root in roots:
        root.mkdir(parents=True, exist_ok=False)
    media, truth, metadata = roots
    record = {'created_at_utc': now(), 'dataset': 'UTFPR Vehicle Speed Measurement',
              'scope': 'Exactly first16MiB of each of two original elementary H.264 sources. First200 complete decoded frames only.',
              'media_root': str(media), 'isolated_truth_root': str(truth), 'metadata_root': str(metadata),
              'truth_access': 'Acquisition/evaluation only; excluded from predictor input/arguments.',
              'state': 'acquiring_provenance_only_no_models', 'files': []}
    def save():
        args.receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    try:
        for stem, size in EXPECTED_XML.items():
            record['files'].append(fetch(BASE+stem+'.h264', media/(stem+'.h264'), exact_bytes=PREFIX_BYTES, cap=PREFIX_BYTES, byte_range=True))
            save()
            record['files'].append(fetch(BASE+stem+'.xml', truth/(stem+'.xml'), exact_bytes=size, cap=256*1024))
            save()
            print(stem, 'prefix and complete isolated XML acquired', flush=True)
        sources = [
            (BASE, 'official-dataset-page.html', 1024*1024),
            (BASE+'Paper_ITS_final.pdf', 'Paper_ITS_final.pdf', 8*1024*1024),
            ('https://api.github.com/repos/dluvizon/vehicle-dsm/commits/master', 'repository-master-commit.json', 512*1024),
        ]
        for url, filename, cap in sources:
            record['files'].append(fetch(url, metadata/filename, cap=cap))
            save()
        commit = json.loads((metadata/'repository-master-commit.json').read_text(encoding='utf-8'))['sha']
        if not re.fullmatch('[a-f0-9]{40}', commit):
            raise ValueError('Invalid author repository commit.')
        record['author_repository_commit'] = commit
        tree_url = f'https://api.github.com/repos/dluvizon/vehicle-dsm/git/trees/{commit}?recursive=1'
        record['files'].append(fetch(tree_url, metadata/'repository-tree.json', cap=2*1024*1024))
        tree = json.loads((metadata/'repository-tree.json').read_text(encoding='utf-8'))
        if tree.get('truncated'):
            raise ValueError('Repository tree truncated.')
        paths = ['dataset/matrix.txt', 'dataset/lanes.xml', 'README.md', 'LICENSE.md']
        paths += [row['path'] for row in tree['tree'] if row['path'].split('/')[-1] in ('speed.cpp', 'matrix.cpp')]
        for relative in paths:
            path = metadata/relative.replace('/', '__')
            url = f'https://raw.githubusercontent.com/dluvizon/vehicle-dsm/{commit}/{relative}'
            record['files'].append(fetch(url, path, cap=1024*1024))
            save()
        if fingerprint(metadata/'dataset__matrix.txt') != MATRIX_SHA256:
            raise ValueError('Calibration differs from independently pinned author matrix.')
        record.update(state='provenance_acquired_no_models', published_fps=30.15,
                      timing_scope='Author-provided nominal timebase. Elementary stream decoder guesses are not capture PTS.',
                      calibration_scope='Author full1920x1080 lane-specific planar matrices; divide homogeneous result by60px/m. No separate lens model.',
                      height_factors='Do not apply plate-height correction factors to bbox bottoms or tire/road points.',
                      license_scope='Official page grants research availability and requests citation. Code MIT does not establish unrestricted video redistribution rights.',
                      citation='Luvizon, Nassu, Minetto. A Video-Based System for Vehicle Speed Measurement in Urban Roadways. IEEE T-ITS18(6):1393-1404,2017. DOI10.1109/TITS.2016.2606369.')
        save()
        print('RECEIPT', args.receipt, flush=True)
    except BaseException as exc:
        record.update(state='stopped_preserving_downloads', error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
