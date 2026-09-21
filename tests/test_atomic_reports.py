import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from ivp.media import write_json


def test_concurrent_status_writers_have_independent_temporary_files(tmp_path):
    target = tmp_path / 'status.json'
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: write_json(target, {'frame':n,'text':'x'*1000}), range(30)))
    assert json.loads(target.read_text())['frame'] in range(30)
    assert not list(tmp_path.glob('*.tmp'))


def test_windows_transient_replace_denial_retries(tmp_path, monkeypatch):
    original = Path.replace
    calls = []
    def replace(source, destination):
        calls.append(source)
        if len(calls) < 3:
            raise PermissionError('Temporarily open in another reader')
        return original(source, destination)
    monkeypatch.setattr(Path, 'replace', replace)
    target = tmp_path / 'status.json'
    write_json(target, {'state':'complete'})
    assert len(calls) == 3
    assert json.loads(target.read_text())['state'] == 'complete'
    assert not list(tmp_path.glob('*.tmp'))


def test_status_reader_recovers_from_transient_windows_denial(tmp_path, monkeypatch):
    from ivp.jobs import read_status
    original = Path.read_text
    calls = []
    write_json(tmp_path/'status.json', {'state':'complete'})
    def read(path, *args, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError('Transient Windows replacement lock')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    assert read_status(tmp_path)['state'] == 'complete'
    assert len(calls) == 3


@pytest.mark.parametrize('latest', ['complete', 'cancelled', 'failed', 'running'])
def test_status_refresh_preserves_worker_update_before_lock(tmp_path, monkeypatch, latest):
    from ivp import jobs
    old = (datetime.now(timezone.utc)-timedelta(seconds=40)).isoformat()
    write_json(tmp_path/'status.json', {'state':'running','updated_at':old})
    fresh = {'state':latest,'updated_at':jobs.now(),'frames':120,'message':'Latest worker update'}

    @contextmanager
    def worker_finishes_before_lock():
        # The poll read an old status just before the worker wrote its final
        # result and released the lock. Recovery must recheck under the lock.
        write_json(tmp_path/'status.json', fresh)
        yield

    monkeypatch.setattr(jobs, 'job_lock', worker_finishes_before_lock)
    assert jobs.refreshed_status(tmp_path) == fresh
    assert jobs.read_status(tmp_path) == fresh


def test_stopped_worker_is_interrupted_without_removing_evidence(tmp_path, monkeypatch):
    from ivp import jobs
    monkeypatch.setattr(jobs, 'RUNS', tmp_path)
    old = (datetime.now(timezone.utc)-timedelta(seconds=40)).isoformat()
    write_json(tmp_path/'status.json', {'state':'running','updated_at':old,'frames':12})
    (tmp_path/'motion-tracks.csv').write_text('retained diagnostic evidence')
    assert jobs.refreshed_status(tmp_path)['state'] == 'interrupted'
    assert jobs.read_status(tmp_path)['frames'] == 12
    assert (tmp_path/'motion-tracks.csv').read_text() == 'retained diagnostic evidence'
