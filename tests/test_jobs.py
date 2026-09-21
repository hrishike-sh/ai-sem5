import contextlib
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import ivp.jobs as jobs
from ivp.media import hidden_process_options, write_json
from ivp.pipeline import Settings


def test_exclusive_lock_rejects_second_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS", tmp_path)
    code = "import ivp.jobs as jobs; import sys,time; from pathlib import Path; jobs.RUNS=Path(sys.argv[1]);\nwith jobs.job_lock():\n print('ready',flush=True)\n time.sleep(15)"
    process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=jobs.ROOT, stdout=subprocess.PIPE, text=True, **hidden_process_options())
    try:
        assert process.stdout.readline().strip() == "ready"
        with pytest.raises(RuntimeError, match="Another video job"):
            with jobs.job_lock():
                pass
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_interrupted_job_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS", tmp_path)
    run = tmp_path / "old"
    run.mkdir()
    old = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    write_json(run / "status.json", {"state": "running", "updated_at": old})
    assert jobs.refreshed_status(run)["state"] == "interrupted"


@pytest.mark.parametrize("latest_state", ["complete", "cancelled", "failed", "queued", "running", "encoding"])
def test_recovery_rechecks_status_after_acquiring_lock(tmp_path, monkeypatch, latest_state):
    monkeypatch.setattr(jobs, "RUNS", tmp_path)
    run = tmp_path / "changing"
    run.mkdir()
    old = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    write_json(run / "status.json", {"state": "running", "updated_at": old})
    latest = {"state": latest_state, "message": "Latest worker status", "progress": .9,
              "updated_at": jobs.now() if latest_state in ("queued", "running", "encoding") else old}
    original_lock = jobs.job_lock

    @contextlib.contextmanager
    def worker_updates_before_lock():
        # The refresh has read the old status; the worker updates before it can lock.
        write_json(run / "status.json", latest)
        with original_lock():
            yield

    monkeypatch.setattr(jobs, "job_lock", worker_updates_before_lock)
    assert jobs.refreshed_status(run) == latest
    assert jobs.read_status(run) == latest


def test_stale_status_is_preserved_while_worker_holds_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS", tmp_path)
    run = tmp_path / "active"
    run.mkdir()
    old = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
    status = {"state": "running", "updated_at": old, "message": "Long inference"}
    write_json(run / "status.json", status)
    with jobs.job_lock():
        assert jobs.refreshed_status(run) == status
    assert jobs.read_status(run) == status


@pytest.mark.parametrize("ai", [False, True], ids=["ivp", "ai"])
def test_worker_launch_failure_is_saved_and_allows_immediate_retry(tmp_path, monkeypatch, ai):
    monkeypatch.setattr(jobs, "RUNS", tmp_path / "runs")
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original upload")
    unusable_python = tmp_path / "unusable-python"
    unusable_python.write_text("not an executable", encoding="utf-8")
    if ai:
        from traffic_ai import runtime

        monkeypatch.setattr(runtime, "python_path", lambda: unusable_python)
    else:
        monkeypatch.setattr(jobs.sys, "executable", str(unusable_python))

    with pytest.raises(RuntimeError, match="Could not start the processing worker") as error:
        jobs.start_video(source, {"kind": "video"}, Settings(), [], ai=ai)

    run = next(jobs.RUNS.glob("*/status.json")).parent
    status = jobs.read_status(run)
    assert status["state"] == "failed"
    assert status["progress"] == 0
    assert isinstance(error.value.__cause__, OSError)
    assert str(error.value.__cause__) in status["message"]
    assert datetime.fromisoformat(status["updated_at"]).tzinfo is not None
    assert (run / "request.json").is_file()
    assert not (run / "process.json").exists()
    assert source.read_bytes() == b"original upload"

    def launch_retry(command, **kwargs):
        assert command[1:3] == ["-m", "traffic_ai.worker" if ai else "ivp.worker"]
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(jobs.subprocess, "Popen", launch_retry)
    retry = jobs.start_video(source, {"kind": "video"}, Settings(), [], ai=ai)
    assert retry != run
    assert jobs.read_status(retry)["state"] == "queued"
    assert (retry / "process.json").is_file()
    assert jobs.read_status(run) == status


def test_worker_log_failure_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "RUNS", tmp_path / "runs")
    original_new_run = jobs.new_run

    def run_with_unwritable_log():
        run = original_new_run()
        (run / "worker.log").mkdir()
        return run

    monkeypatch.setattr(jobs, "new_run", run_with_unwritable_log)
    with pytest.raises(RuntimeError, match="Could not start the processing worker"):
        jobs.start_video(tmp_path / "source.mp4", {"kind": "video"}, Settings(), [])

    run = next(jobs.RUNS.glob("*/status.json")).parent
    assert jobs.read_status(run)["state"] == "failed"
    assert not (run / "process.json").exists()


@pytest.mark.parametrize("payload", [
    None,
    [],
    "running",
    {"state": "queued"},
    {"state": "running", "updated_at": None},
    {"state": "encoding", "updated_at": 123},
    {"state": "running", "updated_at": "not a timestamp"},
    {"state": "encoding", "updated_at": "2026-10-14T08:00:00"},
], ids=["null", "array", "string", "missing-time", "null-time", "numeric-time", "invalid-time", "naive-time"])
def test_unreadable_status_does_not_break_refresh_or_block_new_jobs(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(jobs, "RUNS", tmp_path / "runs")
    run = jobs.RUNS / "unreadable"
    run.mkdir(parents=True)
    write_json(run / "status.json", payload)
    original = (run / "status.json").read_bytes()

    status = jobs.refreshed_status(run)
    assert status["state"] == "unknown"
    assert "unreadable" in status["message"]
    assert "Start a new run" in status["message"]
    assert (run / "status.json").read_bytes() == original

    # Invalid metadata cannot bypass the OS lock protecting a real worker.
    with jobs.job_lock():
        with pytest.raises(RuntimeError, match="Another video job"):
            jobs.start_video(tmp_path / "source.mp4", {"kind": "video"}, Settings(), [])

    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=123))
    started = jobs.start_video(tmp_path / "source.mp4", {"kind": "video"}, Settings(), [])
    assert started != run
    assert jobs.read_status(started)["state"] == "queued"
    assert (run / "status.json").read_bytes() == original


@pytest.mark.parametrize("state", ["complete", "cancelled", "failed"])
def test_terminal_status_remains_readable_without_a_timestamp(tmp_path, state):
    status = {"state": state, "message": "Saved result"}
    write_json(tmp_path / "status.json", status)
    assert jobs.refreshed_status(tmp_path) == status


def test_missing_status_still_waits_for_worker(tmp_path):
    assert jobs.read_status(tmp_path) == {"state": "unknown", "message": "Waiting for job status."}
