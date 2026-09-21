"""Dependency diagnostics must identify the missing item before any job creation."""
from pathlib import Path

import pytest

from ivp import jobs
from ivp.pipeline import Settings
from traffic_ai import runtime


@pytest.mark.parametrize("kind", ["zebra", "motion"])
@pytest.mark.parametrize("missing", ["runtime", "weights"])
def test_vehicle_recovery_names_actual_missing_dependency(tmp_path, monkeypatch, kind, missing):
    executable = tmp_path / "runtime" / "python.exe"
    executable.parent.mkdir()
    if missing == "weights":
        executable.write_bytes(b"test-only placeholder; never executable")
    monkeypatch.setattr(runtime, "python_path", lambda: executable)
    monkeypatch.setattr(runtime, "runtime_root", lambda: tmp_path)
    def no_job(*args, **kwargs):
        pytest.fail("A missing dependency must not acquire a job slot or start a process")
    monkeypatch.setattr(jobs, "job_lock", no_job)
    monkeypatch.setattr(jobs.subprocess, "Popen", no_job)
    kwargs = {"zebra": {}} if kind == "zebra" else {"motion": {"tracking_mode": "vehicles"}}
    with pytest.raises(RuntimeError) as exc:
        jobs.start_video(Path("unused.mp4"), {"kind": "video"}, Settings(), [], **kwargs)
    message = str(exc.value)
    if missing == "runtime":
        assert "Python runtime is missing" in message
        assert str(executable) in message
        assert "Setup-AI.ps1" in message
    else:
        assert "model weights are missing" in message
        assert str(tmp_path / "models" / "yolo11s.pt") in message
        assert "installs dependencies, not these weights" in message
