"""Saved-export audit contracts; generated fixtures are explicitly synthetic."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
import zipfile

import pytest

from traffic_ai import context_replay
from traffic_ai.contracts import Rule
from tests.test_control_association import corridor, lamp, sign


def make_run(tmp_path, camera=True):
    run = tmp_path / "recorded-synthetic-run"
    run.mkdir()
    frames = []
    for index in range(3):
        frame = {"frame": index, "timestamp": index * .2, "scene_id": 0, "scene_revision": 1,
                 "signal_observations": [lamp()], "decisions": [
                     {"rule": rule.value, "verdict": "cannot_assess", "reason": "synthetic recorded input"} for rule in Rule]}
        if camera is not None:
            frame["camera_stable"] = camera
        frames.append(frame)
    counts = Counter(d["rule"] + "/" + d["verdict"] for frame in frames for d in frame["decisions"])
    (run / "report.json").write_text(json.dumps({"synthetic": True, "ai": {"assessment_counts": dict(counts)}}))
    (run / "status.json").write_text(json.dumps({"state": "complete"}))
    (run / "detections.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in frames), encoding="utf-8")
    scene = {"scene_id": 0, "revision": 1, "approaches": [corridor()], "sign_observations": [sign()]}
    with zipfile.ZipFile(run / "scene_features.zip", "w") as archive:
        archive.writestr("0001/scene.json", json.dumps(scene))
    return run, frames, scene


def test_complete_audit_keeps_source_decisions_hashes_labels_and_originals(tmp_path):
    run, frames, _ = make_run(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    output = tmp_path / "audit"
    summary = context_replay.replay(run, output)
    assert summary["frames"] == 3 and summary["source_counts_match_report"]
    assert summary["signal_association_status_counts"] == {"pending": 2, "associated": 1}
    assert summary["sign_association_status_counts"] == {"associated": 1}
    assert summary["fresh_inference"] is False and summary["rule_decisions_recomputed"] is False
    assert summary["provenance"] == {"synthetic": True}
    records = [json.loads(line) for line in (output / "control-associations.jsonl").read_text().splitlines()]
    for raw, record, source in zip(before["detections.jsonl"].splitlines(keepends=True), records, frames):
        assert record["source_line_sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["timestamp"] == source["timestamp"]
        assert record["scene_source"]["member"] == "0001/scene.json"
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    assert not list(output.glob("*.tmp"))


def test_unrecorded_camera_stability_is_unknown_and_never_borrowed_from_verdicts(tmp_path):
    run, _, _ = make_run(tmp_path, camera=None)
    summary = context_replay.replay(run, tmp_path / "audit")
    assert summary["frames_without_recorded_camera_stability"] == 3
    assert summary["signal_association_status_counts"] == {"withheld": 3}
    lines = (tmp_path / "audit/control-associations.jsonl").read_text().splitlines()
    assert all(json.loads(line)["camera_stability"] is None for line in lines)


def test_missing_or_changed_scene_cannot_inherit_signal_confirmation(tmp_path):
    run, frames, scene = make_run(tmp_path)
    extra = deepcopy(frames[-1])
    extra.update(timestamp=.6, frame=3, scene_revision=2)
    frames.append(extra)
    extra = deepcopy(extra)
    extra.update(timestamp=.8, frame=4, scene_revision=3)
    frames.append(extra)
    (run / "detections.jsonl").write_text("".join(json.dumps(frame) + "\n" for frame in frames))
    scene["revision"] = 2
    with zipfile.ZipFile(run / "scene_features.zip", "a") as archive:
        archive.writestr("0002/scene.json", json.dumps(scene))
    summary = context_replay.replay(run, tmp_path / "audit")
    assert summary["frames_without_archived_scene"] == 1
    assert summary["source_counts_match_report"] is False  # Exposed, not silently reconciled.
    lines = [json.loads(line) for line in (tmp_path / "audit/control-associations.jsonl").read_text().splitlines()]
    assert lines[3]["association"]["graphs"]["signal"]["observations"][0]["status"] == "pending"
    assert lines[4]["scene_status"] == "not_recorded"
    assert lines[4]["association"]["graphs"]["signal"]["observations"][0]["status"] == "withheld"


def test_no_overwrite_source_run_write_or_incomplete_input(tmp_path):
    run, _, _ = make_run(tmp_path)
    with pytest.raises(ValueError, match="outside the original"):
        context_replay.replay(run, run / "new-audit")
    output = tmp_path / "audit"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        context_replay.replay(run, output)
    (run / "status.json").write_text(json.dumps({"state": "cancelled"}))
    with pytest.raises(ValueError, match="completed"):
        context_replay.replay(run, tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_changed_source_cannot_publish_completion_marker(tmp_path, monkeypatch):
    run, _, _ = make_run(tmp_path)
    original_sha = context_replay._sha
    calls = Counter()

    def changed(path):
        calls[path.name] += 1
        return "changed" if path.name == "detections.jsonl" and calls[path.name] > 1 else original_sha(path)

    monkeypatch.setattr(context_replay, "_sha", changed)
    with pytest.raises(ValueError, match="source changed"):
        context_replay.replay(run, tmp_path / "audit")
    assert not (tmp_path / "audit/association-summary.json").exists()


def test_cli_runs_without_network_subprocess_or_model_runtime(tmp_path):
    run, _, _ = make_run(tmp_path)
    output = tmp_path / "audit"
    code = """
import runpy,sys
def guard(event,args):
    if event.startswith('socket.') or event == 'subprocess.Popen':
        raise AssertionError('Unexpected external runtime: '+event)
    if event == 'import' and args[0].split('.')[0] in {'torch','transformers','ultralytics','requests','httpx'}:
        raise AssertionError('Unexpected model/provider dependency: '+args[0])
sys.addaudithook(guard)
sys.argv=['traffic_ai.context_replay',sys.argv[1],'--output',sys.argv[2]]
runpy.run_module('traffic_ai.context_replay',run_name='__main__')
"""
    done = subprocess.run([sys.executable, "-c", code, str(run), str(output)], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["frames"] == 3
