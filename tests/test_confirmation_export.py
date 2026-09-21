"""Actual worker export contracts; all model/video processing explicitly mocked.

Default: test the imported worker. For an isolated candidate source snapshot set
IVP_AI_EXPORT_WORKER_SNAPSHOT. Synthetic assessments are not traffic truth.
"""
from copy import deepcopy
import builtins
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ivp.media import write_json
from ivp.pipeline import Settings
from traffic_ai.contracts import Decision, Detection, FrameAnalysis, Rule, Verdict
from traffic_ai.event_lifecycle import LifecycleAwareEventCollector


def strict(text):
    def reject(value):
        raise AssertionError("Nonfinite JSON token: " + value)
    return json.loads(text, parse_constant=reject)


@pytest.fixture
def worker(monkeypatch):
    candidate = os.environ.get("IVP_AI_EXPORT_WORKER_SNAPSHOT")
    if candidate:
        spec = importlib.util.spec_from_file_location("traffic_ai._export_candidate", Path(candidate))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        from traffic_ai import worker as module
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "ultralytics", "requests", "httpx"}:
            raise AssertionError("Export regression cannot import model/network code: " + name)
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("Export regression cannot load a model, connect or launch processes")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(module, "Analyzer", forbidden)
    return module


def request(tmp_path, kind):
    source = tmp_path / "synthetic-source.png"
    Image.fromarray(np.full((80, 120, 3), 80, np.uint8)).save(source)
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "request.json", {"source": str(source), "info": {"kind": kind},
                                      "settings": Settings().to_dict()})
    return source, run


def decision(timestamp, verdict, track=7, rule=Rule.HELMET, box=(40, 20, 60, 40)):
    return Decision(rule, verdict, "Synthetic measured assessment", timestamp, track,
                    box, {"synthetic": True, "measured_pixels": 7.25,
                                      "uncertainty": "Synthetic input; not real traffic truth"})


def install_analyzer(worker, monkeypatch, rows, support="collector", fail_snapshot=False):
    instances = []

    class Collector(LifecycleAwareEventCollector):
        def confirmation_snapshot(self):
            if fail_snapshot:
                raise RuntimeError("Synthetic confirmation export failure")
            instance = instances[-1]
            instance.snapshot_calls.append(instance.index - 1)
            return super().confirmation_snapshot()

    class Analyzer:
        scene_revision, scene_masks = 1, {}

        def __init__(self, status, cancel):
            cancel()
            status("Synthetic export test double")
            self.scene = SimpleNamespace(stationary_camera=True)
            self.index, self.source_frames, self.snapshot_calls, self.created = 0, [], [], []
            if support == "collector":
                self.events = Collector()
            elif support == "no_accessor":
                self.events = object()
            elif support == "noncallable":
                self.events = SimpleNamespace(confirmation_snapshot=None)
            instances.append(self)

        def analyze(self, result, original, video, scene_change):
            timestamp, verdicts, x = rows[self.index]
            assert result.timestamp == timestamp
            vehicle = Detection("motorcycle", .95, (x - 10, 20, x + 10, 40), 7)
            assessments = [decision(timestamp, verdict, track, rule, vehicle.box) for verdict, track, rule in verdicts]
            self.source_frames.append(deepcopy([d.to_dict() for d in assessments]))
            created = []
            if support == "collector":
                created, _ = self.events.update(assessments, timestamp, scene_id=0, image=not video,
                                                 detections=[vehicle], camera_stable=True, scene_revision=1)
            self.created.extend(created)
            self.index += 1
            return result.color_rgb.copy(), FrameAnalysis(timestamp, [vehicle], assessments, 0), created

        def scene_snapshot(self):
            return {"scene_id": 0, "revision": 1, "synthetic": True}

        def summary(self):
            return {"synthetic": True, "events_by_rule": {}, "assessment_counts": {}}

    monkeypatch.setattr(worker, "Analyzer", Analyzer)
    return instances


def records(run):
    return [strict(line) for line in (run / "detections.jsonl").read_text().splitlines()]


def test_video_export_is_current_source_indexed_and_preserves_raw_assessments(worker, monkeypatch, tmp_path):
    rows = [(0., [(Verdict.VIOLATION, 7, Rule.HELMET)], 50),
            (.2, [(Verdict.UNKNOWN, 8, Rule.RIDERS), (Verdict.CLEAR, 7, Rule.HELMET)], 50),
            (.5, [(Verdict.VIOLATION, 7, Rule.HELMET)], 50),
            (.6, [(Verdict.VIOLATION, 7, Rule.HELMET), (Verdict.CLEAR, 7, Rule.HELMET)], 50),
            (.8, [(Verdict.VIOLATION, 7, Rule.HELMET)], 50),
            (1.3, [(Verdict.VIOLATION, 7, Rule.HELMET)], 50),
            (1.4, [(Verdict.UNKNOWN, 7, Rule.HELMET)], 50),
            (1.5, [(Verdict.VIOLATION, 7, Rule.HELMET)], 250)]
    instances = install_analyzer(worker, monkeypatch, rows)
    source, run = request(tmp_path, "video")
    original = source.read_bytes()

    def fake_video(run, frame_hook, final_hook):
        pixels = np.full((80, 120, 3), 80, np.uint8)
        for index, (timestamp, _, _) in enumerate(rows):
            frame_hook(SimpleNamespace(color_rgb=pixels, timestamp=timestamp), pixels, index, 0.)
        write_json(run / "report.json", {"kind": "video", "synthetic": True, "ai": final_hook()})
        worker.update(run, "complete", progress=1, message="Synthetic export test complete")

    monkeypatch.setattr(worker, "process_video", fake_video)
    worker.process_run(run)
    instance, saved = instances[0], records(run)
    assert len(saved) == len(rows) and instance.snapshot_calls == list(range(len(rows)))
    for index, frame in enumerate(saved):
        assert frame["frame"] == index and frame["timestamp"] == rows[index][0]
        assert frame["scene_revision"] == 1
        trace = frame["event_confirmation"]
        assert trace["timestamp"] == frame["timestamp"] == trace["lifecycle"]["timestamp"]
        assert trace["scene_id"] == frame["scene_id"] == 0
        assert trace["lifecycle"]["scene_revision"] == frame["scene_revision"]
        assert json.loads(json.dumps(instance.source_frames[index])) == frame["decisions"]
    assert saved[0]["event_confirmation"]["pending"][0]["observations"] == 1
    assert saved[1]["event_confirmation"]["cleared_pending"][0]["source_decision_indices"] == [1]
    conflict = saved[3]["event_confirmation"]["contradictions"][0]
    assert conflict["violation_decision_indices"] == [0] and conflict["clear_decision_indices"] == [1]
    assert saved[5]["event_confirmation"]["new_events"] == 1
    assert saved[6]["decisions"][0]["verdict"] == Verdict.UNKNOWN.value
    boundary = saved[7]["event_confirmation"]["lifecycle"]["boundaries"][0]
    assert boundary["reason"] == "tracking_position_jump" and boundary["discarded_states"]["active"] == 1
    emitted = [strict(line) for line in (run / "events.jsonl").read_text().splitlines()]
    assert len(emitted) == 1 and emitted[0]["started_at"] == .8 and emitted[0]["last_seen_at"] == 1.3
    assert emitted[0]["evidence"]["synthetic"] is True and emitted[0]["evidence"]["measured_pixels"] == 7.25
    assert strict((run / "status.json").read_text())["state"] == "complete" and source.read_bytes() == original


def test_image_worker_exports_current_confirmation_without_video_requirements(worker, monkeypatch, tmp_path):
    instances = install_analyzer(worker, monkeypatch, [(0., [(Verdict.VIOLATION, 7, Rule.HELMET)], 50)])
    source, run = request(tmp_path, "image")
    original = source.read_bytes()
    worker.process_run(run)
    frame = records(run)[0]
    assert frame["event_confirmation"]["image"] is True
    assert frame["event_confirmation"]["timestamp"] == frame["timestamp"] == 0.
    assert frame["event_confirmation"]["new_events"] == 1 and instances[0].snapshot_calls == [0]
    assert frame["decisions"][0]["evidence"]["synthetic"] is True
    assert source.read_bytes() == original and strict((run / "status.json").read_text())["state"] == "complete"


@pytest.mark.parametrize("support", ["no_events", "no_accessor", "noncallable"])
def test_optional_export_keeps_older_and_mocked_collectors_compatible(worker, monkeypatch, tmp_path, support):
    install_analyzer(worker, monkeypatch, [(0., [(Verdict.UNKNOWN, 7, Rule.HELMET)], 50)], support)
    _, run = request(tmp_path, "image")
    worker.process_run(run)
    frame = records(run)[0]
    assert "event_confirmation" not in frame
    assert frame["decisions"][0]["verdict"] == Verdict.UNKNOWN.value
    assert strict((run / "status.json").read_text())["state"] == "complete"


def test_confirmation_export_failure_cannot_publish_a_complete_report(worker, monkeypatch, tmp_path):
    install_analyzer(worker, monkeypatch, [(0., [(Verdict.UNKNOWN, 7, Rule.HELMET)], 50)], fail_snapshot=True)
    source, run = request(tmp_path, "image")
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="Synthetic confirmation export failure"):
        worker.process_run(run)
    assert strict((run / "status.json").read_text())["state"] == "failed"
    assert not (run / "report.json").exists() and source.read_bytes() == original
