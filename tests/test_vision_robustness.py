"""Malformed-output and wall-budget regressions; model/runtime are test doubles."""
import json

import numpy as np
import pytest

from traffic_ai import bounded_vision as vision


def test_deeply_nested_model_content_returns_explicit_fallback(tmp_path, monkeypatch):
    class Runtime:
        token = "fixture"
        metrics = {"owned_server_stopped": True}
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(vision, "ManagedVisionRuntime", Runtime)
    content = "["*1200+"0"+"]"*1200
    monkeypatch.setattr(vision, "request_json", lambda *a, **k: {
        "choices": [{"finish_reason":"stop", "message":{"content":content}}]})
    pixels = np.zeros((8,8,3),np.uint8)
    pixels[:,:4] = 255
    result = vision.review_image(pixels, output=tmp_path/"nested")
    assert result["state"] == "unavailable" and "observation" not in result
    saved = json.loads((tmp_path/"nested/review.json").read_text())
    assert saved["state"] == "unavailable" and saved["runtime"]["owned_server_stopped"]


def test_response_finishing_after_wall_deadline_is_rejected(monkeypatch):
    clock = [0.]
    class Socket:
        def settimeout(self, *args):
            pass
        def shutdown(self, *args):
            pass
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, *args):
            clock[0] = .02
            return b"{}"
    class Connection:
        sock = Socket()
        def __init__(self, *args, **kwargs):
            pass
        def connect(self):
            pass
        def request(self, *args, **kwargs):
            pass
        def getresponse(self):
            return Response()
        def close(self):
            pass
    class ImmediateThread:
        def __init__(self, target, **kwargs):
            self.target = target
        def start(self):
            self.target()
        def join(self, **kwargs):
            pass
    monkeypatch.setattr(vision.http.client, "HTTPConnection", Connection)
    monkeypatch.setattr(vision.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(vision.time, "monotonic", lambda:clock[0])
    with pytest.raises(vision.VisionUnavailable, match="time budget"):
        vision.request_json("/health", timeout=.01)


def test_health_response_cannot_extend_total_startup_budget(tmp_path, monkeypatch):
    clock = [0.]
    spec = vision.RuntimeSpec(tmp_path/"binary", tmp_path/"weights", tmp_path/"projector")
    spec.binary.write_bytes(b"fixture")
    spec.model.write_bytes(b"GGUF-fixture")
    spec.projector.write_bytes(b"GGUF-fixture")
    class Process:
        pid = 123
        stopped = False
        def poll(self):
            return 0 if self.stopped else None
        def terminate(self):
            self.stopped = True
        def wait(self, **kwargs):
            return 0
    process = Process()
    def health(*args, **kwargs):
        clock[0] = 1.1
        return {"status":"ok"}
    monkeypatch.setattr(vision.subprocess, "Popen", lambda *a, **k:process)
    monkeypatch.setattr(vision.socket, "create_connection", lambda *a, **k:(_ for _ in ()).throw(ConnectionRefusedError()))
    monkeypatch.setattr(vision, "request_json", health)
    monkeypatch.setattr(vision.time, "monotonic", lambda:clock[0])
    runtime = vision.ManagedVisionRuntime(spec,tmp_path,startup_timeout=1)
    with pytest.raises(vision.VisionUnavailable,match="startup exceeded"):
        with runtime:
            pytest.fail("Late readiness must be rejected")
    assert process.stopped and runtime.metrics["owned_server_stopped"] and not runtime.token


@pytest.mark.parametrize("color", [(0,0,0),(230,230,230),(20,100,240)])
def test_uniform_pixels_abstain_without_model_or_scene_clear_claim(tmp_path,monkeypatch,color):
    monkeypatch.setattr(vision.subprocess,"Popen",lambda *a,**k:pytest.fail("Uniform input must not load a model"))
    pixels = np.zeros((72,96,3),np.uint8)
    pixels[:] = color
    original = pixels.copy()
    result = vision.review_image(pixels,output=tmp_path/"uniform",synthetic=True)
    assert result["state"] == "unavailable" and result["phase"] == "input_validation"
    assert result["input_guard"] == "uniform_review_image" and result["inference_attempted"] is False
    assert "does not establish that a scene is clear" in result["message"]
    assert "observation" not in result and "selection" not in result and np.array_equal(pixels,original)
    assert result["runtime"] == {"owned_server_stopped":True}


def test_cancellation_wins_before_uniform_image_guard(tmp_path):
    result = vision.review_image(np.zeros((8,8,3),np.uint8),output=tmp_path/"cancel",cancel=lambda:True)
    assert result["state"] == "cancelled" and not result["inference_attempted"]
