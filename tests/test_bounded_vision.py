"""Bounded local vision control/transport tests; model responses are test doubles."""
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image
import pytest

from traffic_ai import bounded_vision as vision


SELECTION = {"visible_features": ["car", "road", "crosswalk"],
             "uncertainty_codes": ["small_details"]}


def patterned_pixels(height=16, width=16):
    pixels = np.zeros((height,width,3),np.uint8)
    pixels[:,:width//2] = 255
    return pixels


@pytest.mark.parametrize("bad", [
    {**SELECTION, "verdict": "violation"}, {**SELECTION, "visible_features": ["speeding"]},
    {**SELECTION, "visible_features": ["car", "car"]}, {**SELECTION, "uncertainty_codes": ["legally_clear"]},
    {**SELECTION, "sign_text_or_symbols": ["x"*81]}, {**SELECTION, "visible_features": "car"},
])
def test_model_cannot_introduce_spatial_speed_or_rule_fields(bad):
    with pytest.raises(vision.VisionUnavailable):
        vision.validate_selection(bad)


def test_summary_is_constructed_from_categories_without_invented_relationships():
    value = vision.validate_selection(deepcopy(SELECTION))
    assert "car" in value["summary"] and "crossing markings" in value["summary"]
    assert "on the crosswalk" not in value["summary"]
    assert "object-to-marking relationships" in value["uncertainties"][-1]
    empty = vision.validate_selection({"visible_features": [], "uncertainty_codes": []})
    assert "does not establish that the scene is clear" in empty["summary"]


@pytest.fixture
def fake_runtime(monkeypatch):
    instances = []
    class Runtime:
        def __init__(self, spec, folder, **kwargs):
            self.metrics = {"owned_server_stopped": False, "startup_seconds": .01}
            self.token = "fixture-auth"
            instances.append(self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.metrics["owned_server_stopped"] = True
            self.token = ""
    monkeypatch.setattr(vision, "ManagedVisionRuntime", Runtime)
    return instances


def test_complete_record_has_image_provenance_and_no_auth_material(tmp_path, monkeypatch, fake_runtime):
    calls = []
    def response(route, payload=None, **kwargs):
        calls.append((route, payload, kwargs))
        return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(SELECTION)}}]}
    monkeypatch.setattr(vision, "request_json", response)
    pixels = patterned_pixels(640,1280)
    original = pixels.copy()
    record = vision.review_image(pixels, 1.25, output=tmp_path/"review", synthetic=True)
    assert record["state"] == "complete" and record["source_size"] == [1280, 640]
    assert record["review_size"] == [512, 256] and record["synthetic"] is True
    assert record["grounding_enabled"] is False and record["runtime"]["owned_server_stopped"]
    assert "fixture-auth" not in json.dumps(record)
    assert calls[0][1]["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert np.array_equal(pixels, original)
    assert fake_runtime[0].token == ""


@pytest.mark.parametrize("failure", ["timeout", "length", "bad_json", "unsupported_claim"])
def test_failure_has_no_observation_and_stops_owned_runtime(tmp_path, monkeypatch, fake_runtime, failure):
    def response(*args, **kwargs):
        if failure == "timeout":
            raise vision.VisionUnavailable("Local vision request exceeded its time budget.")
        value = {**SELECTION, "speed_kmh": 99} if failure == "unsupported_claim" else SELECTION
        return {"choices": [{"finish_reason": "length" if failure == "length" else "stop",
                             "message": {"content": "{broken" if failure == "bad_json" else json.dumps(value)}}]}
    monkeypatch.setattr(vision, "request_json", response)
    result = vision.review_image(patterned_pixels(), output=tmp_path/"review")
    assert result["state"] == "unavailable" and "observation" not in result
    assert result["runtime"]["owned_server_stopped"]
    assert json.loads((tmp_path/"review/review.json").read_text())["state"] == "unavailable"


def test_cancel_before_start_does_not_spawn_or_invent_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(vision.subprocess, "Popen", lambda *a, **k: pytest.fail("Must not start a model"))
    record = vision.review_image(np.zeros((8,8,3), np.uint8), output=tmp_path/"cancelled", cancel=lambda: True)
    assert record["state"] == "cancelled" and "observation" not in record
    assert record["runtime"]["owned_server_stopped"]


@pytest.mark.parametrize("cleanup_error", [OSError("fixture failure"), subprocess.TimeoutExpired("fixture", 2)])
def test_failed_cleanup_discards_already_parsed_observation(tmp_path, monkeypatch, fake_runtime, cleanup_error):
    runtime_class = vision.ManagedVisionRuntime
    def fail_cleanup(self, *args):
        self.token = ""
        raise cleanup_error
    monkeypatch.setattr(runtime_class, "__exit__", fail_cleanup)
    monkeypatch.setattr(vision, "request_json", lambda *a, **k: {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(SELECTION)}}]})
    result = vision.review_image(patterned_pixels(), output=tmp_path/"failed-cleanup")
    assert result["state"] == "unavailable" and not result["runtime"]["owned_server_stopped"]
    assert "observation" not in result and "selection" not in result
    assert "cleanup could not be confirmed" in result["message"]


def test_existing_listener_is_left_untouched(tmp_path, monkeypatch):
    spec = vision.RuntimeSpec(tmp_path/"binary.exe", tmp_path/"model", tmp_path/"projector")
    spec.binary.write_bytes(b"test")
    spec.model.write_bytes(b"GGUF-test")
    spec.projector.write_bytes(b"GGUF-test")
    class Existing:
        closed = False
        def close(self):
            self.closed = True
    existing = Existing()
    monkeypatch.setattr(vision.socket, "create_connection", lambda *a, **k: existing)
    monkeypatch.setattr(vision.subprocess, "Popen", lambda *a, **k: pytest.fail("Existing server must not be replaced"))
    runtime = vision.ManagedVisionRuntime(spec, tmp_path, startup_timeout=1)
    with pytest.raises(vision.VisionUnavailable, match="busy"):
        runtime.__enter__()
    assert existing.closed and runtime.metrics["owned_server_stopped"]


def test_owned_server_is_cpu_only_hidden_offline_and_token_is_ephemeral(tmp_path, monkeypatch):
    spec = vision.RuntimeSpec(tmp_path/"binary.exe", tmp_path/"model", tmp_path/"projector")
    spec.binary.write_bytes(b"test")
    spec.model.write_bytes(b"GGUF-test")
    spec.projector.write_bytes(b"GGUF-test")
    captured = {}
    class Process:
        pid = 123
        returncode = None
        def poll(self):
            return self.returncode
        def terminate(self):
            self.returncode = 1
        def wait(self, **kwargs):
            return self.returncode
    def start(command, **kwargs):
        captured.update(command=command, **kwargs)
        return Process()
    monkeypatch.setattr(vision.subprocess, "Popen", start)
    monkeypatch.setattr(vision.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(ConnectionRefusedError()))
    monkeypatch.setattr(vision, "request_json", lambda *a, **k: {"status": "ok"})
    previous = os.environ.get("LLAMA_API_KEY")
    with vision.ManagedVisionRuntime(spec, tmp_path, startup_timeout=1) as runtime:
        assert runtime.token and captured["env"]["LLAMA_API_KEY"] == runtime.token
        assert runtime.token not in " ".join(captured["command"])
        assert captured["command"][captured["command"].index("--device")+1] == "none"
        assert "--offline" in captured["command"] and "--no-webui" in captured["command"]
        if os.name == "nt":
            assert captured["creationflags"] == subprocess.CREATE_NO_WINDOW
    assert runtime.metrics["owned_server_stopped"] and not runtime.token
    assert os.environ.get("LLAMA_API_KEY") == previous


@pytest.fixture
def slow_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            if self.path == "/health":
                self.send_response(302)
                self.send_header("Location", "https://example.invalid/not-followed")
                self.end_headers()
        def do_POST(self):
            time.sleep(.6)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"{}")
            except OSError:
                pass
    server = ThreadingHTTPServer(("127.0.0.1",0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_real_transport_cancellation_interrupts_inflight_socket(slow_server):
    cancel = threading.Event()
    timer = threading.Timer(.05, cancel.set)
    timer.start()
    start = time.monotonic()
    try:
        with pytest.raises(vision.VisionCancelled):
            vision.request_json("/v1/chat/completions", {}, timeout=2, cancel=cancel.is_set, port=slow_server.server_port)
    finally:
        timer.cancel()
    assert time.monotonic()-start < 1
    assert not any(t.name == "local-vision-request" and t.is_alive() for t in threading.enumerate())


def test_real_transport_deadline_and_redirect_are_not_followed(slow_server):
    with pytest.raises(vision.VisionUnavailable):
        vision.request_json("/v1/chat/completions", {}, timeout=.03, port=slow_server.server_port)
    with pytest.raises(vision.VisionUnavailable, match="302"):
        vision.request_json("/health", timeout=.3, port=slow_server.server_port)


def test_cancel_cli_saves_explicit_result_without_starting_model(tmp_path):
    source = tmp_path/"synthetic.png"
    Image.new("RGB", (32,32), "black").save(source)
    before = source.read_bytes()
    result = subprocess.run([sys.executable, "scripts/review_local_vision.py", str(source), "--output", str(tmp_path/"out"),
                             "--synthetic", "--cancel-after", "0"], capture_output=True, text=True, timeout=10)
    assert result.returncode == 130, result.stderr
    record = json.loads(result.stdout)
    assert record["state"] == "cancelled" and record["synthetic"] is True
    assert "observation" not in record and source.read_bytes() == before


def test_legacy_app_call_gets_explicit_failure_record(tmp_path, monkeypatch):
    from traffic_ai import local_vision
    monkeypatch.setattr(vision, "review_image", lambda *a, **k: {"state":"unavailable", "message":"budget exhausted"})
    with pytest.raises(vision.VisionUnavailable) as error:
        local_vision.review_frame(np.zeros((8,8,3), np.uint8))
    assert error.value.record["state"] == "unavailable"
