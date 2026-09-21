"""No-model malformed-protocol, snapshot and output-failure checks."""
import base64
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from traffic_ai import bounded_vision as vision


def pixels():
    value=np.zeros((32,48,3),np.uint8)
    value[:,:24]=255
    return value


def completion(content=None,finish="stop"):
    if content is None:content=json.dumps({"visible_features":["road"],"uncertainty_codes":["visibility"]})
    return {"choices":[{"finish_reason":finish,"message":{"content":content}}]}


@pytest.fixture
def no_model(monkeypatch):
    class Runtime:
        token="fixture"
        def __init__(self,*args,**kwargs):self.metrics={"owned_server_stopped":False}
        def __enter__(self):return self
        def __exit__(self,*args):self.metrics["owned_server_stopped"]=True
    monkeypatch.setattr(vision,"ManagedVisionRuntime",Runtime)


@pytest.mark.parametrize("options", [
    {"timeout":0},{"timeout":float("nan")},{"timeout":float("inf")},{"timeout":121},{"timeout":True},
    {"startup_timeout":0},{"startup_timeout":61},{"timestamp":-1},{"timestamp":float("nan")},
    {"timestamp":True},{"synthetic":"true"}])
def test_invalid_parameters_do_not_create_output_or_start_model(tmp_path,monkeypatch,options):
    monkeypatch.setattr(vision.subprocess,"Popen",lambda *a,**k:pytest.fail("Invalid input must not launch"))
    with pytest.raises(ValueError):vision.review_image(pixels(),output=tmp_path/"out",**options)
    assert not (tmp_path/"out").exists()


@pytest.mark.parametrize("response",[
    {"choices":[]},{"choices":{}},{"choices":[{}]},
    completion("null"),completion('{"visible_features":[],"visible_features":["car"],"uncertainty_codes":[]}'),
    completion('{"visible_features":["car"],"uncertainty_codes":[],"speed_kmh":99}'),
    completion("["*1200+"0"+"]"*1200),completion(finish="tool_calls"),
    completion(finish="\ud800"),completion(finish={"unexpected":"object"}),
])
def test_malformed_protocol_returns_parseable_unavailable_record(tmp_path,monkeypatch,no_model,response):
    monkeypatch.setattr(vision,"request_json",lambda *a,**k:response)
    result=vision.review_image(pixels(),output=tmp_path/"out")
    assert result["state"]=="unavailable" and "observation" not in result and "selection" not in result
    saved=json.loads((tmp_path/"out/review.json").read_text(encoding="utf-8"))
    assert saved["state"]=="unavailable" and saved["runtime"]["owned_server_stopped"]
    finish=response.get("choices")
    if isinstance(finish,list) and finish and isinstance(finish[0],dict):
        raw=finish[0].get("finish_reason")
        if raw not in ("stop","length","tool_calls","content_filter","error"):
            assert saved.get("finish_reason")=="invalid"


def test_array_change_after_snapshot_does_not_change_model_or_saved_pixels(tmp_path,monkeypatch,no_model):
    original=pixels()
    captured={}
    def response(route,payload,**kwargs):
        original[:]=100
        url=payload["messages"][1]["content"][1]["image_url"]["url"]
        captured["png"]=base64.b64decode(url.split(",",1)[1])
        return completion()
    monkeypatch.setattr(vision,"request_json",response)
    result=vision.review_image(original,output=tmp_path/"out")
    saved=(tmp_path/"out/review-input.png").read_bytes()
    assert result["state"]=="complete" and saved==captured["png"]
    assert hashlib.sha256(saved).hexdigest()==result["review_image_sha256"]
    assert np.all(original==100)


def test_existing_output_is_refused_without_replacing_saved_attempt(tmp_path,monkeypatch,no_model):
    monkeypatch.setattr(vision,"request_json",lambda *a,**k:completion())
    vision.review_image(pixels(),output=tmp_path/"out")
    before={p.name:p.read_bytes() for p in (tmp_path/"out").iterdir()}
    with pytest.raises(FileExistsError):vision.review_image(pixels(),output=tmp_path/"out")
    assert before=={p.name:p.read_bytes() for p in (tmp_path/"out").iterdir()}


def test_failed_record_write_does_not_return_a_completed_attempt(tmp_path,monkeypatch,no_model):
    monkeypatch.setattr(vision,"request_json",lambda *a,**k:completion())
    real=Path.write_text
    def broken(path,text,**kwargs):
        if path.name=="review.json":
            real(path,"{",**kwargs)
            raise OSError("Synthetic interrupted write")
        return real(path,text,**kwargs)
    monkeypatch.setattr(Path,"write_text",broken)
    with pytest.raises(OSError,match="interrupted write"):
        vision.review_image(pixels(),output=tmp_path/"out")
    with pytest.raises(json.JSONDecodeError):json.loads((tmp_path/"out/review.json").read_text())
