"""HTTP acceptance for imported v2 review cards; no browser/vision rendering.

Run in a primary-source snapshot. The old isolated baseline has no app bridge
and skips this module. All sources are synthetic, providers are test doubles.
"""
from copy import deepcopy
import hashlib
import json
import os
import urllib.error
import urllib.request

from PIL import Image
import pytest

pytest.importorskip("ivp.evidence_reviews", reason="Requires the primary app's imported review bridge")
import web_server
from ivp.evidence_reviews import EvidenceReviews
from ivp.media import write_json
from tests.test_evidence_review import write_run
from tests.test_review_integration import stopped_event as stopped_event
from tests.test_web_server import request, server as server
from tests.test_bounded_vision import fake_runtime as fake_runtime
from traffic_ai.review_cli import validate_review_bundle
from traffic_ai.review_llm import LocalReviewSelector


@pytest.fixture
def review_source(server, stopped_event):
    stop = deepcopy(stopped_event)
    stop["future_field"] = {"kept": True}
    unknown = deepcopy(stop)
    unknown.update(event_id="unknown-condition", timestamp_seconds=6., track_id=2, status="withheld",
                   reason="Applicable condition is unknown.", evidence_timestamp_seconds=6.)
    unknown["measurements"].update(condition="unknown", speed=None, stopped_in_approach=False,
                                  stop_evidence=None, crossing_bracket_seconds=[5.9,6.1])
    (web_server.ROOT/"runs").mkdir(exist_ok=True)
    source = write_run(web_server.ROOT/"runs"/"acceptance-source", [stop, unknown])
    rows = "timestamp_seconds,track_id,speed,speed_unit,speed_status,tracking_state\n"
    rows += "".join(f"{3.5+i/16},1,0.14,km/h,estimated,observed\n" for i in range(17))
    rows += "5.73,1,9.29,km/h,estimated,observed\n6.0,2,,km/h,unavailable,observed\n"
    (source/"motion-tracks.csv").write_text(rows, encoding="utf-8")
    return source, [stop, unknown]


def fingerprints(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}


def ready(server, source, **flags):
    started = request(server, "/api/evidence-review", {"source_run":source.name, **flags})
    server.lab.evidence_reviews.future.result(timeout=15)
    return request(server, "/api/evidence-review?id="+started["id"])


def test_imported_stop_and_unknown_cards_preserve_source_and_download(server, review_source):
    source, events = review_source
    before = fingerprints(source)
    result = ready(server, source)
    assert result["state"] == "complete" and result["source_integrity"]["state"] == "verified"
    cards = result["review"]["events"]
    assert [c["event"] for c in cards] == events
    assert cards[0]["stop_review"]["csv_coverage"] == "count_and_states_match"
    assert cards[0]["stop_context"]["sample_count"] == 17
    assert {"stop_duration", "stop_fitted_speed"} <= {f["id"] for f in cards[0]["facts"]}
    assert cards[1]["event"]["status"] == "withheld"
    artifact = source.parent/"evidence-reviews"/result["id"]/result["review"]["artifact"]
    validate_review_bundle(artifact)
    link = next(d["url"] for d in result["downloads"] if d["name"] == "review.json")
    with urllib.request.urlopen(link, timeout=5) as response:
        assert "attachment" in response.headers["Content-Disposition"]
        full = json.load(response)
    assert [r["event"] for r in full["events"]] == events
    assert before == fingerprints(source)


def test_optional_bad_stop_and_missing_thumbnail_do_not_hide_unknown_case(server, review_source):
    source, events = review_source
    events[0]["measurements"]["stop_evidence"] = {"duration_seconds":"malformed"}
    events[0]["thumbnail"] = "missing.png"
    write_run(source, events)
    result = ready(server, source)
    assert result["state"] == "complete"
    stop, unknown = result["review"]["events"]
    assert stop["stop_review"]["state"] == "inconsistent"
    assert "thumbnail_missing" in {w["code"] for w in stop["warnings"]}
    assert unknown["event"]["status"] == "withheld"


@pytest.mark.parametrize("fault", ["timeout", "unsupported_claim"])
def test_provider_fault_keeps_completed_stop_evidence_and_offline_artifact(server, review_source, monkeypatch, fault):
    source, events = review_source
    before = fingerprints(source)
    def model_response(self, route, payload=None):
        if fault == "timeout":
            raise TimeoutError("Synthetic provider timeout; no live model call")
        if route == "/api/tags":
            return {"models":[{"name":self.model,"digest":"synthetic-test-digest"}]}
        return {"done":True,"message":{"content":json.dumps({"fact_ids":["status"],"question_ids":[],"verdict":"violation"})}}
    monkeypatch.setattr(LocalReviewSelector, "_request", model_response)
    result = ready(server, source, use_model=True)
    assert result["state"] == "complete" and result["model_states"]["unavailable"] == 1
    assert [r["event"] for r in result["review"]["events"]] == events
    assert result["review"]["events"][0]["stop_review"]["state"] == "internally_consistent"
    job = source.parent/"evidence-reviews"/result["id"]
    validate_review_bundle(job/"offline")
    assert result["downloads"] and before == fingerprints(source)


def test_malformed_required_csv_and_active_source_never_publish_success(server, review_source):
    source, _ = review_source
    write_json(source/"status.json", {"state":"running"})
    with pytest.raises(urllib.error.HTTPError) as error:
        request(server, "/api/evidence-review", {"source_run":source.name})
    assert error.value.code == 400
    write_json(source/"status.json", {"state":"complete"})
    (source/"motion-tracks.csv").write_text("broken,csv\n1,2\n")
    before = fingerprints(source)
    result = ready(server, source)
    assert result["state"] == "failed" and not result["downloads"]
    assert "review" not in result and before == fingerprints(source)


def test_repeat_restart_and_source_change_keep_saved_snapshot_explicit(server, review_source):
    source, events = review_source
    first = ready(server, source)
    first_artifact = source.parent/"evidence-reviews"/first["id"]/first["review"]["artifact"]
    saved = fingerprints(first_artifact)
    second = ready(server, source)
    assert second["id"] != first["id"] and saved == fingerprints(first_artifact)
    job = source.parent/"evidence-reviews"/second["id"]
    write_json(job/"status.json", {"state":"running","phase":"local_model"})
    restored = EvidenceReviews(web_server.ROOT, server.lab.executor, server.lab.files).get(second["id"])
    assert restored["state"] == "interrupted" and restored["downloads"]
    assert [r["event"] for r in restored["review"]["events"]] == events
    csv = source/"motion-tracks.csv"
    old = csv.read_bytes(); stat = csv.stat()
    csv.write_bytes(old.replace(b"9.29", b"9.28"))
    os.utime(csv, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    changed = request(server, "/api/evidence-review?id="+first["id"])
    assert changed["source_integrity"]["state"] == "changed"
    assert [r["event"] for r in changed["review"]["events"]] == events
    assert saved == fingerprints(first_artifact)


def test_partial_model_record_write_never_publishes_completed_scene_result(server,tmp_path,monkeypatch,fake_runtime):
    pytest.importorskip("ivp.scene_reviews",reason="Requires the primary scene-review bridge")
    from pathlib import Path
    from traffic_ai import bounded_vision
    source=tmp_path/"pattern.png"
    image=Image.new("RGB",(48,32),"black")
    image.paste("white",(0,0,24,32));image.save(source)
    media=server.lab.register(source)
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(bounded_vision,"request_json",lambda *a,**k:{"choices":[{
        "finish_reason":"stop","message":{"content":json.dumps({"visible_features":["road"],"uncertainty_codes":[]})}}]})
    real=Path.write_text
    def interrupted(path,text,**kwargs):
        if path.name=="review.json" and path.parent.name=="model":
            real(path,"{",**kwargs)
            raise OSError("Synthetic interrupted model record write")
        return real(path,text,**kwargs)
    monkeypatch.setattr(Path,"write_text",interrupted)
    attempt=request(server,"/api/review",{"media":media["id"],"timestamp":0,"settings":{}})
    server.lab.scene_reviews.future.result(timeout=10)
    result=request(server,"/api/review?id="+attempt["id"])
    folder=server.lab.scene_reviews.folder(attempt["id"])
    assert result["state"]=="unavailable" and "review" not in result
    assert not (folder/"review.json").exists()
    assert (folder/"model/review.json").read_text()=="{"
    assert fake_runtime[0].metrics["owned_server_stopped"]
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
