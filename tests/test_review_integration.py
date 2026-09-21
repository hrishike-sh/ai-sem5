"""Current completed-stop contract and failure-path integration (synthetic inputs)."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from traffic_ai.evidence_review import ReviewError, load_review, review_event
from traffic_ai.review_cli import export_review, validate_review_bundle
from traffic_ai.review_llm import _catalog, apply_local_selection
from tests.test_evidence_review import write_run


@pytest.fixture
def stopped_event():
    return {"event_id": "incident-stop", "timestamp_seconds": 5.61, "track_id": 1,
            "type": "line_crossing", "status": "observation", "reason": "Recorded continuous low-motion interval.",
            "synthetic": True, "source_revision": "a"*64,
            "quality": {"tracking_state": "observed", "detector_confidence": .9, "speed_status": "estimated"},
            "measurements": {"speed": 9.29, "speed_unit": "km/h", "condition": "stop_required",
                "stopped_in_approach": True, "crossing_bracket_seconds": [5.53, 5.73],
                "stop_evidence": {"method": "Continuous observed calibrated positions",
                    "start_seconds": 3.5, "end_seconds": 4.5, "duration_seconds": 1., "observed_samples": 17,
                    "position_diameter_bound_m": .42, "maximum_position_diameter_m": 2/3.6,
                    "fitted_speed_kmh": .14, "maximum_fitted_speed_kmh": 2.}},
            "configuration": {"synthetic": True, "calibration_confirmed": True, "real_time_confirmed": True},
            "calibration": {"points": [[0,0],[1,0],[1,1],[0,1]], "width_m": 16, "length_m": 40,
                            "source_revision": "a"*64, "media_id": "LOCAL_PRIVATE_MEDIA_ID"},
            "evidence_timestamp_seconds": 5.73, "thumbnail": None}


def codes(review):
    return {w["code"] for w in review["warnings"]}


def test_current_stop_proof_has_cited_facts_separate_from_crossing_speed(stopped_event):
    review = review_event(stopped_event, {"file": "motion-incidents.jsonl", "line": 1})
    assert review["event"] == stopped_event
    assert review["stop_review"]["state"] == "internally_consistent"
    facts = {f["id"]: f for f in review["facts"]}
    assert facts["stop_duration"]["value"] == 1.
    assert facts["stop_samples"]["value"] == 17
    assert facts["stop_fitted_speed"]["value"] == .14
    assert facts["stop_fitted_speed"]["pointer"] == "/measurements/stop_evidence/fitted_speed_kmh"
    assert facts["speed"]["value"]["speed"] == 9.29
    assert "stop_window_timing" in facts
    assert "exactly stationary" in review["stop_review"]["limitation"]


@pytest.mark.parametrize("problem", ["missing", "null", "string", "duration", "duration_mismatch", "too_few",
    "diameter", "fitted_speed", "inflated_policy", "future", "contradiction", "unconfirmed", "wrong_source"])
def test_missing_malformed_or_unsupported_proof_never_becomes_a_stop_finding(stopped_event, problem):
    proof = stopped_event["measurements"]["stop_evidence"]
    if problem == "missing":
        del stopped_event["measurements"]["stop_evidence"]
    elif problem == "null":
        stopped_event["measurements"]["stop_evidence"] = None
    elif problem == "string":
        stopped_event["measurements"]["stop_evidence"] = "model says stopped"
    elif problem == "duration":
        proof["duration_seconds"] = "1 second"
    elif problem == "duration_mismatch":
        proof["duration_seconds"] = 9
    elif problem == "too_few":
        proof["observed_samples"] = 2
    elif problem == "diameter":
        proof["position_diameter_bound_m"] = 2
    elif problem == "fitted_speed":
        proof["fitted_speed_kmh"] = 7
    elif problem == "inflated_policy":
        proof["maximum_fitted_speed_kmh"] = 20
    elif problem == "future":
        proof.update(start_seconds=10., end_seconds=11.)
    elif problem == "contradiction":
        stopped_event["measurements"]["stopped_in_approach"] = False
    elif problem == "unconfirmed":
        stopped_event["configuration"]["real_time_confirmed"] = False
    elif problem == "wrong_source":
        stopped_event["calibration"]["source_revision"] = "b"*64
    before = deepcopy(stopped_event)
    review = review_event(stopped_event, {})
    assert review["event"] == before
    assert review["event"]["status"] == "observation"
    assert review["stop_review"]["state"] in {"missing", "inconsistent"}
    assert any(code.startswith("stop_") for code in codes(review))


def test_proof_csv_excerpt_covers_earlier_window_and_reports_counts(tmp_path, stopped_event):
    run = write_run(tmp_path, [stopped_event])
    rows = "timestamp_seconds,track_id,speed,speed_unit,speed_status,tracking_state\n"
    rows += "".join(f"{3.5+i/16},1,0.14,km/h,estimated,observed\n" for i in range(17))
    rows += "5.73,1,9.29,km/h,estimated,observed\n"
    (run/"motion-tracks.csv").write_text(rows)
    review = load_review(run)["events"][0]
    context = review["stop_context"]
    assert context["requested_window_seconds"] == [3.5, 4.5]
    assert context["sample_count"] == 17 and context["observed_rows"] == 17
    assert context["samples"][0]["values"]["timestamp_seconds"] == "3.5"
    assert review["stop_review"]["csv_coverage"] == "count_and_states_match"
    assert review["track_context"]["sample_count"] < 18


def test_missing_proof_rows_and_uncertain_tracking_remain_explicit(tmp_path, stopped_event):
    run = write_run(tmp_path, [stopped_event])
    (run/"motion-tracks.csv").write_text("timestamp_seconds,track_id,speed,speed_unit,speed_status,tracking_state\n"
                                       "3.5,1,,km/h,collecting stable track,ambiguous overlap\n")
    review = load_review(run)["events"][0]
    assert review["stop_review"]["csv_coverage"] == "incomplete_or_conflicting"
    assert "stop_csv_evidence_mismatch" in codes(review)
    assert review["event"]["status"] == "observation"


def test_missing_optional_csv_does_not_hide_proof_or_claim_verification(tmp_path, stopped_event):
    write_run(tmp_path, [stopped_event])
    review = load_review(tmp_path)["events"][0]
    assert review["stop_review"]["state"] == "internally_consistent"
    assert review["stop_review"]["csv_coverage"] == "unavailable"
    assert "stop_csv_unavailable" in codes(review)


def test_local_catalog_does_not_include_calibration_identifiers(stopped_event):
    review = review_event(stopped_event, {})
    messages, _ = _catalog(review)
    content = json.dumps(messages)
    assert "LOCAL_PRIVATE_MEDIA_ID" not in content and "a"*64 not in content
    assert "stop_duration" in content


def test_provider_cannot_mutate_original_review_or_add_legal_claims(tmp_path, stopped_event):
    write_run(tmp_path, [stopped_event])
    document = load_review(tmp_path)
    before = deepcopy(document["events"][0])
    class MutatingProvider:
        def select(self, review):
            review["event"]["status"] = "violation"
            review["facts"].append({"id": "convicted", "text": "Illegal act proved"})
            return {"state": "selected", "fact_ids": ["convicted"], "question_ids": []}
    apply_local_selection(document, MutatingProvider())
    after = document["events"][0]
    assert after["event"] == before["event"] and after["facts"] == before["facts"]
    assert after["language_model"]["state"] == "unavailable"


def test_failed_bundle_write_leaves_output_retriable(tmp_path, stopped_event, monkeypatch):
    run, output = tmp_path/"run", tmp_path/"review"
    write_run(run, [stopped_event])
    original = Path.write_text
    def fail_html(path, *args, **kwargs):
        if path.name == "review.html":
            raise OSError("simulated disk write failure")
        return original(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", fail_html)
        with pytest.raises(OSError, match="simulated"):
            export_review(run, output)
    assert not output.exists()
    document = export_review(run, output)
    marker = json.loads((output/"review-complete.json").read_text())
    assert marker["files"]["review.json"] == hashlib.sha256((output/"review.json").read_bytes()).hexdigest()
    assert document["events"][0]["event"] == stopped_event
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ReviewError, match="already exists"):
        export_review(run, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    second = export_review(run, tmp_path/"review-again")
    assert second["events"][0]["event"] == stopped_event


def test_bundle_tampering_and_missing_completion_are_visible(tmp_path, stopped_event):
    run = write_run(tmp_path/"run", [stopped_event])
    output = tmp_path/"review"
    export_review(run, output)
    assert validate_review_bundle(output)["complete"] is True
    (output/"review.html").write_text("changed report")
    with pytest.raises(ReviewError, match="completion hash"):
        validate_review_bundle(output)
    (output/"review-complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        validate_review_bundle(output)


def test_publication_copy_failure_cleans_only_its_own_output(tmp_path, stopped_event, monkeypatch):
    import traffic_ai.review_cli as cli
    run = write_run(tmp_path/"run", [stopped_event])
    output = tmp_path/"review"
    original = cli.shutil.copyfileobj
    calls = []
    def fail_second(source, target, *args, **kwargs):
        calls.append(target.name)
        if len(calls) == 2:
            target.write(b"partial")
            raise OSError("copy failed")
        return original(source, target, *args, **kwargs)
    monkeypatch.setattr(cli.shutil, "copyfileobj", fail_second)
    with pytest.raises(OSError, match="copy failed"):
        export_review(run, output)
    assert not output.exists()


def test_actual_loopback_provider_timeout_retains_completed_offline_report(tmp_path, stopped_event, monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    import time
    from traffic_ai import review_llm

    class SlowInventory(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            time.sleep(.15)
            try:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"models":[]}')
            except OSError:
                pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), SlowInventory)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(review_llm.LocalReviewSelector, "endpoint", f"http://127.0.0.1:{server.server_port}")
    try:
        run = write_run(tmp_path/"run", [stopped_event])
        before = {p.name: p.read_bytes() for p in run.iterdir()}
        output = tmp_path/"review"
        document = export_review(run, output, local_llm=True, llm_timeout=.02)
        assert document["events"][0]["language_model"]["state"] == "unavailable"
        assert document["events"][0]["stop_review"]["state"] == "internally_consistent"
        assert validate_review_bundle(output)["complete"] is True
        assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
