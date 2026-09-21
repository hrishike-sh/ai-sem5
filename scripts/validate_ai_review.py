"""Replay a completed worker run and reviewer faults; never modify source exports.

Provider faults are test doubles. --live-local explicitly checks an already
running local server, without starting it or requesting weights/credentials.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_ai.evidence_review import ReviewError, load_review
from traffic_ai.review_cli import export_review, validate_review_bundle
from traffic_ai.review_llm import LocalReviewSelector


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fixture(folder, source, *, mutation=None):
    """An explicitly synthetic local adapter fixture, not a detector result."""
    folder.mkdir()
    event = deepcopy(source)
    event["synthetic"] = True
    event.setdefault("configuration", {})["synthetic"] = True
    event["thumbnail"] = None
    event["basis"] = "Synthetic reviewer fault fixture; not a fresh detector outcome."
    event["review_fixture"] = {"kind": "synthetic_adapter_fault", "mutation": mutation or "none"}
    if mutation == "missing_stop":
        event["measurements"].pop("stop_evidence", None)
    elif mutation == "malformed_stop":
        event["measurements"]["stop_evidence"] = {"duration_seconds": "invented by model"}
    (folder/"status.json").write_text('{"state":"complete"}', encoding="utf-8")
    (folder/"motion-incidents.jsonl").write_text(json.dumps(event, allow_nan=False)+"\n", encoding="utf-8")
    return folder


def validate(run, output, *, live_local=False, model_file=None):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output == run or run in output.parents or output in run.parents:
        raise ReviewError("Validation output must be a new directory outside the source run.")
    original = load_review(run)
    before = {f["path"]: f["sha256"] for f in original["source"]["files"]}
    output.mkdir(parents=True)
    records = []

    def record(name, kind, check, **details):
        records.append({"name": name, "kind": kind, "passed": bool(check), **details})

    # Exercise the actual CLI as a separate process, including output completion.
    cli = subprocess.run([sys.executable, "-m", "traffic_ai.review_cli", str(run),
                          "--output", str(output/"worker-review")], cwd=ROOT,
                         capture_output=True, text=True, timeout=60)
    if cli.returncode != 0:
        raise ReviewError("Worker review CLI failed: "+cli.stderr)
    validate_review_bundle(output/"worker-review")
    reviewed = json.loads((output/"worker-review/review.json").read_text(encoding="utf-8"))
    record("worker_event_preservation", "actual_saved_worker_export",
           [r["event"] for r in reviewed["events"]] == [r["event"] for r in original["events"]],
           event_count=len(original["events"]))
    stops = [r for r in reviewed["events"] if r["event"].get("measurements", {}).get("stop_evidence")]
    record("current_stop_contract", "actual_saved_worker_export", bool(stops) and all(
        r.get("stop_review", {}).get("state") == "internally_consistent" and
        r["stop_review"]["csv_coverage"] == "count_and_states_match" for r in stops),
        stop_records=len(stops))
    # Repeated invocation must retain the prior bundle exactly.
    bundle_before = {p.name: _sha(p) for p in (output/"worker-review").iterdir()}
    try:
        export_review(run, output/"worker-review")
    except ReviewError:
        refused = True
    else:
        refused = False
    record("repeat_preserves_existing_bundle", "filesystem", refused and bundle_before == {
        p.name: _sha(p) for p in (output/"worker-review").iterdir()})
    again = export_review(run, output/"worker-review-again")
    record("fresh_repeat_preserves_events", "filesystem", [r["event"] for r in again["events"]] ==
           [r["event"] for r in original["events"]])
    seed = (stops or reviewed["events"])[0]["event"] if reviewed["events"] else None
    if seed is None:
        raise ReviewError("This integration validation needs at least one exported event.")
    for mutation in ("missing_stop", "malformed_stop"):
        fixture = _fixture(output/(mutation+"-input"), seed, mutation=mutation)
        doc = export_review(fixture, output/(mutation+"-review"))
        item = doc["events"][0]
        record(mutation, "synthetic_fault_fixture", item.get("stop_review", {}).get("state") in
               ("missing", "inconsistent") and item["event"]["status"] == seed["status"])
    missing = _fixture(output/"missing-status-input", seed)
    (missing/"status.json").unlink()
    try:
        export_review(missing, output/"missing-status-review")
    except ReviewError:
        refused = True
    else:
        refused = False
    record("missing_required_status", "synthetic_fault_fixture", refused and not (output/"missing-status-review").exists())
    broken_csv = _fixture(output/"broken-csv-input", seed)
    (broken_csv/"motion-tracks.csv").write_text("timestamp_seconds,track_id\ninvalid,1\n", encoding="utf-8")
    try:
        export_review(broken_csv, output/"broken-csv-review")
    except ReviewError:
        refused = True
    else:
        refused = False
    record("malformed_track_csv", "synthetic_fault_fixture", refused and not (output/"broken-csv-review").exists())

    for fault in ("timeout", "unsupported_claim", "invalid_json"):
        class FaultProvider(LocalReviewSelector):
            def _request(self, route, payload=None):
                if fault == "timeout":
                    raise TimeoutError("Synthetic timeout; no network request occurred")
                if route == "/api/tags":
                    return {"models": [{"name": self.model, "digest": "synthetic-model-digest"}]}
                value = {"fact_ids": ["status"], "question_ids": [], "verdict": "violation"}
                content = json.dumps(value) if fault == "unsupported_claim" else "{not JSON"
                return {"done": True, "message": {"content": content}}
        with patch("traffic_ai.review_llm.LocalReviewSelector", FaultProvider):
            doc = export_review(run, output/(fault+"-fallback"), local_llm=True)
        validate_review_bundle(output/(fault+"-fallback"))
        record(fault, "mocked_provider_response", doc["events"][0]["language_model"]["state"] == "unavailable" and
               [r["event"] for r in doc["events"]] == [r["event"] for r in original["events"]])
    if live_local:
        doc = export_review(run, output/"live-local", local_llm=True, max_llm_events=4,
                            local_provider="llama-cpp" if model_file else "ollama", model_file=model_file)
        models = [r["language_model"] for r in doc["events"][:4]]
        record("live_local_selection", "live_model_inference", all(m["state"] == "selected" for m in models),
               selections=models, event_data_preserved=[r["event"] for r in doc["events"]] ==
               [r["event"] for r in original["events"]])
    after = {name: _sha(run/name) for name in before}
    record("source_exports_unchanged", "filesystem", before == after)
    result = {"kind": "ai_review_integration_validation", "created_at": datetime.now(timezone.utc).isoformat(),
              "source_run": str(run), "source_manifest": before, "passed": sum(r["passed"] for r in records),
              "total": len(records), "cases": records,
              "note": "Worker exports are replayed read-only. Fault cases and mocked providers do not establish detector accuracy or live model performance."}
    (output/"validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live-local", action="store_true")
    parser.add_argument("--model-file", type=Path)
    args = parser.parse_args()
    if args.model_file and not args.live_local:
        parser.error("--model-file requires --live-local")
    try:
        result = validate(args.run, args.output, live_local=args.live_local, model_file=args.model_file)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print("Validation failed: "+str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"passed": result["passed"], "total": result["total"], "receipt": str(args.output/"validation.json")}))
    return int(result["passed"] != result["total"])


if __name__ == "__main__":
    raise SystemExit(main())
