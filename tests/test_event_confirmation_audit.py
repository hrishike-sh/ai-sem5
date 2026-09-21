import hashlib
import json

from scripts.audit_event_confirmation import audit
from traffic_ai.contracts import Rule, Verdict


def source(tmp_path, rows, rule=Rule.WRONG_WAY):
    run = tmp_path / "synthetic-run"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"state": "complete"}))
    (run / "report.json").write_text(json.dumps({"kind": "video", "synthetic": True, "ai": {}}))
    records = []
    for index, (timestamp, verdicts) in enumerate(rows):
        records.append({"frame": index, "timestamp": timestamp, "scene_id": 0, "decisions": [
            {"rule": rule.value, "verdict": verdict.value, "timestamp": timestamp, "track_id": 1,
             "box": [0, 0, 20, 20], "reason": "Synthetic software fixture", "evidence": {"synthetic": True}}
            for verdict in verdicts]})
    (run / "detections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return run


def test_saved_stream_exposes_premature_event_without_changing_source_labels(tmp_path):
    run = source(tmp_path, [(0., [Verdict.VIOLATION]), (.2, [Verdict.CLEAR]), (.5, [Verdict.VIOLATION])])
    original = {p.name: p.read_bytes() for p in run.iterdir()}
    result = audit(run, tmp_path / "audit.json")
    assert result["baseline_candidate_events"] == 1 and result["contradiction_aware_candidate_events"] == 0
    assert result["cleared_pending_states"] == 1 and result["frames_with_event_differences"] == 1
    assert result["source_verdict_counts"] == {"violation": 2, "no_violation_observed": 1}
    assert result["provenance"] == {"synthetic": True} and result["fresh_inference"] is False
    assert {p.name: p.read_bytes() for p in run.iterdir()} == original
    assert result["sources_sha256"]["detections.jsonl"] == hashlib.sha256(original["detections.jsonl"]).hexdigest()


def test_transition_contradiction_exports_both_source_indices_as_unresolved(tmp_path):
    run = source(tmp_path, [(0., [Verdict.VIOLATION, Verdict.CLEAR])], rule=Rule.RED_LIGHT)
    result = audit(run, tmp_path / "audit.json")
    assert result["baseline_candidate_events"] == 1 and result["contradiction_aware_candidate_events"] == 0
    assert result["contradictory_identity_rule_frames"] == 1
    conflict = result["examples"][0]["confirmation"]["contradictions"][0]
    assert conflict["violation_decision_indices"] == [0] and conflict["clear_decision_indices"] == [1]
