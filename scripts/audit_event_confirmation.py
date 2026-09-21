"""Compare current event confirmation policies on saved AI assessments only.

No detector, media decode or source-rule recomputation. Derived event descriptors
are policy-replay candidates, not replacement source findings or accuracy labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from traffic_ai.contracts import Decision, Rule, Verdict
from traffic_ai.event_continuity import ContradictionAwareEventCollector
from traffic_ai.events import EventCollector


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def descriptor(event):
    return {"track_id": event.track_id, "rule": event.rule.value, "started_at": event.started_at,
            "last_seen_at": event.last_seen_at, "observations": event.observations, "box": list(event.box),
            "evidence_sha256": hashlib.sha256(json.dumps(event.evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()}


def audit(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError("Output must be new and outside the source run")
    files = ("status.json", "report.json", "detections.jsonl")
    hashes = {name: sha(run / name) for name in files}
    if json.loads((run / "status.json").read_text())["state"] != "complete":
        raise ValueError("A completed saved run is required")
    source_report = json.loads((run / "report.json").read_text())
    if "ai" not in source_report or source_report.get("kind") not in ("image", "video"):
        raise ValueError("A saved AI image/video assessment stream is required")
    old, new = EventCollector(), ContradictionAwareEventCollector()
    counts, examples = Counter(), []
    frame_count = old_events = new_events = different_frames = cleared = conflicts = 0
    with (run / "detections.jsonl").open("rb") as stream:
        for line_number, raw in enumerate(stream, 1):
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("Frame record exceeds audit limit")
            if not raw.strip():
                continue
            frame = json.loads(raw)
            decisions = []
            for value in frame["decisions"]:
                evidence = value.get("evidence", {})
                if not isinstance(evidence, dict):
                    raise ValueError("Decision evidence must be an object")
                decisions.append(Decision(Rule(value["rule"]), Verdict(value["verdict"]), value["reason"],
                                          value["timestamp"], value.get("track_id"),
                                          tuple(value["box"]) if value.get("box") is not None else None, evidence))
                counts[value["verdict"]] += 1
            arguments = {"timestamp": frame["timestamp"], "scene_id": frame["scene_id"],
                         "image": source_report["kind"] == "image"}
            before, _ = old.update(decisions, **arguments)
            after, _ = new.update(decisions, **arguments)
            snapshot = new.confirmation_snapshot()
            old_events += len(before)
            new_events += len(after)
            first, second = [descriptor(e) for e in before], [descriptor(e) for e in after]
            changed = first != second
            different_frames += int(changed)
            cleared += len(snapshot["cleared_pending"])
            conflicts += len(snapshot["contradictions"])
            if len(examples) < 64 and (changed or snapshot["cleared_pending"] or snapshot["contradictions"]):
                examples.append({"source_line": line_number, "source_line_sha256": hashlib.sha256(raw).hexdigest(),
                                 "frame": frame.get("frame"), "timestamp": frame["timestamp"],
                                 "baseline_candidates": first, "contradiction_aware_candidates": second,
                                 "confirmation": snapshot})
            frame_count += 1
    if any(sha(run / name) != value for name, value in hashes.items()):
        raise ValueError("Source changed during audit")
    result = {"kind": "event_confirmation_policy_audit", "completed": True, "run_id": run.name,
              "sources_sha256": hashes, "source_verdict_counts": dict(counts), "frames": frame_count,
              "baseline_candidate_events": old_events, "contradiction_aware_candidate_events": new_events,
              "frames_with_event_differences": different_frames, "cleared_pending_states": cleared,
              "contradictory_identity_rule_frames": conflicts, "examples": examples,
              "policy": {"persistence_seconds": old.persistence, "gap_seconds": old.gap,
                         "basis": "Current default source code; historical per-job policy was not recorded"},
              "provenance": {key: source_report[key] for key in ("synthetic", "provenance") if key in source_report}
                            or {"status": "not_recorded"},
              "source_decisions_recomputed": False, "fresh_inference": False, "sources_unchanged": True,
              "limits": ["These are current-policy replay candidates, not replacement source findings.",
                         "Synthetic fixtures and archived model verdicts are not independent traffic truth.",
                         "Event IDs are omitted because UUID generation is unrelated to semantic differences."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    temporary.rename(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(json.dumps({key: result[key] for key in ("completed", "frames", "baseline_candidate_events",
                                                  "contradiction_aware_candidate_events", "fresh_inference")}))


if __name__ == "__main__":
    main()
