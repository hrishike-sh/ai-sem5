"""Replay automatic control associations from existing AI exports, without inference.

Exports an additive audit, never replacement rule decisions. Missing recorded
camera stability stays unknown. Only scene JSON is read from the ZIP; no media,
model, subprocess or network is needed.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .control_association import ConflictAwareApproachResolver, POLICY


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(data):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(data, parse_constant=invalid)


def _write(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def replay(run, output):
    """Read a completed saved run and write a new, independently hashed audit."""
    run, output = Path(run).resolve(), Path(output).resolve()
    if output == run or output.is_relative_to(run):
        raise ValueError("The audit output must be outside the original run")
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing audit output")
    paths = {name: run / name for name in ("detections.jsonl", "scene_features.zip", "report.json", "status.json")}
    if _json(paths["status.json"].read_bytes()).get("state") != "complete":
        raise ValueError("Only completed saved AI runs are accepted")
    sources = {name: {"sha256": _sha(path), "bytes": path.stat().st_size} for name, path in paths.items()}
    report = _json(paths["report.json"].read_bytes())
    if "ai" not in report:
        raise ValueError("The run has no recorded AI output")
    scenes = {}
    with zipfile.ZipFile(paths["scene_features.zip"]) as archive:
        for info in archive.infolist():
            if not re.fullmatch(r"[0-9]+/scene\.json", info.filename):
                continue
            if info.file_size > 8 * 1024 * 1024:
                raise ValueError("Scene JSON exceeds the audit size limit")
            data = archive.read(info)
            scene = _json(data)
            key = (scene["scene_id"], scene["revision"])
            if any(type(value) is not int or value < 0 for value in key) or key in scenes:
                raise ValueError("Invalid or duplicate archived scene identity")
            scenes[key] = (scene, {"member": info.filename, "sha256": hashlib.sha256(data).hexdigest()})
    output.mkdir(parents=True, exist_ok=False)
    frame_count, unknown_camera, missing_scene = 0, 0, 0
    counts, signal_status, sign_status = Counter(), Counter(), Counter()
    resolver, active_key, used_scenes = None, None, set()
    audit_path = output / "control-associations.jsonl"
    with paths["detections.jsonl"].open("rb") as stream, audit_path.open("x", encoding="utf-8") as audit:
        for line_number, raw in enumerate(stream, 1):
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("Frame JSON exceeds the audit size limit")
            if not raw.strip():
                continue
            frame = _json(raw)
            key = (frame.get("scene_id"), frame.get("scene_revision"))
            if any(type(value) is not int or value < 0 for value in key):
                key = None
            archived = scenes.get(key)
            if resolver is None or key != active_key:
                scene = archived[0] if archived else {}
                resolver = ConflictAwareApproachResolver(scene.get("approaches", []))
                resolver.assign_signs(deepcopy(scene.get("sign_observations", [])))
                active_key = key
                if archived:
                    used_scenes.add(key)
                    sign_status.update(r["status"] for r in resolver.graphs["sign"]["observations"])
            if not archived:
                missing_scene += 1
            stable = frame.get("camera_stable")
            known_stability = type(stable) is bool
            unknown_camera += int(not known_stability)
            signals = deepcopy(frame.get("signal_observations", []))
            resolver.update_signals(signals, frame["timestamp"], stable if known_stability else False)
            snapshot = resolver.association_snapshot()
            signal_status.update(r["status"] for r in snapshot["graphs"]["signal"]["observations"])
            for decision in frame.get("decisions", []):
                counts[f"{decision['rule']}/{decision['verdict']}"] += 1
            _write(audit, {"frame": frame.get("frame"), "timestamp": frame["timestamp"],
                           "scene_id": frame.get("scene_id"), "scene_revision": frame.get("scene_revision"),
                           "source_line": line_number, "source_line_sha256": hashlib.sha256(raw).hexdigest(),
                           "scene_source": archived[1] if archived else None,
                           "camera_stability": stable if known_stability else None,
                           "camera_stability_status": "recorded" if known_stability else "not_recorded",
                           "scene_status": "recorded" if archived else "not_recorded",
                           "association": snapshot, "rule_decisions_recomputed": False})
            frame_count += 1
    if any(_sha(path) != sources[name]["sha256"] for name, path in paths.items()):
        raise ValueError("A source changed during replay; the audit is incomplete")
    summary = {"kind": "automatic_control_association_audit", "version": 1, "policy": POLICY,
               "sources": sources, "frames": frame_count, "scenes_used": len(used_scenes),
               "frames_without_recorded_camera_stability": unknown_camera, "frames_without_archived_scene": missing_scene,
               "signal_association_status_counts": dict(signal_status), "sign_association_status_counts": dict(sign_status),
               "source_assessment_counts": dict(counts), "source_counts_match_report": dict(counts) == report["ai"].get("assessment_counts", {}),
               "provenance": {name: report[name] for name in ("synthetic", "provenance", "synthetic_provenance") if name in report}
                             or {"status": "not_recorded"},
               "rule_decisions_recomputed": False, "fresh_inference": False,
               "output": {"filename": audit_path.name, "sha256": _sha(audit_path), "bytes": audit_path.stat().st_size},
               "limitations": ["Geometric association candidates are not verified legal applicability or detector accuracy.",
                               "Missing camera stability is unknown and never assumed from the clip or verdicts.",
                               "Existing source decisions are counted unchanged; no new legal verdicts are generated."]}
    # Presence of this last-written file is the completion marker. On interruption
    # or failure, partial output may remain for diagnosis without a marker.
    marker = output / ".association-summary.tmp"
    with marker.open("x", encoding="utf-8") as stream:
        _write(stream, summary)
    marker.rename(output / "association-summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Existing completed AI run directory")
    parser.add_argument("--output", required=True, type=Path, help="New audit directory outside the original run")
    args = parser.parse_args()
    summary = replay(args.run, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "frames": summary["frames"],
                      "fresh_inference": False, "source_counts_match_report": summary["source_counts_match_report"]}))


if __name__ == "__main__":
    main()
