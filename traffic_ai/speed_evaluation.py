"""Post-run scoring of synthetic speed fixtures; never runs or supplies a predictor.

Predictions are read and hashed before separate evaluation-only truth is opened.
Only declared virtual-plane tests are supported; no real-road accuracy inferred.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if math.isfinite(result) else None


def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def iou(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0., right - left) * max(0., bottom - top)
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (first, second)]
    return overlap / max(1e-12, sum(areas) - overlap)


def mapped_box(row, transform, camera):
    x, y, width, height = [float(row[key]) for key in ("bbox_x", "bbox_y", "bbox_width", "bbox_height")]
    if not all(math.isfinite(v) for v in (x, y, width, height)) or min(width, height) <= 0:
        raise ValueError("Invalid observed box")
    mapping = transform @ np.linalg.inv(camera)
    corners = np.array([[x, y, 1.], [x + width, y, 1.], [x + width, y + height, 1.], [x, y + height, 1.]]) @ mapping.T
    if np.any(np.abs(corners[:, 2]) < 1e-9):
        raise ValueError("Invalid coordinate mapping")
    points = corners[:, :2] / corners[:, 2, None]
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite mapped observation")
    return [float(points[:, 0].min()), float(points[:, 1].min()),
            float(points[:, 0].max()), float(points[:, 1].max())]


def seal_prediction(entry, runs_root):
    """Load only completed prediction artifacts; this function has no truth path."""
    if entry.get("http_status") == 400 and not entry.get("run_id"):
        return {"rejected_http_400_recorded": True, "entry": entry, "hashes": {}}
    identifier = entry.get("run_id")
    if not isinstance(identifier, str) or not identifier or Path(identifier).name != identifier:
        raise ValueError("A single run directory name is required")
    root = runs_root.resolve()
    run = (root / identifier).resolve()
    if not run.is_relative_to(root):
        raise ValueError("Run escapes supplied root")
    names = ("request.json", "status.json", "coordinates.json", "motion-tracks.csv", "motion-activity.csv", "motion-incidents.jsonl")
    hashes = {name: sha(run / name) for name in names}
    request = json.loads((run / "request.json").read_text())
    if json.loads((run / "status.json").read_text()).get("state") != "complete":
        raise ValueError("Prediction run must already be complete")
    source = Path(request["source"])
    source_hash = sha(source) if source.is_file() else None
    return {"entry": entry, "run": run, "hashes": hashes, "request": request,
            "input_source_sha256": source_hash, "tracks": read_csv(run / "motion-tracks.csv"),
            "activity": read_csv(run / "motion-activity.csv"),
            "coordinates": json.loads((run / "coordinates.json").read_text()),
            "events": [json.loads(line) for line in (run / "motion-incidents.jsonl").read_text().splitlines() if line.strip()]}


def score(spec, prediction, truth, protocol):
    """Pure post-hoc scoring; raw prediction speeds are never recomputed from truth."""
    fps, frame_count = spec["media"]["fps"], spec["media"]["frames"]
    expected_hash = spec["video_sha256"]
    checks = {"recorded_input_hash_matches": prediction["entry"].get("input_sha256") == expected_hash}
    if prediction.get("rejected_http_400_recorded"):
        checks["expected_invalid_calibration"] = spec["calibration_state"] == "invalid"
        return {"checks": checks, "passed": all(checks.values()), "evidence": "Recorded HTTP rejection, not a fresh request",
                "accepted_speed_frames": 0, "metrics": None}
    checks["actual_uploaded_bytes_match"] = prediction["input_source_sha256"] == expected_hash
    if not all(checks.values()):
        return {"checks": checks, "passed": False, "accepted_speed_frames": 0,
                "metrics": None, "error": "Input hash contract failed; no accuracy metrics computed"}
    if spec["calibration_state"] == "invalid":
        checks["invalid_calibration_rejected"] = False
        return {"checks": checks, "passed": False, "accepted_speed_frames": 0,
                "metrics": None, "error": "Invalid calibration was accepted instead of rejected"}
    request = prediction["request"]
    configured = request["motion"].get("calibration")
    checks["configured_test_threshold"] = request["motion"]["incidents"]["speed_limit_kmh"] == protocol["test_threshold_kmh"]
    if spec["calibration_state"] == "absent":
        checks["calibration_absent"] = configured is None
    else:
        reference = spec["calibration"]
        checks["declared_calibration_matches"] = configured is not None and all(configured.get(k) == reference[k] for k in ("points", "width_m", "length_m"))
    activity = prediction["activity"]
    tolerance = protocol["timing_match_tolerance_seconds"]
    checks["all_decoded_frame_times_match"] = len(activity) == frame_count and all(
        number(row.get("timestamp_seconds")) is not None and abs(float(row["timestamp_seconds"]) - index / fps) <= tolerance
        for index, row in enumerate(activity))
    if not checks["all_decoded_frame_times_match"]:
        return {"checks": checks, "passed": False, "accepted_speed_frames": 0,
                "metrics": None, "error": "Decoded timing contract failed; no frame association attempted"}
    transform = np.asarray(prediction["coordinates"]["transform_to_source_pixel_centres"], dtype=float)
    if transform.shape != (3, 3) or not np.isfinite(transform).all():
        raise ValueError("Coordinate transform must be finite 3 by 3")
    per_frame, malformed_times = defaultdict(list), 0
    for row in prediction["tracks"]:
        timestamp = number(row.get("timestamp_seconds"))
        if timestamp is None:
            malformed_times += 1
            continue
        index = round(timestamp * fps)
        if not 0 <= index < len(activity) or abs(timestamp - index / fps) > tolerance:
            malformed_times += 1
            continue
        per_frame[index].append(row)
    checks["prediction_timestamps_bound_to_frames"] = malformed_times == 0
    candidates = [event for event in prediction["events"] if event.get("type") == "speeding" and event.get("status") == "candidate"]
    physical_rows = [row for row in prediction["tracks"] if row.get("speed_unit") == "km/h" and number(row.get("speed")) is not None]
    occlusion = spec.get("occlusion_half_open_seconds")
    if occlusion:
        start, end = occlusion
        checks["no_physical_speed_in_gap_or_initial_recovery"] = not any(start <= float(r["timestamp_seconds"]) < end + .5 for r in physical_rows)
        checks["no_speed_candidate_in_gap_or_initial_recovery"] = not any(start <= float(e["timestamp_seconds"]) < end + .5 for e in candidates)
    if spec["calibration_state"] == "absent":
        checks["no_physical_speeds"] = not physical_rows
        checks["no_speeding_candidate"] = not candidates
        return {"checks": checks, "passed": all(checks.values()), "accepted_speed_frames": 0,
                "metrics": None, "pixel_speed_rows_allowed": True}
    errors, eligible, ambiguous, matched, statuses = [], 0, 0, 0, Counter()
    recovery_eligible = recovery_accepted = 0
    used_ids = set()
    accepted_by_frame = {}
    mapping_failures = 0
    for index, row in enumerate(truth):
        timestamp = float(row["timestamp_seconds"])
        if row["eligible_observation"] != "True" or timestamp < 1.:
            continue
        if occlusion and timestamp < occlusion[1] + .5 and timestamp >= occlusion[0]:
            continue
        eligible += 1
        recovering = bool(occlusion and timestamp >= occlusion[1] + .5)
        recovery_eligible += int(recovering)
        target_box = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
        frame = activity[index]
        camera = np.eye(3)
        try:
            camera[:2] = np.array([float(frame[f"camera_m{k}"]) for k in range(6)]).reshape(2, 3)
            qualified = [candidate for candidate in per_frame[index]
                         if iou(mapped_box(candidate, transform, camera), target_box) >= protocol["association"]["minimum_iou"]]
        except (KeyError, ValueError, np.linalg.LinAlgError):
            mapping_failures += 1
            continue
        if len(qualified) != 1:
            ambiguous += int(len(qualified) > 1)
            continue
        matched += 1
        candidate = qualified[0]
        statuses[candidate.get("speed_status", "not_recorded")] += 1
        value = number(candidate.get("speed"))
        if value is not None and candidate.get("speed_unit") == "km/h" and candidate.get("speed_status") == "estimated":
            errors.append(abs(value - float(row["declared_simulated_kmh"])))
            used_ids.add(candidate["track_id"])
            accepted_by_frame[index] = candidate
            recovery_accepted += int(recovering)
    coverage = len(errors) / eligible if eligible else 0.
    metrics = {"eligible_reference_frames": eligible, "matched_observed_frames": matched,
               "accepted_speed_frames": len(errors), "accepted_eligible_frame_coverage": coverage,
               "ambiguous_reference_frames": ambiguous, "coordinate_mapping_failures": mapping_failures,
               "mean_absolute_error_kmh": statistics.mean(errors) if errors else None,
               "p95_absolute_error_kmh": quantile(errors, .95), "max_absolute_error_kmh": max(errors) if errors else None,
               "matched_status_counts": dict(statuses), "accepted_track_ids": sorted(used_ids),
               "recovery_eligible_frames": recovery_eligible, "recovery_accepted_frames": recovery_accepted}
    gates = protocol["metric_gates"]
    checks["nonzero_reference_and_accepted_samples"] = bool(eligible and errors)
    checks["coverage_target"] = coverage >= gates["accepted_eligible_frame_coverage_min"]
    checks["mean_error_target"] = bool(errors) and metrics["mean_absolute_error_kmh"] <= gates["mean_absolute_error_kmh_max"]
    checks["p95_error_target"] = bool(errors) and metrics["p95_absolute_error_kmh"] <= gates["p95_absolute_error_kmh_max"]
    checks["coordinate_mapping_available"] = mapping_failures == 0
    candidate_bindings = []
    for event in candidates:
        stamp = number(event.get("timestamp_seconds"))
        index = round(stamp * fps) if stamp is not None else -1
        observed = accepted_by_frame.get(index)
        measurements = event.get("measurements", {})
        value = number(measurements.get("speed"))
        threshold = protocol["test_threshold_kmh"]
        valid = (stamp is not None and abs(stamp - index / fps) <= tolerance and observed is not None
                 and str(event.get("track_id")) == observed["track_id"] and event.get("synthetic") is True
                 and value is not None and measurements.get("speed_unit") == "km/h"
                 and measurements.get("threshold_kmh") == threshold
                 and value > threshold + max(2., threshold * .05)
                 and abs(value - float(observed["speed"])) <= .011
                 and (number(measurements.get("sustained_seconds")) or 0) >= .5 - 1e-6
                 and (number(measurements.get("observed_samples")) or 0) >= 3)
        candidate_bindings.append(bool(valid))
    checks["speed_candidates_bound_to_eligible_target_and_measurements"] = all(candidate_bindings)
    checks["speed_candidate_expected"] = bool(candidates) if spec["expected"]["declared_speed_kmh"] > protocol["test_threshold_kmh"] else not candidates
    if occlusion:
        checks["recovery_reference_window_available"] = recovery_eligible > 0
        checks["recovery_coverage_target"] = recovery_eligible > 0 and recovery_accepted / recovery_eligible >= gates["accepted_eligible_frame_coverage_min"]
    return {"checks": checks, "passed": all(checks.values()), "metrics": metrics,
            "speed_candidate_count": len(candidates), "accepted_speed_frames": len(errors)}


def evaluate(packs, records, runs_root, protocol_path, output, audits):
    output = Path(output).resolve()
    roots = [Path(pack).resolve() for pack in packs]
    runs_root = Path(runs_root).resolve()
    if output.exists() or any(output.is_relative_to(root) for root in roots + [runs_root]):
        raise ValueError("Output must be new and outside all prediction/fixture sources")
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    specs, sources = {}, {str(protocol_path): sha(protocol_path)}
    audit_by_manifest = {}
    for audit_path in audits:
        path = Path(audit_path).resolve()
        sources[str(path)] = sha(path)
        audit = json.loads(path.read_text())
        if audit.get("completed") is not True or audit.get("source_unchanged") is not True:
            raise ValueError("A completed read-only fixture audit is required")
        audit["source_hashes"] = {name.replace("\\", "/"): value for name, value in audit["source_hashes"].items()}
        audit_by_manifest[audit["manifest_sha256"]] = audit
    for root in roots:
        path = root / "manifest.json"
        sources[str(path)] = sha(path)
        audit = audit_by_manifest.get(sources[str(path)])
        if audit is None:
            raise ValueError("No matching fixture audit for manifest")
        manifest = json.loads(path.read_text())
        if manifest.get("synthetic") is not True:
            raise ValueError("Only explicitly synthetic fixture packs are supported")
        for case in manifest["scenarios"]:
            if case["id"] in specs:
                raise ValueError("Scenario IDs must be unique across supplied packs")
            spec = {**case, "media": manifest["media"], "root": root}
            spec["audit_hashes"] = audit["source_hashes"]
            config_path = root / case["config"]
            if not config_path.resolve().is_relative_to(root):
                raise ValueError("Fixture configuration escapes source root")
            sources[str(config_path)] = sha(config_path)
            if sources[str(config_path)] != audit["source_hashes"].get(case["config"]):
                raise ValueError("Configuration changed since fixture audit")
            spec["calibration"] = json.loads(config_path.read_text())["calibration"]
            video_path = root / case["video"]
            if not video_path.resolve().is_relative_to(root):
                raise ValueError("Fixture video escapes source root")
            sources[str(video_path)] = sha(video_path)
            if sources[str(video_path)] != case["video_sha256"]:
                raise ValueError("Fixture video no longer matches declared hash")
            specs[case["id"]] = spec
    selected = protocol["videos"] + protocol.get("calibration_negative_cases", ["speed_36_absent_calibration", "speed_36_invalid_calibration"])
    entries = {}
    for record in records:
        path = Path(record).resolve()
        sources[str(path)] = sha(path)
        document = json.loads(path.read_text())
        for entry in document["cases"]:
            if entry["scenario"] not in selected:
                continue
            if entry["scenario"] in entries:
                raise ValueError("Duplicate prediction scenario; select a single record per case")
            entries[entry["scenario"]] = entry
    sealed, loading_errors, dataflow = {}, {}, []
    # No evaluation_only file is opened above or in this prediction sealing stage.
    for identifier in selected:
        if identifier in entries:
            try:
                sealed[identifier] = seal_prediction(entries[identifier], runs_root)
            except (ValueError, OSError, KeyError) as error:
                loading_errors[identifier] = str(error)
    dataflow.append({"stage": "prediction_artifacts_sealed", "cases": sorted(sealed)})
    results = []
    for identifier in selected:
        if identifier not in specs:
            results.append({"scenario": identifier, "state": "missing_fixture", "passed": False})
            continue
        if identifier not in sealed:
            results.append({"scenario": identifier, "state": "prediction_load_failed" if identifier in loading_errors else "pending_prediction",
                            "error": loading_errors.get(identifier), "passed": False})
            continue
        spec, prediction = specs[identifier], sealed[identifier]
        truth_path = spec["root"] / spec["ground_truth"]
        if not truth_path.resolve().is_relative_to(spec["root"]):
            raise ValueError("Evaluation truth escapes fixture root")
        sources[str(truth_path)] = sha(truth_path)
        if sources[str(truth_path)] != spec["audit_hashes"].get(spec["ground_truth"]):
            raise ValueError("Evaluation truth changed since independent fixture audit")
        truth = read_csv(truth_path)
        dataflow.append({"stage": "evaluation_truth_opened", "case": identifier, "truth_sha256": sources[str(truth_path)]})
        result = score(spec, prediction, truth, protocol)
        results.append({"scenario": identifier, "state": "scored", "run_id": prediction["entry"].get("run_id"),
                        "prediction_hashes": prediction["hashes"], **result})
    for prediction in sealed.values():
        if "run" in prediction:
            if any(sha(prediction["run"] / name) != expected for name, expected in prediction["hashes"].items()):
                raise ValueError("Prediction artifacts changed during evaluation")
    if any(sha(Path(path)) != expected for path, expected in sources.items()):
        raise ValueError("Fixture/reference sources changed during evaluation")
    result = {"kind": "post_run_synthetic_speed_suite", "synthetic": True, "evaluation_only": True,
              "protocol_sha256": sources[str(protocol_path)], "cases": results, "dataflow": dataflow,
              "passed": sum(case["passed"] for case in results), "total": len(results),
              "all_passed": all(case["passed"] for case in results), "source_hashes": sources,
              "sources_unchanged": True, "new_inference": False,
              "limits": ["Declared simulated scale, not measured road accuracy.",
                         "Metrics score exported observations; evaluation truth was opened only after predictions were sealed.",
                         "Historical predictor truth separation is declared by its runner, not retroactively proven here.",
                         "Thresholds are fixed engineering targets; no actual legal limit inferred."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result
