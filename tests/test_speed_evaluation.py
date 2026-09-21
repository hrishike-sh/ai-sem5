"""Post-run evaluator regressions. Toy exports test scoring, not detector accuracy."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from traffic_ai import speed_evaluation as module


def fixture(speed=36):
    calibration = {"points": [[0, 0], [1, 0], [1, 1], [0, 1]], "width_m": 100, "length_m": 100}
    spec = {"media": {"fps": 10, "frames": 30}, "video_sha256": "unit-input-hash",
            "calibration": calibration, "calibration_state": "valid", "occlusion_half_open_seconds": None,
            "expected": {"declared_speed_kmh": speed}}
    protocol = {"test_threshold_kmh": 30, "timing_match_tolerance_seconds": .0001,
                "association": {"minimum_iou": .25},
                "metric_gates": {"accepted_eligible_frame_coverage_min": .5,
                                 "mean_absolute_error_kmh_max": 1., "p95_absolute_error_kmh_max": 2.}}
    truth, tracks, activity = [], [], []
    for index in range(30):
        stamp = index / 10
        truth.append({"frame": str(index), "timestamp_seconds": str(stamp), "eligible_observation": "True",
                      "x1": "40", "x2": "60", "y1": str(20 + index), "y2": str(40 + index),
                      "declared_simulated_kmh": str(speed)})
        # Source pixels = 2 * unregistered pixels + .5. Camera adds (3,5).
        tracks.append({"timestamp_seconds": str(stamp), "track_id": "7", "bbox_x": "22.75",
                       "bbox_y": str((20 + index - .5) / 2 + 5), "bbox_width": "10", "bbox_height": "10",
                       "speed": str(speed), "speed_unit": "km/h", "speed_status": "estimated"})
        activity.append({"timestamp_seconds": str(stamp), **{f"camera_m{k}": str(value)
                         for k, value in enumerate((1, 0, 3, 0, 1, 5))}})
    candidate = {"type": "speeding", "status": "candidate", "timestamp_seconds": 1.5,
                 "track_id": 7, "synthetic": True,
                 "measurements": {"speed": speed, "speed_unit": "km/h", "threshold_kmh": 30,
                                  "sustained_seconds": .5, "observed_samples": 6}}
    prediction = {"entry": {"input_sha256": "unit-input-hash"}, "input_source_sha256": "unit-input-hash",
                  "request": {"motion": {"calibration": calibration, "incidents": {"speed_limit_kmh": 30}}},
                  "coordinates": {"transform_to_source_pixel_centres": [[2, 0, .5], [0, 2, .5], [0, 0, 1]]},
                  "tracks": tracks, "activity": activity, "events": [candidate] if speed > 30 else []}
    return spec, prediction, truth, protocol


def test_scaling_and_camera_mapping_score_exported_speeds_without_recomputing_them():
    spec, prediction, truth, protocol = fixture()
    before = deepcopy(prediction)
    result = module.score(spec, prediction, truth, protocol)
    assert result["passed"] and result["metrics"]["accepted_eligible_frame_coverage"] == 1.
    assert result["metrics"]["mean_absolute_error_kmh"] == 0.
    assert result["checks"]["speed_candidates_bound_to_eligible_target_and_measurements"]
    assert prediction == before


def test_zero_accepted_samples_cannot_pass_on_a_positive_event_alone():
    spec, prediction, truth, protocol = fixture()
    for row in prediction["tracks"]:
        row.update(speed="", speed_status="collecting stable track")
    result = module.score(spec, prediction, truth, protocol)
    assert not result["passed"] and result["metrics"]["accepted_eligible_frame_coverage"] == 0.
    assert result["metrics"]["mean_absolute_error_kmh"] is None
    assert not result["checks"]["nonzero_reference_and_accepted_samples"]


def test_inaccurate_numeric_exports_fail_fixed_error_gates():
    spec, prediction, truth, protocol = fixture()
    for row in prediction["tracks"]:
        row["speed"] = "42"
    result = module.score(spec, prediction, truth, protocol)
    assert result["metrics"]["mean_absolute_error_kmh"] == 6.
    assert not result["checks"]["mean_error_target"] and not result["checks"]["p95_error_target"]


def test_ambiguous_duplicate_predictions_reduce_coverage_instead_of_selecting_best_speed():
    spec, prediction, truth, protocol = fixture()
    prediction["tracks"] += [{**row, "track_id": "8"} for row in prediction["tracks"]]
    result = module.score(spec, prediction, truth, protocol)
    assert not result["passed"] and result["metrics"]["ambiguous_reference_frames"] == 20
    assert result["metrics"]["accepted_speed_frames"] == 0


def test_bad_frame_timing_prevents_association_and_scoring():
    spec, prediction, truth, protocol = fixture()
    prediction["activity"].pop()
    result = module.score(spec, prediction, truth, protocol)
    assert not result["passed"] and result["metrics"] is None
    assert not result["checks"]["all_decoded_frame_times_match"]


@pytest.mark.parametrize("unit,expected", [("px/s", True), ("km/h", False)])
def test_absent_calibration_allows_pixels_and_never_physical_speed(unit, expected):
    spec, prediction, truth, protocol = fixture()
    spec["calibration_state"] = "absent"
    prediction["request"]["motion"]["calibration"] = None
    prediction["events"] = []
    for row in prediction["tracks"]:
        row["speed_unit"] = unit
    result = module.score(spec, prediction, truth, protocol)
    assert result["passed"] is expected and result["metrics"] is None


def test_exact_test_threshold_has_no_required_candidate():
    spec, prediction, truth, protocol = fixture(30)
    result = module.score(spec, prediction, truth, protocol)
    assert result["passed"] and result["speed_candidate_count"] == 0


def test_wrong_candidate_identity_or_measurement_does_not_pass_good_numeric_coverage():
    spec, prediction, truth, protocol = fixture()
    prediction["events"][0]["track_id"] = 99
    result = module.score(spec, prediction, truth, protocol)
    assert result["metrics"]["accepted_eligible_frame_coverage"] == 1.
    assert not result["checks"]["speed_candidates_bound_to_eligible_target_and_measurements"]


def test_input_hash_mismatch_prevents_accuracy_metrics():
    spec, prediction, truth, protocol = fixture()
    prediction["input_source_sha256"] = "different-input"
    result = module.score(spec, prediction, truth, protocol)
    assert not result["passed"] and result["metrics"] is None
    assert not result["checks"]["actual_uploaded_bytes_match"]


def test_invalid_calibration_cannot_pass_with_plausible_numeric_predictions():
    spec, prediction, truth, protocol = fixture()
    spec["calibration_state"] = "invalid"
    result = module.score(spec, prediction, truth, protocol)
    assert not result["passed"] and not result["checks"]["invalid_calibration_rejected"]
    assert result["metrics"] is None


def write_csv(path, rows):
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value))


def disk_fixture(tmp_path):
    spec, prediction, truth, protocol = fixture(30)
    pack, runs = tmp_path / "pack", tmp_path / "runs"
    pack.mkdir()
    run = runs / "unit-run"
    run.mkdir(parents=True)
    (pack / "unit.mp4").write_bytes(b"SYNTHETIC SCORER UNIT BYTES, not a decoded video")
    video_hash = module.sha(pack / "unit.mp4")
    write_json(pack / "config.json", {"calibration": spec["calibration"]})
    write_csv(pack / "truth.csv", truth)
    case = {**spec, "id": "unit", "video": "unit.mp4", "video_sha256": video_hash,
            "config": "config.json", "ground_truth": "truth.csv"}
    case.pop("media")
    write_json(pack / "manifest.json", {"synthetic": True, "media": spec["media"], "scenarios": [case]})
    write_json(run / "request.json", {**prediction["request"], "source": str(pack / "unit.mp4")})
    write_json(run / "status.json", {"state": "complete"})
    write_json(run / "coordinates.json", prediction["coordinates"])
    write_csv(run / "motion-tracks.csv", prediction["tracks"])
    write_csv(run / "motion-activity.csv", prediction["activity"])
    (run / "motion-incidents.jsonl").write_text("")
    protocol["videos"] = ["unit"]
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "record.json", {"cases": [{"scenario": "unit", "run_id": "unit-run", "input_sha256": video_hash}]})
    write_json(tmp_path / "audit.json", {"completed": True, "source_unchanged": True,
               "manifest_sha256": module.sha(pack / "manifest.json"),
               "source_hashes": {name: module.sha(pack / name) for name in ("config.json", "truth.csv", "unit.mp4")}})
    return pack, runs


def test_all_prediction_artifacts_are_loaded_before_evaluation_truth(tmp_path, monkeypatch):
    pack, runs = disk_fixture(tmp_path)
    log, original = [], module.read_csv

    def watched(path):
        log.append(Path(path).name)
        return original(path)

    monkeypatch.setattr(module, "read_csv", watched)
    result = module.evaluate([pack], [tmp_path / "record.json"], runs, tmp_path / "protocol.json",
                             tmp_path / "result.json", [tmp_path / "audit.json"])
    assert log.index("motion-tracks.csv") < log.index("truth.csv")
    assert log.index("motion-activity.csv") < log.index("truth.csv")
    assert result["dataflow"][0]["stage"] == "prediction_artifacts_sealed"
    assert result["cases"][0]["passed"] and not result["all_passed"]  # Two missing calibration cases remain explicit.


def test_truth_changed_after_audit_is_rejected_without_an_output(tmp_path):
    pack, runs = disk_fixture(tmp_path)
    (pack / "truth.csv").write_text("tampered evaluation input")
    with pytest.raises(ValueError, match="truth changed"):
        module.evaluate([pack], [tmp_path / "record.json"], runs, tmp_path / "protocol.json",
                        tmp_path / "result.json", [tmp_path / "audit.json"])
    assert not (tmp_path / "result.json").exists()


def test_explicit_subset_protocol_does_not_claim_unselected_calibration_controls(tmp_path):
    pack, runs = disk_fixture(tmp_path)
    protocol = json.loads((tmp_path / "protocol.json").read_text())
    protocol["calibration_negative_cases"] = []
    write_json(tmp_path / "protocol.json", protocol)
    result = module.evaluate([pack], [tmp_path / "record.json"], runs, tmp_path / "protocol.json",
                             tmp_path / "result.json", [tmp_path / "audit.json"])
    assert result["total"] == result["passed"] == 1 and result["all_passed"]
    assert [case["scenario"] for case in result["cases"]] == ["unit"]
