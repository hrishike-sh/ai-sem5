"""Read-only audit of simulated geometry, fixture hashes and decoded timestamps."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Fixture reference escapes its root")
    return path


def analytic(trajectory, timestamp):
    speed, initial = trajectory["speed_px_s"], trajectory["initial_bottom_y"]
    mode = trajectory["mode"]
    if mode == "linear":
        return initial + speed * timestamp, speed
    stop = trajectory["stop_bottom_y"]
    stop_at = (stop - initial) / speed
    if timestamp < stop_at:
        return initial + speed * timestamp, speed
    if mode == "stop_forever" or timestamp < stop_at + trajectory["stop_duration_seconds"]:
        return float(stop), 0.
    return stop + speed * (timestamp - stop_at - trajectory["stop_duration_seconds"]), speed


def check_opaque_mask(video, rectangle, interval, fps):
    left, top, width, height = rectangle
    start, end = interval
    selected = {max(0, math.ceil(start * fps) - 1), math.ceil(start * fps), math.ceil(end * fps) - 1, math.ceil(end * fps)}
    capture, samples, index = cv2.VideoCapture(str(video)), [], 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in selected:
                patch = frame[top + 8:top + height - 8, left + 8:left + width - 8]
                fraction = float(np.mean(np.max(np.abs(patch.astype(float) - 64), axis=2) < 8))
                masked = start <= index / fps < end
                assert fraction > .99 if masked else fraction < .95, "Rendered opaque mask timing differs from metadata"
                samples.append({"frame": index, "mask_expected": masked, "gray64_fraction": fraction})
            index += 1
    finally:
        capture.release()
    assert len(samples) == len(selected)
    return samples


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root):
        raise ValueError("Audit output must be new and outside the source pack")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("synthetic") is not True:
        raise ValueError("This audit accepts explicitly synthetic fixture packs")
    media, calibration = manifest["media"], manifest["virtual_calibration"]
    width, height, fps, count = (media[k] for k in ("width", "height", "fps", "frames"))
    expected_y_scale = height * (calibration["points"][2][1] - calibration["points"][0][1]) / calibration["length_m"]
    assert math.isclose(expected_y_scale, calibration["pixels_per_m_y"], abs_tol=1e-9)
    xs, ys = [p[0] for p in calibration["points"]], [p[1] for p in calibration["points"]]
    assert {tuple(p) for p in calibration["points"]} == {(x, y) for x in (min(xs), max(xs)) for y in (min(ys), max(ys))}, "This audit requires the declared axis-aligned virtual test plane"
    pinned = {manifest_path: sha(manifest_path)}
    for name, spec in manifest["assets"].items():
        path = contained(root, name)
        pinned[path] = sha(path)
        assert pinned[path] == spec["sha256"], name
    results = []
    for case in manifest["scenarios"]:
        video, config_path, truth_path = [contained(root, case[k]) for k in ("video", "config", "ground_truth")]
        for path in (video, config_path, truth_path):
            pinned[path] = sha(path)
        assert pinned[video] == case["video_sha256"], case["id"]
        config = json.loads(config_path.read_text())
        assert config["synthetic"] is True and config["incidents"]["synthetic"] is True
        command = ["ffprobe", "-v", "error", "-show_frames", "-show_streams", "-show_format",
                   "-show_entries", "frame=best_effort_timestamp_time:stream=codec_type,codec_name,width,height,pix_fmt,avg_frame_rate:format=duration",
                   "-of", "json", str(video)]
        probe = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        streams = probe["streams"]
        assert len(streams) == 1 and streams[0]["codec_type"] == "video"
        stream = streams[0]
        assert (stream["width"], stream["height"], stream["codec_name"]) == (width, height, "h264")
        assert stream["avg_frame_rate"] == f"{fps}/1"
        pts = [float(row["best_effort_timestamp_time"]) for row in probe["frames"]]
        assert len(pts) == count and abs(float(probe["format"]["duration"]) - count / fps) < 1e-6
        maximum_pts_error = max(abs(t - index / fps) for index, t in enumerate(pts))
        assert maximum_pts_error < 1e-6
        with truth_path.open(newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == count
        eligible, post_recovery = [], []
        occlusion = case["occlusion_half_open_seconds"]
        for index, row in enumerate(rows):
            stamp = index / fps
            assert int(row["frame"]) == index and abs(float(row["timestamp_seconds"]) - stamp) < 1e-8
            expected_y, speed = analytic(case["trajectory"], stamp)
            rendered_y = int(row["bottom_centre_y"])
            assert abs(float(row["analytic_bottom_y"]) - expected_y) < 1e-6
            assert rendered_y == math.floor(expected_y - 160 + 1e-8) + 160
            assert abs(float(row["declared_simulated_kmh"]) - speed / expected_y_scale * 3.6) < 1e-6
            assert int(row["x2"]) - int(row["x1"]) == 100 and int(row["y2"]) - int(row["y1"]) == 160
            masked = bool(occlusion and occlusion[0] <= stamp < occlusion[1])
            assert (row["occluded"] == "True") == masked
            x1, y1, x2, y2 = [int(row[key]) for key in ("x1", "y1", "x2", "y2")]
            full = 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            within = min(xs) <= (x1 + x2) / 2 / width <= max(xs) and min(ys) <= y2 / height <= max(ys)
            caption_overlap = y1 < 28 and y2 > 0
            assert (row["fully_in_frame"] == "True") == full
            assert (row["within_calibration"] == "True") == within
            assert (row["caption_overlap"] == "True") == caption_overlap
            assert (row["eligible_observation"] == "True") == (full and not masked and y1 >= 28 and within)
            mask_rectangle = manifest["composition"].get("occlusion_mask_xywh")
            if masked and mask_rectangle is not None:
                mx, my, mw, mh = mask_rectangle
                assert mx <= int(row["x1"]) < int(row["x2"]) <= mx + mw
                assert my <= int(row["y1"]) < int(row["y2"]) <= my + mh, "Declared full occlusion does not cover the sprite"
            if row["eligible_observation"] == "True":
                assert row["fully_in_frame"] == row["within_calibration"] == "True" and not masked
                eligible.append(stamp)
                if occlusion and stamp >= occlusion[1] + .5:
                    post_recovery.append(stamp)
        results.append({"scenario": case["id"], "video_sha256": pinned[video], "frames": len(pts),
                        "max_pts_error_seconds": maximum_pts_error, "truth_rows": len(rows),
                        "eligible_frames": len(eligible), "last_eligible_timestamp": max(eligible) if eligible else None,
                        "eligible_frames_after_occlusion_plus_0_5_seconds": len(post_recovery),
                        "post_recovery_available": bool(post_recovery) if occlusion else None})
        rectangle = manifest["composition"].get("occlusion_mask_xywh")
        if rectangle is not None and occlusion:
            results[-1]["rendered_mask_pixel_checks"] = check_opaque_mask(video, rectangle, occlusion, fps)
    assert all(sha(path) == expected for path, expected in pinned.items()), "Fixture source changed during audit"
    result = {"kind": "synthetic_speed_fixture_audit", "synthetic": True, "completed": True,
              "manifest_sha256": pinned[manifest_path], "source_hashes": {p.relative_to(root).as_posix(): value for p, value in pinned.items()},
              "calibration_meaning": "Declared virtual plane; no physical measurement inferred from generated appearance",
              "scenarios": results, "source_unchanged": True, "inference_run": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, indent=2, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pack, args.output)
    print(json.dumps({"completed": result["completed"], "scenarios": len(result["scenarios"]), "source_unchanged": True}))


if __name__ == "__main__":
    main()
