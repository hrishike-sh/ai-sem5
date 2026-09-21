"""Controlled camera-reference probe using a retained video's first decoded frame.

Generated occlusion/translation is explicitly synthetic, not traffic ground truth.
No images are written and no perception model is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import cv2
import numpy as np

from traffic_ai.scene_lifecycle import RefreshAwareCameraStability
from traffic_ai.segmentation import CameraStability


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def probe(original):
    height, width = original.shape[:2]
    translated = cv2.warpAffine(original, np.float32([[1, 0, 24], [0, 1, 0]]), (width, height))
    sequence = [(0., original), (.1, original), (1.2, original), (1.3, np.zeros_like(original)),
                (1.4, translated), (1.5, translated), (2.6, translated)]
    base, improved = CameraStability(), RefreshAwareCameraStability()
    improved.acknowledge_scene(0, 1)
    observations = []
    for timestamp, pixels in sequence:
        before, after = base.observe(pixels, timestamp), improved.observe(pixels, timestamp)
        same_displacement = (before[1] is None and after[1] is None or
                             before[1] is not None and after[1] is not None and abs(before[1] - after[1]) < 1e-6)
        observations.append({"timestamp": timestamp, "baseline_stable": bool(before[0]), "stable": bool(after[0]),
                             "baseline_measured_displacement": before[1], "measured_displacement": after[1],
                             "measurements_preserved": bool(before[0]) == bool(after[0]) and same_displacement,
                             "scene_refresh_required": improved.scene_refresh_required,
                             "context_ready": improved.context_ready})
    before_ack = improved.lifecycle_snapshot()
    old_revision_accepted = improved.acknowledge_scene(0, 1)
    new_revision_accepted = improved.acknowledge_scene(0, 2)
    checks = {"initial_context_becomes_ready": observations[2]["context_ready"],
              "original_displacement_gate_misses_reference_loss": not any(
                  o["measured_displacement"] is not None and o["measured_displacement"] > 1.5 for o in observations),
              "camera_measurements_preserved": all(o["measurements_preserved"] for o in observations),
              "pending_after_occlusion_and_reacquisition": all(o["scene_refresh_required"] and not o["context_ready"] for o in observations[3:]),
              "new_reference_becomes_stable": observations[-1]["stable"],
              "old_revision_cannot_acknowledge_refresh": not old_revision_accepted,
              "new_revision_acknowledges_refresh": new_revision_accepted and improved.context_ready}
    return {"checks": checks, "observations": observations, "before_acknowledgement": before_ack,
            "after_acknowledgement": improved.lifecycle_snapshot()}


def validate(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite a previous lifecycle probe")
    if output == run or output.is_relative_to(run):
        raise ValueError("Probe output must be outside the retained run")
    request_path, status_path = run / "request.json", run / "status.json"
    if json.loads(status_path.read_text())["state"] != "complete":
        raise ValueError("A completed retained run is required")
    request = json.loads(request_path.read_text())
    if request["info"]["kind"] != "video":
        raise ValueError("This probe requires an existing video source")
    source = Path(request["source"]).resolve()
    if not source.is_file():
        raise ValueError("The retained local source video is unavailable")
    protected = {p: digest(p) for p in (request_path, status_path, source)}
    with av.open(str(source)) as container:
        original = next(container.decode(video=0)).to_ndarray(format="rgb24")
    result = probe(original)
    unchanged = all(digest(path) == value for path, value in protected.items())
    result["checks"]["retained_source_and_run_metadata_unchanged"] = unchanged
    receipt = {"kind": "controlled_scene_reference_lifecycle_probe", "passed": all(result["checks"].values()),
               "source_run": run.name, "source_sha256": protected[source],
               "request_sha256": protected[request_path], "status_sha256": protected[status_path],
               "decoded_first_frame_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
               "frame_shape": list(original.shape), "synthetic_derivative": True,
               "recipe": "Repeat first decoded RGB frame, insert a zero-filled occlusion, then translate RGB 24 pixels right with zero border; timestamps are controlled test inputs.",
               "fresh_perception_inference": False, "frames_written": 0, **result,
               "limitations": ["This controlled derivative tests camera-reference continuity, not actual camera motion accuracy or traffic-rule correctness.",
                               "Decoded RGB is local and transient. No private pixels, model calls or external transmission are needed.",
                               "Original camera authenticity and synthetic provenance are not established by this test."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = validate(args.run, args.output)
    print(json.dumps({"passed": receipt["passed"], "checks": len(receipt["checks"]), "output": str(args.output.resolve()),
                      "synthetic_derivative": True, "fresh_perception_inference": False}))
    raise SystemExit(int(not receipt["passed"]))


if __name__ == "__main__":
    main()
