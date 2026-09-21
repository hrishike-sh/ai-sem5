"""Fixed development experiment on retained inputs and one pinned local model.

CPU-only, two threads, no network/downloads/installers and no image exports.
The exact local Windows version probe is permitted; other subprocesses are not.
This is not an independent accuracy benchmark. Cases and acceptance are written
before the first model call; failures remain in the completed report.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from ivp.media import load_image
from traffic_ai.turn_sign_context import ContextPaddedTurnDetector
from traffic_ai.turn_signs import TURN_MEANINGS, TurnSignDetector, turn_sign_observation


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def evaluate(primary, weights, output):
    primary, weights, output = Path(primary).resolve(), Path(weights).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Refusing to replace an existing experiment")
    if not weights.is_file() or digest(weights) != TurnSignDetector.SHA256:
        raise ValueError("Only the existing pinned yahyagul checkpoint is accepted; no download fallback")
    photo = "temp/qa/turn_signs/No_left_turn_sign_with_stoplight_(17821040904).jpg"
    cases = [{"id": "photograph", "file": photo, "kind": "known_development_photograph", "crop": None, "expected": "do_not_turn_l"},
             {"id": "documented_crop", "file": photo, "kind": "crop_of_same_development_photograph", "crop": [242, 475, 630, 840], "expected": "do_not_turn_l"},
             {"id": "rider_negative", "file": "demo/indian_riders.png", "kind": "known_development_negative", "crop": None, "expected": None},
             {"id": "bus_negative", "file": "temp/qa/traffic_bus.jpg", "kind": "known_development_negative", "crop": None, "expected": None},
             {"id": "other_sign", "file": "temp/qa/signs/00052.png", "kind": "known_no_vehicles_sign_not_turn_prohibition", "crop": None, "expected": None},
             {"id": "illustration", "file": "temp/qa/turn_signs/Road_Sign_No_Left_Turn.jpg", "kind": "illustration_not_real_photograph", "crop": None, "expected": "do_not_turn_l", "diagnostic_only": True}]
    for case in cases:
        case["source_sha256"] = digest(primary / case["file"])
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"kind": "fixed_context_padding_development_protocol", "cases": cases,
                "weights_sha256": TurnSignDetector.SHA256, "device": "cpu", "threads": 2,
                "model_score_threshold": .8, "raw_confidence_threshold": .25, "imgsz": 960,
                "padding_factors": [1., 2.], "padding_rgb": [114, 114, 114], "max_extra_canvas_pixels": 4_000_000,
                "acceptance": "Recover the documented crop; preserve full photo; no accepted turn prohibition on three retained negatives. Illustration remains diagnostic-only.",
                "scope": "Known development cases only, no general precision/recall or legal applicability claim."}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    progress = {"stage": "protocol_written", "model_loaded": False, "calls_started": 0,
                "calls_completed": 0, "active_case": None}

    def write_progress(stage):
        progress["stage"] = stage
        temporary = output / ".progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        temporary.replace(output / "progress.json")

    write_progress("import_runtime")
    # These are task-local runtime settings, not provider credentials or shared config.
    os.environ["YOLO_CONFIG_DIR"] = str(output / "ultralytics")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    (output / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
    (output / "matplotlib").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["HF_HUB_OFFLINE"] = "1"
    blocked = []
    allowed_probes = []
    comspec = os.environ.get("COMSPEC", "").casefold()

    def guard(event, args):
        # Windows platform.machine() uses gethostname(), a local OS read. Socket
        # construction, DNS, connections and subprocesses remain forbidden.
        local_ver = (event == "subprocess.Popen" and comspec and str(args[0]).casefold() == comspec
                     and isinstance(args[1], str) and args[1].casefold() in
                     {comspec + ' /c "ver"', '"' + comspec + '" /c "ver"'})
        if local_ver:
            allowed_probes.append("local Windows cmd /c ver")
        if (event.startswith("socket.") and event != "socket.gethostname"
                or event == "subprocess.Popen" and not local_ver):
            blocked.append(event)
            write_progress("blocked_" + event)
            raise RuntimeError("Network/subprocess forbidden in retained-model experiment: " + event)

    sys.addaudithook(guard)
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import torch_utils

    torch_utils.NUM_THREADS = 2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    model = YOLO(str(weights))
    progress["model_loaded"] = True
    write_progress("model_loaded")
    if not set(TURN_MEANINGS) <= set(model.names.values()):
        raise ValueError("Pinned checkpoint labels do not contain supported explicit prohibitions")
    started = perf_counter()

    def cancel():
        if perf_counter() - started > 120:
            raise TimeoutError("Fixed experiment exceeded its 120-second deadline")

    class ExistingLocalDetector:
        manifest = {"repository": TurnSignDetector.MODEL, "revision": TurnSignDetector.REVISION,
                    "sha256": TurnSignDetector.SHA256, "device": "cpu", "threads": 2}

        def detect(self, rgb):
            cancel()
            progress["calls_started"] += 1
            write_progress("inference_started")
            result = model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), device="cpu", imgsz=960,
                                   conf=.25, verbose=False, save=False, batch=1, half=False)[0]
            progress["calls_completed"] += 1
            write_progress("inference_completed")
            if result.boxes is None:
                return []
            return [turn_sign_observation(rgb, box, model.names[int(label)], score)
                    for box, label, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]

    detector = ExistingLocalDetector()
    results = []
    for case in cases:
        cancel()
        progress["active_case"] = case["id"]
        rgb = load_image(primary / case["file"])
        if case["crop"] is not None:
            x1, y1, x2, y2 = case["crop"]
            rgb = rgb[y1:y2, x1:x2].copy()
        adapter = ContextPaddedTurnDetector(detector, cancel=cancel)
        began = perf_counter()
        observations = adapter.detect(rgb)
        recognised = [o for o in observations if o["status"] == "recognised_candidate"]
        baseline = [h for o in observations for h in o["detector_hypotheses"] if h["pass_index"] == 0]
        expected = case["expected"]
        passed = (not recognised if expected is None else bool(recognised) and all(o["label"] == expected for o in recognised))
        result = {"case": deepcopy(case), "acceptance_passed": bool(passed), "seconds": perf_counter() - began,
                  "model_calls": adapter.last_audit["model_calls"], "observations": observations,
                  "audit": deepcopy(adapter.last_audit), "baseline_hypotheses": baseline}
        results.append(result)
        (output / (case["id"] + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"case": case["id"], "acceptance_passed": bool(passed), "recognised": [{"label": o["label"], "score": o["model_score"]} for o in recognised],
                          "model_calls": result["model_calls"], "seconds": result["seconds"]}), flush=True)
    unchanged = digest(weights) == TurnSignDetector.SHA256 and all(digest(primary / c["file"]) == c["source_sha256"] for c in cases)
    report = {"kind": "context_padding_development_evaluation", "completed": True,
              "acceptance_passed": unchanged and all(r["acceptance_passed"] for r in results if not r["case"].get("diagnostic_only")),
              "protocol": protocol, "model": detector.manifest, "results": results, "sources_and_weights_unchanged": unchanged,
              "fresh_local_inference": True, "network_or_subprocess_attempts": blocked, "seconds": perf_counter() - started,
              "allowed_local_platform_reads": allowed_probes,
              "limits": ["All inputs are retained development cases; two positive views are the same photograph, not independent positives.",
                         "Illustration is separate and not a real-world accuracy sample.",
                         "Raw model scores remain uncalibrated; sign approach/exception applicability is never inferred."]}
    temporary = output / ".report.tmp"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.rename(output / "report.json")
    write_progress("complete")
    print(json.dumps({"completed": True, "acceptance_passed": report["acceptance_passed"], "fresh_local_inference": True,
                      "model_calls": sum(r["model_calls"] for r in results), "output": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args.primary, args.weights, args.output)


if __name__ == "__main__":
    main()
