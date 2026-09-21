"""Bounded live vision controls using only explicitly declared synthetic images."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import threading

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivp.media import load_image
from traffic_ai import bounded_vision as vision


def _input_hash(rgb):
    image = Image.fromarray(rgb).convert("RGB")
    image.thumbnail((512,512))
    stream = io.BytesIO(); image.save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("car_image", type=Path)
    parser.add_argument("empty_image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true", required=True)
    parser.add_argument("--positive-record", type=Path, help="Reuse a prior live result only when its image hash/protocol match")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new validation output directory")
    args.output.mkdir(parents=True)
    car, empty = load_image(args.car_image), load_image(args.empty_image)
    before = {str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.car_image,args.empty_image)}
    checks = []
    def check(name, kind, passed, record):
        checks.append({"name":name,"kind":kind,"passed":bool(passed),"record":record})
    if args.positive_record:
        positive = json.loads(args.positive_record.read_text(encoding="utf-8"))
        assert positive["synthetic"] is True and positive["review_image_sha256"] == _input_hash(car)
        vision.validate_selection(positive["selection"])
        positive_kind = "replayed_live_vision_record"
    else:
        positive = vision.review_image(car, output=args.output/"car", synthetic=True)
        positive_kind = "live_vision_inference"
    check("synthetic_car_presence", positive_kind, positive["state"]=="complete" and
          "car" in positive.get("selection",{}).get("visible_features",[]), positive)
    negative = vision.review_image(empty, output=args.output/"empty", synthetic=True)
    uniform = negative.get("input_guard")=="uniform_review_image" and negative["state"]=="unavailable" and "observation" not in negative
    check("empty_image_control", "deterministic_uniform_input_abstention" if uniform else "live_vision_inference", uniform or (
          negative["state"]=="complete" and set(negative.get("selection",{}).get("visible_features",[])) <= {"unknown"}), negative)
    cancelled = threading.Event()
    original = vision.request_json
    timer = None
    def request_with_cancel(route, *values, **options):
        nonlocal timer
        if route == "/v1/chat/completions":
            timer = threading.Timer(.3, cancelled.set)
            timer.daemon = True; timer.start()
        return original(route, *values, **options)
    vision.request_json = request_with_cancel
    try:
        stopped = vision.review_image(car, output=args.output/"cancel", synthetic=True, cancel=cancelled.is_set)
    finally:
        vision.request_json = original
        if timer:timer.cancel()
    check("inflight_cancellation", "live_runtime_cancellation", stopped["state"]=="cancelled" and
          stopped["phase"]=="inference" and "observation" not in stopped and
          stopped["runtime"]["owned_server_stopped"], stopped)
    spec = vision.RuntimeSpec(args.output/"missing-binary",args.output/"missing-model",args.output/"missing-projector")
    missing = vision.review_image(car,output=args.output/"missing",synthetic=True,spec=spec)
    check("missing_runtime_fallback", "local_missing_resource", missing["state"]=="unavailable" and
          "observation" not in missing and missing["runtime"]["owned_server_stopped"], missing)
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in before.items())
    checks.append({"name":"source_images_unchanged","kind":"filesystem","passed":unchanged})
    result = {"kind":"bounded_live_vision_acceptance","created_at":datetime.now(timezone.utc).isoformat(),
              "synthetic":True,"checks":checks,"passed":sum(c["passed"] for c in checks),"total":len(checks),
              "note":"These controls verify runtime completion, simple feature behavior and cleanup. They do not validate traffic-rule, sign, grounding or detector accuracy."}
    (args.output/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"passed":result["passed"],"total":result["total"],"receipt":str(args.output/"validation.json")}))
    return int(result["passed"]!=result["total"])


if __name__ == "__main__":
    raise SystemExit(main())
