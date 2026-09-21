"""Run one local CPU-only image overview, with bounded runtime and cancellation."""
import argparse
import json
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ivp.media import load_image
from traffic_ai.bounded_vision import review_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument("--synthetic", action="store_true", help="Explicitly label this input as synthetic")
    parser.add_argument("--timestamp", type=float, default=0.)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--startup-timeout", type=float, default=30.)
    parser.add_argument("--cancel-after", type=float, help="Controlled acceptance check: request cancellation after this many seconds")
    args = parser.parse_args()
    if args.cancel_after is not None and args.cancel_after < 0:
        parser.error("--cancel-after must be nonnegative")
    cancelled = threading.Event()
    timer = None
    if args.cancel_after is not None:
        timer = threading.Timer(args.cancel_after, cancelled.set)
        timer.daemon = True
        timer.start()
    try:
        record = review_image(load_image(args.image), args.timestamp, output=args.output,
                              synthetic=True if args.synthetic else None, timeout=args.timeout,
                              startup_timeout=args.startup_timeout, cancel=cancelled.is_set)
    except (OSError, ValueError) as exc:
        print("Vision input/output error: "+str(exc), file=sys.stderr)
        return 2
    finally:
        if timer:
            timer.cancel()
    print(json.dumps(record, ensure_ascii=False))
    return {"complete": 0, "unavailable": 2, "cancelled": 130}[record["state"]]


if __name__ == "__main__":
    raise SystemExit(main())
