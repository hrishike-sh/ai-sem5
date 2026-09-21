import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from ivp.media import load_image, write_json
from .perception import Perception


def main():
    parser = argparse.ArgumentParser(description="Run local pretrained perception on an image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-helmet", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    rgb = load_image(args.image)
    model = Perception(helmet=not args.without_helmet)
    detections = model.detect(rgb)
    report = {"source": str(args.image.resolve()), "device": model.device,
              "models": model.manifest,
              "elapsed_seconds": perf_counter() - started,
              "detections": [asdict(d) for d in detections],
              "note": "Perception smoke test only; detections are not violation decisions"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
