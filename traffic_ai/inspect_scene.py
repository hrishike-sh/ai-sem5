import argparse
from dataclasses import asdict
from pathlib import Path

import cv2

from ivp.media import load_image, png_bytes, write_json
from .scene import SceneProposer


def main():
    parser = argparse.ArgumentParser(description="Inspect grounded scene proposals independently of rule decisions")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--prompt", default="a pedestrian crossing. a crosswalk. a stop line. a traffic light. a traffic sign.")
    args = parser.parse_args()
    rgb = load_image(args.image)
    scale = min(1., 960 / max(rgb.shape[:2]))
    rgb = cv2.resize(rgb, (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)))
    proposal = SceneProposer().propose(rgb, args.prompt, args.threshold)
    output = rgb.copy()
    for detection in proposal.detections:
        x1, y1, x2, y2 = map(round, detection.box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(output, f"{detection.label}: {detection.confidence:.2f}", (max(0, x1), max(16, y1)), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 180, 0), 1)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "proposals.json", {"source": str(args.image), "prompt": args.prompt, "threshold": args.threshold,
                                               "size": [rgb.shape[1], rgb.shape[0]], "proposal": asdict(proposal)})
    (args.output / "proposals.png").write_bytes(png_bytes(output))
    print([(d.label, round(d.confidence, 3)) for d in proposal.detections])


if __name__ == "__main__":
    main()
