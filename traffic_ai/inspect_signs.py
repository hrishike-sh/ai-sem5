import argparse
from pathlib import Path

from ivp.media import load_image, write_json
from .contracts import Detection
from .signs import SignClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate sign-crop classification independently of scene applicability")
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    model = SignClassifier()
    observations = []
    for path in args.images:
        rgb = load_image(path)
        proposal = Detection("traffic sign", 1., (0, 0, rgb.shape[1], rgb.shape[0]))
        result = model.classify(rgb, proposal)
        result["source"] = str(path)
        observations.append(result)
        print(path.name, result["label"], result["status"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"model": model.MODEL, "revision": model.REVISION, "observations": observations,
                             "note": "Known sign crops supplied directly; this does not test full-scene sign localisation or road applicability"})


if __name__ == "__main__":
    main()
