import numpy as np

from .runtime import configure_cache
from .signs import colour_shape_support


TURN_MEANINGS = {"do_not_turn_l": ["left"], "do_not_turn_r": ["right"], "do_not_u_turn": ["u_turn"]}


def turn_sign_observation(rgb, box, label, score):
    x1, y1, x2, y2 = map(int, box)
    crop = rgb[max(0, y1):max(0, min(rgb.shape[0], y2)), max(0, x1):max(0, min(rgb.shape[1], x2))]
    supported = label in TURN_MEANINGS
    accepted = supported and score >= .8 and min(crop.shape[:2], default=0) >= 24 and colour_shape_support(crop, 15)
    return {"box": list(box), "label": label, "model_score": float(score),
            "status": "recognised_candidate" if accepted else "cannot_assess",
            "restriction": None, "prohibited_turns": TURN_MEANINGS[label] if accepted else [],
            "associated_approach": None, "proposal_source": TurnSignDetector.MODEL,
            "reason": "Explicit prohibition label and red circular shape agree; applicability unverified" if accepted
                      else "Unsupported sign meaning or insufficient score, size, or colour/shape support"}


class TurnSignDetector:
    MODEL = "yahyagul/traffic-sign-yolov8"
    REVISION = "15be2681d3b6e666f45c744219aab84e97a2f1b4"
    SHA256 = "93793cc99ffda2a10e09860e3a7c6383f7c25d470dd604c4555b5b07253a6e2b"

    def __init__(self, device="cpu"):
        home = configure_cache()
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from ultralytics import YOLO
        from pathlib import Path
        from .perception import Perception

        path = home / "models" / "turn-yahyagul-15be2681.pt"
        if not path.exists():
            try:
                path = hf_hub_download(self.MODEL, "best_roboflow.pt", revision=self.REVISION, local_files_only=True)
            except LocalEntryNotFoundError:
                path = hf_hub_download(self.MODEL, "best_roboflow.pt", revision=self.REVISION)
        fingerprint = Perception._fingerprint(Path(path))
        if fingerprint["sha256"] != self.SHA256:
            raise ValueError("Turn-sign weights failed checksum verification; download the pinned checkpoint again")
        self.model = YOLO(path)
        self.device = device
        if not set(TURN_MEANINGS) <= set(self.model.names.values()):
            raise ValueError("Turn-sign checkpoint labels do not match the supported meanings")
        self.manifest = {"repository": self.MODEL, "revision": self.REVISION,
                         **fingerprint, "device": device}

    def detect(self, rgb):
        result = self.model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), device=self.device, imgsz=960, conf=.25, verbose=False)[0]
        if result.boxes is None:
            return []
        return [turn_sign_observation(rgb, box, self.model.names[int(cls)], score)
                for box, cls, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.cls.cpu().tolist(), result.boxes.conf.cpu().tolist())]


def main():
    import argparse
    from pathlib import Path
    import torch
    from ivp.media import load_image, write_json

    parser = argparse.ArgumentParser(description="Inspect explicit turn-prohibition sign observations")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    detector = TurnSignDetector("cuda:0" if torch.cuda.is_available() else "cpu")
    observations = {str(path): detector.detect(load_image(path)) for path in args.images}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {"model": detector.manifest, "observations": observations})
    print(args.output.resolve())


if __name__ == "__main__":
    main()
