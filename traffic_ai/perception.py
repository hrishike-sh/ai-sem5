import numpy as np
import hashlib
from pathlib import Path

from .contracts import Detection
from .runtime import configure_cache


class Perception:
    def __init__(self, device=None, helmet=True):
        home = configure_cache()
        from .local_models import verified_local_models
        paths, self.manifest = verified_local_models(helmet=helmet)
        import torch
        from ultralytics import YOLO

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        models = home / "models"
        models.mkdir(exist_ok=True)
        self.road_users = YOLO(str(paths['road_users']))
        self.helmet = None
        if helmet:
            self.helmet = YOLO(str(paths['helmet']))
            self.manifest['helmet']['labels'] = self.helmet.names

    @staticmethod
    def _fingerprint(path):
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        return {"filename": path.name, "sha256": digest, "bytes": path.stat().st_size}

    def reset_tracks(self):
        predictor = self.road_users.predictor
        if predictor is not None and hasattr(predictor, "trackers"):
            for tracker in predictor.trackers:
                tracker.reset()

    def detect(self, rgb, track=False):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Perception expects a uint8 RGB frame")
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        arguments = dict(source=bgr, device=self.device, conf=.25, imgsz=960, verbose=False, classes=[0, 1, 2, 3, 5, 7, 9, 11])
        if track:
            results = self.road_users.track(**arguments, persist=True, tracker="bytetrack.yaml")
        else:
            results = self.road_users.predict(**arguments)
        detections = self._convert(results[0], "yolo11s")
        if self.helmet is not None:
            heads = self.helmet.predict(source=bgr, device=self.device, conf=.4, imgsz=960, verbose=False)[0]
            detections.extend(self._convert(heads, "iam-tsr/yolov8n-helmet-detection"))
        return detections

    @staticmethod
    def _convert(result, model):
        if result.boxes is None:
            return []
        boxes = result.boxes
        ids = boxes.id.cpu().tolist() if boxes.id is not None else [None] * len(boxes)
        def label_for(cls):
            label = str(result.names[int(cls)]).strip().lower().replace(" ", "_")
            if model == "iam-tsr/yolov8n-helmet-detection":
                label = {"without_helmet": "no_helmet", "with_helmet": "helmet"}.get(label, label)
            return label

        return [Detection(label_for(cls), float(conf), tuple(map(float, box)), int(track) if track is not None else None, model)
                for box, conf, cls, track in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), ids)]
