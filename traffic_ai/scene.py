from dataclasses import dataclass

from PIL import Image

from .contracts import Detection
from .runtime import cached_pretrained, configure_cache


@dataclass
class SceneProposal:
    detections: list[Detection]
    model: str
    revision: str | None


class SceneProposer:
    MODEL = "IDEA-Research/grounding-dino-tiny"
    REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    PROMPT = "a crosswalk. a stop line. a sidewalk. a traffic light. a traffic sign. a road arrow."

    def __init__(self, device="cpu"):
        configure_cache()
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = cached_pretrained(AutoProcessor, self.MODEL, self.REVISION)
        self.model = cached_pretrained(AutoModelForZeroShotObjectDetection, self.MODEL, self.REVISION).to(device).eval()
        self.device = device

    def propose(self, rgb, prompt=None, threshold=.25):
        import torch

        image = Image.fromarray(rgb)
        inputs = self.processor(images=image, text=prompt or self.PROMPT, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        import inspect

        postprocess = self.processor.post_process_grounded_object_detection
        threshold_name = "threshold" if "threshold" in inspect.signature(postprocess).parameters else "box_threshold"
        result = postprocess(outputs, inputs.input_ids, **{threshold_name: threshold}, text_threshold=.25, target_sizes=[rgb.shape[:2]])[0]
        labels = result.get("text_labels", result.get("labels"))
        detections = [Detection(str(label), float(score), tuple(map(float, box)), model=self.MODEL)
                      for box, score, label in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), labels) if str(label).strip()]
        return SceneProposal(detections, self.MODEL, getattr(self.model.config, "_commit_hash", None))
