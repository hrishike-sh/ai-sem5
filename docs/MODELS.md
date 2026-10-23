# Models, provenance and limitations

[Back to README](../README.md) | [Runtime and OCR configuration](CONFIGURATION.md)

The app integrates pretrained third-party models. User-facing feature labels do not change their identities, checkpoint provenance or licenses. No weights are redistributed here, and dependency installation does not supply them. Use only weights you are authorized to use. Repository license labels do not settle all derivative obligations; each model and dependency retains its upstream terms. No model license is granted by this project.

Paths below are relative to the model runtime (`.ai-runtime/` by default), unless `ai-runtime.json` explicitly maps a supported model key to another absolute path. Loaders verify the specified fingerprints before loading the primary image models.

## Subject and helmet models

| Model | Default location | SHA-256 | Bytes |
| --- | --- | --- | --- |
| Ultralytics YOLO11s | `models/yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` | 19,313,732 |
| `iam-tsr/yolov8n-helmet-detection`, revision `59b68b03b60314badb0c19ebcc3b95baa8241cf8` | `huggingface/hub/models--iam-tsr--yolov8n-helmet-detection/snapshots/59b68b03b60314badb0c19ebcc3b95baa8241cf8/best.pt` | `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d` | 6,199,658 |

The pins and lookup rules are in [`local_models.py`](../traffic_ai/local_models.py). Subject localization uses [`Perception`](../traffic_ai/perception.py), selecting CUDA when available and CPU otherwise, with inference size 960. It retains person, motorcycle, bicycle, car, bus and truck boxes and numbers at most twelve subjects. These IDs are local to the image, not identities or cross-frame tracks.

The [helmet checkpoint](https://huggingface.co/iam-tsr/yolov8n-helmet-detection) has the actual classes **With Helmet** and **Without Helmet**. Its card records training at image size 224; the image-review specialist uses full-context CPU inference at 960 to retain small heads. It matches head candidates to the upper portion of a person box, with a heuristic candidate-score threshold of 0.50 and inference floor of 0.10. Competing classes, multiple heads, ambiguous matches, low scores and missing detections remain explicit. A missing detection never establishes no helmet. Duplicate predictions do not add people or vehicles. The current helmet loader also verifies the road-user checkpoint, so both pinned files are needed for this check.

The helmet repository card labels the checkpoint MIT, and its linked dataset is labeled CC BY 4.0. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) uses AGPL-3.0 or Enterprise terms; upstream obligations also require review for YOLO-derived checkpoints and redistribution.

## Vehicle subtype model

[IISc UVH-26](https://huggingface.co/iisc-aim/UVH-26), YOLOv11-S, supplies fourteen labels: Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, tempo-traveller, bicycle, Van and Others.

- Revision: `4a22412775adb6f97f22735647afee976b4638a0`.
- Default path: `models/uvh26-yolo11s-4a224127.pt`; optional config key: `models.vehicle_types`.
- SHA-256: `12dbeb8f6fa207d19446fa652fc4313359fbec267027c8c01641004dcf1d0150`.
- Size: 57,142,241 bytes.
- [Official pinned file](https://huggingface.co/iisc-aim/UVH-26/blob/4a22412775adb6f97f22735647afee976b4638a0/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt); [authors' technical report](https://arxiv.org/abs/2511.02563).

[`vehicle_types.py`](../traffic_ai/vehicle_types.py) uses the original selected image on CPU at its training size of 640, with an inference floor of 0.25 and candidate threshold of 0.50. Proposal counts are separate from numbered-box counts. Geometric matches, subtype conflicts and unmatched proposals remain candidates. No weights are downloaded automatically. The repository card declares Apache 2.0 and explicitly retains upstream Ultralytics AGPLv3 obligations.

## Plate detection and OCR

[`plate_models.py`](../traffic_ai/plate_models.py) integrates [Morse Tech Lab's YOLO11n plate detector](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) and [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR) English recognition. Detection and recognition run locally on CPU. The detector runs at size 960; recognition uses grayscale crops bounded to 640x160 and an English alphanumeric allowlist. At most twenty plate proposals are processed. Plates smaller than 32x12 pixels are not sent to recognition.

| File under `models/plates/` | Official source | SHA-256 |
| --- | --- | --- |
| `detector.pt` | [YOLO11n checkpoint at revision `251a30d7daedca065f56e04b0af04052c907c68f`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/251a30d7daedca065f56e04b0af04052c907c68f/license-plate-finetune-v1n.pt) | `0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2` |
| `english_g2.pth` | Extract this exact file from [EasyOCR's official archive](https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip) | `e2272681d9d67a04e2dff396b6e95077bc19001f8f6d3593c307b9852e1c29e8` |

The adapter requires EasyOCR 1.7.2 and records upstream revision `c4f3cd7225efd4f85451bd8b4a7646ae9a092420`. The detector is labeled AGPL-3.0 and EasyOCR Apache-2.0. The detector publisher warns of training/test contamination; published metrics are not a reliable field estimate. Both weight hashes are checked, and missing files are never downloaded automatically. See [optional OCR installation](CONFIGURATION.md#optional-plate-ocr).

A candidate reading uses heuristic detector/OCR thresholds of 0.50/0.80. Small, unreadable and low-confidence readings remain uncertain. Boxes are associated only with a uniquely containing numbered vehicle; overlapping or unmatched candidates remain unassigned. Manual transcription retains the original OCR and clears old LLM/reviewer results. Original predictions and corrections persist in saved reviews and JSON. OCR strings and corrections are excluded from LLM text, although plate pixels may remain in explicitly approved images. Plate text does not establish an offense, valid registration, ownership or vehicle history; there is no database lookup.

## Auxiliary video and scene models

The optional full analyzer uses additional pinned model adapters. These are separate from the five image-review checks.

| Purpose and source | Model ID | Revision |
| --- | --- | --- |
| Scene proposals, [`scene.py`](../traffic_ai/scene.py) | `IDEA-Research/grounding-dino-tiny` | `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| Segmentation, [`segmentation.py`](../traffic_ai/segmentation.py) | `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` | `21b3847fae21ddee674abd31129307b6a1235bd9` |
| Sign classification, [`signs.py`](../traffic_ai/signs.py) | `bazyl/gtsrb-model` | `95f6c95c765e1705d7f92161a5a42a40a5bd8338` |
| Turn-sign proposals, [`turn_signs.py`](../traffic_ai/turn_signs.py) | `yahyagul/traffic-sign-yolov8` | `15be2681d3b6e666f45c744219aab84e97a2f1b4` |

The turn-sign adapter uses `best_roboflow.pt`, SHA-256 `93793cc99ffda2a10e09860e3a7c6383f7c25d470dd604c4555b5b07253a6e2b`, with optional `models.turn_sign` configuration. Its [repository card](https://huggingface.co/yahyagul/traffic-sign-yolov8) is labeled MIT. [SegFormer's license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) limits use to noncommercial research or evaluation. Review each upstream model and dependency's terms before reuse or redistribution.

Auxiliary loaders can download pinned models unless offline configuration prevents it. An explicit snapshot map supplies local directories for matching revisions; see [configuration](CONFIGURATION.md#reuse-an-existing-interpreter-and-weights).

Optional scene overview uses the bounded CPU runtime configured by `RuntimeSpec` in [`bounded_vision.py`](../traffic_ai/bounded_vision.py), with local files under `<runtime>/ollama/`. Its recorded model identity is `qwen3-vl:4b`; absence of the runtime/model/projector reports unavailable. It emits unverified feature categories, not geometry, speed, signal applicability or incident decisions. It remains separate from detection.

## Recorded local checks

These are small development checks retained from the project notes, not a labeled accuracy benchmark or a claim of training new models:

- Helmet: the flyover frame produced a raw no-helmet false candidate on a billboard and a helmet candidate on a rider. The rider image produced overlapping head detections and a low-score conflicting class. Results were scale-sensitive, and duplicate predictions occurred.
- Vehicle subtypes: the supplied flyover frame produced seven qualifying proposals; the rider image produced two and missed other visible vehicles. A false Three-wheeler proposal scored 0.497, below the 0.50 threshold. Size 960 produced worse subtype confusion in these checks, so size 640 was retained.
- Signs: the existing `yahyagul/traffic-sign-yolov8` checkpoint returned no candidates on those two samples or a synthetic sign probe. Sign/signal recognition was not added to the primary image workflow.
- Plates: the two real samples produced low-confidence, incorrect or incomplete OCR; neither became an accepted reading. A clearly labeled high-resolution synthetic text fixture was transcribed exactly but remained below the 0.80 threshold. This demonstrates execution and uncertainty handling, not reliable real-world recognition.
- Video rules: three four-second runs on permitted real footage produced zero supported candidates. Raw detections included a billboard face mistaken for an unhelmeted head and a car region mistaken for a helmet; association/rule checks did not promote those to supported violations. Zero supported candidates does not prove that all events were found.

The [demo guide](DEMO.md) records additional walkthroughs and their limits. Synthetic geometry and mocked model tests verify software behavior; their results are not real-traffic detection or OCR accuracy.
