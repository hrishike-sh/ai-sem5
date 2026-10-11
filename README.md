# ViolationLens · AI Sem 5

Local traffic evidence review. Upload a permitted image or select a video frame, number people and vehicles locally, preview their boxes and crops, and ask Astra for findings tied to each subject ID using your ChatGPT plan. Full-clip rule analysis remains optional.

## Run

Install Python 3.11, uv, and FFmpeg (both `ffmpeg` and `ffprobe` on PATH). On Windows:

```powershell
.\Setup-AI.ps1
.\Start.cmd
```

The app runs at http://127.0.0.1:8521. Close its terminal to stop the server. It uses its own `uploads/`, `runs/`, and `.ai-runtime/` directories. `TRAFFIC_AI_HOME` can select another dedicated runtime directory. The retained `ivp/` package supplies shared image processing, media handling, jobs and evidence storage.

## Review workflow

Drop one JPG or PNG, or choose it with the file picker. After you acknowledge permission, the same image is uploaded once and automatically processed by the available local features: subject localization, helmet evidence, vehicle-class counts, and vehicle subtype proposals. Each section shows its own progress, results or failure; a combined report includes successful evidence and pending or unavailable checks. For video, select a timestamp and choose **Upload and preview**. No hosted request runs automatically.

Inspect the numbered boxes: P IDs represent people and V IDs represent vehicles, not verified riders or identities. Correct missing or wrong boxes by drawing, editing coordinates/type, or deleting them. Manual boxes also work when the subject detector is unavailable; the independent vehicle subtype specialist can still inspect the image. At most twelve subjects are numbered; a truncated detection is visibly reported. Replacing the image clears the visible results and ignores stale responses from the prior image.

Inspect the exact annotated context and ID-linked crops, approve sending them to OpenAI, then choose **Send numbered subjects to Astra**. One context and up to twelve padded crops are sent; each JPEG is bounded to 768 pixels per edge. Crops use the bound original-resolution frame. This uses account-visible `gpt-6-astra` with no silent model substitution or paid API fallback. Source paths and reviewer notes are excluded. These are within-image IDs, not identities or cross-frame tracking.

Astra must return every submitted ID once, distinguish riders from pedestrians, and link each apparent rider to a submitted two-wheeler. Each finding cites that subject's own crop and shared context. Possible missing helmet requires a visible uncovered head on an apparent motorcycle rider; an obscured or ambiguous head is unclear. Possible triple riding requires three distinct visible people associated with the same motorcycle, with consistent group references. Zebra occupancy is only an observation. Speed, wrong-way movement, red-light crossing, obstruction and parking legality cannot be established by this single-frame flow. There are no legal citations, fines or determinations of guilt.

Review the separate per-ID observations, suspected issues and uncertainty, then save Accept or Reject with a note and export JSON. A status of no visible violation means no supported visible concern was selected, not verified compliance. **Edit boxes** lets you correct a saved assessment; saving changes clears its model result and reviewer decision and requires a fresh preview and consent. Reopening unchanged saved results sends no new model request. Older unnumbered assessments remain readable, but new requests require subject boxes.

Expand **Optional local detection and rule review** for the existing local pipeline:

1. Upload a permitted MP4, MOV or AVI clip and run detection. Missing model files are shown as setup requirements.
2. Open timestamped candidate cards, inspect recorded facts and missing context, and load before/during/after frames. Frame timestamps are actual decoded positions; unavailable frames are labeled.
3. Accept or reject a candidate with an optional note, then export the local review JSON. Human decisions are stored separately from the original algorithm output and are invalidated if that output changes.

For a short walkthrough, see the [demo guide](docs/DEMO.md). Optional ChatGPT explanations are described below; the review workflow also works without them.

For a repeatable demonstration, expand **Try a synthetic rule demo** and choose opposing motion, following motion, or unknown direction. The app generates a labeled three-second rectangle animation and evaluates its known coordinates with the existing geometric rule engine. This works without detector weights or ChatGPT. It demonstrates configured-rule checks, evidence frames, saved human decisions and export; it is not learned detection or a real-traffic benchmark.

The optional deterministic pipeline prioritizes explicit bare-head evidence and wrong-way movement with independent direction context. A missing helmet detection alone is insufficient. Red-signal cases require an associated signal and temporal context; zebra occupancy alone does not establish prohibited obstruction. Its automatic triple-riding rule remains deferred; the numbered Astra review offers a separate fallible visual assessment. `supported`, `needs_review` and `insufficient_evidence` describe evidence, not guilt or fines. There is no automatic external reporting.

The advanced detection and overview interface remains available at `/studio`. Saved results and uploads stay in this repository's local runtime directories.

## Local specialist evidence

After boxes are numbered or corrected, the app runs the pinned `iam-tsr/yolov8n-helmet-detection` checkpoint locally on the original selected frame. Its actual classes are **With Helmet** and **Without Helmet**. Each matched person card shows the raw model score and any competing predictions. Scores are not calibrated probabilities or measured accuracy. A heuristic 0.50 threshold distinguishes candidates from low-confidence predictions; conflicting classes, overlapping person matches, missing detections and unavailable weights remain explicit. A missing detection never establishes no helmet. Person-to-motorcycle overlaps are unconfirmed geometric candidates, not verified rider identities.

The checkpoint was trained at image size 224; this app uses full-context CPU inference at size 960 to retain small heads. Local checks showed strong scale sensitivity and duplicate predictions. Duplicates do not create additional people or vehicle counts. The accompanying **vehicle breakdown** counts numbered motorcycle, bicycle, car, bus and truck boxes in this image, distinguishing detector proposals from manual corrections and displaying any twelve-subject truncation. It is not a unique traffic-volume measurement.

Astra receives the exact previews plus these bound local predictions. It cannot replace scores or turn absent/uncertain evidence into a missing-helmet issue. A suspected missing-helmet observation additionally needs a supported local no-helmet prediction and a visible rider association to the same motorcycle. Local evidence can be exported without Astra. Correcting boxes clears both local evidence and the previous Astra assessment; the specialist reruns before another send.

Two real local samples were inspected without hosted transmission. The flyover frame contained a raw no-helmet false candidate on a billboard and a helmet candidate on a rider. The supplied rider image produced overlapping head detections and a low-score conflicting class. These are useful failure cases, not a labeled accuracy benchmark. The existing `yahyagul/traffic-sign-yolov8` checkpoint returned no candidates on the two samples or a synthetic sign probe, so sign/signal recognition was not added to the primary workflow. No plate detector or OCR quality was established.

Model cards label the [helmet checkpoint](https://huggingface.co/iam-tsr/yolov8n-helmet-detection) and [sign checkpoint](https://huggingface.co/yahyagul/traffic-sign-yolov8) repositories MIT; the helmet card's linked dataset is labeled CC BY 4.0. These labels do not settle derivative obligations. [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/) uses AGPL-3.0 or Enterprise terms. Weights remain external to this repository.

The additional [IISc UVH-26](https://huggingface.co/iisc-aim/UVH-26) YOLOv11-S specialist proposes fourteen vehicle types: Hatchback, Sedan, SUV, MUV, Bus, Truck, Three-wheeler, Two-wheeler, LCV, Mini-bus, tempo-traveller, bicycle, Van and Others. It uses the original selected image at its training size of 640 pixels. Raw scores below 0.50 remain low-confidence evidence; qualifying proposal counts are separate from numbered-box counts. Subtypes and geometric matches remain candidates, and unmatched or conflicting predictions are visible. The supplied flyover frame produced seven qualifying proposals; the rider image produced two, missing other visible vehicles. A false Three-wheeler proposal at 0.497 was below the threshold. At 960 pixels the model produced worse subtype confusions in these checks, so that setting was not adopted. These two samples establish feasibility, not accuracy.

The optional UVH weight file is `models/uvh26-yolo11s-4a224127.pt`, or the ignored `ai-runtime.json` `models.vehicle_types` path. It must match revision `4a22412775adb6f97f22735647afee976b4638a0`, SHA-256 `12dbeb8f6fa207d19446fa652fc4313359fbec267027c8c01641004dcf1d0150` (57,142,241 bytes). The [official file](https://huggingface.co/iisc-aim/UVH-26/blob/4a22412775adb6f97f22735647afee976b4638a0/weights/YOLOv11-S/UVH-26-MV-YOLOv11-S.pt) hash was checked before local use. The repository card declares Apache 2.0 and explicitly retains upstream Ultralytics AGPLv3 obligations. See the authors' [technical report](https://arxiv.org/abs/2511.02563). No weights are redistributed or downloaded automatically by this feature.

## Optional ChatGPT plan explanation

The Windows prototype implements the official [Sign in with ChatGPT flow for locally hosted apps](https://developers.openai.com/siwc/token-sharing-open-source/sign-in). In the review page, choose **Connect ChatGPT** and explicitly approve opening sign-in to authorize ViolationLens. This creates a separate app grant; it does not reuse Codex or browser credentials. The grant is encrypted with Windows DPAPI under `%LOCALAPPDATA%/ViolationLens/auth`, outside the repository. Installing dependencies or opening the page does not authorize access.

After sign-in, each card offers two explicit choices:

- **Preview frames and facts for ChatGPT** prepares the exact images and structured evidence locally. Inspect that preview, then choose **Send selected frames and facts to ChatGPT** to send it. A request contains at most three JPEG images, each at most 768 pixels on its longest edge. Preparation may include context images and subject crops; the full clip is not sent. JPEG metadata is stripped, but crops or blur do not guarantee anonymity, and identities may remain visible.
- **Send structured facts only to ChatGPT** sends the card's canonical evidence labels and values, rule/status, identifiers, and missing context without images. This remains an explicit choice when images are unavailable; a failed image request does not silently switch to facts-only.

Reviewer notes and source paths are excluded from both requests. The adapter queries models available to the connected account and accepts completed structured responses tied to the supplied evidence IDs. The factual explanation uses saved canonical facts and retains missing-fact warnings. Image observations appear separately as fallible model interpretations; they do not change recorded findings or establish speed, permitted direction, identity, guilt, or a legal violation.

Successful explanations are saved with the original evidence fingerprint and included in the review export. Reopening a run shows **Saved explanation**, its image descriptors, and any model observations. Reloading does not send the evidence again or run new inference. Sending new image evidence requires a current exact preview and another explicit send action.

The app uses ChatGPT plan usage and has no paid API fallback. Account eligibility, available models and a live inference request must be verified after authorization. Offline tests use clearly labeled mock grants and responses; they do not demonstrate live access. Usage and limits are managed in [ChatGPT settings](https://chatgpt.com/settings/usage). A failed or incomplete stream produces an error, not a fabricated explanation.

## Models

Dependencies do not include model weights. Supply weights you are authorized to use under the runtime directory:

| Model | Location | SHA-256 |
| --- | --- | --- |
| Ultralytics YOLO11s | `models/yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| `iam-tsr/yolov8n-helmet-detection`, revision `59b68b03b60314badb0c19ebcc3b95baa8241cf8` | `huggingface/hub/models--iam-tsr--yolov8n-helmet-detection/snapshots/59b68b03b60314badb0c19ebcc3b95baa8241cf8/best.pt` | `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d` |

The detector checks these fingerprints before loading. The full analyzer also uses pinned Grounding DINO, segmentation and sign-recognition models; their identifiers and revisions are in `traffic_ai/scene.py`, `segmentation.py`, `signs.py`, and `turn_signs.py`. Each model and dependency retains its upstream licensing terms; no model license is granted here. In particular, [SegFormer's license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) limits use to noncommercial research or evaluation. Ultralytics dependencies and YOLO-derived checkpoints also require review of their upstream terms before redistribution.

To reuse an existing local interpreter and weights without copying them, create ignored `ai-runtime.json` with absolute local paths. Keep `home` set to an existing dedicated directory for this app. Optional fields are `python`, `models` (keys `road_users`, `helmet`, `turn_sign`, `vehicle_types`), and `pretrained_snapshots` (model ID mapped to `{ "revision": "<pinned commit>", "path": "<local snapshot directory>" }`). Set `isolate_caches: true` and `offline: true` to keep generated caches under `home` and prevent model downloads. Referenced model files and snapshots are read only. The default setup can download missing auxiliary models unless offline mode is enabled.

Scene overview is optional and uses the bounded CPU runtime specified by `VisionConfig` in `traffic_ai/bounded_vision.py`, under `.ai-runtime/ollama/`. Without those local files it reports unavailable. Detection remains separate from this overview.

## Scope

Detection covers road users and helmets, with tracking and evidence-based assessments for nine traffic-rule categories. Saved results include media, individual decisions, event evidence and source-coordinate information. Unknown context remains `cannot_assess`. General real-traffic accuracy and fully automatic signal/sign applicability have not been established. Scene overview categories are unverified and do not determine violations.

Three actual four-second runs on permitted real footage produced **zero supported candidates**. Raw detections included a billboard face mistaken for an unhelmeted head and a car region mistaken for a helmet; association and rule checks did not promote those detections to supported violations. These small checks illustrate limitations and withholding behavior, not an accuracy benchmark or proof that every event was found.

Provide your own permitted image or video. No demo media, private uploads, cached weights, credentials or generated results are bundled.

## Development

```powershell
.\.ai-runtime\venv\Scripts\python.exe -m pytest
node --test tests/*.test.cjs
.\.ai-runtime\venv\Scripts\python.exe -m compileall -q ivp traffic_ai web_server.py launch.py
```

The Python suite generates its own media fixtures. Model calls are stubbed in the automated suite; running it does not establish model accuracy. There is no frontend bundler or deployment configuration.
