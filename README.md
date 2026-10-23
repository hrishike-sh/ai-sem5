# ViolationLens - AI Sem 5

ViolationLens is a local traffic-image review app. One uploaded image feeds five checks, with editable subject boxes, saved evidence and JSON export. An optional LLM review uses a connected ChatGPT plan only after an explicit send action. Video detection and rule review are available separately.

This project integrates pretrained models and evidence checks. Its results are candidates for human review; it does not establish legal violations, identities or fines. Real-world accuracy has not been established.

## Run on Windows

Prerequisites: Python 3.11, [uv](https://docs.astral.sh/uv/), and FFmpeg with both `ffmpeg` and `ffprobe` on `PATH`. From the repository directory:

```powershell
.\Setup-AI.ps1
.\Start.cmd
```

Open **http://127.0.0.1:8521**. Press Ctrl+C in the server terminal to stop it.

`Setup-AI.ps1` creates a new `.ai-runtime/venv`, installs `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from the CUDA 12.8 wheel index, then installs `requirements-ai.txt`. It refuses to overwrite an existing virtual environment and writes the runtime location to ignored `ai-runtime.json`. It installs dependencies, **not model weights or the optional OCR packages**.

The CUDA wheel choice does not make every check run on a GPU. Subject detection selects CUDA when available and otherwise uses CPU; helmet, vehicle subtype and plate checks explicitly use CPU. See [configuration and OCR setup](docs/CONFIGURATION.md) and [required weights, provenance and licenses](docs/MODELS.md). Missing weights are shown as unavailable checks.

## Review an image

Drop a permitted JPG or PNG, or choose it with the file picker. Local processing starts automatically. For MP4, MOV or AVI, choose a timestamp and select **Upload and preview** to review one decoded frame.

| Check | What it provides |
| --- | --- |
| Subject localization | Editable person and vehicle boxes with within-image P/V IDs; at most twelve subjects, with truncation disclosed. |
| Helmet evidence | Local helmet/no-helmet candidates, raw scores, conflicts and unmatched detections. |
| Vehicle class counts | Counts of numbered motorcycle, bicycle, car, bus and truck boxes, distinguishing detector proposals from manual corrections. |
| Vehicle subtype evidence | Separate proposals for fourteen types, including two-wheelers, three-wheelers, hatchbacks and buses. |
| Plate detection and reading | Candidate plate boxes, English OCR and manual transcription with the original reading retained. |

Each check reports its own progress, result or failure. The combined table joins subject IDs, apparent rider associations, plate candidates, observations and review status. An unavailable check is not evidence of compliance. Vehicle counts describe this image, not unique traffic volume.

Inspect the shared image and correct missed or incorrect boxes. Manual boxes work when subject detection is unavailable. P IDs denote people, not confirmed riders or identities; geometric overlap alone does not establish a rider association. Plate candidates attach to a numbered vehicle only when the containment match is unambiguous.

Local evidence can be exported without an LLM request. Saved image reviews can be reopened from the page. After an LLM assessment, record **Accept assessment** or **Reject assessment** and an optional note; that decision is stored separately. Saving box changes, rerunning specialist checks or correcting a plate clears the previous LLM assessment and reviewer decision. Inspect the updated previews before sending again. Reopening unchanged saved results does not trigger another model request.

## Optional LLM review

Choose **Connect ChatGPT**, approve sign-in, then inspect the exact outbound images before selecting **Send to LLM**. Connecting or uploading alone does not send evidence to OpenAI.

The image review sends one annotated context image and up to twelve padded subject crops, each bounded to 768 pixels per edge, with subject identifiers and bound local evidence. Crops use the original selected frame. Reviewer notes, source paths, OCR strings and manual plate transcriptions are excluded from the text payload. Faces and plate pixels may remain visible in the approved images; metadata removal and cropping do not guarantee anonymity.

The implementation requests `gpt-6-astra`, with no silent model substitution or paid API fallback. Account access and plan limits must be checked after authorization. The separate app grant is encrypted with Windows DPAPI outside the repository. See [authentication and request details](docs/CONFIGURATION.md#chatgpt-connection).

The assessment ties observations and uncertainty to each submitted subject. A possible missing-helmet issue needs supported local no-helmet evidence and a visible association with the same motorcycle. Missing detections and ambiguous heads remain uncertain. A three-rider suspicion needs three distinct visible people associated with one motorcycle. A model score alone never creates a violation.

## Optional video workflow

Expand **Optional local detection and rule review** to upload a clip, run the existing detector pipeline and inspect timestamped candidates with before/during/after frames. Save accept/reject decisions and export the review JSON. Decisions are invalidated if the underlying algorithm output changes.

**Try a synthetic rule demo** generates a labeled three-second rectangle animation for opposing, following or unknown direction. It exercises configured geometry, evidence frames, decisions and export without detector weights or ChatGPT. It is not a real-traffic benchmark. The advanced detection and scene-overview interface is at **http://127.0.0.1:8521/studio**. See the [demo guide](docs/DEMO.md) for walkthroughs and recorded checks.

## Limits and recorded checks

A single image cannot establish speed, permitted travel direction, red-light crossing, obstruction or parking legality. Zebra occupancy is an observation. The optional video rules need independent direction, signal and temporal context; unknown context remains explicit. Automatic triple-riding detection in that rule pipeline remains deferred. There is no owner lookup or automatic external reporting.

The small local checks exposed missed vehicles, helmet false positives and conflicting predictions. On two real samples, plate crops produced incorrect, incomplete or low-confidence OCR; neither became an accepted reading. A synthetic text fixture was transcribed correctly but stayed below the heuristic 0.80 OCR threshold. Three four-second real-video runs produced zero supported candidates. These checks demonstrate execution and withholding behavior, not accuracy, recall or proof that every event was found. Raw scores are not calibrated probabilities. Details are retained in [model notes](docs/MODELS.md#recorded-local-checks).

## Files and development

Uploads, saved video runs, saved image reviews and temporary previews live in repository-local `uploads/`, `runs/`, `direct-reviews/` and `temp/`. `.ai-runtime/` holds the default interpreter, weights and caches. `TRAFFIC_AI_HOME` changes the model runtime location, not those saved-evidence folders. Private media, credentials, cached weights and generated results are ignored and are not bundled.

`web/` contains the browser UI, `web_server.py` serves it, and `traffic_ai/` contains the review and model adapters. The retained `ivp/` package supplies shared image processing, media handling, jobs and evidence storage. There is no frontend bundler or deployment configuration. Advanced utilities in `scripts/` have their own input requirements; historical validation utilities may require external fixtures that are not bundled.

For the default environment:

```powershell
.\.ai-runtime\venv\Scripts\python.exe -m pytest
node --test tests/*.test.cjs
.\.ai-runtime\venv\Scripts\python.exe -m compileall -q ivp traffic_ai web_server.py launch.py
```

Use the configured interpreter if it differs; the browser tests require Node.js. Regression coverage uses generated media fixtures and mocked model responses, alongside runtime and readiness checks. Passing tests verifies software behavior, not model accuracy or live ChatGPT access.
