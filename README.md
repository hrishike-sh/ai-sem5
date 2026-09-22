# AI Sem 5

Local traffic-image and video analysis. Run detection and rule assessment, inspect saved evidence, or request an optional local scene overview.

## Run

Install Python 3.11, uv, and FFmpeg (both `ffmpeg` and `ffprobe` on PATH). On Windows:

```powershell
.\Setup-AI.ps1
.\Start.cmd
```

The app runs at http://127.0.0.1:8521. Close its terminal to stop the server. It uses its own `uploads/`, `runs/`, and `.ai-runtime/` directories. `TRAFFIC_AI_HOME` can select another dedicated runtime directory. The retained `ivp/` package supplies shared image processing, media handling, jobs and evidence storage.

## Models

Dependencies do not include model weights. Supply weights you are authorized to use under the runtime directory:

| Model | Location | SHA-256 |
| --- | --- | --- |
| Ultralytics YOLO11s | `models/yolo11s.pt` | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| `iam-tsr/yolov8n-helmet-detection`, revision `59b68b03b60314badb0c19ebcc3b95baa8241cf8` | `huggingface/hub/models--iam-tsr--yolov8n-helmet-detection/snapshots/59b68b03b60314badb0c19ebcc3b95baa8241cf8/best.pt` | `c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d` |

The detector checks these fingerprints before loading. The full analyzer also uses pinned Grounding DINO, segmentation and sign-recognition models; their identifiers and revisions are in `traffic_ai/scene.py`, `segmentation.py`, `signs.py`, and `turn_signs.py`. These adapters may download missing model files on first use. Each model and dependency retains its upstream licensing terms; no model license is granted here.

Scene overview is optional and uses the bounded CPU runtime specified by `VisionConfig` in `traffic_ai/bounded_vision.py`, under `.ai-runtime/ollama/`. Without those local files it reports unavailable. Detection remains separate from this overview.

## Scope

Detection covers road users and helmets, with tracking and evidence-based assessments for nine traffic-rule categories. Saved results include media, individual decisions, event evidence and source-coordinate information. Unknown context remains `cannot_assess`. General real-traffic accuracy and fully automatic signal/sign applicability have not been established. Scene overview categories are unverified and do not determine violations.

Provide your own permitted image or video. No demo media, private uploads, cached weights, credentials or generated results are bundled.

## Development

```powershell
.\.ai-runtime\venv\Scripts\python.exe -m pytest
node --test tests/*.test.cjs
.\.ai-runtime\venv\Scripts\python.exe -m compileall -q ivp traffic_ai web_server.py launch.py
```

The Python suite generates its own media fixtures. Model calls are stubbed in the automated suite; running it does not establish model accuracy. There is no frontend bundler or deployment configuration.
