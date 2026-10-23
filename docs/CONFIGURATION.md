# Configuration

[Back to README](../README.md) | [Model weights and licenses](MODELS.md)

## Runtime and storage

Run the scripts from the repository directory. `Start.cmd` invokes `Start.ps1`, which starts `web_server.py` on port 8521 using the configured interpreter.

The model runtime directory is selected in this order: `TRAFFIC_AI_HOME`, `ai-runtime.json` field `home`, then repository-local `.ai-runtime/`. The interpreter is `ai-runtime.json` field `python` when supplied, otherwise `<runtime>/venv/Scripts/python.exe` on Windows. The launcher also prepends `<runtime>/site-packages` to `PYTHONPATH` when that directory exists.

`uploads/`, `runs/`, `direct-reviews/` and `temp/` stay under the repository. Moving the runtime does not move media or saved reviews. The app grant is stored separately under `%LOCALAPPDATA%/ViolationLens/auth`.

## Dependency setup

`Setup-AI.ps1` uses uv to create a Python 3.11 environment, installs `torch==2.11.0+cu128` and `torchvision==0.26.0+cu128` from `https://download.pytorch.org/whl/cu128`, then installs the pinned `requirements-ai.txt`. It does not install model weights or OCR extras. It rejects an existing `venv` and writes `ai-runtime.json` with the chosen `home`; use an existing configuration directly instead of rerunning setup over it.

For a new dedicated runtime location:

```powershell
.\Setup-AI.ps1 -RuntimeHome 'C:\ViolationLensRuntime'
.\Start.cmd
```

The supplied installer has no CPU-wheel switch. CUDA-capable PyTorch can still execute CPU inference. `Perception` chooses `cuda:0` when available, otherwise `cpu`; the image helmet, subtype and plate specialists explicitly select CPU. Optional bounded scene overview is CPU-only.

## Reuse an existing interpreter and weights

Create ignored `ai-runtime.json` with absolute local paths. The `home` directory must already exist, be writable and be dedicated to the app; it cannot be the filesystem root, user home or repository root. `python`, if supplied, must point to an existing interpreter. Example structure (replace every example path before use):

```json
{
  "home": "C:\\ViolationLensRuntime",
  "python": "C:\\ViolationLensRuntime\\venv\\Scripts\\python.exe",
  "models": {
    "road_users": "C:\\ModelStore\\yolo11s.pt",
    "helmet": "C:\\ModelStore\\helmet\\best.pt",
    "turn_sign": "C:\\ModelStore\\best_roboflow.pt",
    "vehicle_types": "C:\\ModelStore\\uvh26-yolo11s-4a224127.pt"
  },
  "offline": true,
  "isolate_caches": true
}
```

Only `home`, `python`, `models`, `pretrained_snapshots`, `offline` and `isolate_caches` are accepted. Optional fields can be omitted. `models` accepts only the four keys above; plate weights have fixed paths under the runtime instead.

`pretrained_snapshots` maps an exact model ID to an object containing `revision` (a 40-character commit hash) and `path` (an absolute local snapshot directory). Match the revisions in [MODELS.md](MODELS.md#auxiliary-video-and-scene-models). Referenced weights and snapshots are read without copying or modifying them. A configured snapshot map prevents automatic download fallback, including for missing mapped files.

`isolate_caches: true` keeps generated Hugging Face, Torch, Ultralytics, XDG and Matplotlib caches under the dedicated runtime and enables Hugging Face/Transformers offline behavior. `offline: true` also enables offline model loading. These flags concern model loading; they do not disable an explicitly requested ChatGPT connection. Without offline configuration, optional auxiliary model loaders may download pinned files. The main image weight loaders do not download missing weights.

## Optional plate OCR

With the default runtime, install the optional recognition packages alongside the base environment:

```powershell
uv pip install --python .\.ai-runtime\venv\Scripts\python.exe --target .\.ai-runtime\site-packages --no-deps -r requirements-ocr.txt
```

Substitute the configured interpreter and runtime paths if different. This preserves the base PyTorch, torchvision and OpenCV packages. The optional requirements pin EasyOCR 1.7.2 and its additional dependencies. Restart the app after installation so the launcher adds the target directory to the import path.

Supply `<runtime>/models/plates/detector.pt` and `<runtime>/models/plates/english_g2.pth` from the exact sources in [MODELS.md](MODELS.md#plate-detection-and-ocr). Both hashes are checked before loading; EasyOCR downloads are disabled. Recognition is English-only and uses an alphanumeric allowlist. Missing packages or weights produce an unavailable check rather than a successful empty result.

## ChatGPT connection

The Windows prototype implements the [Sign in with ChatGPT flow for locally hosted apps](https://developers.openai.com/siwc/token-sharing-open-source/sign-in). **Connect ChatGPT** asks for approval before opening sign-in. It creates a separate app grant and does not reuse Codex or browser credentials. Windows DPAPI encrypts the stored grant under `%LOCALAPPDATA%/ViolationLens/auth`. Installing dependencies or opening the page does not authorize access.

The main numbered-image review requests `gpt-6-astra`. It requires an exact preview and an explicit **Send to LLM** action. There is no silent model substitution or paid API fallback. See the [README](../README.md#optional-llm-review) for the outbound image and text boundaries.

Video candidate cards have a separate explanation workflow:

- **Preview frames and facts for ChatGPT**, then **Send selected frames and facts to ChatGPT**, sends at most three prepared JPEG images, each at most 768 pixels on its longest edge, plus structured evidence. It does not send the full clip.
- **Send structured facts only to ChatGPT** sends canonical labels and values, rule/status, identifiers and missing context without images. This is an explicit choice; a failed image request does not silently fall back to facts-only.

Both exclude reviewer notes and source paths. The video explanation adapter queries account-available models and accepts completed structured responses tied to supplied evidence IDs. Canonical facts and missing-context warnings remain separate from fallible image observations. Successful explanations are saved with evidence fingerprints and included in exports; reopening a saved explanation does not send it again.

Account eligibility, available models and live inference must be verified after authorization. Offline tests use mock grants and responses and do not prove live access. Usage is managed in [ChatGPT settings](https://chatgpt.com/settings/usage). Failed or incomplete responses remain errors.
