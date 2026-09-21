import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def runtime_root():
    configured = os.environ.get("TRAFFIC_AI_HOME")
    config = ROOT / "ai-runtime.json"
    if not configured and config.exists():
        configured = json.loads(config.read_text(encoding="utf-8-sig"))["home"]
    return Path(configured) if configured else ROOT / ".ai-runtime"


def configure_cache():
    home = runtime_root()
    home.mkdir(parents=True, exist_ok=True)
    (home / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(home / "huggingface"))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(home / "ultralytics"))
    os.environ.setdefault("TORCH_HOME", str(home / "torch"))
    return home


def python_path():
    return runtime_root() / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def cached_pretrained(factory, model, revision, **kwargs):
    try:
        return factory.from_pretrained(model, revision=revision, local_files_only=True, **kwargs)
    except OSError:
        return factory.from_pretrained(model, revision=revision, **kwargs)
