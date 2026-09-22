"""Read-only, pinned local weight lookup. Never downloads or constructs models."""
import hashlib
from pathlib import Path
from .runtime import runtime_root

HELMET_REPO='iam-tsr/yolov8n-helmet-detection'
HELMET_REVISION='59b68b03b60314badb0c19ebcc3b95baa8241cf8'
PINS={
    'road_users':('85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5',19313732),
    'helmet':('c8eb324e365cf4faeab491d9cc301535ec745171b55e8b1acadea62be5101a9d',6199658),
}


def verified_local_models(helmet=True):
    home=runtime_root()
    paths={'road_users':home/'models'/'yolo11s.pt'}
    if helmet:
        cached=home/'huggingface'/'hub'/'models--iam-tsr--yolov8n-helmet-detection'
        paths['helmet']=cached/'snapshots'/HELMET_REVISION/'best.pt'
    manifest={}
    for key,path in paths.items():
        expected,size=PINS[key]
        if not path.is_file():
            raise RuntimeError(f'{key} model unavailable locally. Supply the documented pinned weights (README.md); Setup-AI installs dependencies only. No download was attempted.')
        with path.open('rb') as stream:
            actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if path.stat().st_size!=size or actual!=expected:
            raise RuntimeError(f'{key} model fingerprint is unsupported. Restore the pinned local model; no download was attempted.')
        manifest[key]={'filename':'best.pt' if key=='helmet' else path.name,'sha256':actual,'bytes':size}
        if key=='helmet':
            manifest[key].update(repository=HELMET_REPO,revision=HELMET_REVISION)
    return paths,manifest


def helmet_availability():
    try:
        verified_local_models()
        return {'ready':True,'message':'Pinned vehicle and helmet models are available locally.'}
    except (OSError,RuntimeError) as exc:
        return {'ready':False,'message':str(exc)}
