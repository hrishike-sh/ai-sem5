"""Read-only startup diagnostics; no downloads, model loads or auth requests."""
import hashlib
import importlib.util
from pathlib import Path

from .local_models import verified_local_models
from .runtime import runtime_config, runtime_root, python_path, model_path


SNAPSHOTS = (
    ('scene', 'IDEA-Research/grounding-dino-tiny', 'a2bb814dd30d776dcf7e30523b00659f4f141c71',
     ('config.json', 'model.safetensors', 'tokenizer_config.json', 'vocab.txt', 'preprocessor_config.json')),
    ('segmentation', 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024', '21b3847fae21ddee674abd31129307b6a1235bd9',
     ('config.json', 'pytorch_model.bin', 'preprocessor_config.json')),
    ('sign_classifier', 'bazyl/gtsrb-model', '95f6c95c765e1705d7f92161a5a42a40a5bd8338',
     ('config.json', 'pytorch_model.bin', 'preprocessor_config.json')),
)
TURN_HASH = '93793cc99ffda2a10e09860e3a7c6383f7c25d470dd604c4555b5b07253a6e2b'


def pipeline_readiness():
    blockers, warnings = [], []
    def block(identity, message):
        blockers.append({'id': identity, 'message': message})
    try:
        config, home, executable = runtime_config(), runtime_root(), python_path()
    except (ValueError, OSError):
        return {'ready': False, 'blockers': [{'id': 'configuration', 'message': 'Local runtime configuration is invalid. Check ai-runtime.json paths and fields.'}], 'warnings': []}
    if not executable.is_file():
        block('python', 'The configured Python runtime is missing. Run Setup-AI.ps1 or select an existing compatible interpreter.')
    missing = []
    for name in ('numpy', 'cv2', 'av', 'torch', 'torchvision', 'ultralytics', 'transformers', 'lap'):
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError, AttributeError):
            missing.append(name)
    if missing:
        block('dependencies', 'Required local dependencies are missing: ' + ', '.join(missing) + '. Run Setup-AI.ps1 in a dedicated runtime.')
    try:
        verified_local_models()
    except (ValueError, OSError, RuntimeError) as exc:
        block('perception', str(exc))
    for identity, model, revision, filenames in SNAPSHOTS:
        entry = config.get('pretrained_snapshots', {}).get(model)
        path = Path(entry['path']) if entry and entry['revision'] == revision else (
            home / 'huggingface' / 'hub' / ('models--' + model.replace('/', '--')) / 'snapshots' / revision)
        if any(not (path / filename).is_file() for filename in filenames):
            block(identity, f'The pinned {identity.replace("_", " ")} model is incomplete locally. Configure its existing snapshot before detection.')
    turn = model_path('turn_sign', home / 'models' / 'turn-yahyagul-15be2681.pt')
    try:
        with turn.open('rb') as stream:
            valid_turn = hashlib.file_digest(stream, 'sha256').hexdigest() == TURN_HASH
    except OSError:
        valid_turn = False
    if not valid_turn:
        block('turn_sign', 'The pinned turn-sign checkpoint is missing or does not match its expected hash.')
    if not config.get('offline') and not config.get('isolate_caches'):
        warnings.append({'id': 'network', 'message': 'Offline mode is not enabled; auxiliary loaders can attempt downloads. Configure offline mode for a fixed local demo.'})
    warnings.append({'id': 'scope', 'message': 'Model files being available does not establish accuracy. Review uncertainty and source evidence.'})
    return {'ready': not blockers, 'blockers': blockers, 'warnings': warnings}
