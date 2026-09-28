import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
MODEL_KEYS = {'road_users', 'helmet', 'turn_sign'}


def _absolute_path(value, field):
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError(f'ai-runtime.json {field} must be an absolute local path.')
    path = Path(value)
    if not path.is_absolute() or value.startswith(('\\\\', '//')):
        raise ValueError(f'ai-runtime.json {field} must be an absolute local path.')
    # Keep snapshot filenames such as best.pt: Hugging Face snapshots can link
    # to extensionless blobs, while model loaders select their format by suffix.
    # Resolve symlinks only for containment checks, not for loader-facing paths.
    return Path(os.path.abspath(path))


def runtime_config():
    """Read optional local configuration without creating or changing files.

    Model files may be absent so readiness checks can explain missing weights.
    Snapshot directories may be absent; loaders then try the existing local HF
    cache, never an online fallback for an explicitly configured snapshot map.
    """
    path = ROOT / 'ai-runtime.json'
    if not path.exists():
        return {}
    try:
        with path.open('rb') as stream:
            raw = stream.read(64 * 1024 + 1)
        if len(raw) > 64 * 1024:
            raise ValueError('ai-runtime.json exceeds the 64 KiB configuration limit.')
        config = json.loads(raw.decode('utf-8-sig'))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('ai-runtime.json must contain a valid JSON object.') from exc
    allowed = {'home', 'python', 'models', 'pretrained_snapshots', 'offline', 'isolate_caches'}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError('ai-runtime.json contains unsupported configuration fields.')
    for flag in ('offline', 'isolate_caches'):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError(f'ai-runtime.json {flag} must be a boolean.')
    if 'home' in config:
        home = _absolute_path(config['home'], 'home')
        if not home.is_dir() or not os.access(home, os.W_OK):
            raise ValueError('ai-runtime.json home must be an existing writable runtime directory.')
        resolved_home = home.resolve()
        if resolved_home in (Path(resolved_home.anchor), Path.home().resolve(), ROOT.resolve()):
            raise ValueError('ai-runtime.json home must be a dedicated runtime directory.')
        config['home'] = str(home)
    if 'python' in config:
        executable = _absolute_path(config['python'], 'python')
        if not executable.is_file():
            raise ValueError('ai-runtime.json python must reference an existing interpreter file.')
        config['python'] = str(executable)
    if 'models' in config:
        models = config['models']
        if not isinstance(models, dict) or set(models) - MODEL_KEYS:
            raise ValueError('ai-runtime.json models supports road_users, helmet and turn_sign paths only.')
        for key, value in models.items():
            model = _absolute_path(value, 'models.' + key)
            if model.exists() and not model.is_file():
                raise ValueError(f'ai-runtime.json models.{key} must reference a model file.')
            models[key] = str(model)
    if 'pretrained_snapshots' in config:
        snapshots = config['pretrained_snapshots']
        if not isinstance(snapshots, dict) or len(snapshots) > 64:
            raise ValueError('ai-runtime.json pretrained_snapshots must be a bounded model mapping.')
        for model, entry in snapshots.items():
            if (not isinstance(model, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', model)
                    or not isinstance(entry, dict) or set(entry) != {'revision', 'path'}
                    or not isinstance(entry['revision'], str) or not re.fullmatch(r'[a-fA-F0-9]{40}', entry['revision'])):
                raise ValueError('Pretrained snapshots need a model ID, pinned 40-digit commit revision and local path.')
            snapshot = _absolute_path(entry['path'], 'pretrained_snapshots.' + model)
            if snapshot.exists() and not snapshot.is_dir():
                raise ValueError('Pretrained snapshot paths must reference directories.')
            entry['path'] = str(snapshot)
    return config


def runtime_root():
    configured = os.environ.get("TRAFFIC_AI_HOME")
    if not configured:
        configured = runtime_config().get('home')
    return Path(configured) if configured else ROOT / ".ai-runtime"


def configure_cache():
    config = runtime_config()
    home = runtime_root()
    paths = {'HF_HOME': home / 'huggingface', 'YOLO_CONFIG_DIR': home / 'ultralytics', 'TORCH_HOME': home / 'torch'}
    if config.get('isolate_caches'):
        paths.update(HF_HUB_CACHE=home / 'huggingface' / 'hub',
                     HUGGINGFACE_HUB_CACHE=home / 'huggingface' / 'hub',
                     XDG_CACHE_HOME=home / 'cache', MPLCONFIGDIR=home / 'matplotlib')
        if any(not path.resolve().is_relative_to(home.resolve()) for path in
               (*paths.values(), home / 'ultralytics' / 'Ultralytics')):
            raise ValueError('Isolated cache paths must stay inside the dedicated runtime directory.')
    home.mkdir(parents=True, exist_ok=True)
    (home / "ultralytics" / "Ultralytics").mkdir(parents=True, exist_ok=True)
    for key, value in paths.items():
        if config.get('isolate_caches'):
            os.environ[key] = str(value)
        else:
            os.environ.setdefault(key, str(value))
    if config.get('offline') or config.get('isolate_caches'):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return home


def python_path():
    configured = runtime_config().get('python')
    if configured:
        return Path(configured)
    return runtime_root() / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def model_path(key, default):
    if key not in MODEL_KEYS:
        raise ValueError('Unknown local model key.')
    return Path(runtime_config().get('models', {}).get(key, default))


def cached_pretrained(factory, model, revision, **kwargs):
    config = runtime_config()
    if kwargs.get('trust_remote_code') not in (None, False):
        raise ValueError('Remote model code is not enabled by this runtime.')
    local_only = kwargs.pop('local_files_only', False)
    if type(local_only) is not bool:
        raise ValueError('local_files_only must be a boolean.')
    snapshots = config.get('pretrained_snapshots', {})
    entry = snapshots.get(model)
    if entry is not None and entry['revision'].lower() == str(revision).lower() and Path(entry['path']).is_dir():
        try:
            return factory.from_pretrained(entry['path'], revision=revision, local_files_only=True, **kwargs)
        except OSError:
            # A partial explicit snapshot can still be present in the existing
            # local cache. Never silently download missing mapped snapshot files.
            pass
    try:
        return factory.from_pretrained(model, revision=revision, local_files_only=True, **kwargs)
    except OSError:
        if (config.get('offline') or config.get('isolate_caches') or 'pretrained_snapshots' in config
                or local_only or os.environ.get('HF_HUB_OFFLINE') == '1' or os.environ.get('TRANSFORMERS_OFFLINE') == '1'):
            raise
        return factory.from_pretrained(model, revision=revision, **kwargs)
