"""Local configuration contracts using inert files; no model or interpreter runs."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from traffic_ai import local_models, runtime


CACHE_KEYS = ('HF_HOME', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TORCH_HOME',
              'YOLO_CONFIG_DIR', 'XDG_CACHE_HOME', 'MPLCONFIGDIR')
REVISION = 'a' * 40


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    monkeypatch.setattr(runtime, 'ROOT', project)
    for key in (*CACHE_KEYS, 'TRAFFIC_AI_HOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
        monkeypatch.delenv(key, raising=False)
    return project


def configure(project, data):
    (project / 'ai-runtime.json').write_text(json.dumps(data), encoding='utf-8')


def test_config_accessor_does_not_create_runtime_or_modify_config(isolated):
    assert runtime.runtime_config() == {}
    assert runtime.runtime_root() == isolated / '.ai-runtime'
    assert not (isolated / '.ai-runtime').exists()
    home = isolated / 'dedicated'
    home.mkdir()
    executable = isolated / 'python.exe'
    executable.write_bytes(b'inert interpreter placeholder, never executed')
    configure(isolated, {'home': str(home), 'python': str(executable), 'offline': True})
    original = (isolated / 'ai-runtime.json').read_bytes()
    assert runtime.runtime_config()['home'] == str(home)
    assert runtime.python_path() == executable
    assert not list(home.iterdir())
    assert (isolated / 'ai-runtime.json').read_bytes() == original


def test_defaults_preserve_legacy_runtime_and_env_override(isolated, monkeypatch):
    home = isolated / 'environment-home'
    monkeypatch.setenv('TRAFFIC_AI_HOME', str(home))
    assert runtime.runtime_root() == home
    assert runtime.python_path() == home / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    external = str(isolated / 'existing-cache')
    monkeypatch.setenv('HF_HOME', external)
    assert runtime.configure_cache() == home
    assert os.environ['HF_HOME'] == external
    assert 'HF_HUB_OFFLINE' not in os.environ


def test_isolation_overrides_all_inherited_cache_locations_before_loading(isolated, monkeypatch):
    home = isolated / 'dedicated'
    home.mkdir()
    configure(isolated, {'home': str(home), 'isolate_caches': True, 'offline': True})
    for key in CACHE_KEYS:
        monkeypatch.setenv(key, str(isolated.parent / 'read-only-shared' / key))
    monkeypatch.setenv('HF_HUB_OFFLINE', '0')
    monkeypatch.setenv('TRANSFORMERS_OFFLINE', '0')
    assert runtime.configure_cache() == home
    assert all(Path(os.environ[key]).is_relative_to(home) for key in CACHE_KEYS)
    assert os.environ['HF_HUB_CACHE'] == os.environ['HUGGINGFACE_HUB_CACHE']
    assert os.environ['HF_HUB_OFFLINE'] == os.environ['TRANSFORMERS_OFFLINE'] == '1'
    assert not (isolated.parent / 'read-only-shared').exists()


@pytest.mark.parametrize('data', [[], {'unknown': True}, {'offline': 'true'}, {'isolate_caches': 1},
                                {'home': 'relative'}, {'python': 'relative'}, {'models': []},
                                {'home': '//server/shared-runtime'},
                                {'models': {'unknown': 'x'}}, {'models': {'helmet': 'relative'}},
                                {'pretrained_snapshots': []},
                                {'pretrained_snapshots': {'owner/model': {'revision': 'main', 'path': '/missing'}}}])
def test_invalid_configuration_rejected_without_creating_files(isolated, data):
    configure(isolated, data)
    with pytest.raises(ValueError):
        runtime.runtime_config()
    assert sorted(path.name for path in isolated.iterdir()) == ['ai-runtime.json']


def test_missing_home_interpreter_and_directory_model_paths_rejected(isolated):
    for data in ({'home': str(isolated / 'missing')}, {'python': str(isolated / 'missing.exe')},
                 {'home': str(isolated)}, {'models': {'helmet': str(isolated)}}):
        configure(isolated, data)
        with pytest.raises(ValueError):
            runtime.runtime_config()


def test_bounded_config_size_and_invalid_json(isolated):
    path = isolated / 'ai-runtime.json'
    for raw in (b'not JSON', b'x' * (64 * 1024 + 1)):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            runtime.runtime_config()


def test_model_paths_optional_and_absence_is_reported_by_existing_pin_checks(isolated):
    mapped = isolated / 'shared' / 'road.pt'
    configure(isolated, {'models': {'road_users': str(mapped)}})
    assert runtime.model_path('road_users', isolated / 'fallback') == mapped
    assert runtime.model_path('helmet', isolated / 'fallback') == isolated / 'fallback'
    with pytest.raises(RuntimeError, match='unavailable locally'):
        local_models.verified_local_models(helmet=False)


def test_snapshot_model_symlink_preserves_pt_suffix_and_pinned_content(isolated, monkeypatch):
    blob = isolated / 'hub/blobs' / ('b' * 64)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b'SYNTHETIC inert snapshot blob; no model is loaded')
    checkpoint = isolated / 'hub/snapshots' / REVISION / 'best.pt'
    checkpoint.parent.mkdir(parents=True)
    try:
        checkpoint.symlink_to(blob)
    except (OSError, NotImplementedError) as exc:
        if os.name == 'nt':
            pytest.skip(f'Windows account cannot create symlinks: {exc}')
        raise
    configure(isolated, {'models': {'road_users': str(checkpoint)}})
    monkeypatch.setitem(local_models.PINS, 'road_users',
                        (hashlib.sha256(blob.read_bytes()).hexdigest(), blob.stat().st_size))
    configured = runtime.model_path('road_users', isolated / 'fallback.pt')
    assert configured == checkpoint and configured.suffix == '.pt'
    assert configured.resolve() == blob
    paths, manifest = local_models.verified_local_models(helmet=False)
    assert paths['road_users'] == checkpoint
    assert manifest['road_users']['filename'] == 'best.pt'
    assert checkpoint.is_symlink()


def test_absolute_paths_normalize_lexically_without_resolving_loader_filename(isolated, monkeypatch):
    checkpoint = isolated / 'snapshots' / REVISION / 'best.pt'
    value = str(checkpoint.parent / '..' / REVISION / 'best.pt')
    def forbidden_resolve(*args, **kwargs):
        pytest.fail('Loader-facing path normalization must not dereference symlinks')
    monkeypatch.setattr(Path, 'resolve', forbidden_resolve)
    normalized = runtime._absolute_path(value, 'models.helmet')
    assert normalized == checkpoint and normalized.name == 'best.pt'


def test_mapped_weights_still_require_exact_pins_and_are_never_changed(isolated, monkeypatch):
    road, helmet = isolated / 'road.pt', isolated / 'helmet.pt'
    road.write_bytes(b'SYNTHETIC road weight placeholder')
    helmet.write_bytes(b'SYNTHETIC helmet weight placeholder')
    configure(isolated, {'models': {'road_users': str(road), 'helmet': str(helmet)}})
    monkeypatch.setattr(local_models, 'PINS', {key: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size)
                                             for key, path in [('road_users', road), ('helmet', helmet)]})
    paths, manifest = local_models.verified_local_models()
    assert paths == {'road_users': road, 'helmet': helmet}
    assert manifest['helmet']['revision'] == local_models.HELMET_REVISION
    original = road.read_bytes()
    monkeypatch.setitem(local_models.PINS, 'road_users', ('0' * 64, len(original)))
    with pytest.raises(RuntimeError, match='fingerprint is unsupported'):
        local_models.verified_local_models()
    assert road.read_bytes() == original


def test_explicit_runtime_root_injection_keeps_existing_pincheck_hook(isolated, monkeypatch):
    configure(isolated, {'models': {'road_users': str(isolated / 'configured-but-unused.pt')}})
    fixture_home = isolated / 'fixture-home'
    fixture_path = fixture_home / 'models/yolo11s.pt'
    fixture_path.parent.mkdir(parents=True)
    fixture_path.write_bytes(b'SYNTHETIC injected weight placeholder')
    monkeypatch.setattr(local_models, 'runtime_root', lambda: fixture_home)
    monkeypatch.setitem(local_models.PINS, 'road_users', (hashlib.sha256(fixture_path.read_bytes()).hexdigest(), fixture_path.stat().st_size))
    assert local_models.verified_local_models(helmet=False)[0]['road_users'] == fixture_path


class Factory:
    def __init__(self, successful=None):
        self.calls = []
        self.successful = successful

    def from_pretrained(self, path, **kwargs):
        self.calls.append((path, kwargs))
        if path == self.successful or kwargs.get('local_files_only') is not True:
            return 'inert-loaded-result'
        raise OSError('Synthetic local cache miss')


def test_mapped_pinned_snapshot_is_loaded_locally_with_options_preserved(isolated):
    snapshot = isolated / 'snapshot'
    snapshot.mkdir()
    configure(isolated, {'offline': True, 'pretrained_snapshots': {'owner/model': {'revision': REVISION, 'path': str(snapshot)}}})
    factory = Factory(str(snapshot))
    assert runtime.cached_pretrained(factory, 'owner/model', REVISION, use_fast=False, trust_remote_code=False) == 'inert-loaded-result'
    assert factory.calls == [(str(snapshot), {'revision': REVISION, 'local_files_only': True, 'use_fast': False, 'trust_remote_code': False})]


@pytest.mark.parametrize('kind', ['missing', 'different_revision', 'partial', 'unmapped'])
def test_configured_snapshot_misses_can_only_try_current_local_cache(isolated, kind):
    snapshot = isolated / 'snapshot'
    if kind != 'missing':
        snapshot.mkdir()
    configure(isolated, {'pretrained_snapshots': {'owner/model': {'revision': REVISION, 'path': str(snapshot)}}})
    factory = Factory()
    model = 'owner/other' if kind == 'unmapped' else 'owner/model'
    revision = 'b' * 40 if kind == 'different_revision' else REVISION
    with pytest.raises(OSError, match='Synthetic local cache miss'):
        runtime.cached_pretrained(factory, model, revision)
    assert all(kwargs['local_files_only'] is True for _, kwargs in factory.calls)
    assert factory.calls[-1][0] == model
    assert len(factory.calls) == (2 if kind == 'partial' else 1)


def test_offline_unmapped_model_does_not_make_online_fallback(isolated):
    configure(isolated, {'offline': True})
    factory = Factory()
    with pytest.raises(OSError):
        runtime.cached_pretrained(factory, 'owner/model', REVISION)
    assert len(factory.calls) == 1
    assert factory.calls[0][1]['local_files_only'] is True


def test_isolation_alone_forces_local_loading_before_configure_cache(isolated):
    configure(isolated, {'isolate_caches': True})
    factory = Factory()
    with pytest.raises(OSError):
        runtime.cached_pretrained(factory, 'owner/model', REVISION)
    assert len(factory.calls) == 1 and factory.calls[0][1]['local_files_only'] is True


def test_default_online_fallback_remains_compatible_when_not_restricted(isolated):
    factory = Factory()
    assert runtime.cached_pretrained(factory, 'owner/model', REVISION) == 'inert-loaded-result'
    assert factory.calls == [('owner/model', {'revision': REVISION, 'local_files_only': True}),
                             ('owner/model', {'revision': REVISION})]


def test_explicit_local_only_and_inherited_offline_are_respected(isolated, monkeypatch):
    for kwargs in ({'local_files_only': True}, {}):
        if not kwargs:
            monkeypatch.setenv('HF_HUB_OFFLINE', '1')
        factory = Factory()
        with pytest.raises(OSError):
            runtime.cached_pretrained(factory, 'owner/model', REVISION, **kwargs)
        assert len(factory.calls) == 1


def test_remote_code_cannot_be_enabled(isolated):
    factory = Factory()
    with pytest.raises(ValueError, match='Remote model code'):
        runtime.cached_pretrained(factory, 'owner/model', REVISION, trust_remote_code=True)
    assert factory.calls == []


def test_existing_scene_and_segformer_revision_constants_pass_config_validation(isolated):
    from traffic_ai.scene import SceneProposer
    from traffic_ai.segmentation import RoadSegmenter

    snapshots = {}
    for index, detector in enumerate((SceneProposer, RoadSegmenter)):
        snapshot = isolated / f'inert-snapshot-{index}'
        snapshot.mkdir()
        assert len(detector.REVISION) == 40
        snapshots[detector.MODEL] = {'revision': detector.REVISION, 'path': str(snapshot)}
    configure(isolated, {'offline': True, 'pretrained_snapshots': snapshots})
    assert runtime.runtime_config()['pretrained_snapshots'] == snapshots


@pytest.mark.parametrize('flag', ['offline', 'isolate_caches'])
def test_turn_sign_local_cache_miss_cannot_fall_back_online(isolated, monkeypatch, flag):
    from traffic_ai.turn_signs import TurnSignDetector

    class LocalMiss(OSError):
        pass

    calls = []
    def download(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs.get('local_files_only') is True, 'Online fallback must never be attempted'
        raise LocalMiss('Synthetic local-cache miss; no network operation')

    def forbidden_model(*args, **kwargs):
        pytest.fail('Missing offline weights must not construct a model')

    configure(isolated, {flag: True})
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    monkeypatch.setitem(sys.modules, 'huggingface_hub.errors', SimpleNamespace(LocalEntryNotFoundError=LocalMiss))
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=forbidden_model))
    with pytest.raises(RuntimeError, match='offline runtime; no download was attempted'):
        TurnSignDetector()
    assert calls == [((TurnSignDetector.MODEL, 'best_roboflow.pt'),
                      {'revision': TurnSignDetector.REVISION, 'local_files_only': True})]


@pytest.mark.parametrize('correct_pin', [True, False])
def test_turn_sign_configured_read_only_path_is_verified_before_model_construction(isolated, monkeypatch, correct_pin):
    from traffic_ai.perception import Perception
    from traffic_ai.turn_signs import TURN_MEANINGS, TurnSignDetector

    checkpoint = isolated / 'read-only-models' / 'turn.pt'
    checkpoint.parent.mkdir()
    original = b'SYNTHETIC inert turn checkpoint fixture; never a real model'
    checkpoint.write_bytes(original)
    configure(isolated, {'offline': True, 'models': {'turn_sign': str(checkpoint)}})
    calls = []

    def fingerprint(path):
        calls.append(('fingerprint', path))
        assert path == checkpoint
        return {'filename': path.name, 'sha256': TurnSignDetector.SHA256 if correct_pin else '0' * 64,
                'bytes': len(original)}

    def model(path):
        calls.append(('model', path))
        assert calls[0] == ('fingerprint', checkpoint)
        return SimpleNamespace(names=dict(enumerate(TURN_MEANINGS)))

    def forbidden_download(*args, **kwargs):
        pytest.fail('Configured existing checkpoint must not invoke cache or network download')

    monkeypatch.setattr(Perception, '_fingerprint', staticmethod(fingerprint))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=forbidden_download))
    monkeypatch.setitem(sys.modules, 'huggingface_hub.errors', SimpleNamespace(LocalEntryNotFoundError=OSError))
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=model))
    checkpoint.chmod(0o444)
    try:
        if correct_pin:
            detector = TurnSignDetector()
            assert calls == [('fingerprint', checkpoint), ('model', checkpoint)]
            assert detector.manifest['sha256'] == TurnSignDetector.SHA256
            assert detector.manifest['revision'] == TurnSignDetector.REVISION
        else:
            with pytest.raises(ValueError, match='checksum verification'):
                TurnSignDetector()
            assert calls == [('fingerprint', checkpoint)]
        assert checkpoint.read_bytes() == original
    finally:
        checkpoint.chmod(0o666)
