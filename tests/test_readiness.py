"""Startup diagnostics use synthetic files and never load model weights."""
import hashlib

import pytest

from traffic_ai import readiness


@pytest.fixture
def ready(tmp_path, monkeypatch):
    interpreter=tmp_path/'python.exe'; interpreter.write_bytes(b'fixture interpreter')
    turn=tmp_path/'turn.pt'; turn.write_bytes(b'fixture model')
    for _, model, revision, names in readiness.SNAPSHOTS:
        folder=tmp_path/'huggingface/hub'/('models--'+model.replace('/','--'))/'snapshots'/revision
        folder.mkdir(parents=True)
        for name in names: (folder/name).write_bytes(b'fixture')
    monkeypatch.setattr(readiness,'runtime_config',lambda:{'offline':True})
    monkeypatch.setattr(readiness,'runtime_root',lambda:tmp_path)
    monkeypatch.setattr(readiness,'python_path',lambda:interpreter)
    monkeypatch.setattr(readiness,'model_path',lambda *args:turn)
    monkeypatch.setattr(readiness,'TURN_HASH',hashlib.sha256(turn.read_bytes()).hexdigest())
    monkeypatch.setattr(readiness,'verified_local_models',lambda:({},{}))
    monkeypatch.setattr(readiness.importlib.util,'find_spec',lambda name:object())
    return tmp_path,turn


def test_ready_only_when_all_local_prerequisites_present(ready):
    assert readiness.pipeline_readiness()['ready'] is True


def test_missing_auxiliary_checkpoint_blocks_ready(ready):
    root,_=ready
    next(root.rglob('model.safetensors')).unlink()
    state=readiness.pipeline_readiness()
    assert not state['ready'] and state['blockers'][0]['id']=='scene'


def test_changed_turn_checkpoint_blocks_ready(ready):
    _,turn=ready; turn.write_bytes(b'changed')
    assert any(b['id']=='turn_sign' for b in readiness.pipeline_readiness()['blockers'])


def test_missing_dependencies_report_actionable_names(ready,monkeypatch):
    monkeypatch.setattr(readiness.importlib.util,'find_spec',lambda name:None if name=='torch' else object())
    state=readiness.pipeline_readiness()
    assert not state['ready'] and 'torch' in state['blockers'][0]['message']


def test_invalid_config_reports_without_model_loading(monkeypatch):
    def invalid(): raise ValueError('invalid')
    monkeypatch.setattr(readiness,'runtime_config',invalid)
    assert readiness.pipeline_readiness()['blockers'][0]['id']=='configuration'
