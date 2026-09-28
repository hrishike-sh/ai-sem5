import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import traffic_ai.local_models as local
from traffic_ai.perception import Perception


@pytest.fixture
def cached(tmp_path,monkeypatch):
    monkeypatch.setattr(local,'runtime_root',lambda:tmp_path)
    monkeypatch.setattr('traffic_ai.perception.configure_cache',lambda:tmp_path)
    road=tmp_path/'models'/'yolo11s.pt';road.parent.mkdir();road.write_bytes(b'road fixture')
    helmet=tmp_path/'huggingface/hub/models--iam-tsr--yolov8n-helmet-detection/snapshots'/local.HELMET_REVISION/'best.pt'
    helmet.parent.mkdir(parents=True);helmet.write_bytes(b'helmet fixture')
    monkeypatch.setattr(local,'PINS',{k:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size) for k,p in [('road_users',road),('helmet',helmet)]})
    constructed=[]
    def model(path):
        constructed.append(path);return SimpleNamespace(names={0:'With Helmet',1:'Without Helmet'})
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=model))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False)))
    return road,helmet,constructed


@pytest.mark.parametrize('bad',['road_missing','helmet_missing','helmet_changed','road_changed'])
def test_bad_local_weights_fail_before_any_model_construction(cached,bad):
    road,helmet,constructed=cached
    p=road if bad.startswith('road') else helmet
    if bad.endswith('missing'):p.unlink()
    else:p.write_bytes(b'wrong content')
    with pytest.raises(RuntimeError,match='unavailable|unsupported'):
        Perception(helmet=True)
    assert constructed==[]
    assert local.helmet_availability()['ready'] is False


def test_verified_both_before_loading_and_manifest(cached):
    road,helmet,constructed=cached
    detector=Perception(helmet=True)
    assert constructed==[str(road),str(helmet)]
    assert detector.manifest['helmet']['revision']==local.HELMET_REVISION
    assert detector.manifest['helmet']['labels'][1]=='Without Helmet'


def test_vehicle_only_path_does_not_require_helmet(cached):
    road,helmet,constructed=cached;helmet.unlink()
    detector=Perception(helmet=False)
    assert constructed==[str(road)] and detector.helmet is None


@pytest.mark.parametrize('visible,expected',[(0,'cpu'),(1,'cuda:0')])
def test_cuda_probe_requires_a_visible_device(cached,monkeypatch,visible,expected):
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda:True,device_count=lambda:visible)))
    assert Perception(helmet=False).device==expected
