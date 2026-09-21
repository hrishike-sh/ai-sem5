"""Resource configuration only: these tests never load a model or spawn a worker."""
from types import SimpleNamespace

import pytest

from ivp import performance


def test_affinity_limits_available_threads(monkeypatch):
    monkeypatch.setattr(performance.os, 'sched_getaffinity', lambda pid: {2, 3, 7}, raising=False)
    monkeypatch.setattr(performance.os, 'cpu_count', lambda: 20)
    assert performance.available_cpu_threads() == 3


@pytest.mark.parametrize('count, expected', [(16, 16), (20, 20), (1, 1), (None, 1), (0, 1)])
def test_windows_uses_detected_logical_processors(monkeypatch, count, expected):
    monkeypatch.delattr(performance.os, 'sched_getaffinity', raising=False)
    monkeypatch.setattr(performance.os, 'cpu_count', lambda: count)
    assert performance.available_cpu_threads() == expected


@pytest.mark.parametrize('failure', [OSError, NotImplementedError])
def test_unavailable_affinity_falls_back(monkeypatch, failure):
    def unavailable(pid):
        raise failure()
    monkeypatch.setattr(performance.os, 'sched_getaffinity', unavailable, raising=False)
    monkeypatch.setattr(performance.os, 'cpu_count', lambda: 16)
    assert performance.available_cpu_threads() == 16


def test_opencv_and_torch_use_same_hardware_budget(monkeypatch):
    import cv2
    calls = []
    monkeypatch.setattr(performance, 'available_cpu_threads', lambda: 16)
    monkeypatch.setattr(cv2, 'setNumThreads', lambda n: calls.append(('opencv', n)))
    monkeypatch.setattr(cv2, 'getNumThreads', lambda: 16)
    torch = SimpleNamespace(set_num_threads=lambda n: calls.append(('torch', n)),
                            get_num_threads=lambda: 16)
    runtime = performance.configure_opencv_threads()
    assert performance.configure_torch_threads(torch) == 16
    assert calls == [('opencv', 16), ('torch', 16)]
    assert runtime['available_logical_processors'] == runtime['opencv_threads'] == 16


@pytest.mark.parametrize('cuda', [False, True])
def test_vehicle_loading_restores_thread_budget_after_dependency_import(tmp_path, monkeypatch, cuda):
    import sys
    import cv2
    from traffic_ai import runtime
    from traffic_ai.vehicle_motion import VehicleTracker
    cache = tmp_path / 'models'
    cache.mkdir()
    (cache / 'yolo11s.pt').write_bytes(b'fixture; never passed to a real loader')
    monkeypatch.setattr(runtime, 'configure_cache', lambda: tmp_path)
    monkeypatch.setattr(performance, 'available_cpu_threads', lambda: 16)
    configured = {'opencv': 1, 'torch': 2}  # dependency import may disable OpenCV threading
    monkeypatch.setattr(cv2, 'setNumThreads', lambda n: configured.update(opencv=n))
    monkeypatch.setattr(cv2, 'getNumThreads', lambda: configured['opencv'])
    torch = SimpleNamespace(set_num_threads=lambda n: configured.update(torch=n),
        get_num_threads=lambda: configured['torch'],
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda n: 'fixture GPU'))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=lambda path: object()))
    monkeypatch.setitem(sys.modules, 'traffic_ai.robust_recovery',
                        SimpleNamespace(RobustObservedRecovery=lambda settings: object()))
    tracker = VehicleTracker()
    tracker._load()
    assert configured == {'opencv': 16, 'torch': 16}
    assert tracker.device == ('cuda:0' if cuda else 'cpu')
    assert tracker.manifest['opencv_threads'] == tracker.manifest['cpu_threads'] == 16
    assert tracker.manifest['gpu_name'] == ('fixture GPU' if cuda else None)
