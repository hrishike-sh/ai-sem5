"""Synthetic pixels and guarded Analyzer double; no model or network calls."""
import json

import numpy as np
import pytest

from ivp.media import png_bytes, source_revision, write_json
from tests.test_confirmation_export import worker as worker, request, install_analyzer


@pytest.mark.parametrize('phase', ['before', 'load', 'analyze', 'summary', 'report'])
@pytest.mark.parametrize('action', ['replace', 'remove'])
def test_changed_image_source_cannot_publish_completed_analysis(worker, monkeypatch, tmp_path, phase, action):
    source, run = request(tmp_path, 'image')
    saved = json.loads((run / 'request.json').read_text())
    saved['info']['source_revision'] = source_revision(source)
    write_json(run / 'request.json', saved)
    install_analyzer(worker, monkeypatch, [(0., [], 50)])
    original_analyzer = worker.Analyzer
    original_load = worker.load_image
    original_write = worker.write_json
    calls = []

    def change():
        if action == 'remove':
            source.unlink()
        else:
            source.write_bytes(png_bytes(np.full((80, 120, 3), 255, np.uint8)))

    class ObservedAnalyzer(original_analyzer):
        def __init__(self, *args, **kwargs):
            calls.append('construct')
            super().__init__(*args, **kwargs)

        def analyze(self, *args, **kwargs):
            calls.append('analyze')
            result = super().analyze(*args, **kwargs)
            if phase == 'analyze':
                change()
            return result

        def summary(self):
            result = super().summary()
            if phase == 'summary':
                change()
            return result

    def load(path):
        rgb = original_load(path)
        if phase == 'load':
            change()
        return rgb

    def write(path, value):
        original_write(path, value)
        if phase == 'report' and path.name == 'report.json':
            change()

    monkeypatch.setattr(worker, 'Analyzer', ObservedAnalyzer)
    monkeypatch.setattr(worker, 'load_image', load)
    monkeypatch.setattr(worker, 'write_json', write)
    if phase == 'before':
        change()
    with pytest.raises(ValueError, match='Source image'):
        worker.process_run(run)
    assert json.loads((run / 'status.json').read_text())['state'] == 'failed'
    assert not (run / 'report.json').exists()
    assert not (run / 'processed.png').exists()
    if phase == 'before':
        assert not calls
    if phase == 'load':
        assert 'analyze' not in calls


def test_legacy_image_request_binds_revision_at_start(worker, monkeypatch, tmp_path):
    source, run = request(tmp_path, 'image')
    install_analyzer(worker, monkeypatch, [(0., [], 50)])
    original = worker.Analyzer

    class ChangedDuringLoad(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            source.write_bytes(png_bytes(np.full((80, 120, 3), 255, np.uint8)))

    monkeypatch.setattr(worker, 'Analyzer', ChangedDuringLoad)
    with pytest.raises(ValueError, match='Source image changed'):
        worker.process_run(run)
    assert json.loads((run / 'status.json').read_text())['state'] == 'failed'
    assert not (run / 'report.json').exists()


def test_image_cancellation_after_analysis_withholds_final_publication(worker, monkeypatch, tmp_path):
    source, run = request(tmp_path, 'image')
    original_bytes = source.read_bytes()
    install_analyzer(worker, monkeypatch, [(0., [], 50)])
    original = worker.Analyzer

    class CancelAfterAnalysis(original):
        def analyze(self, *args, **kwargs):
            result = super().analyze(*args, **kwargs)
            (run / 'cancel.request').touch()
            return result

    monkeypatch.setattr(worker, 'Analyzer', CancelAfterAnalysis)
    worker.process_run(run)
    assert json.loads((run / 'status.json').read_text())['state'] == 'cancelled'
    assert not (run / 'report.json').exists()
    assert not (run / 'processed.png').exists()
    assert source.read_bytes() == original_bytes
