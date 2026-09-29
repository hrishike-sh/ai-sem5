"""Synthetic contract tests only; no permitted source, model or network is used."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import validate_real_demo as demo
from ivp import media


@pytest.mark.parametrize('name', ['../outside', 'UPPERCASE', 'a/b', 'a\\b', '', 'x' * 65, None])
def test_sample_ids_cannot_escape_owned_output_prefix(name):
    with pytest.raises(ValueError):
        demo.sample_name(name)


def test_owned_paths_reject_traversal(tmp_path):
    with pytest.raises(ValueError, match='inside'):
        demo.owned_path(tmp_path, '../outside')


@pytest.mark.parametrize('options', [(-1, 4, 5, 960), (9, 2, 5, 960), (0, 9, 5, 960),
                                    (0, 4, 11, 960), (0, 4, 5, 1920), (0, 4, 5.5, 960),
                                    (float('nan'), 4, 5, 960), (0, float('inf'), 5, 960),
                                    (False, 4, 5, 960)])
def test_transform_is_finite_bounded_and_inside_source(options):
    info = {'duration': 10., 'width': 2160, 'height': 3840}
    with pytest.raises(ValueError):
        demo.validate_transform(info, *options)


def test_transform_preserves_aspect_without_upscaling():
    assert demo.validate_transform({'duration': 10, 'width': 2160, 'height': 3840}, 1, 4, 5, 960) == (540, 960)
    assert demo.validate_transform({'duration': 10, 'width': 321, 'height': 181}, 1, 4, 5, 960) == (320, 180)


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    source = tmp_path / 'synthetic-original.mp4'
    source.write_bytes(b'SYNTHETIC inert video placeholder; not real footage')
    info = {'kind': 'video', 'width': 1280, 'height': 720, 'duration': 10., 'fps': 30.,
            'rotation': 0, 'stream_index': 0}
    def probe(path):
        return {**info, **({'duration': 4., 'width': 960, 'height': 540, 'fps': 5., 'frame_count': 20}
                          if Path(path).name.startswith('demo-') else {})}
    monkeypatch.setattr(media, 'probe_video', probe)
    monkeypatch.setattr(media, 'preview_frame', lambda path, info, stamp: (np.zeros((20, 30, 3), np.uint8), stamp + .02))
    monkeypatch.setattr(media, 'tool_path', lambda name: 'inert-ffmpeg')
    monkeypatch.setattr(media, 'hidden_process_options', lambda: {})
    return project, source


def fake_encoder(monkeypatch, mutate=None):
    calls = []
    def encode(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[-1]).write_bytes(b'SYNTHETIC inert derived clip')
        if mutate:
            mutate()
        return SimpleNamespace(returncode=0, stderr='Synthetic encoder test double; no media processing.')
    monkeypatch.setattr(demo.subprocess, 'run', encode)
    return calls


def test_preparation_records_reproduction_and_source_integrity(synthetic_source, monkeypatch):
    root, source = synthetic_source
    original = source.read_bytes()
    calls = fake_encoder(monkeypatch)
    result = demo.prepare_clip(source, root, 'synthetic-test', offset=2, duration=4, fps=5, max_side=960)
    assert source.read_bytes() == original
    assert result['source']['sha256'] == demo.sha256(source)
    assert result['transform']['first_decoded_source_frame_at_or_after_offset'] == 2.02
    assert result['transform']['source_offset_seconds'] == 2
    assert 'Approximate original-source time' in result['transform']['timestamp_mapping']
    assert result['clip']['info']['source_revision'] == media.source_revision(result['clip']['path'])
    assert Path(result['clip']['path']) == root / 'uploads/demo-synthetic-test.mp4'
    command, options = calls[0]
    assert '-n' in command and '-y' not in command
    assert command[command.index('-ss') + 1] == '2'
    assert options['timeout'] == 180
    assert not options.get('shell')
    assert demo.read_manifest(root, 'synthetic-test')['clip']['sha256'] == result['clip']['sha256']


def test_preparation_never_overwrites_completed_clip(synthetic_source, monkeypatch):
    root, source = synthetic_source
    calls = fake_encoder(monkeypatch)
    result = demo.prepare_clip(source, root, 'synthetic-test')
    original_clip = Path(result['clip']['path']).read_bytes()
    with pytest.raises(ValueError, match='already exists'):
        demo.prepare_clip(source, root, 'synthetic-test')
    assert len(calls) == 1
    assert Path(result['clip']['path']).read_bytes() == original_clip


def test_changed_original_is_rejected_after_encoder(synthetic_source, monkeypatch):
    root, source = synthetic_source
    fake_encoder(monkeypatch, lambda: source.write_bytes(b'SYNTHETIC mutation during mocked encoding'))
    with pytest.raises(ValueError, match='Original source changed'):
        demo.prepare_clip(source, root, 'synthetic-test')
    assert not (root / 'validation/real-demo/synthetic-test/manifest.json').exists()


def test_changed_derived_clip_is_rejected_before_any_analysis(synthetic_source, monkeypatch):
    root, source = synthetic_source
    fake_encoder(monkeypatch)
    result = demo.prepare_clip(source, root, 'synthetic-test')
    Path(result['clip']['path']).write_bytes(b'SYNTHETIC modified output')
    with pytest.raises(ValueError, match='Derived clip changed'):
        demo.analyze_clip(root, 'synthetic-test')
    assert not (root / 'runs').exists()


def test_source_separated_manifest_cannot_point_at_other_media(synthetic_source, monkeypatch):
    root, source = synthetic_source
    fake_encoder(monkeypatch)
    result = demo.prepare_clip(source, root, 'synthetic-test')
    result['clip']['path'] = str(source)
    path = root / 'validation/real-demo/synthetic-test/manifest.json'
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='bind'):
        demo.read_manifest(root, 'synthetic-test')


def test_inspection_records_actual_timestamps_and_pixel_provenance(synthetic_source):
    root, source = synthetic_source
    original = source.read_bytes()
    result = demo.extract_inspection(source, root, 'synthetic-inspect', [1, 3])
    assert source.read_bytes() == original
    assert [frame['actual_source_seconds'] for frame in result['frames']] == [1.02, 3.02]
    for frame in result['frames']:
        assert frame['image_sha256'] == demo.sha256(root / frame['image'])
    assert result['annotations'] == []
    assert 'not a labeled benchmark' in result['notice']


def test_analysis_requires_explicit_offline_isolated_runtime(synthetic_source, monkeypatch):
    from traffic_ai import runtime
    root, source = synthetic_source
    fake_encoder(monkeypatch)
    demo.prepare_clip(source, root, 'synthetic-test')
    monkeypatch.setattr(runtime, 'runtime_config', lambda: {'offline': False, 'isolate_caches': True})
    with pytest.raises(ValueError, match='explicit offline'):
        demo.analyze_clip(root, 'synthetic-test')
    assert not (root / 'runs').exists()
