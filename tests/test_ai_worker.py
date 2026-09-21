import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from ivp.jobs import cancel_job, read_status
from ivp.media import load_image, write_json
from ivp.pipeline import Settings
from traffic_ai.contracts import Detection, FrameAnalysis
from traffic_ai.worker import process_run


class FakeAnalyzer:
    def __init__(self, status, cancel):
        cancel()
        status("Test model")

    def analyze(self, result, original, video, scene_change):
        assert not video
        assert original.shape == (80, 120, 3)
        assert original[0, 0].tolist() == [80, 80, 80]
        output = result.color_rgb.copy()
        output[5:10, 5:10] = [0, 255, 0]
        return output, FrameAnalysis(0, [Detection("car", .9, (15, 15, 35, 35))], []), []

    def summary(self):
        return {"events_by_rule": {}, "assessment_counts": {}, "test_double": True}


@pytest.mark.parametrize("source_mode", ["rgb8", "gray16"])
def test_image_worker_writes_reopenable_media_and_mapping(tmp_path, monkeypatch, source_mode):
    from traffic_ai import worker

    monkeypatch.setattr(worker, "Analyzer", FakeAnalyzer)
    source = tmp_path / "original.png"
    pixels = np.full((80, 120), 80 * 257, np.uint16) if source_mode == "gray16" else np.full((80, 120, 3), 80, np.uint8)
    Image.fromarray(pixels).save(source)
    before = source.read_bytes()
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "request.json", {"source": str(source), "info": {"kind": "image"}, "settings": Settings(max_width=60).to_dict()})
    process_run(run)
    assert json.loads((run / "status.json").read_text())["state"] == "complete"
    output = load_image(run / "processed.png")
    assert output.shape == (40, 60, 3)
    assert output[7, 7].tolist() == [0, 255, 0]
    preview = load_image(run / "detections.png")
    assert preview[25, 15].tolist() == [0, 200, 255]
    assert preview[7, 7].tolist() == [80, 80, 80]
    assert output[25, 15].tolist() == [80, 80, 80]
    coordinates = json.loads((run / "coordinates.json").read_text())
    assert coordinates["transform_to_source_pixel_centres"] == [[2., 0., .5], [0., 2., .5], [0., 0., 1.]]
    assert source.read_bytes() == before
    assert len((run / "detections.jsonl").read_text().splitlines()) == 1


def test_cancel_before_model_loading(tmp_path, monkeypatch):
    from traffic_ai import worker

    monkeypatch.setattr(worker, "Analyzer", lambda **kwargs: (_ for _ in ()).throw(AssertionError("Should not load")))
    write_json(tmp_path / "request.json", {})
    (tmp_path / "cancel.request").touch()
    process_run(tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "cancelled"


@pytest.fixture
def image_request(tmp_path):
    source = tmp_path / "original.png"
    Image.fromarray(np.full((80, 120, 3), 80, np.uint8)).save(source)
    run = tmp_path / "run"
    run.mkdir()
    write_json(run / "request.json", {"source": str(source), "info": {"kind": "image"},
               "settings": Settings().to_dict()})
    return source, run


@pytest.mark.parametrize("phase", ["summary", "report"])
def test_cancel_during_image_finalization_does_not_publish_result(image_request, monkeypatch, phase):
    from traffic_ai import worker

    source, run = image_request
    original = source.read_bytes()
    original_write = worker.write_json

    class FinishingAnalyzer(FakeAnalyzer):
        def summary(self):
            if phase == "summary":
                cancel_job(run)
            return super().summary()

    def write_then_cancel(path, payload):
        original_write(path, payload)
        if phase == "report" and path == run / "report.json":
            cancel_job(run)

    monkeypatch.setattr(worker, "Analyzer", FinishingAnalyzer)
    monkeypatch.setattr(worker, "write_json", write_then_cancel)
    process_run(run)
    assert read_status(run)["state"] == "cancelled"
    assert not (run / "processed.png").exists()
    assert not (run / "report.json").exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("folder,archive_name", [("evidence", "evidence.zip"), ("scene_features", "scene_features.zip")])
@pytest.mark.parametrize("cancel", [True, False], ids=["cancel", "complete"])
def test_archive_packaging_honors_cancellation(image_request, monkeypatch, folder, archive_name, cancel):
    from traffic_ai import worker

    source, run = image_request
    original = source.read_bytes()
    contents = run / folder
    if folder == "scene_features":
        contents /= "0001"
    contents.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (5, 5), "gray").save(contents / f"{index}.png")
    original_write = zipfile.ZipFile.write
    written = []

    def write_then_cancel(archive, filename, *args, **kwargs):
        result = original_write(archive, filename, *args, **kwargs)
        if archive.filename == str(run / archive_name):
            written.append(filename)
            if cancel:
                cancel_job(run)
        return result

    monkeypatch.setattr(worker, "Analyzer", FakeAnalyzer)
    monkeypatch.setattr(zipfile.ZipFile, "write", write_then_cancel)
    process_run(run)
    expected_count = 1 if cancel else 3
    assert len(written) == expected_count
    with zipfile.ZipFile(run / archive_name) as archive:
        assert len(archive.namelist()) == expected_count
    assert read_status(run)["state"] == ("cancelled" if cancel else "complete")
    assert (run / "processed.png").exists() == (not cancel)
    assert (run / "report.json").exists() == (not cancel)
    assert source.read_bytes() == original
