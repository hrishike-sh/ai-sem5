import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from ivp.jobs import cancel_job, read_status
from ivp.media import png_bytes, write_json
from ivp.motion_worker import run_motion
from tests.test_video import request_run, source as source


@pytest.mark.parametrize("phase", ["motion", "incident", "coordinates", None],
                         ids=["motion-file", "incident-thumbnail", "metadata", "complete"])
def test_motion_evidence_packaging_stops_after_the_current_file(source, tmp_path, monkeypatch, phase):
    run = request_run(tmp_path, source)
    request = json.loads((run / "request.json").read_text())
    request["motion"] = {"tracking_mode": "ivp"}
    write_json(run / "request.json", request)
    original_source = source.read_bytes()
    original_request = (run / "request.json").read_bytes()
    # Stand-ins for thumbnails already emitted by the incident evidence writer.
    for number in range(2):
        (run / f"incident-fixture-{number}.png").write_bytes(png_bytes(np.full((8, 8, 3), 80, np.uint8)))
    original_write = zipfile.ZipFile.write
    written = []
    after_cancel = []
    cancelled_on = []

    def write_then_cancel(archive, filename, *args, **kwargs):
        if Path(archive.filename) != run / "motion-evidence.zip":
            return original_write(archive, filename, *args, **kwargs)
        name = Path(filename).name
        if (run / "cancel.request").exists():
            after_cancel.append(name)
        result = original_write(archive, filename, *args, **kwargs)
        written.append(name)
        selected = ((phase == "motion" and name.startswith("motion-")) or
                    (phase == "incident" and name.startswith("incident-")) or
                    (phase == "coordinates" and name == "coordinates.json"))
        if selected and not cancelled_on:
            cancelled_on.append(name)
            cancel_job(run)
        return result

    monkeypatch.setattr(zipfile.ZipFile, "write", write_then_cancel)
    run_motion(run)
    assert not after_cancel
    if phase is None:
        assert read_status(run)["state"] == "complete"
        assert {"coordinates.json", "request.json", "incident-fixture-0.png", "incident-fixture-1.png"} <= set(written)
        assert (run / "processed.mp4").is_file()
        with zipfile.ZipFile(run / "motion-evidence.zip") as archive:
            assert archive.testzip() is None
            assert archive.namelist() == written
    else:
        assert cancelled_on
        assert written[-1] == cancelled_on[0]
        assert read_status(run)["state"] == "cancelled"
        for name in ("processed.mp4", "video.partial.mp4", "report.json", "motion-evidence.zip"):
            assert not (run / name).exists()
    assert source.read_bytes() == original_source
    assert (run / "request.json").read_bytes() == original_request
