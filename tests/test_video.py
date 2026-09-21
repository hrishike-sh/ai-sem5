import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from types import SimpleNamespace

import av
import numpy as np
import pytest

from ivp.jobs import cancel_job, read_status
from ivp.media import MediaError, hidden_process_options, preview_frame, probe_video, tool_path, video_frames, write_json
from ivp.pipeline import Settings
from ivp.worker import process_video


def ffmpeg(*args):
    subprocess.run([tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True, **hidden_process_options())


def request_run(tmp_path, source, settings=None, masks=None):
    run = tmp_path / ("run_" + str(len(list(tmp_path.glob("run_*")))))
    run.mkdir()
    write_json(run / "request.json", {"source": str(source), "info": probe_video(source),
                "settings": (settings or Settings()).to_dict(), "selected_masks": masks or []})
    return run


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=1.5", "-f", "lavfi", "-i", "sine=frequency=500:duration=1.5", "-c:v", "libx264", "-c:a", "aac", "-shortest", path)
    return path


def stamps(path):
    with av.open(str(path)) as container:
        return [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]


def ffmpeg_display_frames(path, info):
    result = subprocess.run([tool_path("ffmpeg"), "-hide_banner", "-loglevel", "error", "-i", str(path),
                             "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                            check=True, capture_output=True, **hidden_process_options())
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, info["height"], info["width"], 3)


def test_audio_timing_masks_and_output(source, tmp_path):
    run = request_run(tmp_path, source, Settings(gamma=1.2), ["Edges"])
    process_video(run)
    assert read_status(run)["state"] == "complete"
    out = run / "processed.mp4"
    info = probe_video(out)
    assert info["has_audio"]
    assert info["codec"] == "h264"
    assert info["duration"] == pytest.approx(probe_video(source)["duration"], abs=0.09)
    assert stamps(out) == pytest.approx(stamps(source), abs=2 / 90000)
    with av.open(str(out)) as container:
        audio = list(container.decode(audio=0))
        assert sum(frame.samples for frame in audio) > 40000
    with zipfile.ZipFile(run / "masks.zip") as archive:
        assert len(archive.namelist()) == 3
        manifest = json.loads(archive.read('manifest.json'))
        assert manifest['status'] == 'review_only' and manifest['eligible_for_rules'] is False
        assert manifest['source_binding']['source_revision']
    rows = list(csv.DictReader((run / "measurements.csv").open()))
    assert len(rows) == 18


def test_source_changed_after_preview_is_rejected(source, tmp_path):
    from ivp.media import source_revision
    run = request_run(tmp_path, source)
    request = json.loads((run/'request.json').read_text())
    request['info']['source_revision'] = source_revision(source)
    write_json(run/'request.json', request)
    source.touch()
    with pytest.raises(ValueError, match='Source video changed'):
        process_video(run)
    assert read_status(run)['state'] == 'failed'
    assert not (run/'processed.mp4').exists()


def test_cancel_during_report_packaging(source, tmp_path):
    run = request_run(tmp_path, source)
    def finish():
        (run/'motion-evidence.zip').write_bytes(b'pending evidence')
        (run/'cancel.request').touch()
        return {'ready': True}
    process_video(run, final_hook=finish)
    assert read_status(run)['state'] == 'cancelled'
    assert not any((run/name).exists() for name in ('processed.mp4','report.json','motion-evidence.zip'))


def test_ai_frame_hook_preserves_audio_and_presentation_times(source, tmp_path):
    run = request_run(tmp_path, source)
    calls = []

    def hook(result, original, index, distance):
        calls.append((index, result.timestamp))
        assert np.array_equal(result.color_rgb, original)
        output = result.color_rgb.copy()
        output[10:30, 10:30] = [0, 255, 0]
        return output

    process_video(run, frame_hook=hook, final_hook=lambda: {"frames": len(calls)})
    out = run / "processed.mp4"
    assert probe_video(out)["has_audio"]
    assert stamps(out) == pytest.approx(stamps(source), abs=2 / 90000)
    frame, _ = preview_frame(out, probe_video(out), 0)
    assert frame[20, 20, 1] > 240 and frame[20, 20, 0] < 20
    assert json.loads((run / "report.json").read_text())["ai"]["frames"] == 18


def test_variable_frame_rate_is_not_flattened(tmp_path):
    source = tmp_path / "variable.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25:duration=1", "-vf", "select='eq(mod(n,3),0)+eq(n,1)'", "-fps_mode", "vfr", "-c:v", "libx264", source)
    run = request_run(tmp_path, source)
    process_video(run)
    input_times, output_times = stamps(source), stamps(run / "processed.mp4")
    assert len(set(round(x, 3) for x in np.diff(input_times))) > 1
    assert output_times == pytest.approx(input_times, abs=2 / 90000)
    decoded = list(video_frames(source, probe_video(source)))
    actual_endpoint = decoded[-1][1] + decoded[-1][2]
    assert probe_video(run / "processed.mp4")["duration"] == pytest.approx(actual_endpoint, abs=0.001)
    assert not probe_video(run / "processed.mp4")["has_audio"]


def test_preview_and_source_unchanged(source, tmp_path):
    import hashlib
    original = hashlib.sha256(source.read_bytes()).hexdigest()
    info = probe_video(source)
    rgb, timestamp = preview_frame(source, info, 0.6)
    assert rgb.shape == (120, 160, 3)
    assert 0.6 <= timestamp < 0.7
    run = request_run(tmp_path, source)
    process_video(run)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original


def test_cancel_before_start(source, tmp_path):
    run = request_run(tmp_path, source)
    (run / "cancel.request").touch()
    process_video(run)
    assert read_status(run)["state"] == "cancelled"
    assert not (run / "processed.mp4").exists()


def test_broken_video_and_missing_ffmpeg(tmp_path, source, monkeypatch):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"invalid")
    with pytest.raises(MediaError):
        probe_video(bad)
    run = request_run(tmp_path, source)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(MediaError, match="ffmpeg is missing"):
        process_video(run)
    assert read_status(run)["state"] == "failed"


def test_rotation_metadata_applied_once(source, tmp_path):
    rotated = tmp_path / "rotated.mp4"
    ffmpeg("-display_rotation", "90", "-i", source, "-c", "copy", rotated)
    info = probe_video(rotated)
    assert info["width"] == 120
    assert info["height"] == 160
    preview, _ = preview_frame(rotated, info, 0)
    run = request_run(tmp_path, rotated)
    process_video(run)
    result_info = probe_video(run / "processed.mp4")
    assert result_info["rotation"] == 0
    output, _ = preview_frame(run / "processed.mp4", result_info, 0)
    assert output.shape == preview.shape
    assert np.abs(output.astype(float) - preview.astype(float)).mean() < 8


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("flip", [None, "h", "v"], ids=["rotation-only", "horizontal", "vertical"])
def test_display_orientation_matches_ffmpeg(source, tmp_path, rotation, flip):
    oriented = tmp_path / "oriented.mp4"
    flip_args = ["-display_" + flip + "flip"] if flip else []
    ffmpeg("-display_rotation", rotation, *flip_args, "-i", source, "-c", "copy", oriented)
    info = probe_video(oriented)
    expected = ffmpeg_display_frames(oriented, info)
    preview, timestamp = preview_frame(oriented, info, 0)
    assert timestamp == 0
    assert np.array_equal(preview, expected[0])
    frames = list(video_frames(oriented, info))
    assert np.array_equal(np.stack([rgb for rgb, *_ in frames]), expected)
    assert [stamp for _, stamp, *_ in frames] == pytest.approx(stamps(oriented), abs=2 / 90000)
    if flip is None:
        legacy_info = {key: value for key, value in info.items() if key != "reflected"}
        assert np.array_equal(preview_frame(oriented, legacy_info, 0)[0], expected[0])
        assert np.array_equal(next(video_frames(oriented, legacy_info))[0], expected[0])


@pytest.mark.parametrize("rotation,flip", [(0, "h"), (90, "h"), (0, "v"), (90, "v")])
def test_reflected_export_and_ai_frames_keep_display_orientation(source, tmp_path, rotation, flip):
    oriented = tmp_path / "reflected.mp4"
    ffmpeg("-display_rotation", rotation, "-display_" + flip + "flip", "-i", source, "-c", "copy", oriented)
    original = oriented.read_bytes()
    info = probe_video(oriented)
    expected = ffmpeg_display_frames(oriented, info)
    run = request_run(tmp_path, oriented)

    def hook(result, original_rgb, index, distance):
        assert np.array_equal(original_rgb, expected[index])
        assert np.array_equal(result.color_rgb, expected[index])
        return result.color_rgb

    process_video(run, frame_hook=hook)
    assert read_status(run)["state"] == "complete"
    output = run / "processed.mp4"
    output_info = probe_video(output)
    assert output_info["rotation"] == 0
    assert output_info["reflected"] is False
    assert output_info["has_audio"]
    output_frames = np.stack([rgb for rgb, *_ in video_frames(output, output_info)])
    assert output_frames.shape == expected.shape
    assert np.abs(output_frames.astype(float) - expected.astype(float)).mean() < 8
    assert stamps(output) == pytest.approx(stamps(oriented), abs=2 / 90000)
    assert oriented.read_bytes() == original


def test_odd_resize_padding(source, tmp_path):
    run = request_run(tmp_path, source, Settings(max_width=101))
    process_video(run)
    assert probe_video(run / "processed.mp4")["width"] == 102
    coords = json.loads((run / "coordinates.json").read_text())
    assert coords["processed_size"] == [101, 76]
    assert coords["encoded_size"] == [102, 76]


def test_cancel_during_processing(source, tmp_path, monkeypatch):
    import ivp.worker as worker
    run = request_run(tmp_path, source)
    original = worker.process_frame
    calls = []

    def process_then_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(1)
        if len(calls) == 3:
            (run / "cancel.request").touch()
        return result

    monkeypatch.setattr(worker, "process_frame", process_then_cancel)
    process_video(run)
    assert len(calls) == 3
    assert read_status(run)["state"] == "cancelled"
    assert not (run / "processed.mp4").exists()


@pytest.mark.parametrize("stubborn", [False, True])
@pytest.mark.parametrize("cancelled", [False, True], ids=["monitor-error", "cancel"])
def test_mux_shutdown_preserves_cancellation_or_monitoring_error(source, tmp_path, monkeypatch, stubborn, cancelled):
    import ivp.worker as worker

    run = request_run(tmp_path, source)
    original = source.read_bytes()
    started = False
    original_check = worker.check_cancel

    class Muxer:
        returncode = None
        terminated = False
        killed = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

        def wait(self, timeout):
            if stubborn and not self.killed:
                raise subprocess.TimeoutExpired("test-muxer", timeout)
            self.returncode = -9 if self.killed else -15
            return self.returncode

    process = Muxer()

    def start_muxer(*args, **kwargs):
        nonlocal started
        started = True
        (run / "processed.mp4").write_bytes(b"Incomplete mux output")
        if cancelled:
            cancel_job(run)
        return process

    def check_cancel(path):
        if started and not cancelled:
            raise RuntimeError("Mux monitoring interrupted")
        original_check(path)

    monkeypatch.setattr(worker.subprocess, "Popen", start_muxer)
    monkeypatch.setattr(worker, "check_cancel", check_cancel)
    if cancelled:
        process_video(run)
        assert read_status(run)["state"] == "cancelled"
        for name in ("processed.mp4", "video.partial.mp4", "report.json"):
            assert not (run / name).exists()
    else:
        with pytest.raises(RuntimeError, match="Mux monitoring interrupted"):
            process_video(run)
        assert read_status(run)["state"] == "failed"
        assert read_status(run)["message"] == "Mux monitoring interrupted"
    assert process.terminated
    assert process.killed == stubborn
    assert process.poll() is not None
    assert source.read_bytes() == original


@pytest.mark.skipif(os.name == "nt", reason="This case exercises a real child ignoring POSIX SIGTERM.")
def test_cancelled_muxer_that_ignores_termination_is_reaped(source, tmp_path, monkeypatch):
    import ivp.worker as worker

    run = request_run(tmp_path, source)
    original = source.read_bytes()
    original_popen = subprocess.Popen
    children = []
    waits = []

    def start_muxer(*args, **kwargs):
        ready = run / "muxer.ready"
        code = ("import signal,sys; from pathlib import Path; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                "Path(sys.argv[1]).write_text('ready'); sys.stdin.read()")
        child = original_popen([sys.executable, "-c", code, str(ready)], stdin=subprocess.PIPE, **kwargs)
        children.append(child)
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert child.poll() is None and time.monotonic() < deadline
            time.sleep(.01)
        (run / "processed.mp4").write_bytes(b"Incomplete mux output")
        cancel_job(run)

        def wait(timeout):
            waits.append(timeout)
            # Exercise a real ignored signal without spending ten seconds per test.
            return child.wait(timeout=.1 if len(waits) == 1 else timeout)

        return SimpleNamespace(poll=child.poll, terminate=child.terminate, kill=child.kill, wait=wait)

    monkeypatch.setattr(worker.subprocess, "Popen", start_muxer)
    try:
        process_video(run)
        assert read_status(run)["state"] == "cancelled"
        assert len(children) == 1
        assert children[0].poll() is not None
        assert len(waits) == 2
        for name in ("processed.mp4", "video.partial.mp4", "report.json"):
            assert not (run / name).exists()
        assert source.read_bytes() == original
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()


@pytest.mark.parametrize("phase", ["hook", "report"])
def test_cancel_during_finalization_does_not_publish_video(source, tmp_path, monkeypatch, phase):
    import ivp.worker as worker

    run = request_run(tmp_path, source)
    original = source.read_bytes()
    original_write = worker.write_json

    def finish():
        if phase == "hook":
            cancel_job(run)
        return {"test_double": True}

    def write_then_cancel(path, payload):
        original_write(path, payload)
        if phase == "report" and path == run / "report.json":
            cancel_job(run)

    monkeypatch.setattr(worker, "write_json", write_then_cancel)
    process_video(run, final_hook=finish)
    assert read_status(run)["state"] == "cancelled"
    for name in ("processed.mp4", "video.partial.mp4", "report.json"):
        assert not (run / name).exists()
    assert source.read_bytes() == original


def first_audio_time(path):
    with av.open(str(path)) as container:
        frame = next(container.decode(audio=0))
        return float(frame.pts * frame.time_base)


def test_delayed_audio_keeps_offset(tmp_path):
    source = tmp_path / "delayed_audio.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2", "-itsoffset", "0.4",
           "-f", "lavfi", "-i", "sine=frequency=600:duration=1.3", "-c:v", "libx264", "-c:a", "aac", source)
    run = request_run(tmp_path, source)
    process_video(run)
    assert first_audio_time(run / "processed.mp4") == pytest.approx(first_audio_time(source), abs=0.05)


def test_nonzero_video_start_normalised(tmp_path):
    source = tmp_path / "nonzero.mp4"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=1.5", "-f", "lavfi", "-i",
           "sine=frequency=600:duration=1.5", "-c:v", "libx264", "-c:a", "aac", "-output_ts_offset", "2", source)
    source_times = stamps(source)
    assert source_times[0] >= 1.9
    run = request_run(tmp_path, source)
    process_video(run)
    assert stamps(run / "processed.mp4") == pytest.approx([stamp - source_times[0] for stamp in source_times], abs=2 / 90000)
    assert first_audio_time(run / "processed.mp4") == pytest.approx(0, abs=0.05)


def test_encoder_failure_exposes_failed_status(source, tmp_path, monkeypatch):
    import ivp.worker as worker
    run = request_run(tmp_path, source)

    def fail_encoder(*args, **kwargs):
        raise RuntimeError("Deliberate encoder failure")

    monkeypatch.setattr(worker.av, "open", fail_encoder)
    with pytest.raises(RuntimeError, match="encoder failure"):
        process_video(run)
    assert read_status(run)["state"] == "failed"
