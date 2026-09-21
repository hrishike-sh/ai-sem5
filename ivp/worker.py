import csv
import json
import subprocess
import sys
import time
import zipfile
from fractions import Fraction
from pathlib import Path

import av
import cv2

from .jobs import job_lock, now
from .media import hidden_process_options, png_bytes, tool_path, video_frames, write_json, source_revision
from .pipeline import Settings, process_frame, scene_distance, scene_signature
from .marking_policy import mask_export_manifest, review_only_policy


class Cancelled(Exception):
    pass


def check_cancel(run):
    if (run / "cancel.request").exists():
        raise Cancelled()


def check_source(source, expected_revision=None):
    try:
        revision = source_revision(source)
    except OSError as exc:
        raise ValueError('Source video is no longer available. Reopen it and recalibrate before retrying.') from exc
    if expected_revision and revision != expected_revision:
        raise ValueError('Source video changed after preview. Reopen it and recalibrate before retrying.')
    return revision


def update(run, state, **kwargs):
    write_json(run / "status.json", {"state": state, "updated_at": now(), **kwargs})


def process_video(run, frame_hook=None, final_hook=None, final_key='ai'):
    request = json.loads((run / "request.json").read_text(encoding="utf-8"))
    info = request["info"]
    settings = Settings(**request["settings"])
    source = Path(request["source"])
    masks = request["selected_masks"]
    encoder = None
    first_absolute = 0
    frame_count = 0
    previous = None
    total_changes = 0
    last_report = 0
    next_sample = 0
    total_duration = 0
    begin = time.monotonic()
    sums = {"brightness": 0.0, "contrast": 0.0, "laplacian_variance": 0.0}
    output_size = None
    packet_durations = {}
    try:
        tool_path("ffmpeg")
        check_cancel(run)
        # Legacy requests lack a stored revision; bind this run once at startup.
        expected_revision = check_source(source, info.get('source_revision'))
        update(run, "running", progress=0, message="Processing frames.")
        with (run / "measurements.csv").open("w", newline="", encoding="utf-8") as metrics_file, \
             (run / "scene_changes.csv").open("w", newline="", encoding="utf-8") as scene_file, \
             zipfile.ZipFile(run / "masks.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            writer = csv.writer(metrics_file)
            writer.writerow(["frame", "timestamp_seconds", "duration_seconds", "brightness_before", "brightness_after", "contrast_before", "contrast_after", "blur_before", "blur_after", "scene_distance"])
            scene_writer = csv.writer(scene_file)
            scene_writer.writerow(["frame", "timestamp_seconds", "distance", "interpretation"])
            for rgb, stamp, frame_duration, absolute in video_frames(source, info):
                check_cancel(run)
                sample = bool(masks) and stamp >= next_sample
                result = process_frame(rgb, settings, stamp, include_features=sample, include_stages=False)
                if encoder is None:
                    first_absolute = absolute
                    encoder = av.open(str(run / "video.partial.mp4"), "w")
                    stream = encoder.add_stream("libx264", rate=Fraction(str(info["fps"] or 25)).limit_denominator(100000))
                    height, width = result.color_rgb.shape[:2]
                    stream.width, stream.height = width + width % 2, height + height % 2
                    stream.pix_fmt = "yuv420p"
                    stream.time_base = Fraction(1, 90000)
                    stream.codec_context.time_base = Fraction(1, 90000)
                    stream.codec_context.max_b_frames = 0
                    stream.options = {"crf": "18", "preset": "fast"}
                    output_size = [stream.width, stream.height]
                    write_json(run / "coordinates.json", {"source_size": result.source_size,
                               "processed_size": [width, height], "encoded_size": output_size,
                               "transform_to_source_pixel_centres": result.transform_to_source.tolist(),
                               "source_orientation": "display-oriented RGB; EXIF/display rotation already applied",
                               "padding": "Right/bottom edge padding to even dimensions for H.264, if necessary."})
                    archive.writestr('manifest.json', json.dumps(mask_export_manifest(masks, {
                        'source_name': source.name,
                        'source_revision': expected_revision,
                        'timestamp_scope': 'Each PNG folder records source frame index and relative presentation seconds.',
                        'transform_to_source_pixel_centres': result.transform_to_source.tolist()}), indent=2))
                signature = scene_signature(rgb)
                distance = scene_distance(previous, signature)
                if distance >= settings.scene_threshold:
                    total_changes += 1
                    scene_writer.writerow([frame_count, stamp, distance, "Possible scene or appearance change"])
                previous = signature
                array = frame_hook(result, rgb, frame_count, distance) if frame_hook else result.color_rgb
                if array.shape[:2] != (height, width):
                    raise ValueError("Frame dimensions changed within the video. Re-export it at a fixed resolution.")
                if array.shape[1] != stream.width or array.shape[0] != stream.height:
                    array = cv2.copyMakeBorder(array, 0, stream.height - array.shape[0], 0, stream.width - array.shape[1], cv2.BORDER_REPLICATE)
                frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                frame.pts = round(stamp * 90000)
                frame.time_base = Fraction(1, 90000)
                frame.duration = max(1, round(frame_duration * 90000))
                packet_durations[frame.pts] = frame.duration
                for packet in stream.encode(frame):
                    packet.duration = packet_durations.pop(packet.pts)
                    encoder.mux(packet)
                before, after = result.measurements["original"], result.measurements["processed"]
                writer.writerow([frame_count, stamp, frame_duration, before["brightness"], after["brightness"], before["contrast"], after["contrast"], before["laplacian_variance"], after["laplacian_variance"], distance])
                for key in sums:
                    sums[key] += after[key]
                if sample:
                    for name in masks:
                        if name in result.features:
                            archive.writestr(f"{frame_count:08d}_{stamp:.3f}s/{name.replace(' ', '_')}.png", png_bytes(result.features[name]))
                    next_sample = stamp + 1.0
                total_duration = stamp + frame_duration
                frame_count += 1
                if time.monotonic() - last_report >= 0.5:
                    update(run, "running", progress=min(stamp / max(info["duration"], 0.001) * 0.95, 0.95),
                           frames=frame_count, timestamp=stamp, fps=frame_count / max(time.monotonic() - begin, 0.001), message="Processing frames.")
                    last_report = time.monotonic()
            if encoder is None:
                raise ValueError("The video contains no readable frames.")
            for packet in stream.encode():
                packet.duration = packet_durations.pop(packet.pts)
                encoder.mux(packet)
            encoder.close()
            encoder = None
        check_cancel(run)
        update(run, "encoding", progress=0.97, frames=frame_count, message="Finalising MP4 and preserving audio.")
        # Audio opens the source path again, independently of the decoder handle.
        check_cancel(run)
        check_source(source, expected_revision)
        command = [tool_path("ffmpeg"), "-hide_banner", "-y", "-copyts", "-i", str(run / "video.partial.mp4"),
                   "-itsoffset", str(-first_absolute), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                   "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-af", "atrim=start=0", "-t", str(total_duration),
                   "-map_metadata", "-1", "-movflags", "+faststart", str(run / "processed.mp4")]
        with (run / "ffmpeg.log").open("wb") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, **hidden_process_options())
            try:
                while process.poll() is None:
                    check_cancel(run)
                    time.sleep(0.1)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                raise
        if process.returncode:
            detail = (run / "ffmpeg.log").read_text(encoding="utf-8", errors="replace")[-700:]
            raise RuntimeError("FFmpeg could not finalise the MP4. " + detail)
        check_cancel(run)
        check_source(source, expected_revision)
        report = {"schema_version": 1, "kind": "video", "source": source.name, "source_info": info,
                  "settings": settings.to_dict(), "frames": frame_count, "duration_seconds": total_duration,
                  "elapsed_seconds": time.monotonic() - begin, "scene_change_candidates": total_changes,
                  "mean_processed_quality": {k: v / frame_count for k, v in sums.items()},
                  "mask_export": {"names": masks, "sampling": "First frame at or after each next 1-second interval",
                                  "marking_policy": review_only_policy()},
                  "timestamp_policy": "Original frame presentation intervals preserved at 1/90000-second resolution; start normalised to zero.",
                  "audio_policy": "First audio track retained as AAC with original relative timestamps; negative pre-roll trimmed.",
                  "limitations": ["Quality and scene changes are heuristics.", "Road and colour masks do not classify traffic rules."]}
        if final_hook:
            report[final_key] = final_hook()
        check_cancel(run)
        check_source(source, expected_revision)
        write_json(run / "report.json", report)
        check_cancel(run)
        check_source(source, expected_revision)
        (run / "video.partial.mp4").unlink(missing_ok=True)
        update(run, "complete", progress=1, frames=frame_count, message="Video is ready.", duration=total_duration)
    except Cancelled:
        if encoder is not None:
            encoder.close()
            encoder = None
        for name in ("video.partial.mp4", "processed.mp4", "report.json", "motion-evidence.zip"):
            (run / name).unlink(missing_ok=True)
        update(run, "cancelled", progress=0, frames=frame_count, message="Cancelled. Original upload is unchanged.")
    except Exception as exc:
        update(run, "failed", progress=0, frames=frame_count, message=str(exc))
        raise
    finally:
        if encoder is not None:
            encoder.close()


def main():
    run = Path(sys.argv[1]).resolve()
    for attempt in range(20):
        try:
            with job_lock():
                process_video(run)
            return
        except RuntimeError as exc:
            if "Another video job" not in str(exc):
                raise
            time.sleep(0.2)
    update(run, "failed", progress=0, message="Another video worker is already active.")


if __name__ == "__main__":
    main()
