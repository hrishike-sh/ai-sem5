import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class MediaError(ValueError):
    pass


def source_revision(path):
    """Local file revision, not a content hash: detects replacement since preview."""
    path = Path(path).resolve()
    stat = path.stat()
    return hashlib.sha256(f'{path}|{stat.st_size}|{stat.st_mtime_ns}'.encode()).hexdigest()


def tool_path(name):
    path = shutil.which(name)
    if not path:
        raise MediaError(f"{name} is missing. Install FFmpeg, add its bin folder to PATH, then restart the app.")
    return path


def save_upload(upload, uploads_dir):
    buffer = upload.getbuffer()
    digest = hashlib.sha256(buffer).hexdigest()
    folder = Path(uploads_dir) / digest[:20]
    folder.mkdir(parents=True, exist_ok=True)
    name = Path(upload.name.replace("\\", "/")).name
    path = folder / name
    if not path.exists():
        with tempfile.TemporaryDirectory(prefix=".upload-", dir=folder) as staging:
            temporary = Path(staging) / "source"
            with temporary.open("wb") as file:
                file.write(buffer)
            if not path.exists():
                temporary.replace(path)
    return path


def hidden_process_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if hasattr(subprocess, "CREATE_NO_WINDOW") else {}


def _display_reflected(matrix):
    try:
        rows = [[int(value) for value in row.split(":", 1)[1].split()]
                for row in matrix.strip().splitlines()]
        if len(rows) != 3 or any(len(row) != 3 for row in rows):
            raise ValueError("Expected a 3 by 3 display matrix.")
    except (ValueError, IndexError, AttributeError) as exc:
        raise MediaError("Cannot read the video's display orientation. Re-export it as an H.264 MP4.") from exc
    return rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0] < 0


def probe_video(path):
    command = [tool_path("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30, **hidden_process_options())
        if result.returncode:
            raise MediaError("Cannot decode this video. Try exporting it as an H.264 MP4. " + result.stderr[-400:])
        info = json.loads(result.stdout)
        stream = next((s for s in info["streams"] if s["codec_type"] == "video" and not s.get("disposition", {}).get("attached_pic")), None)
        if not stream:
            raise MediaError("The selected file does not contain a video stream.")
        rotation = float(stream.get("tags", {}).get("rotate", 0))
        reflected = False
        for side in stream.get("side_data_list", []):
            if "rotation" in side:
                rotation = float(side["rotation"])
            if "displaymatrix" in side:
                reflected = _display_reflected(side["displaymatrix"])
        rate = stream.get("avg_frame_rate", "0/1")
        num, den = map(float, rate.split("/"))
        fps = num / den if den else 0
        width, height = stream["width"], stream["height"]
        if round(rotation / 90) % 2:
            width, height = height, width
        duration = float(stream.get("duration") or info.get("format", {}).get("duration") or 0)
        return {"kind": "video", "width": width, "height": height, "encoded_width": stream["width"],
                "encoded_height": stream["height"], "duration": duration, "fps": fps,
                "rotation": rotation, "reflected": reflected, "stream_index": stream["index"],
                "frame_count": int(stream.get("nb_frames", 0) or 0),
                "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
                "codec": stream.get("codec_name", "unknown"), "size_bytes": Path(path).stat().st_size}
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, OSError) as exc:
        raise MediaError(f"Could not inspect video: {exc}") from exc


def load_image(path):
    try:
        with Image.open(path) as image:
            limit = Image.MAX_IMAGE_PIXELS
            if limit is not None and image.width * image.height > limit:
                raise MediaError("Image exceeds the safe pixel limit. Resize it before uploading.")
            image = ImageOps.exif_transpose(image)
            if image.mode in ("I;16", "I;16L", "I;16B", "I;16N"):
                samples = np.asarray(image, dtype=np.uint32)
                # Round the full unsigned 16-bit range to 8 bits.
                gray = ((samples + 128) // 257).astype(np.uint8)
                transparency = image.info.get("transparency")
                if transparency is not None:
                    gray[samples == transparency] = 255
                image = Image.fromarray(gray)
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                layer = image.convert("RGBA")
                background = Image.new("RGBA", layer.size, "white")
                image = Image.alpha_composite(background, layer)
            rgb = np.asarray(image.convert("RGB")).copy()
            if min(rgb.shape[:2]) < 2:
                raise MediaError("Image dimensions must be at least 2 × 2 pixels.")
            return rgb
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise MediaError("Cannot read this image. Use a valid JPG or PNG; very large images should be resized first.") from exc


def orient_video(rgb, rotation, reflected=False):
    # A reflected display matrix needs a vertical flip before its rotation.
    if reflected:
        rgb = np.flipud(rgb)
    quarter_turns = round(rotation / 90) % 4
    return np.ascontiguousarray(np.rot90(rgb, quarter_turns)) if quarter_turns or reflected else rgb


def video_frames(path, info):
    try:
        with av.open(str(path)) as container:
            stream = container.streams[info["stream_index"]]
            first = None
            last = None
            pending = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise MediaError("Video frames have no presentation timestamps. Re-export the source as MP4.")
                absolute = float(frame.pts * frame.time_base)
                if first is None:
                    first = absolute
                stamp = absolute - first
                if last is not None and stamp <= last:
                    raise MediaError("Video has repeated or out-of-order timestamps. Re-export it before processing.")
                if pending is not None:
                    old_frame, old_stamp, old_absolute = pending
                    yield orient_video(old_frame.to_ndarray(format="rgb24"), info["rotation"], info.get("reflected", False)), old_stamp, stamp - old_stamp, old_absolute
                last = stamp
                pending = (frame, stamp, absolute)
            if pending is not None:
                frame, stamp, absolute = pending
                duration = float(frame.duration * frame.time_base) if frame.duration else 1 / (info["fps"] or 25)
                yield orient_video(frame.to_ndarray(format="rgb24"), info["rotation"], info.get("reflected", False)), stamp, duration, absolute
    except av.FFmpegError as exc:
        raise MediaError(f"Video decoding failed: {exc}. Try converting the source to H.264 MP4.") from exc


def preview_frame(path, info, seconds):
    try:
        with av.open(str(path)) as container:
            stream = container.streams[info["stream_index"]]
            start = stream.start_time or 0
            target = start + int(seconds / stream.time_base)
            if seconds > 0:
                container.seek(target, stream=stream, backward=True)
            last = None
            for frame in container.decode(stream):
                stamp = float((frame.pts - start) * stream.time_base) if frame.pts is not None else 0
                last = (orient_video(frame.to_ndarray(format="rgb24"), info["rotation"], info.get("reflected", False)), max(0, stamp))
                if stamp >= seconds:
                    return last
            if last:
                return last
        raise MediaError("No readable frame at this position. Move the slider earlier.")
    except av.FFmpegError as exc:
        raise MediaError(f"Cannot preview this position: {exc}") from exc


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def write_json(path, payload):
    path = Path(path)
    # Each writer owns its temporary file. Windows readers/virus scanners can
    # briefly hold the destination open, so retry the atomic replacement.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
        for attempt in range(8):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.02 * (attempt + 1))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
