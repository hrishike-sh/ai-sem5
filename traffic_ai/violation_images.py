"""Bounded, in-memory preparation of server-selected RGB evidence frames.

No URLs, filenames, detection models, network calls, or credential access. The
caller owns frame/source binding and supplies any crop/redaction coordinates.
Cropping and supplied-region blur reduce exposure; they do not anonymize images.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import hashlib
import io
import json
import math

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError


MAX_IMAGES = 3
MAX_EDGE = 768
MAX_IMAGE_BYTES = 256 * 1024
MAX_SOURCE_PIXELS = 40_000_000
DATA_URL_PREFIX = "data:image/jpeg;base64,"
PRIVACY_NOTICE = "JPEG metadata was removed. Identities may remain visible; cropping or blur does not guarantee anonymity."
_DESCRIPTOR_KEYS = {"id", "timestamp", "role", "data_url", "sha256", "width", "height", "byte_length", "privacy"}


class EvidenceImageError(ValueError):
    def __init__(self, message, code="invalid_images"):
        super().__init__(message)
        self.code = code


def _frame_id(value):
    return type(value) is int or isinstance(value, str) and 0 < len(value) <= 200


def _metadata(frame_id, timestamp, role):
    if not _frame_id(frame_id):
        raise EvidenceImageError("An exact server frame ID is required.")
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp) or timestamp < 0:
        raise EvidenceImageError("A finite, nonnegative frame timestamp is required.")
    if role not in ("context", "subject_crop"):
        raise EvidenceImageError("Unsupported evidence image role.")


def _box(box, width, height):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) is not int for value in box)):
        raise EvidenceImageError("Image regions must contain four integer source coordinates.")
    left, top, right, bottom = box
    if not 0 <= left < right <= width or not 0 <= top < bottom <= height:
        raise EvidenceImageError("Image region lies outside the source frame.")
    return left, top, right, bottom


def _check_jpeg_segments(raw):
    """Reject metadata anywhere in the JPEG, including after entropy scans.

    Pillow's header metadata inventory alone does not cover trailing bytes or
    application markers after a scan. Only a standard, thumbnail-free JFIF
    header is allowed; compressed pixels are still decoded separately by Pillow.
    """
    position = 2
    while position < len(raw):
        if raw[position] != 255:
            break
        while position < len(raw) and raw[position] == 255:
            position += 1
        if position >= len(raw):
            break
        marker = raw[position]
        position += 1
        if marker == 0xD9:
            if position == len(raw):
                return
            break
        if marker in (0, 1, 0xD8) or 0xD0 <= marker <= 0xD7 or position + 2 > len(raw):
            break
        length = int.from_bytes(raw[position:position + 2], "big")
        if length < 2 or position + length > len(raw):
            break
        payload = raw[position + 2:position + length]
        if marker == 0xFE or 0xE1 <= marker <= 0xEF:
            raise EvidenceImageError("Evidence JPEG still contains unsupported metadata.")
        if marker == 0xE0 and (len(payload) != 14 or not payload.startswith(b"JFIF\x00") or payload[-2:] != b"\x00\x00"):
            raise EvidenceImageError("Evidence JPEG contains unsupported application data.")
        position += length
        if marker == 0xDA:
            # In compressed scan data FF00 is an escaped pixel byte; restart
            # markers are standalone. Every other marker returns to parsing.
            while position < len(raw):
                position = raw.find(b"\xff", position)
                if position < 0 or position + 1 >= len(raw):
                    raise EvidenceImageError("Evidence JPEG contains an incomplete pixel scan.")
                following = raw[position + 1]
                if following == 0 or 0xD0 <= following <= 0xD7:
                    position += 2
                else:
                    break
    raise EvidenceImageError("Evidence JPEG contains malformed or trailing data.")


def prepare_evidence_image(rgb, *, frame_id, timestamp, role="context", crop_box=None, blur_regions=()):
    """Return a metadata-free JPEG descriptor without writing media to disk.

    ``rgb`` is PIL RGB or uint8 HxWx3 RGB (not OpenCV BGR). Coordinates are
    integer, source-space (left, top, right, bottom), with exclusive right/bottom.
    Pass a trusted crop with role=subject_crop; otherwise preserve full context.
    ``blur_regions`` are supplied detections, not a claim that all identities were
    detected. Preview these exact encoded bytes before obtaining send consent.
    """
    _metadata(frame_id, timestamp, role)
    if isinstance(rgb, Image.Image):
        if rgb.mode != "RGB":
            raise EvidenceImageError("Evidence images must be decoded RGB pixels.")
        width, height = rgb.size
    elif isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3:
        height, width = rgb.shape[:2]
    else:
        raise EvidenceImageError("Evidence images must be decoded RGB pixels.")
    if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
        raise EvidenceImageError("Source frame exceeds the pixel limit.")
    if (role == "subject_crop") != (crop_box is not None):
        raise EvidenceImageError("A subject crop requires a trusted source region and crop role.")
    crop = _box(crop_box, width, height) if crop_box is not None else (0, 0, width, height)
    if not isinstance(blur_regions, (list, tuple)) or len(blur_regions) > 100:
        raise EvidenceImageError("Too many supplied redaction regions.")
    boxes = [_box(region, width, height) for region in blur_regions]
    source = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    # Copy pixels into a new image to discard EXIF, ICC, comments and other info.
    pixels = source.crop(crop)
    image = Image.new("RGB", pixels.size)
    image.paste(pixels)
    applied = 0
    for left, top, right, bottom in boxes:
        clipped = (max(left, crop[0]) - crop[0], max(top, crop[1]) - crop[1],
                   min(right, crop[2]) - crop[0], min(bottom, crop[3]) - crop[1])
        if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
            continue
        region = image.crop(clipped)
        radius = max(8, min(region.size) / 5)
        image.paste(region.filter(ImageFilter.GaussianBlur(radius)), clipped[:2])
        applied += 1
    image.thumbnail((MAX_EDGE, MAX_EDGE), Image.Resampling.LANCZOS)
    encoded = b""
    for quality in (85, 70, 55, 40):
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = buffer.getvalue()
        if len(encoded) <= MAX_IMAGE_BYTES:
            break
    if len(encoded) > MAX_IMAGE_BYTES:
        raise EvidenceImageError("Prepared image exceeds the byte limit.")
    return {"id": frame_id, "timestamp": timestamp, "role": role,
            "data_url": DATA_URL_PREFIX + base64.b64encode(encoded).decode("ascii"),
            "sha256": hashlib.sha256(encoded).hexdigest(), "width": image.width, "height": image.height,
            "byte_length": len(encoded),
            "privacy": {"metadata_stripped": True, "blurred_regions": applied, "notice": PRIVACY_NOTICE}}


def validate_evidence_images(images, allowed_frame_ids):
    """Validate server-prepared descriptors and exact JPEG bytes; never fetch URLs.

    This validates representation and card membership, not provenance. Only the
    server may call this with pixels loaded from its current source revision.
    """
    if images is None:
        return []
    if not isinstance(images, list) or not 1 <= len(images) <= MAX_IMAGES:
        raise EvidenceImageError("Supply one to three prepared evidence images, or None for facts only.")
    allowed = {(type(value), value) for value in allowed_frame_ids}
    seen = set()
    validated = []
    for item in images:
        if not isinstance(item, dict) or set(item) != _DESCRIPTOR_KEYS:
            raise EvidenceImageError("Evidence image descriptor contains unsupported fields.")
        _metadata(item["id"], item["timestamp"], item["role"])
        key = (type(item["id"]), item["id"])
        if key not in allowed or key in seen:
            raise EvidenceImageError("Evidence image has an unknown or repeated frame ID.")
        seen.add(key)
        url = item["data_url"]
        if (not isinstance(url, str) or not url.startswith(DATA_URL_PREFIX)
                or len(url) > len(DATA_URL_PREFIX) + 4 * ((MAX_IMAGE_BYTES + 2) // 3)):
            raise EvidenceImageError("Only bounded, embedded JPEG evidence images are accepted.")
        try:
            raw = base64.b64decode(url[len(DATA_URL_PREFIX):], validate=True)
        except (ValueError, binascii.Error):
            raise EvidenceImageError("Evidence image is not valid base64 JPEG data.") from None
        if (not raw or len(raw) > MAX_IMAGE_BYTES or not raw.startswith(b"\xff\xd8")
                or not raw.endswith(b"\xff\xd9") or base64.b64encode(raw).decode("ascii") != url[len(DATA_URL_PREFIX):]
                or type(item["byte_length"]) is not int or item["byte_length"] != len(raw)
                or item["sha256"] != hashlib.sha256(raw).hexdigest()):
            raise EvidenceImageError("Evidence image bytes do not match their descriptor.")
        if any(type(item[key]) is not int or not 1 <= item[key] <= MAX_EDGE for key in ("width", "height")):
            raise EvidenceImageError("Prepared evidence image exceeds the dimension limit.")
        _check_jpeg_segments(raw)
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "JPEG" or image.mode != "RGB" or image.size != (item["width"], item["height"]):
                    raise EvidenceImageError("JPEG dimensions or color mode do not match the descriptor.")
                if any(marker != "APP0" or not value.startswith(b"JFIF\x00") for marker, value in image.applist):
                    raise EvidenceImageError("Evidence JPEG still contains unsupported metadata.")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
            if isinstance(error, EvidenceImageError):
                raise
            raise EvidenceImageError("Evidence JPEG could not be decoded safely.") from None
        privacy = item["privacy"]
        if (not isinstance(privacy, dict) or set(privacy) != {"metadata_stripped", "blurred_regions", "notice"}
                or privacy["metadata_stripped"] is not True or type(privacy["blurred_regions"]) is not int
                or not 0 <= privacy["blurred_regions"] <= 100 or privacy["notice"] != PRIVACY_NOTICE):
            raise EvidenceImageError("Evidence image privacy description is invalid.")
        validated.append(deepcopy(item))
    return validated


def image_descriptors(images):
    """Safe metadata for previews/results. Does not include image bytes or paths."""
    return [{key: deepcopy(value) for key, value in item.items() if key != "data_url"} for item in images]


def image_fingerprint(images):
    """Bind an explanation to the exact selected JPEG hashes and descriptors."""
    raw = json.dumps(image_descriptors(images), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
