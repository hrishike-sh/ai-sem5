"""Synthetic RGB fixtures only; no source footage, filesystem media or network."""
import base64
from copy import deepcopy
import hashlib
import io

import numpy as np
from PIL import Image
import pytest

from traffic_ai.violation_images import (
    DATA_URL_PREFIX, EvidenceImageError, MAX_EDGE, MAX_IMAGE_BYTES, PRIVACY_NOTICE,
    image_descriptors, image_fingerprint, prepare_evidence_image, validate_evidence_images,
)


def prepared(**kwargs):
    return prepare_evidence_image(Image.new("RGB", (48, 32), "orange"), frame_id=7, timestamp=1.25, **kwargs)


def jpeg_bytes(item):
    return base64.b64decode(item["data_url"].split(",", 1)[1])


def replace_bytes(item, raw):
    item["data_url"] = DATA_URL_PREFIX + base64.b64encode(raw).decode("ascii")
    item["sha256"] = hashlib.sha256(raw).hexdigest()
    item["byte_length"] = len(raw)


def test_preparation_is_bounded_metadata_free_rgb_and_does_not_mutate_source():
    rng = np.random.default_rng(18)
    rgb = rng.integers(0, 256, (900, 1200, 3), dtype=np.uint8)
    source = Image.fromarray(rgb)
    source.info["exif"] = b"private synthetic EXIF"
    source.info["icc_profile"] = b"private synthetic ICC"
    original = source.tobytes()
    result = prepare_evidence_image(source, frame_id="frame-7", timestamp=1.5)
    assert source.tobytes() == original and source.info["exif"] == b"private synthetic EXIF"
    assert result["width"] == MAX_EDGE and result["height"] == 576
    assert 0 < result["byte_length"] <= MAX_IMAGE_BYTES
    assert b"private synthetic" not in jpeg_bytes(result)
    with Image.open(io.BytesIO(jpeg_bytes(result))) as image:
        assert image.format == "JPEG" and image.mode == "RGB"
        assert not image.getexif() and "icc_profile" not in image.info
    assert result["privacy"]["notice"] == PRIVACY_NOTICE
    assert result["privacy"]["blurred_regions"] == 0
    assert validate_evidence_images([result], ["frame-7"]) == [result]


def test_trusted_crop_and_supplied_blur_use_source_coordinates():
    rgb = np.zeros((60, 80, 3), dtype=np.uint8)
    rgb[20:40, 30:50] = np.indices((20, 20)).sum(axis=0)[:, :, None] % 2 * 255
    original = rgb.copy()
    plain = prepare_evidence_image(rgb, frame_id=7, timestamp=1, role="subject_crop", crop_box=(20, 10, 60, 50))
    blurred = prepare_evidence_image(rgb, frame_id=7, timestamp=1, role="subject_crop",
                                    crop_box=(20, 10, 60, 50), blur_regions=[(30, 20, 50, 40), (0, 0, 10, 10)])
    assert np.array_equal(rgb, original)
    assert (blurred["width"], blurred["height"]) == (40, 40)
    assert blurred["privacy"]["blurred_regions"] == 1
    with Image.open(io.BytesIO(jpeg_bytes(plain))) as a, Image.open(io.BytesIO(jpeg_bytes(blurred))) as b:
        assert np.asarray(b)[10:30, 10:30].std() < np.asarray(a)[10:30, 10:30].std() / 2
    assert validate_evidence_images([blurred], [7]) == [blurred]


@pytest.mark.parametrize("options", [
    {"frame_id": True}, {"frame_id": ""}, {"timestamp": -1}, {"timestamp": float("nan")},
    {"timestamp": True}, {"role": "client_url"}, {"role": "subject_crop"},
    {"crop_box": (0, 0, 2, 2)}, {"role": "subject_crop", "crop_box": (-1, 0, 10, 10)},
    {"role": "subject_crop", "crop_box": (0, 0, 1000, 10)},
    {"role": "subject_crop", "crop_box": (0, 0, 2.5, 10)},
    {"blur_regions": [(0, 0, 0, 10)]}, {"blur_regions": [(0, 0, 2, 2)] * 101},
], ids=["bool-id", "empty-id", "negative-time", "nan-time", "bool-time", "role", "missing-crop", "crop-role",
        "negative-crop", "outside-crop", "float-crop", "empty-blur", "many-blurs"])
def test_invalid_preparation_parameters_fail_closed(options):
    args = {"frame_id": 7, "timestamp": 1}
    args.update(options)
    with pytest.raises(EvidenceImageError):
        prepare_evidence_image(Image.new("RGB", (48, 32)), **args)


@pytest.mark.parametrize("source", ["https://example.test/frame.jpg", b"raw encoded JPEG", np.zeros((2, 2)),
                                   np.zeros((2, 2, 3), dtype=np.float32), Image.new("RGBA", (2, 2))],
                         ids=["url", "bytes", "grayscale", "float-array", "rgba"])
def test_only_decoded_rgb_sources_are_accepted(source):
    with pytest.raises(EvidenceImageError):
        prepare_evidence_image(source, frame_id=7, timestamp=1)


@pytest.mark.parametrize("change", [
    {"id": "7"}, {"id": True}, {"timestamp": float("inf")}, {"width": 769}, {"height": True},
    {"sha256": "0" * 64}, {"byte_length": 1}, {"data_url": "https://example.test/image.jpg"},
    {"data_url": DATA_URL_PREFIX + "!not-base64!"}, {"source_path": "C:\\private\\footage.mp4"},
    {"privacy": {"anonymized": True}}, {"role": "untrusted"},
], ids=["typed-id", "bool-id", "infinite-time", "large-edge", "bool-height", "hash", "bytes",
        "remote-url", "base64", "extra-field", "privacy", "role"])
def test_descriptor_tampering_is_rejected(change):
    item = prepared()
    item.update(change)
    with pytest.raises(EvidenceImageError):
        validate_evidence_images([item], [7])


@pytest.mark.parametrize("case", ["empty", "too-many", "duplicate", "tuple", "unknown"])
def test_bundle_limits_and_card_binding(case):
    item = prepared()
    values = {"empty": [], "too-many": [item] * 4, "duplicate": [item, item], "tuple": (item,), "unknown": [item]}
    with pytest.raises(EvidenceImageError):
        validate_evidence_images(values[case], [8] if case == "unknown" else [7])
    assert validate_evidence_images(None, [7]) == []


@pytest.mark.parametrize("case", ["png", "exif", "truncated", "oversized", "dimensions", "comment", "trailing", "late-comment"])
def test_encoded_image_format_and_metadata_cannot_be_forged(case):
    item = prepared()
    raw = jpeg_bytes(item)
    if case in {"png", "exif", "dimensions"}:
        buffer = io.BytesIO()
        image = Image.new("RGB", (49 if case == "dimensions" else 48, 32), "orange")
        options = {"exif": b"Exif\x00\x00synthetic-private-data"} if case == "exif" else {}
        image.save(buffer, format="PNG" if case == "png" else "JPEG", **options)
        raw = buffer.getvalue()
    elif case == "truncated":
        raw = raw[:60] + b"\xff\xd9"
    elif case == "oversized":
        raw = b"\xff\xd8" + b"x" * MAX_IMAGE_BYTES + b"\xff\xd9"
    elif case == "comment":
        raw = raw[:2] + b"\xff\xfe\x00\x09private" + raw[2:]
    elif case == "trailing":
        raw += b"private trailing metadata\xff\xd9"
    elif case == "late-comment":
        raw = raw[:-2] + b"\xff\xfe\x00\x09private\xff\xd9"
    replace_bytes(item, raw)
    with pytest.raises(EvidenceImageError):
        validate_evidence_images([item], [7])


def test_safe_descriptors_are_detached_and_fingerprint_binds_selection():
    item = prepared()
    validated = validate_evidence_images([item], [7])
    descriptors = image_descriptors(validated)
    assert "data_url" not in descriptors[0]
    initial = image_fingerprint(validated)
    changed = deepcopy(validated)
    changed[0]["timestamp"] = 3
    assert image_fingerprint(changed) != initial
    descriptors[0]["privacy"]["notice"] = "changed"
    item["timestamp"] = 99
    assert validated[0]["privacy"]["notice"] == PRIVACY_NOTICE and validated[0]["timestamp"] == 1.25
