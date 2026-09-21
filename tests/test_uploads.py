import contextlib
import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ivp.media import load_image, save_upload


@pytest.fixture
def upload():
    buffer = io.BytesIO()
    Image.new("RGB", (24, 18), (20, 80, 140)).save(buffer, format="PNG")
    buffer.name = "street.png"
    return buffer


def upload_path(uploads, upload):
    digest = hashlib.sha256(upload.getbuffer()).hexdigest()
    return uploads / digest[:20] / Path(upload.name.replace("\\", "/")).name


@pytest.mark.parametrize("phase", ["write", "close"])
def test_failed_upload_is_not_published_and_can_be_retried(tmp_path, upload, monkeypatch, phase):
    uploads = tmp_path / "uploads"
    expected = upload_path(uploads, upload)
    original_open = Path.open

    @contextlib.contextmanager
    def failing_file(file):
        def write(buffer):
            file.write(buffer[:20] if phase == "write" else buffer)
            file.flush()
            if phase == "write":
                raise OSError("Upload write interrupted")
        with file:
            yield SimpleNamespace(write=write)
        if phase == "close":
            raise OSError("Upload close interrupted")

    def interrupt_write(path, mode="r", *args, **kwargs):
        file = original_open(path, mode, *args, **kwargs)
        if mode == "wb" and path.is_relative_to(uploads):
            return failing_file(file)
        return file

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", interrupt_write)
        with pytest.raises(OSError, match="Upload .* interrupted"):
            save_upload(upload, uploads)
    assert not expected.exists()
    assert not any(path.is_file() for path in uploads.rglob("*"))
    saved = save_upload(upload, uploads)
    assert saved == expected
    assert saved.read_bytes() == upload.getvalue()
    assert load_image(saved).shape == (18, 24, 3)


def test_overlapping_uploads_never_reuse_an_incomplete_source(tmp_path, upload, monkeypatch):
    uploads = tmp_path / "uploads"
    original_open = Path.open
    writing = threading.Event()
    release = threading.Event()

    @contextlib.contextmanager
    def slow_file(file):
        def write(buffer):
            file.write(buffer[:20])
            file.flush()
            writing.set()
            assert release.wait(timeout=5)
            file.write(buffer[20:])
        with file:
            yield SimpleNamespace(write=write)

    def delay_first_write(path, mode="r", *args, **kwargs):
        file = original_open(path, mode, *args, **kwargs)
        if mode == "wb" and path.is_relative_to(uploads) and not writing.is_set():
            return slow_file(file)
        return file

    monkeypatch.setattr(Path, "open", delay_first_write)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(save_upload, upload, uploads)
        try:
            assert writing.wait(timeout=5)
            second = save_upload(upload, uploads)
            assert second.read_bytes() == upload.getvalue()
            assert load_image(second).shape == (18, 24, 3)
            original = second.stat()
        finally:
            release.set()
        assert first.result(timeout=5) == second
    assert second.read_bytes() == upload.getvalue()
    assert second.stat().st_mtime_ns == original.st_mtime_ns
    assert [path for path in uploads.rglob("*") if path.is_file()] == [second]


def test_completed_upload_is_reused_without_writing(tmp_path, upload, monkeypatch):
    uploads = tmp_path / "uploads"
    saved = save_upload(upload, uploads)
    original = saved.stat()
    original_open = Path.open

    def reject_write(path, mode="r", *args, **kwargs):
        if mode == "wb":
            pytest.fail("Reusing a complete upload should not write it again")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_write)
    assert save_upload(upload, uploads) == saved
    assert saved.read_bytes() == upload.getvalue()
    assert saved.stat().st_mtime_ns == original.st_mtime_ns


def test_same_filename_with_different_contents_preserves_both_uploads(tmp_path, upload):
    uploads = tmp_path / "uploads"
    first = save_upload(upload, uploads)
    another = io.BytesIO()
    Image.new("RGB", (24, 18), (200, 10, 30)).save(another, format="PNG")
    another.name = "C:\\camera\\street.png"
    second = save_upload(another, uploads)
    assert first != second
    assert first.name == second.name == "street.png"
    assert first.read_bytes() == upload.getvalue()
    assert second.read_bytes() == another.getvalue()
