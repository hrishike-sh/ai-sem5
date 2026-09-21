import contextlib
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from ivp.downloads import FileServer


@pytest.fixture
def server():
    instance = FileServer()
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("range_value", [None, "bytes=2-5"], ids=["full", "range"])
def test_download_headers_match_the_file_opened(tmp_path, monkeypatch, server, range_value):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"original file contents")
    replacement = b"replacement"
    original_open = Path.open
    changed = False

    def change_before_open(file_path, mode="r", *args, **kwargs):
        nonlocal changed
        if file_path == path and mode == "rb" and not changed:
            changed = True
            with original_open(path, "wb") as output:
                output.write(replacement)
        return original_open(file_path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", change_before_open)
    request = urllib.request.Request(server.url(path), headers={"Range": range_value} if range_value else {})
    with urllib.request.urlopen(request, timeout=3) as response:
        received = response.read()
        expected = replacement[2:6] if range_value else replacement
        assert response.status == (206 if range_value else 200)
        assert received == expected
        assert int(response.headers["Content-Length"]) == len(expected)
        if range_value:
            assert response.headers["Content-Range"] == "bytes 2-5/11"
    assert changed


def test_file_disappearing_before_open_returns_404(tmp_path, monkeypatch, server):
    path = tmp_path / "removed.bin"
    path.write_bytes(b"original file contents")
    original_open = Path.open

    def remove_before_open(file_path, mode="r", *args, **kwargs):
        if file_path == path and mode == "rb":
            path.unlink()
        return original_open(file_path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", remove_before_open)
    with pytest.raises(urllib.error.HTTPError) as error:
        with urllib.request.urlopen(server.url(path), timeout=3) as response:
            response.read()
    assert error.value.code == 404
    error.value.close()


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize("phase", ["open", "stat"])
def test_unreadable_file_returns_404(tmp_path, monkeypatch, server, method, phase):
    path = tmp_path / "unreadable.bin"
    path.write_bytes(b"original file contents")
    original_open = Path.open
    original_is_file = Path.is_file

    def deny_open(file_path, mode="r", *args, **kwargs):
        if file_path == path and mode == "rb":
            raise PermissionError("File is no longer readable")
        return original_open(file_path, mode, *args, **kwargs)

    def deny_stat(file_path):
        if file_path == path:
            raise PermissionError("File metadata is no longer readable")
        return original_is_file(file_path)

    if phase == "open":
        monkeypatch.setattr(Path, "open", deny_open)
    else:
        monkeypatch.setattr(Path, "is_file", deny_stat)
    request = urllib.request.Request(server.url(path), method=method)
    with pytest.raises(urllib.error.HTTPError) as error:
        with urllib.request.urlopen(request, timeout=3) as response:
            response.read()
    assert error.value.code == 404
    error.value.close()


@pytest.mark.skipif(os.name == "nt", reason="Windows prevents replacing this open file.")
def test_replacement_after_open_uses_the_open_files_size(tmp_path, monkeypatch, server):
    path = tmp_path / "replaced.bin"
    original = b"original file contents"
    path.write_bytes(original)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"new")
    original_open = Path.open
    changed = False

    def replace_after_open(file_path, mode="r", *args, **kwargs):
        nonlocal changed
        file = original_open(file_path, mode, *args, **kwargs)
        if file_path == path and mode == "rb" and not changed:
            changed = True
            replacement.replace(path)
        return file

    monkeypatch.setattr(Path, "open", replace_after_open)
    with urllib.request.urlopen(server.url(path), timeout=3) as response:
        assert response.read() == original
        assert int(response.headers["Content-Length"]) == len(original)
    assert path.read_bytes() == b"new"


@pytest.mark.parametrize("method,range_value", [("HEAD", None), ("GET", "bytes=999-")])
def test_open_file_is_closed_for_head_and_invalid_range(tmp_path, monkeypatch, server, method, range_value):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"original file contents")
    original_open = Path.open
    opened = []
    closed = threading.Event()

    @contextlib.contextmanager
    def track_close(file):
        try:
            with file as handle:
                yield handle
        finally:
            closed.set()

    def record_open(file_path, mode="r", *args, **kwargs):
        file = original_open(file_path, mode, *args, **kwargs)
        if file_path == path and mode == "rb":
            opened.append(file)
            return track_close(file)
        return file

    monkeypatch.setattr(Path, "open", record_open)
    request = urllib.request.Request(server.url(path), method=method,
                                     headers={"Range": range_value} if range_value else {})
    if range_value:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=3)
        assert error.value.code == 416
        error.value.close()
    else:
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 200
            assert response.read() == b""
    if opened:
        assert closed.wait(timeout=3)
    assert all(file.closed for file in opened)
