"""Prove rejected range/status/redirect headers never lead to a body read."""
from email.message import Message
import urllib.error

import pytest

from scripts.acquire_utfpr_prefix import NoRedirect, PREFIX_BYTES, fetch


class Response:
    def __init__(self, status=206, content_range='bytes 0-16777215/900000000', length=PREFIX_BYTES, url='https://example.org/clip'):
        self.status, self.url, self.read_calls, self.closed = status, url, [], False
        self.headers = Message()
        self.headers['Content-Length'] = str(length)
        self.headers['Content-Range'] = content_range
    def __enter__(self):return self
    def __exit__(self, *args):return False
    def geturl(self):return self.url
    def close(self):self.closed = True
    def read(self, n):
        self.read_calls.append(n)
        return b'x'*n


class Opener:
    def __init__(self, response):self.response = response
    def open(self, request, timeout):
        assert request.get_header('Range') == 'bytes=0-16777215'
        return self.response


@pytest.mark.parametrize('response', [Response(status=200), Response(content_range='bytes 1-16777216/900000000'),
                                    Response(content_range='bytes 0-999/900000000'), Response(length=PREFIX_BYTES+1),
                                    Response(url='https://other.example/clip')])
def test_rejected_response_never_consumes_body_or_creates_file(tmp_path, response):
    with pytest.raises(ValueError):
        fetch('https://example.org/clip', tmp_path/'clip.h264', exact_bytes=PREFIX_BYTES,
              cap=PREFIX_BYTES, byte_range=True, opener=Opener(response))
    assert response.read_calls == [] and list(tmp_path.iterdir()) == []


def test_validated_prefix_never_reads_more_than_exact_cap(tmp_path):
    response = Response()
    destination = tmp_path/'clip.h264'
    record = fetch('https://example.org/clip', destination, exact_bytes=PREFIX_BYTES,
                   cap=PREFIX_BYTES, byte_range=True, opener=Opener(response))
    assert sum(response.read_calls) == PREFIX_BYTES
    assert destination.stat().st_size == PREFIX_BYTES
    assert record['response_status'] == 206 and record['bytes_read'] == PREFIX_BYTES
    with pytest.raises(FileExistsError):
        fetch('https://example.org/clip', destination, exact_bytes=PREFIX_BYTES, cap=PREFIX_BYTES,
              byte_range=True, opener=Opener(Response()))


def test_redirect_callback_refuses_before_reading_the_response():
    response = Response(status=302)
    with pytest.raises(ValueError, match='Redirect refused'):
        NoRedirect().redirect_request(None, response, 302, 'Found', {}, 'https://other.example/whole-video')
    assert response.read_calls == []
    assert response.closed


def test_http_error_is_closed_without_reading_body(tmp_path):
    response = Response(status=464)
    class ErrorOpener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 464, '', response.headers, response)
    with pytest.raises(urllib.error.HTTPError):
        fetch('https://example.org/clip', tmp_path/'clip.h264', exact_bytes=PREFIX_BYTES,
              cap=PREFIX_BYTES, byte_range=True, opener=ErrorOpener())
    assert response.closed and response.read_calls == [] and list(tmp_path.iterdir()) == []
