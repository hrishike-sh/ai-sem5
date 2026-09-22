import urllib.error
import urllib.request
import pytest
from ivp.downloads import FileServer


def test_streamed_download_range_and_unregistered_path(tmp_path):
    path = tmp_path / "sample.mp4"
    path.write_bytes(bytes(range(256)) * 100)
    server = FileServer()
    try:
        url = server.url(path)
        req = urllib.request.Request(url, headers={"Range": "bytes=20-39"})
        with urllib.request.urlopen(req) as result:
            assert result.status == 206
            assert result.read() == path.read_bytes()[20:40]
            assert result.headers["Content-Range"] == "bytes 20-39/25600"
        with urllib.request.urlopen(server.url(path, download=True)) as result:
            assert "attachment" in result.headers["Content-Disposition"]
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url.replace(server.token, "invalid-token"))
        assert error.value.code == 404
    finally:
        server.close()
