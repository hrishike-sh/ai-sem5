import mimetypes
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, urlparse


class FileServer:
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        self.token = secrets.token_urlsafe(24)
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                self.serve(False)

            def do_GET(self):
                self.serve(True)

            def serve(self, body):
                parts = urlparse(self.path).path.split("/")
                if len(parts) != 5 or parts[1] != owner.token or parts[2] not in ("view", "download"):
                    self.send_error(404)
                    return
                with owner.lock:
                    path = owner.files.get(parts[3])
                if path is None:
                    self.send_error(404)
                    return
                try:
                    file = path.open("rb") if path.is_file() else None
                except OSError:
                    file = None
                if file is None:
                    self.send_error(404)
                    return
                with file as handle:
                    self.serve_file(handle, path, body, parts[2] == "download")

            def serve_file(self, file, path, body, download):
                size = os.fstat(file.fileno()).st_size
                start, end = 0, max(0, size - 1)
                range_value = self.headers.get("Range")
                if range_value:
                    try:
                        unit, span = range_value.split("=", 1)
                        if unit != "bytes" or "," in span:
                            raise ValueError()
                        left, right = span.split("-", 1)
                        if left:
                            start = int(left)
                            end = min(int(right), size - 1) if right else size - 1
                        else:
                            start = max(0, size - int(right))
                            end = size - 1
                        if start < 0 or start >= size or start > end:
                            raise ValueError()
                    except ValueError:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.end_headers()
                        return
                self.send_response(206 if range_value else 200)
                self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(end - start + 1 if size else 0))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "private, max-age=0")
                self.send_header("X-Content-Type-Options", "nosniff")
                if range_value:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                if download:
                    self.send_header("Content-Disposition", "attachment; filename*=UTF-8''" + quote(path.name))
                self.end_headers()
                if body:
                    try:
                        file.seek(start)
                        remaining = end - start + 1 if size else 0
                        while remaining:
                            chunk = file.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path, download=False):
        path = Path(path).resolve()
        with self.lock:
            key = next((k for k, value in self.files.items() if value == path), None)
            if key is None:
                key = secrets.token_urlsafe(18)
                self.files[key] = path
        mode = "download" if download else "view"
        return f"http://127.0.0.1:{self.server.server_port}/{self.token}/{mode}/{key}/{quote(path.name)}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
