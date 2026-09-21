import argparse
import contextlib
import errno
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8511"


def healthy():
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=1) as response:
            return response.status == 200 and json.load(response).get("app") == "traffic-lab-web"
    except (OSError, ValueError):
        return False


@contextlib.contextmanager
def startup_lock(timeout=35):
    (ROOT / "temp").mkdir(exist_ok=True)
    with (ROOT / "temp" / ".startup.lock").open("a+b") as lock:
        if lock.seek(0, 2) == 0:
            lock.write(b"0")
            lock.flush()
        if os.name == "nt":
            import msvcrt

            def acquire():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(lock, fcntl.LOCK_UN)

        deadline = time.monotonic() + timeout
        while True:
            try:
                acquire()
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                if time.monotonic() >= deadline:
                    raise SystemExit("Another Traffic Lab launcher is still starting. Wait for it to finish, then retry Start.cmd.") from exc
                time.sleep(.1)
        try:
            yield
        finally:
            release()


def _cleanup_failed_start(process):
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
    saved = ROOT / "temp" / "server.json"
    with contextlib.suppress(OSError, ValueError):
        record = json.loads(saved.read_text(encoding="utf-8"))
        if isinstance(record, dict) and record.get("pid") == process.pid:
            saved.unlink()


def _ensure_server():
    if not healthy():
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", 8511)) == 0:
                raise SystemExit("Port 8511 is occupied by another service. Close it before starting Traffic Lab.")
        (ROOT / "temp").mkdir(exist_ok=True)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
        process = None
        try:
            with (ROOT / "temp" / "server.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, str(ROOT / "web_server.py"), "--port", "8511"],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, **options)
            (ROOT / "temp" / "server.json").write_text(json.dumps({"pid": process.pid, "url": URL}), encoding="utf-8")
            for _ in range(120):
                if healthy():
                    break
                if process.poll() is not None:
                    raise SystemExit("Server stopped during startup. See temp/server.log; rerun Setup.ps1 if dependencies are missing.")
                time.sleep(0.25)
            else:
                raise SystemExit("Server is taking longer than expected. Check temp/server.log and retry Start.cmd.")
        except BaseException:
            if process is not None:
                _cleanup_failed_start(process)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        raise SystemExit("Use the isolated Python 3.11 environment: double-click Start.cmd.")
    with startup_lock():
        _ensure_server()
    print("Traffic Lab is running at " + URL)
    if not args.no_browser:
        webbrowser.open(URL)


if __name__ == "__main__":
    main()
