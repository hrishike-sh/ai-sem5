import json
import os
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path('H:/CodexStorage/ivp-ai/ollama')
ENDPOINT = 'http://127.0.0.1:11435'
MODEL = 'qwen3-vl:4b'


def environment():
    env = os.environ.copy()
    paths = {'USERPROFILE': ROOT / 'profile', 'LOCALAPPDATA': ROOT / 'profile/AppData/Local',
             'APPDATA': ROOT / 'profile/AppData/Roaming', 'TEMP': ROOT / 'temp', 'TMP': ROOT / 'temp',
             'OLLAMA_MODELS': ROOT / 'models', 'CUDA_CACHE_PATH': ROOT / 'cuda-cache'}
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(OLLAMA_HOST='127.0.0.1:11435', OLLAMA_NO_CLOUD='1', OLLAMA_NUM_PARALLEL='1',
               OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_KEEP_ALIVE='0', OLLAMA_CONTEXT_LENGTH='4096')
    return env


def request(route, payload=None, timeout=5):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(ENDPOINT + route, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ensure_running():
    try:
        return request('/api/version')
    except (OSError, urllib.error.URLError):
        pass
    binary = ROOT / 'runtime/ollama.exe'
    if not binary.is_file():
        raise RuntimeError('Local vision runtime is missing. Run Setup-Ollama.ps1; installation stays on H:.')
    with (ROOT / 'server.log').open('ab') as log:
        process = subprocess.Popen([str(binary), 'serve'], cwd=ROOT, env=environment(),
                                   stdout=log, stderr=log, creationflags=subprocess.CREATE_NO_WINDOW)
    (ROOT / 'process.json').write_text(json.dumps({'pid': process.pid, 'executable': str(binary)}))
    for _ in range(50):
        if process.poll() is not None:
            raise RuntimeError('Ollama could not start; see H:/CodexStorage/ivp-ai/ollama/server.log')
        try:
            return request('/api/version')
        except (OSError, urllib.error.URLError):
            time.sleep(.2)
    raise RuntimeError('Ollama startup timed out; inspect its H: server log before retrying.')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'pull'])
    args = parser.parse_args()
    if args.action == 'status':
        print(json.dumps(request('/api/tags')))
        return
    print(json.dumps(ensure_running()), flush=True)
    if args.action == 'pull':
        data = json.dumps({'model': MODEL, 'stream': True}).encode()
        req = urllib.request.Request(ENDPOINT + '/api/pull', data=data, headers={'Content-Type': 'application/json'})
        last = None
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                status = json.loads(line)
                if 'error' in status:
                    raise RuntimeError(status['error'])
                progress = (status.get('status'), status.get('completed', 0) // (128 * 1024 * 1024))
                if progress != last:
                    print(json.dumps(status), flush=True)
                    last = progress
        print(json.dumps(request('/api/tags')), flush=True)


if __name__ == '__main__':
    main()
