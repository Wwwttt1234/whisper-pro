"""Run inside the managed environment; activation scripts are unnecessary."""
import hashlib
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path
from core.storage import read_json, write_json


def setup_digest(root):
    return hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest()


def setup_ready(root, environment):
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return python.is_file() and read_json(environment / ".whisper-setup.json") == setup_digest(root)


def launch():
    root = Path(__file__).resolve().parent
    environment = Path(sys.prefix)
    managed = os.environ.get("UV_PROJECT_ENVIRONMENT")
    first_start = managed and environment.resolve() == Path(managed).resolve() and not setup_ready(root, environment)
    if first_start:
        print("Checking the installed application libraries...", flush=True)
        for module in ("streamlit", "av", "faster_whisper"):
            __import__(module)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = [sys.executable, "-m", "streamlit", "run", str(root / "app.py"), "--server.address", "127.0.0.1", "--server.port", str(port), "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.fileWatcherType", "none", "--server.maxUploadSize", os.environ.get("WHISPER_PRO_MAX_UPLOAD_MB", "2048")]
    process = subprocess.Popen(args, cwd=root, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    def stop(*_):
        if process.poll() is None:
            process.terminate()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        deadline = time.monotonic() + 90
        while process.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("تعذر بدء الواجهة خلال المهلة")
            time.sleep(0.25)
        if process.poll() is not None:
            raise RuntimeError("تعذر بدء البرنامج")
        if first_start:
            write_json(environment / ".whisper-setup.json", setup_digest(root))
        url = f"http://127.0.0.1:{port}"
        print("WHISPER_PRO_URL=" + url, flush=True)
        if not os.environ.get("WHISPER_PRO_NO_BROWSER"):
            webbrowser.open(url)
        return process.wait()
    finally:
        stop()


if __name__ == "__main__":
    sys.exit(launch())
