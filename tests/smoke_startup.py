"""Run the actual source launcher before and after moving a clean checkout."""
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def stop(process):
    if sys.platform != "win32":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        return

    import psutil

    if process.poll() is not None:
        return
    parent = psutil.Process(process.pid)
    processes = [parent, *parent.children(recursive=True)]
    for child in reversed(processes):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, pending = psutil.wait_procs(processes, timeout=5)
    for child in pending:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(pending, timeout=5)


def startup_command(source):
    if sys.platform == "win32":
        return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(source / "bootstrap.ps1")]
    return ["sh", str(source / ("Start.command" if sys.platform == "darwin" else "Start.sh"))]


def startup_environment(data):
    env = {**os.environ, "WHISPER_PRO_DATA_DIR": str(data), "WHISPER_PRO_NO_BROWSER": "1", "WHISPER_PRO_DISABLE_AUTOCHECK": "1"}
    env.pop("WHISPER_PRO_SETUP_YES", None)
    return env


def check_decline(source, data):
    result = subprocess.run(startup_command(source), cwd=source.parent, env=startup_environment(data), input=b"n\n", capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not data.exists(), "Declining setup still created application files"


def check_startup(source, data, log, *, first=False):
    with log.open("wb") as output:
        process = subprocess.Popen(startup_command(source), cwd=source.parent, env=startup_environment(data), stdin=subprocess.PIPE if first else subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=sys.platform != "win32")
        try:
            if first:
                with process.stdin:
                    process.stdin.write(b"y\n")
            deadline = time.monotonic() + 600
            while process.poll() is None and time.monotonic() < deadline:
                contents = log.read_bytes()
                if match := re.search(rb"WHISPER_PRO_URL=(http://127\.0\.0\.1:\d+)", contents):
                    with urllib.request.urlopen(match[1].decode() + "/_stcore/health", timeout=5) as response:
                        if response.status == 200:
                            return
                time.sleep(0.25)
            raise RuntimeError(log.read_bytes().decode(errors="replace")[-4000:])
        finally:
            stop(process)


def main():
    with TemporaryDirectory(prefix="whisper-startup-") as temp:
        work = Path(temp).resolve()
        original = work / "original source"
        relocated = work / "moved source"
        data = work / "app data"
        original.mkdir()
        for name in ("app.py", "launcher.py", "bootstrap.ps1", "Start.sh", "Start.command", "uv.lock", "pyproject.toml", "README.md"):
            shutil.copy2(ROOT / name, original / name)
        for name in ("core", "engine", "ui", "utils"):
            shutil.copytree(ROOT / name, original / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # A copied checkout may have a stale environment pointing to the old OS/path.
        stale = original / ".venv"
        stale.mkdir()
        stale_config = stale / "pyvenv.cfg"
        stale_config.write_text("home = /missing/old/python\n", encoding="utf-8")
        check_decline(original, data)
        check_startup(original, data, work / "first-start.log", first=True)
        config = data / "app-environment-v2" / "pyvenv.cfg"
        created = config.stat().st_mtime_ns
        assert stale_config.read_text(encoding="utf-8") == "home = /missing/old/python\n"
        original.rename(relocated)
        check_startup(relocated, data, work / "moved-start.log")
        assert config.stat().st_mtime_ns == created, "Moving source rebuilt the app environment"
        print("Startup OK on a clean checkout and after moving source; the app environment was reused.")


if __name__ == "__main__":
    main()
