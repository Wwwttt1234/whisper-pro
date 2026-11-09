"""Build a native launcher with a bundled application source snapshot."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--archive", choices=["Windows-x64", "Linux-x64", "Linux-arm64", "macOS-arm64", "macOS-x64"])
options = parser.parse_args()
staging = (ROOT / "build/release-source").resolve()
if not staging.is_relative_to(ROOT) or staging.parent != (ROOT / "build").resolve():
    raise RuntimeError("Unexpected build path")
if staging.exists():
    shutil.rmtree(staging)
payload = staging / "app"
payload.mkdir(parents=True)
for name in ["app.py", "launcher.py", "pyproject.toml", "uv.lock", "LICENSE", "README.md", "IMPLEMENTATION_REPORT_AR.md"]:
    shutil.copy2(ROOT / name, payload / name)
for name in ["core", "engine", "ui", "utils"]:
    shutil.copytree(ROOT / name, payload / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
mode = "--onedir" if sys.platform == "darwin" else "--onefile"
command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", mode, "--windowed", "--name", "WhisperPro", "--add-data", str(payload) + os.pathsep + "app", str(ROOT / "desktop_launcher.py")]
subprocess.run(command, cwd=ROOT, check=True)
dist = ROOT / "dist"
entry = dist / {"win32": "WhisperPro.exe", "darwin": "WhisperPro.app", "linux": "WhisperPro"}[sys.platform]
binary = entry / "Contents/MacOS/WhisperPro" if sys.platform == "darwin" else entry
with tempfile.TemporaryDirectory(dir=staging) as temporary:
    manifest = Path(temporary) / "manifest.json"
    subprocess.run([str(binary), "--self-test", str(manifest)], check=True, timeout=120)
    actual = json.loads(manifest.read_text(encoding="utf-8"))
    if actual["lock_sha256"] != hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest():
        raise RuntimeError("The built launcher contains a different dependency lock.")

documents = ["LICENSE", "README.md", "GUIDE_AR.md", "SECURITY.md", "IMPLEMENTATION_REPORT_AR.md", "PROJECT_AUDIT_AR.md"]
for name in documents:
    shutil.copy2(ROOT / name, dist / name)

if options.archive:
    packages = ROOT / "packages"
    packages.mkdir(exist_ok=True)
    files = [entry, *(dist / name for name in documents)]
    archive_name = "WhisperPro-" + options.archive
    if sys.platform == "win32":
        with zipfile.ZipFile(packages / (archive_name + ".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
            for file in files:
                archive.write(file, file.name)
    else:
        with tarfile.open(packages / (archive_name + ".tar.gz"), "w:gz") as archive:
            for file in files:
                archive.add(file, arcname=file.name)
