"""Locations shared by the launcher, UI and isolated inference workers."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    override = os.environ.get("WHISPER_PRO_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "WhisperPro"
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/WhisperPro"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "whisper-pro"


def location(name: str) -> Path:
    path = data_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
