import os
import pytest
import tempfile

if os.environ.get("WHISPER_PRO_TEST_TMP"):
    tempfile.tempdir = os.environ["WHISPER_PRO_TEST_TMP"]

os.environ["WHISPER_PRO_DISABLE_AUTOCHECK"] = "1"
os.environ["WHISPER_PRO_NO_BROWSER"] = "1"


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setenv("WHISPER_PRO_DATA_DIR", str(tmp_path / "data"))
    from core import database
    monkeypatch.setattr(database, "DEFAULT_DB_PATH", str(tmp_path / "history.db"))
