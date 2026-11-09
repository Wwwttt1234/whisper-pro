import sys
import threading
import time
import wave
from pathlib import Path

from core.storage import write_json, read_json
from core.settings import GENERAL
from core.jobs import JobManager


def test_cancelled_worker_never_writes_success(tmp_path):
    from engine.worker import execute
    write_json(tmp_path / "job.json", {"general": GENERAL, "profile": {}})
    write_json(tmp_path / "control.json", {"cancel": True})
    assert execute(tmp_path / "job.json") == 2
    assert read_json(tmp_path / "result.json")["status"] == "cancelled"
    assert not list(tmp_path.glob("*.srt"))


def test_worker_uses_current_settings_and_exports(tmp_path, monkeypatch):
    from engine.worker import execute
    import engine.adapters
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as stream:
        stream.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * 16000)
    seen = {}
    def fake(engine, path, model, profile, general, wav, folder, emit, checkpoint, **kwargs):
        seen.update(profile)
        return {"segments": [{"start": 0, "end": 1, "text": "test"}], "language": "en", "duration": 1}
    monkeypatch.setattr(engine.adapters, "transcribe", fake)
    job = {"source": str(source), "engine": "faster_whisper", "model_path": "unused", "model": {"name": "test"}, "general": dict(GENERAL), "profile": {"threads": 7}, "out_dir": str(tmp_path / "output"), "out_name": "result"}
    write_json(tmp_path / "job.json", job)
    assert execute(tmp_path / "job.json") == 0
    assert seen["threads"] == 7
    result = read_json(tmp_path / "result.json")
    assert result["status"] == "completed" and Path(result["written_files"][0]).is_file()


def test_real_process_pause_resume_cancel_and_history(tmp_path, monkeypatch):
    from core.database import Database
    import core.jobs as jobs_module
    import engine.runtime
    import engine.catalog
    writing_history = threading.Event()
    allow_history_write = threading.Event()
    update_session = Database.update_session

    def delayed_update(self, *args, **kwargs):
        writing_history.set()
        assert allow_history_write.wait(10)
        return update_session(self, *args, **kwargs)

    monkeypatch.setattr(Database, "update_session", delayed_update)
    worker_folder = tmp_path / "project" / "engine"
    worker_folder.mkdir(parents=True)
    worker_folder.joinpath("worker.py").write_text('import sys,time,json,pathlib\np=pathlib.Path(sys.argv[1]).parent\nfor i in range(1000):\n (p/"alive.txt").write_text(str(i))\n time.sleep(.05)\n', encoding="utf-8")
    monkeypatch.setattr(jobs_module, "ROOT", worker_folder.parent)
    monkeypatch.setattr(engine.runtime.Runtimes, "ensure", lambda *a, **k: sys.executable)
    monkeypatch.setattr(engine.catalog.Catalog, "ensure", lambda *a, **k: "fake-model")
    source = tmp_path / "video.wav"
    source.write_bytes(b"media")
    manager = JobManager()
    general = {**GENERAL, "output_dir": str(tmp_path / "out"), "beside_source": False}
    identity = manager.submit(source, "faster_whisper", {"id": "test", "name": "test"}, {}, general)
    try:
        deadline = time.monotonic() + 10
        while manager.jobs[identity]["status"] != "running" and time.monotonic() < deadline:
            time.sleep(.05)
        path = Path(manager.jobs[identity]["folder"]) / "alive.txt"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert path.exists()
        manager.pause(identity)
        before = path.read_text()
        time.sleep(.2)
        assert path.read_text() == before
        manager.pause(identity, False)
        time.sleep(.2)
        assert path.read_text() != before
        manager.cancel(identity)
        assert writing_history.wait(10)
        assert manager.jobs[identity]["status"] == "cancelling"
        assert Database().get_sessions()[0]["status"] == "running"
        allow_history_write.set()
        deadline = time.monotonic() + 10
        while manager.jobs[identity]["status"] == "cancelling" and time.monotonic() < deadline:
            time.sleep(.05)
        assert manager.jobs[identity]["status"] == "cancelled"
        assert manager.jobs[identity]["process"].poll() is not None
        assert Database().get_sessions()[0]["status"] == "cancelled"
    finally:
        allow_history_write.set()
        manager.cancel(identity)
        manager.pool.shutdown(wait=True)
