import json
import sys
import threading
import time
import wave
from pathlib import Path

from core.storage import write_json, read_json
from core.settings import GENERAL
from core.jobs import JobManager


def test_live_segments_survive_log_bursts_and_partial_events(tmp_path, monkeypatch):
    import core.jobs as jobs_module

    monkeypatch.setattr(jobs_module.time, "monotonic", lambda: 20.0)
    manager = JobManager()
    manager.jobs["test"] = {"id": "test", "folder": str(tmp_path), "status": "running", "started_at": 1.0, "logs": []}
    first = {"kind": "segment", "start": 0, "end": 2, "text": "المقطع الأول"}
    second = {"kind": "segment", "start": 2, "end": 4, "text": "المقطع الثاني"}
    path = tmp_path / "events.jsonl"
    events = [{"kind": "phase", "phase": "transcribe", "time": 5}, first]
    events.extend({"kind": "log", "message": "x" * 1000} for _ in range(90))
    events.extend([{"kind": "progress", "progress": 0.25, "time": 10}, {"kind": "progress", "progress": 0.5, "time": 15}])
    path.write_text("\n".join(map(json.dumps, [*events, second])), encoding="utf-8")
    snapshot = manager.snapshots()[0]
    assert [row["text"] for row in snapshot["live_segments"]] == [first["text"]]
    assert snapshot["elapsed"] == 19
    assert 0.5 < snapshot["progress"] < 1
    assert snapshot["remaining"] == 10
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    manager.jobs["test"].update(status="paused", paused_at=20.0)
    monkeypatch.setattr(jobs_module.time, "monotonic", lambda: 30.0)
    for _ in range(2):
        snapshot = manager.snapshots()[0]
        assert [row["text"] for row in snapshot["live_segments"]] == [first["text"], second["text"]]
        assert snapshot["elapsed"] == 19
        assert snapshot["remaining"] == 10
    manager.jobs["test"].update(status="completed", ended_at=31.0, pauses=[(20.0, 30.0)], paused_at=None)
    monkeypatch.setattr(jobs_module.time, "monotonic", lambda: 50.0)
    snapshot = manager.snapshots()[0]
    assert (snapshot["elapsed"], snapshot["progress"], snapshot["remaining"]) == (20, 1, 0)


def test_file_history_includes_engine_and_old_records(tmp_path):
    from core.database import Database

    db = Database(str(tmp_path / "history-test.db"))
    old = db.create_session({"selected_model": "old model"})
    db.add_file(old, {"file_name": "old.wav", "elapsed": 3})
    current = db.create_session({"engine": "vosk", "selected_model": "new model"})
    db.add_file(current, {"file_name": "new.mp4", "elapsed": 8.5, "model": "custom model"})
    new_row, old_row = db.get_file_history()
    assert (new_row["file_name"], new_row["elapsed"], new_row["engine"], new_row["model"]) == ("new.mp4", 8.5, "vosk", "custom model")
    assert len(new_row["created_at"]) == 19
    assert (old_row["engine"], old_row["model"]) == (None, "old model")


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
    identity = manager.submit(source, "faster_whisper", {"id": "test", "name": "test"}, {}, general, display_name="الفيديو الأصلي.wav")
    try:
        deadline = time.monotonic() + 10
        while manager.jobs[identity]["status"] != "running" and time.monotonic() < deadline:
            time.sleep(.05)
        path = Path(manager.jobs[identity]["folder"]) / "alive.txt"
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert path.exists()
        manager.pause(identity)
        paused_elapsed = manager.snapshots()[0]["elapsed"]
        before = path.read_text()
        time.sleep(.2)
        assert path.read_text() == before
        assert manager.snapshots()[0]["elapsed"] == paused_elapsed
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
        row = Database().get_file_history()[0]
        assert row["file_name"] == "الفيديو الأصلي.wav"
        assert row["elapsed"] == manager.snapshots()[0]["elapsed"] > paused_elapsed
    finally:
        allow_history_write.set()
        manager.cancel(identity)
        manager.pool.shutdown(wait=True)
