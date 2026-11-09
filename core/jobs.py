"""Background orchestration; per-file workers keep UI responsive and settings isolated."""
import copy
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.paths import ROOT, location
from core.storage import read_json, write_json
from core.database import Database
from core.settings import validate_profile
from engine.catalog import Catalog
from engine.runtime import Runtimes, child_env, stop_tree

TERMINAL = {"completed", "no_speech", "cancelled", "skipped", "error"}


class JobManager:
    def __init__(self, parallel=1):
        self.lock = threading.RLock()
        self.jobs = {}
        self.pool = ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="whisper-job")
        self.parallel = parallel

    def submit(self, source, engine, model, profile, general, token="", mode="transcribe", subtitle_stream=None, channel=None, display_name=None):
        source = Path(source).expanduser().resolve()
        if not source.is_file() or source.stat().st_size == 0:
            raise ValueError("ملف الوسائط غير موجود أو فارغ")
        profile = validate_profile(engine, model, profile)
        identity = uuid.uuid4().hex
        folder = location("jobs") / identity
        folder.mkdir(parents=True)
        out_dir = Path(general["output_dir"]).expanduser() if general.get("output_dir") and not general.get("beside_source") else source.parent
        if not general.get("beside_source") and not general.get("output_dir"):
            out_dir = location("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        name = Path(display_name or source.name).stem + (f"_channel{channel + 1}" if channel is not None else "")
        if general.get("output_subfolder"):
            subfolder = general.get("output_subfolder_name", "{original_name}_output").replace("{original_name}", name).strip()
            if not subfolder or subfolder in {".", ".."} or any(c in subfolder for c in "/\\\x00"):
                raise ValueError("اسم المجلد الفرعي غير صالح")
            out_dir = out_dir / subfolder
            out_dir.mkdir(parents=True, exist_ok=True)
        job = {"id": identity, "source": str(source), "engine": engine, "model": copy.deepcopy(model), "profile": profile, "general": copy.deepcopy(general), "out_dir": str(out_dir), "out_name": name, "folder": str(folder), "mode": mode, "subtitle_stream": subtitle_stream, "channel": channel, "status": "queued", "progress": 0.0, "created_at": time.time(), "logs": []}
        with self.lock:
            self.jobs[identity] = job
        self.pool.submit(self._run, identity, token)
        return identity

    def _patch(self, identity, **fields):
        with self.lock:
            self.jobs[identity].update(fields)

    def _log(self, identity, text, token=""):
        if token:
            text = text.replace(token, "[رمز مخفي]")
        with self.lock:
            job = self.jobs[identity]
            job["logs"] = (job["logs"] + [text])[-150:]

    def _run(self, identity, token):
        job = self.jobs[identity]
        folder = Path(job["folder"])
        cancelled = lambda: bool(job.get("cancel_requested") or read_json(folder / "control.json", {}).get("cancel"))
        log = lambda text: self._log(identity, text, token)
        session = None
        try:
            if cancelled():
                raise InterruptedError()
            self._patch(identity, status="preparing")
            runtime = Runtimes()
            python = sys.executable
            if job["mode"] != "extract":
                executable = runtime.ensure(job["engine"], job["profile"], log, cancelled)
                if job["engine"] == "crisp":
                    job["native_binary"] = executable
                else:
                    python = executable
                job["model_path"] = Catalog().ensure(job["model"], job["profile"], progress=lambda a, b, msg: self._patch(identity, stage=msg, progress=(a / b if b else 0) * 0.1), cancelled=cancelled, token=token, models_dir=job["general"].get("models_dir", ""))
                post_runtimes = {}
                post_profile = {**job["profile"], "device": job["general"].get("post_device", "cpu"), "dtype": job["general"].get("post_dtype", "float32"), "native_binary": "", "runtime_python": "", "llama_binary": job["general"].get("llama_binary", "")}
                if job["general"].get("alignment") in {"bfa", "whisperx"}:
                    key = job["general"]["alignment"]
                    post_runtimes[key] = runtime.ensure(key, post_profile, log, cancelled)
                if job["general"].get("translation_engine") in {"transformers", "llama"}:
                    key = "translation" if job["general"]["translation_engine"] == "transformers" else "llama"
                    post_runtimes[key] = runtime.ensure(key, post_profile, log, cancelled)
                    translation = Catalog().get(job["general"].get("translation_model", ""))
                    if not translation:
                        raise ValueError("اختر نموذج ترجمة من قائمة النماذج")
                    job["translation_model"] = translation
                    job["translation_path"] = Catalog().ensure(translation, job["profile"], cancelled=cancelled, token=token, models_dir=job["general"].get("models_dir", ""))
                job["post_runtimes"] = post_runtimes
            if cancelled():
                raise InterruptedError()
            db = Database()
            session = db.create_session({"selected_model": job["model"]["name"], "device": job["profile"].get("device", "cpu"), "language": job["general"]["language"], "task": job["general"]["task"], "engine": job["engine"], "profile": job["profile"]}, status="running")
            # Runtime-only objects and tokens never go into the job document.
            persisted = {k: v for k, v in job.items() if k not in {"process", "logs"}}
            write_json(folder / "job.json", persisted)
            env = child_env()
            if token:
                env["HF_TOKEN"] = token
            with open(folder / "worker.log", "w", encoding="utf-8") as worker_log:
                process = subprocess.Popen([python, str(ROOT / "engine/worker.py"), str(folder / "job.json")], cwd=ROOT, env=env, stdout=worker_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            self._patch(identity, process=process, status="running", stage="التفريغ", started_at=time.time())
            while process.poll() is None:
                if cancelled():
                    stop_tree(process)
                    break
                time.sleep(0.2)
            result = read_json(folder / "result.json", {})
            with self.lock:
                status = job.get("cancel_status", "cancelled") if cancelled() else result.get("status", "error")
                error = result.get("error", "")
                if status == "error" and not error:
                    error = (folder / "worker.log").read_text(encoding="utf-8", errors="replace")[-4000:]
                if token:
                    error = error.replace(token, "[رمز مخفي]")
                db.add_file(session, {"file_name": source_name(job), "file_path": job["source"], "model": job["model"]["name"], "language": result.get("language", ""), "duration": result.get("duration", 0), "segments": len(result.get("segments", [])), "elapsed": result.get("elapsed", 0), "output_dir": job["out_dir"], "status": status, "error_message": error, "text_preview": result.get("text", "")[:500]})
                db.update_session(session, status=status, total_files=1, total_duration=result.get("duration", 0), total_elapsed=result.get("elapsed", 0), total_segments=len(result.get("segments", [])))
                self._patch(identity, status=status, result=result, error=error, ended_at=time.time(), progress=1.0 if status in {"completed", "no_speech"} else job.get("progress", 0))
        except InterruptedError:
            self._patch(identity, status=job.get("cancel_status", "cancelled"), ended_at=time.time())
        except Exception as exc:
            error = str(exc).replace(token, "[رمز مخفي]") if token else str(exc)
            self._patch(identity, status="error", error=error, ended_at=time.time())
            if session:
                Database().update_session(session, status="error")

    def snapshots(self):
        with self.lock:
            snapshots = []
            for job in self.jobs.values():
                item = {k: copy.deepcopy(v) for k, v in job.items() if k != "process"}
                events = Path(job["folder"]) / "events.jsonl"
                if events.exists():
                    with open(events, "rb") as stream:
                        offset = max(0, events.stat().st_size - 65536)
                        stream.seek(offset)
                        lines = stream.read().decode("utf-8", errors="replace").splitlines()
                    for line in lines[-50:]:
                        try:
                            import json
                            event = json.loads(line)
                            if event.get("kind") == "progress":
                                item["progress"] = max(0, min(1, event.get("progress", 0)))
                            if event.get("kind") == "log":
                                item["logs"].append(event.get("message", ""))
                        except (ValueError, TypeError):
                            pass
                snapshots.append(item)
            return snapshots

    def cancel(self, identity, skipped=False):
        with self.lock:
            job = self.jobs[identity]
            if job["status"] in TERMINAL:
                return
            job["cancel_status"] = "skipped" if skipped else "cancelled"
            job["cancel_requested"] = True
            job["status"] = "cancelling"
            process = job.get("process")
            if process:
                stop_tree(process)
            write_json(Path(job["folder"]) / "control.json", {"cancel": True})

    def pause(self, identity, paused=True):
        import psutil
        with self.lock:
            job = self.jobs[identity]
            process = job.get("process")
            if not process or job["status"] not in {"running", "paused"}:
                return
            if paused:
                write_json(Path(job["folder"]) / "control.json", {"pause": True})
            try:
                parent = psutil.Process(process.pid)
                children = parent.children(recursive=True)
                for item in ([parent] + children if paused else children + [parent]):
                    try:
                        item.suspend() if paused else item.resume()
                    except psutil.Error:
                        pass
                if not paused:
                    write_json(Path(job["folder"]) / "control.json", {"pause": False})
                job["status"] = "paused" if paused else "running"
            except psutil.Error:
                pass


def source_name(job):
    return Path(job["source"]).name
