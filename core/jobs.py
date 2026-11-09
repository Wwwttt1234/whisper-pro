"""Background orchestration; per-file workers keep UI responsive and settings isolated."""
import copy
import json
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.paths import ROOT, location
from core.storage import read_json, write_json
from core.database import Database
from core.settings import validate_profile
from engine.catalog import Catalog
from engine.runtime import Runtimes, child_env, stop_tree

TERMINAL = {"completed", "no_speech", "cancelled", "skipped", "error"}
PHASES = {
    "extract": ("استخراج مسار الترجمة", 0.10),
    "decode": ("تجهيز الصوت", 0.10),
    "transcribe": ("التفريغ", 0.20),
    "postprocess": ("المحاذاة أو الترجمة", 0.95),
    "export": ("حفظ ملفات الإخراج", 0.98),
}


def active_elapsed(job, now):
    if "started_at" not in job:
        return 0.0
    stop = min(now, job.get("ended_at", now), job.get("paused_at") or now)
    paused = sum(max(0.0, min(end, stop) - start) for start, end in job.get("pauses", ()))
    return max(0.0, stop - job["started_at"] - paused)


class JobEvents:
    def __init__(self):
        self.offset = 0
        self.logs = deque(maxlen=35)
        self.segments = []
        self.phase = None
        self.progress = 0.0
        self.audio_fraction = 0.0
        self.remaining = None
        self.sample = None

    def read(self, job):
        path = Path(job["folder"]) / "events.jsonl"
        try:
            stream = path.open("rb")
        except FileNotFoundError:
            return
        with stream:
            stream.seek(self.offset)
            while (line := stream.readline()).endswith(b"\n"):
                self.offset += len(line)
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("kind")
                if kind == "log":
                    self.logs.append(event["message"])
                elif kind == "segment":
                    self.segments.append({key: event[key] for key in ("start", "end", "text")})
                elif kind == "phase" and event["phase"] in PHASES:
                    self.phase = event["phase"]
                    self.progress = max(self.progress, PHASES[self.phase][1])
                    self.remaining = None
                elif kind == "decode":
                    self.progress = max(self.progress, 0.10 + 0.10 * event["progress"])
                elif kind == "progress" and self.phase == "transcribe":
                    self._audio_progress(event["progress"], active_elapsed(job, event["time"]))

    def _audio_progress(self, fraction, current):
        fraction = max(0.0, min(1.0, fraction))
        if fraction <= self.audio_fraction:
            return
        self.audio_fraction = fraction
        self.progress = max(self.progress, 0.20 + 0.75 * fraction)
        if fraction == 1:
            self.remaining = None
            return
        # Use the rate since the first usable sample to exclude model startup.
        if self.sample is None:
            self.sample = (current, fraction)
            return
        previous, start_fraction = self.sample
        if current > previous:
            self.remaining = (current - previous) * (1 - fraction) / (fraction - start_fraction)


class JobManager:
    def __init__(self, parallel=1):
        self.lock = threading.RLock()
        self.jobs = {}
        self.events = {}
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
        job = {"id": identity, "source": str(source), "display_name": display_name or source.name, "engine": engine, "model": copy.deepcopy(model), "profile": profile, "general": copy.deepcopy(general), "out_dir": str(out_dir), "out_name": name, "folder": str(folder), "mode": mode, "subtitle_stream": subtitle_stream, "channel": channel, "status": "queued", "progress": 0.0, "created_at": time.time(), "logs": []}
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
        result, status, error = {}, "error", ""
        try:
            if cancelled():
                raise InterruptedError()
            self._patch(identity, status="preparing", started_at=time.monotonic())
            db = Database()
            session = db.create_session({"selected_model": job["model"]["name"] if job["mode"] != "extract" else None, "device": job["profile"].get("device", "cpu"), "language": job["general"]["language"], "task": job["general"]["task"], "engine": job["engine"] if job["mode"] != "extract" else "extract", "profile": job["profile"]}, status="running")
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
            # Runtime-only objects and tokens never go into the job document.
            persisted = {k: v for k, v in job.items() if k not in {"process", "logs"}}
            write_json(folder / "job.json", persisted)
            env = child_env()
            if token:
                env["HF_TOKEN"] = token
            with open(folder / "worker.log", "w", encoding="utf-8") as worker_log:
                process = subprocess.Popen([python, str(ROOT / "engine/worker.py"), str(folder / "job.json")], cwd=ROOT, env=env, stdout=worker_log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            self._patch(identity, process=process, status="running", stage="بدء المعالجة")
            while process.poll() is None:
                if cancelled():
                    stop_tree(process)
                    break
                time.sleep(0.2)
            result = read_json(folder / "result.json", {})
            status = result.get("status", "error")
            error = result.get("error", "")
            if status == "error" and not error and not cancelled():
                error = (folder / "worker.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        except InterruptedError:
            status = "cancelled"
        except Exception as exc:
            status, error = "error", str(exc)
        with self.lock:
            if cancelled():
                status = job.get("cancel_status", "cancelled")
            ended_at = time.monotonic()
            elapsed = active_elapsed(job, ended_at)
            if token:
                error = error.replace(token, "[رمز مخفي]")
            if session is not None:
                try:
                    model = job["model"]["name"] if job["mode"] != "extract" else None
                    duration = result.get("duration", 0)
                    segments = len(result.get("segments", []))
                    db.add_file(session, {"file_name": job["display_name"], "file_path": job["source"], "model": model, "language": result.get("language", ""), "duration": duration, "segments": segments, "elapsed": elapsed, "output_dir": job["out_dir"], "status": status, "error_message": error, "text_preview": result.get("text", "")[:500]})
                    db.update_session(session, status=status, total_files=1, total_duration=duration, total_elapsed=elapsed, total_segments=segments)
                except Exception as exc:
                    log("تعذّر حفظ السجل: " + str(exc))
            self._patch(identity, status=status, result=result, error=error, ended_at=ended_at)

    def snapshots(self):
        with self.lock:
            snapshots = []
            for job in self.jobs.values():
                item = {k: copy.deepcopy(v) for k, v in job.items() if k != "process"}
                if job["id"] not in self.events:
                    self.events[job["id"]] = JobEvents()
                events = self.events[job["id"]]
                events.read(job)
                item["live_segments"] = events.segments.copy()
                item["elapsed"] = active_elapsed(job, time.monotonic())
                item["logs"].extend(events.logs)
                if events.phase:
                    item["stage"] = PHASES[events.phase][0]
                complete = job["status"] in {"completed", "no_speech"}
                item["progress"] = 1.0 if complete else max(job.get("progress", 0), events.progress)
                item["remaining"] = 0.0 if complete else None if job["status"] in TERMINAL else events.remaining
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
            if not process or job["status"] != ("running" if paused else "paused"):
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
                    job.setdefault("pauses", []).append((job.pop("paused_at"), time.monotonic()))
                else:
                    job["paused_at"] = time.monotonic()
                job["status"] = "paused" if paused else "running"
            except psutil.Error:
                pass
