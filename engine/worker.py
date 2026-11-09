"""Subprocess entry point: no Streamlit state and no tokens in job files."""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.storage import read_json, write_json
from core.media import prepare_wav, extract_subtitles
from core.outputs import write_all_outputs, ass_style
from core.subtitles import layout_segments


def execute(job_file):
    job_file = Path(job_file)
    job = read_json(job_file)
    folder = job_file.parent
    events = folder / "events.jsonl"
    control = folder / "control.json"
    started = time.time()
    last_percent = {}

    def emit(kind, **values):
        if kind in {"decode", "progress"}:
            percent = int(100 * max(0, min(1, values["progress"])))
            if percent <= last_percent.get(kind, -1):
                return
            last_percent[kind] = percent
        entry = {"kind": kind, "time": time.monotonic(), **values}
        with open(events, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")

    def checkpoint():
        state = read_json(control, {})
        if state.get("cancel"):
            raise InterruptedError("ألغى المستخدم المهمة")
        while state.get("pause"):
            time.sleep(0.15)
            state = read_json(control, {})
            if state.get("cancel"):
                raise InterruptedError("ألغى المستخدم المهمة")

    try:
        general, profile = job["general"], job["profile"]
        if profile.get("local_files_only"):
            from core.offline import enable
            enable()
        checkpoint()
        if job.get("mode") == "extract":
            emit("phase", phase="extract")
            segments = extract_subtitles(job["source"], job["subtitle_stream"], folder)
            result = {"segments": segments, "language": general.get("language", "unknown"), "duration": max((s["end"] for s in segments), default=0), "timestamp_quality": "embedded"}
        else:
            from engine.adapters import transcribe
            wav = folder / "audio.wav"
            emit("phase", phase="decode")
            duration = prepare_wav(job["source"], wav, general.get("audio_stream", 0), job.get("channel"), checkpoint, lambda p: emit("decode", progress=p))
            checkpoint()
            emit("phase", phase="transcribe")
            result = transcribe(job["engine"], job["model_path"], job["model"], profile, general, wav, folder, emit, checkpoint, binary=job.get("native_binary"))
            result["duration"] = duration
            write_json(folder / "raw_transcript.json", result)
            if general.get("alignment", "none") != "none" or general.get("translation_engine", "none") != "none":
                from engine.postprocess import run_stages
                emit("phase", phase="postprocess")
                result = run_stages(result, job, wav, folder, emit, checkpoint)
        checkpoint()
        emit("phase", phase="export")
        segments, estimated = layout_segments(result["segments"], general)
        formats = {"out_" + fmt: True for fmt in general["formats"]}
        formats.update(output_suffix=general.get("output_suffix", ""), overwrite=general.get("overwrite", False))
        paths = write_all_outputs(job["out_dir"], job["out_name"], segments, formats, ass_style(general))
        result.update(segments=segments, elapsed=time.time() - started, written_files=paths, text=" ".join(s["text"].replace("\n", " ") for s in segments), estimated_splits=estimated, status="completed" if segments else "no_speech")
        write_json(folder / "result.json", result)
        emit("done", status=result["status"], progress=1.0)
        return 0
    except InterruptedError as exc:
        write_json(folder / "result.json", {"status": "cancelled", "error": str(exc), "elapsed": time.time() - started})
        emit("done", status="cancelled")
        return 2
    except Exception as exc:
        write_json(folder / "result.json", {"status": "error", "error": str(exc), "elapsed": time.time() - started})
        emit("error", message=str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(execute(sys.argv[1]))
