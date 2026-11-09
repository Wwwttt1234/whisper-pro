"""Decode media with bundled PyAV libraries; no system FFmpeg for ASR."""
import re
import shutil
import subprocess
import wave
from pathlib import Path

MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".flv", ".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".mpeg", ".mpg"}


def inspect_media(path):
    import av
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("ملف الوسائط غير موجود أو فارغ")
    with av.open(str(path)) as container:
        streams = [{"index": s.index, "type": s.type, "codec": s.codec_context.name, "language": s.metadata.get("language", ""), "title": s.metadata.get("title", ""), "channels": getattr(s.codec_context, "channels", None)} for s in container.streams if s.type in {"audio", "video", "subtitle"}]
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        return {"duration": duration, "streams": streams, "size": path.stat().st_size}


def prepare_wav(source, destination, audio_stream=0, channel=None, checkpoint=None, progress=None):
    import av
    import numpy as np
    with av.open(str(source)) as container:
        if audio_stream >= len(container.streams.audio):
            raise ValueError("مسار الصوت المختار غير موجود")
        stream = container.streams.audio[audio_stream]
        if channel is not None and channel >= stream.codec_context.channels:
            raise ValueError("الملف لا يحتوي القناة المطلوبة")
        layout = "stereo" if channel is not None else "mono"
        resampler = av.AudioResampler(format="s16p", layout=layout, rate=16000)
        samples = 0
        duration = float(container.duration / av.time_base) if container.duration else 0
        with wave.open(str(destination), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            def write(frames):
                nonlocal samples
                for frame in frames:
                    array = frame.to_ndarray()
                    values = array[channel if channel is not None else 0]
                    writer.writeframes(np.asarray(values, dtype="<i2").tobytes())
                    samples += len(values)
            for frame in container.decode(stream):
                if checkpoint:
                    checkpoint()
                write(resampler.resample(frame))
                if progress and duration:
                    progress(min(samples / 16000 / duration, 1.0))
            write(resampler.resample(None))
    return samples / 16000


def audio_chunks(path, seconds=30, checkpoint=None):
    import numpy as np
    with wave.open(str(path), "rb") as reader:
        rate = reader.getframerate()
        offset = 0
        while True:
            if checkpoint:
                checkpoint()
            raw = reader.readframes(int(rate * seconds))
            if not raw:
                break
            array = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
            yield array, offset / rate, len(array) / rate
            offset += len(array)


def read_srt(text):
    segments = []
    pattern = r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})[^\n]*\n([\s\S]*?)(?=\n\s*\n|$)"
    for match in re.finditer(pattern, text.replace("\r\n", "\n")):
        vals = [int(x) for x in match.groups()[:8]]
        start = vals[0] * 3600 + vals[1] * 60 + vals[2] + vals[3] / 1000
        end = vals[4] * 3600 + vals[5] * 60 + vals[6] + vals[7] / 1000
        segments.append({"start": start, "end": end, "text": match[9].strip(), "words": []})
    return segments


def extract_subtitles(source, stream_index, work_dir):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("استخراج المسار المضمن يحتاج FFmpeg. التفريغ الصوتي لا يحتاجه؛ يمكن اختيار مسار ffmpeg جاهز ضمن النظام.")
    info = inspect_media(source)
    stream = next((s for s in info["streams"] if s["index"] == stream_index and s["type"] == "subtitle"), None)
    if not stream:
        raise ValueError("مسار الترجمة المختار غير موجود")
    if stream["codec"] in {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"}:
        raise ValueError("هذا مسار ترجمة صوري يحتاج OCR؛ لا يمكن تحويله إلى نص بتفريغ الصوت")
    target = Path(work_dir) / "embedded.srt"
    result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-i", str(source), "-map", f"0:{stream_index}", "-c:s", "srt", "-y", str(target)], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return read_srt(target.read_text(encoding="utf-8-sig"))
