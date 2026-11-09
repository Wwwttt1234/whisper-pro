import json
import wave
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from core.outputs import write_all_outputs, format_srt_time, format_ass_time
from core.storage import extract_archive
from core.subtitles import layout_segments
from core.media import prepare_wav, audio_chunks, inspect_media, read_srt
from engine.adapters import native_segments
from engine.postprocess import translated_items


def test_six_formats_and_existing_file_protected(tmp_path):
    original = tmp_path / "clip.srt"
    original.write_text("user subtitle", encoding="utf-8")
    rows = [{"start": 0, "end": 2.25, "text": "نص عربي\nالسطر الثاني", "words": []}]
    files = write_all_outputs(str(tmp_path), "clip", rows, {"out_" + fmt: True for fmt in ["srt", "vtt", "ass", "txt", "json", "csv"]})
    assert original.read_text(encoding="utf-8") == "user subtitle"
    assert len(files) == 6 and all(Path(f).stat().st_size for f in files)
    assert all(Path(f).stem == "clip_1" for f in files)
    assert json.loads((tmp_path / "clip_1.json").read_text(encoding="utf-8"))[0]["text"] == rows[0]["text"]
    assert "\\N" in (tmp_path / "clip_1.ass").read_text(encoding="utf-8")
    assert read_srt((tmp_path / "clip_1.srt").read_text(encoding="utf-8"))[0]["end"] == 2.25


def test_concurrent_exports_reserve_different_names(tmp_path):
    row = [{"start": 0, "end": 1, "text": "test"}]
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda _: write_all_outputs(str(tmp_path), "same", row, {"out_srt": True})[0], range(3)))
    assert len(set(paths)) == 3


def test_time_rounding_carries_to_next_second():
    assert format_srt_time(59.9999) == "00:01:00,000"
    assert format_ass_time(59.9999) == "0:01:00.00"
    assert format_srt_time(-1) == "00:00:00,000"


def test_archive_cannot_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../escape.txt", "bad")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "target")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("target, retain", [
    ("https://huggingface.co/weights", True),
    ("https://huggingface.co:8443/weights", False),
    ("https://cdn.example.org/weights", False),
    ("http://huggingface.co/weights", None),
])
def test_model_redirect_protects_credentials_and_tls(target, retain):
    from urllib.request import Request
    from core.storage import _SafeRedirect

    request = Request("https://huggingface.co/model", headers={"Authorization": "Bearer fixture"})
    handler = _SafeRedirect()
    if retain is None:
        with pytest.raises(ValueError, match="HTTPS"):
            handler.redirect_request(request, None, 302, "Found", {}, target)
    else:
        redirected = handler.redirect_request(request, None, 302, "Found", {}, target)
        assert ("Authorization" in redirected.headers) is retain


def test_stereo_decoding_and_timing(tmp_path):
    import numpy as np
    source = tmp_path / "stereo.wav"
    samples = np.column_stack([np.full(16000, 1000, dtype="int16"), np.full(16000, 2000, dtype="int16")])
    with wave.open(str(source), "wb") as stream:
        stream.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
        stream.writeframes(samples.astype("<i2").tobytes())
    target = tmp_path / "left.wav"
    duration = prepare_wav(source, target, channel=0)
    assert duration == pytest.approx(1)
    chunks = list(audio_chunks(target, seconds=0.25))
    assert len(chunks) == 4
    assert chunks[-1][1] == pytest.approx(0.75)
    assert chunks[0][0].mean() == pytest.approx(1000 / 32768, abs=0.001)
    assert inspect_media(source)["streams"][0]["channels"] == 2


def test_native_timestamp_units_and_words():
    data = {"transcription": [{"offsets": {"from": 1000, "to": 2000}, "text": "hello", "words": [{"text": "hello", "t0": 100, "t1": 200}], "speaker": "S1"}]}
    value = native_segments(data)[0]
    assert value["start"] == 1 and value["end"] == 2
    assert value["words"][0]["start"] == 1 and value["speaker"] == "S1"


def test_translation_must_preserve_all_ids():
    assert translated_items('[{"id": 0, "text": "مرحبا"}]', [0]) == ["مرحبا"]
    with pytest.raises(ValueError):
        translated_items('[{"id": 1, "text": "wrong"}]', [0])
    with pytest.raises(ValueError):
        translated_items('[]', [0])


def test_word_layout_and_offset_preserve_timing():
    row = {"start": 1, "end": 3, "text": "hello world", "words": [{"word": "hello", "start": 1, "end": 1.5}, {"word": "world", "start": 2, "end": 3}]}
    values, estimated = layout_segments([row], {"max_line_length": 10, "max_lines": 1, "time_offset": 2})
    assert len(values) == 2 and not estimated
    assert values[0]["start"] == 3 and values[1]["start"] == 4


def test_incomplete_alignment_never_drops_transcript():
    row = {"start": 0, "end": 3, "text": "Hello number 123 world", "words": [{"word": "Hello", "start": 0, "end": 1}, {"word": "world", "start": 2, "end": 3}]}
    values, estimated = layout_segments([row], {"max_line_length": 10, "max_lines": 1})
    assert " ".join(v["text"] for v in values) == row["text"]
    assert estimated and not any(v["words"] for v in values)


def test_word_wrapping_respects_lines_and_chinese_spacing():
    text = "你好世界"
    row = {"start": 0, "end": 4, "text": text, "words": [{"word": char, "start": i, "end": i + 1} for i, char in enumerate(text)]}
    values, _ = layout_segments([row], {"max_line_length": 2, "max_lines": 1})
    assert "".join(v["text"] for v in values) == text
    assert len(values) == 2
    text = "aaaaaa bbbbbb cccccc dddddd"
    row = {"start": 0, "end": 4, "text": text, "words": [{"word": char, "start": i, "end": i + 1} for i, char in enumerate(text.split())]}
    values, _ = layout_segments([row], {"max_line_length": 10, "max_lines": 2})
    assert all(len(v["text"].splitlines()) <= 2 for v in values)


def test_bfa_uses_absolute_word_times_and_keeps_skipped_segments(monkeypatch):
    import sys
    import types
    from engine.postprocess import align
    original = [{"start": 0, "end": 1, "text": "missing"}, {"start": 2, "end": 3, "text": "hello"}]
    class FakeBFA:
        def __init__(self, **kwargs):
            pass
        def load_audio(self, path):
            return None
        def process_segments(self, *args):
            return [{"segments": [{**original[1], "words_ts": [{"word": "hello", "start_ms": 2100, "end_ms": 2800}]}]}]
    monkeypatch.setitem(sys.modules, "bournemouth_aligner", types.SimpleNamespace(BFA=FakeBFA))
    result = align({"segments": original}, {"general": {"alignment": "bfa"}}, "unused")
    assert result["segments"][0]["text"] == "missing"
    assert result["segments"][1]["words"][0]["start"] == 2.1
    assert result["alignment_partial"]
