
"""
كتابة مخرجات التفريغ بصيغ مختلفة
"""
import os
import json
import csv
import logging
import tempfile
import shutil
import html
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)


def format_srt_time(seconds: float) -> str:
    """تحويل الثواني لصيغة SRT: HH:MM:SS,mmm"""
    total = max(0, round(seconds * 1000))
    seconds, ms = divmod(total, 1000)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    """تحويل الثواني لصيغة VTT: HH:MM:SS.mmm"""
    return format_srt_time(seconds).replace(",", ".")


def format_ass_time(seconds: float) -> str:
    """تحويل الثواني لصيغة ASS: H:MM:SS.cc"""
    seconds, cs = divmod(max(0, round(seconds * 100)), 100)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def write_srt(out_path: str, segments: List[Dict]):
    """كتابة ملف SRT"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_time(seg["start"])
        end = format_srt_time(seg["end"])
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.debug(f"Written SRT: {out_path}")


def write_txt(out_path: str, segments: List[Dict]):
    """كتابة ملف TXT"""
    lines = [seg["text"].strip() for seg in segments]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.debug(f"Written TXT: {out_path}")


def write_vtt(out_path: str, segments: List[Dict]):
    """كتابة ملف VTT"""
    lines = ["WEBVTT\n\n"]
    for seg in segments:
        start = format_vtt_time(seg["start"])
        end = format_vtt_time(seg["end"])
        text = html.escape(seg["text"].strip(), quote=False)
        lines.append(f"{start} --> {end}\n{text}\n\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    logger.debug(f"Written VTT: {out_path}")


def write_json(out_path: str, segments: List[Dict]):
    """كتابة ملف JSON"""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    logger.debug(f"Written JSON: {out_path}")


def ass_style(config):
    def color(value):
        value = value.lstrip("#")
        return "&H00" + value[4:6] + value[2:4] + value[0:2]
    return {"font_name": config.get("ass_font_name", "Arial"), "font_size": config.get("ass_font_size", 16), "primary_color": color(config.get("ass_text_color", "#FFFFFF")), "outline_color": color(config.get("ass_outline_color", "#000000")), **{key: config.get("ass_" + key, default) for key, default in {"bold": False, "italic": False, "outline": 2, "shadow": 1, "alignment": 2, "margin_l": 10, "margin_r": 10, "margin_v": 10, "width": 384, "height": 288}.items()}}


def write_ass(out_path: str, segments: List[Dict], style_config: Dict = None):
    """كتابة ملف ASS"""
    style = style_config or {}
    font_name = style.get("font_name", "Arial")
    font_size = style.get("font_size", 16)
    primary_color = style.get("primary_color", "&H00FFFFFF")

    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {style.get('width', 384)}\n"
        f"PlayResY: {style.get('height', 288)}\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default, {font_name}, {font_size}, {primary_color}, "
        f"&H000000FF, {style.get('outline_color', '&H00000000')}, &H00000000, {-1 if style.get('bold') else 0}, {-1 if style.get('italic') else 0}, 0, 0, 100, 100, 0, 0, "
        f"1, {style.get('outline', 2)}, {style.get('shadow', 1)}, {style.get('alignment', 2)}, {style.get('margin_l', 10)}, {style.get('margin_r', 10)}, {style.get('margin_v', 10)}, 1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    dialogue_lines = []
    for seg in segments:
        start = format_ass_time(seg["start"])
        end = format_ass_time(seg["end"])
        text = seg["text"].strip().replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        speaker = seg.get("speaker", "")
        dialogue_lines.append(
            f"Dialogue: 0,{start},{end},Default,{speaker},0,0,0,,{text}"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(dialogue_lines))
    logger.debug(f"Written ASS: {out_path}")


def write_csv(out_path: str, segments: List[Dict]):
    """كتابة ملف CSV"""
    fieldnames = ["id", "start", "end", "text", "words", "speaker"]
    # إزالة الحقول غير الموجودة
    available_fields = [f for f in fieldnames if any(f in seg for seg in segments)]
    if not available_fields:
        available_fields = ["id", "start", "end", "text"]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=available_fields, extrasaction="ignore")
        writer.writeheader()
        for i, seg in enumerate(segments, 1):
            row = {
                "id": i,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": seg.get("text", "").strip(),
                "words": " ".join(w.get("word", "") for w in seg.get("words", [])),
                "speaker": seg.get("speaker", ""),
            }
            writer.writerow({k: row.get(k, "") for k in available_fields})

    logger.debug(f"Written CSV: {out_path}")


def write_all_outputs(out_dir: str, base_name: str, segments: List[Dict], 
                      output_config: Dict, style_config: Dict = None) -> List[str]:
    """
    كتابة جميع المخرجات المطلوبة

    Args:
        out_dir: مجلد الإخراج
        base_name: اسم الملف الأساسي (بدون امتداد)
        segments: قائمة المقاطع
        output_config: إعدادات الإخراج (out_srt, out_txt, ...)
        style_config: إعدادات التنسيق (لـ ASS)

    Returns:
        قائمة مسارات الملفات المكتوبة
    """
    os.makedirs(out_dir, exist_ok=True)
    written_files = []

    suffix = output_config.get("output_suffix", "").strip()
    if any(c in base_name + suffix for c in '/\\\x00') or base_name in {".", ".."}:
        raise ValueError("اسم إخراج غير صالح")
    out_name = f"{base_name}{suffix}" if suffix else base_name

    writers = {"srt": write_srt, "txt": write_txt, "vtt": write_vtt, "json": write_json, "ass": write_ass, "csv": write_csv}
    formats = [fmt for fmt in writers if output_config.get("out_" + fmt)]
    if not formats:
        raise ValueError("اختر صيغة إخراج واحدة على الأقل")
    original = out_name
    counter = 0
    reservation = None
    while True:
        out_name = original + (f"_{counter}" if counter else "")
        if not output_config.get("overwrite", False) and any(os.path.exists(os.path.join(out_dir, out_name + "." + fmt)) for fmt in formats):
            counter += 1
            continue
        reservation = os.path.join(out_dir, out_name + ".whisper-output.lock")
        try:
            fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            break
        except FileExistsError:
            counter += 1
    staging = tempfile.mkdtemp(prefix=".whisper-output-", dir=out_dir)
    try:
        for fmt in formats:
            temporary = os.path.join(staging, out_name + "." + fmt)
            if fmt == "ass":
                writers[fmt](temporary, segments, style_config)
            else:
                writers[fmt](temporary, segments)
        for fmt in formats:
            path = os.path.join(out_dir, out_name + "." + fmt)
            os.replace(os.path.join(staging, out_name + "." + fmt), path)
            written_files.append(path)
        return written_files
    finally:
        if Path(staging).resolve().parent != Path(out_dir).resolve():
            raise RuntimeError("Unexpected output staging directory")
        shutil.rmtree(staging)
        if reservation:
            os.unlink(reservation)
