"""Versioned user preferences. Tokens are deliberately session-only."""
import copy
import json
import math
import re

from core.paths import ROOT, data_dir
from core.storage import read_json, write_json
from engine.specs import ENGINES, FIELDS, fields_for, defaults_for

GENERAL = {
    "language": "auto", "task": "transcribe", "formats": ["srt"],
    "output_dir": "", "beside_source": True, "output_suffix": "",
    "overwrite": False, "parallel_files": 1, "max_line_length": 42,
    "max_lines": 2, "max_cue_seconds": 7.0, "min_cue_seconds": 0.5,
    "time_offset": 0.0, "split_stereo": False, "audio_stream": 0,
    "auto_check_models": True, "catalog_interval_hours": 24,
    "models_dir": "", "translation_engine": "none", "translation_model": "",
    "target_language": "Arabic", "translation_context": "", "glossary": "",
    "translation_temperature": 0.0, "translation_max_tokens": 1024,
    "translation_batch": 8, "translation_service_url": "http://127.0.0.1:8080",
    "alignment": "none", "alignment_language": "en-us", "theme": "dark",
    "post_device": "cpu", "post_dtype": "float32", "llama_binary": "",
    "llama_context": 8192, "llama_gpu_layers": 0, "translation_threads": 4,
    "output_subfolder": False, "output_subfolder_name": "{original_name}_output",
    "ass_font_name": "Arial", "ass_font_size": 16, "ass_text_color": "#FFFFFF",
    "ass_outline_color": "#000000", "ass_bold": False, "ass_italic": False,
    "ass_outline": 2.0, "ass_shadow": 1.0, "ass_alignment": 2,
    "ass_margin_l": 10, "ass_margin_r": 10, "ass_margin_v": 10,
    "ass_width": 384, "ass_height": 288,
}


def initial_settings():
    return {"schema_version": 2, "engine": "faster_whisper", "selection": {}, "profiles": {}, "general": copy.deepcopy(GENERAL)}


def validate_field(field, value):
    expected = type(field.default)
    if expected is bool:
        if type(value) is not bool:
            raise ValueError(f"{field.label}: يلزم اختيار نعم أو لا")
    elif expected in (int, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field.label}: يلزم رقم محدود")
        if expected is int and int(value) != value:
            raise ValueError(f"{field.label}: يلزم عدد صحيح")
        value = expected(value)
        if field.minimum is not None and value < field.minimum or field.maximum is not None and value > field.maximum:
            raise ValueError(f"{field.label}: القيمة خارج النطاق")
    elif not isinstance(value, str):
        raise ValueError(f"{field.label}: يلزم نص")
    if field.choices and value not in field.choices:
        raise ValueError(f"{field.label}: اختيار غير متاح")
    return value


def validate_profile(engine, model, values):
    result = defaults_for(engine, model)
    for field in fields_for(engine, model):
        if field.key in values:
            result[field.key] = validate_field(field, values[field.key])
    if result.get("vad_filter") and result.get("vad_neg_threshold", 0) >= result.get("vad_threshold", 1):
        raise ValueError("عتبة نهاية الكلام يجب أن تقل عن حساسية وجود الكلام")
    if result.get("min_speakers", 1) > result.get("max_speakers", 32):
        raise ValueError("أقل عدد متحدثين يتجاوز الحد الأعلى")
    if result.get("max_len") and not result.get("token_timestamps"):
        raise ValueError("تحديد طول مقطع whisper.cpp يحتاج تفعيل توقيت الرموز")
    if result.get("do_sample") and result.get("temperature", 0) <= 0:
        raise ValueError("أخذ العينات يحتاج حرارة أكبر من صفر، مثل0.2")
    if result.get("hotwords_file") and result.get("decoding_method") != "modified_beam_search":
        raise ValueError("مصطلحات Sherpa تحتاج modified_beam_search")
    for key in ("temperature_list", "clip_timestamps", "suppress_tokens"):
        if result.get(key, "").strip():
            try:
                vals = [float(s.strip()) for s in result[key].split(",")]
                if not all(math.isfinite(v) for v in vals):
                    raise ValueError()
                if key == "temperature_list" and any(not 0 <= v <= 1 for v in vals):
                    raise ValueError()
                if key == "suppress_tokens" and any(int(v) != v for v in vals):
                    raise ValueError()
                if key == "clip_timestamps" and (len(vals) % 2 or any(a < 0 or b <= a for a, b in zip(vals[::2], vals[1::2]))):
                    raise ValueError()
            except ValueError:
                raise ValueError(f"قائمة غير صالحة: {key}") from None
    if result.get("grammar"):
        grammar = json.loads(result["grammar"])
        if not isinstance(grammar, list) or not all(isinstance(s, str) for s in grammar):
            raise ValueError("قاموس Vosk يجب أن يكون قائمة عبارات نصية")
    return result


def validate_settings(value):
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise ValueError("ملف إعدادات غير مدعوم؛ استخدم ملفًا صدّره هذا الإصدار")
    clean = initial_settings()
    if value.get("engine") not in ENGINES:
        raise ValueError("محرك غير معروف")
    clean["engine"] = value["engine"]
    for name in ("selection", "profiles", "general"):
        if not isinstance(value.get(name, {}), dict):
            raise ValueError(f"قسم إعدادات غير صالح: {name}")
    for key, default in GENERAL.items():
        item = value.get("general", {}).get(key, copy.deepcopy(default))
        if isinstance(default, float):
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"قيمة غير صالحة: {key}")
            item = float(item)
        elif type(item) is not type(default):
            raise ValueError(f"نوع غير صالح: {key}")
        clean["general"][key] = item
    general = clean["general"]
    if not general["formats"] or any(v not in {"srt", "vtt", "ass", "json", "csv", "txt"} for v in general["formats"]):
        raise ValueError("اختر صيغة إخراج صحيحة واحدة على الأقل")
    for key, low, high in [("parallel_files", 1, 8), ("max_line_length", 10, 200), ("max_lines", 1, 5), ("max_cue_seconds", 1, 60), ("min_cue_seconds", 0, 10), ("translation_batch", 1, 64), ("translation_max_tokens", 64, 8192), ("catalog_interval_hours", 1, 720), ("audio_stream", 0, 100)]:
        if not low <= general[key] <= high:
            raise ValueError(f"قيمة خارج الحدود: {key}")
    if general["translation_engine"] not in {"none", "transformers", "llama"} or general["alignment"] not in {"none", "bfa", "whisperx"}:
        raise ValueError("إضافة غير معروفة")
    if general["post_device"] not in {"cpu", "cuda"} or general["post_dtype"] not in {"float32", "float16", "bfloat16"}:
        raise ValueError("جهاز أو دقة مرحلة إضافية غير صالحة")
    if not 512 <= general["llama_context"] <= 131072 or not 0 <= general["llama_gpu_layers"] <= 999 or not 1 <= general["translation_threads"] <= 256:
        raise ValueError("قيمة إعداد llama.cpp خارج الحدود")
    for key, low, high in [("ass_font_size", 8, 200), ("ass_outline", 0, 20), ("ass_shadow", 0, 20), ("ass_alignment", 1, 9), ("ass_margin_l", 0, 2000), ("ass_margin_r", 0, 2000), ("ass_margin_v", 0, 2000), ("ass_width", 128, 7680), ("ass_height", 128, 4320)]:
        if not low <= general[key] <= high:
            raise ValueError("قيمة تنسيق ASS خارج الحدود: " + key)
    if not general["ass_font_name"].strip() or any(c in general["ass_font_name"] for c in ",\r\n\x00"):
        raise ValueError("اسم خط ASS غير صالح")
    if any(not re.fullmatch(r"#[0-9a-fA-F]{6}", general[key]) for key in ["ass_text_color", "ass_outline_color"]):
        raise ValueError("لون ASS غير صالح")
    known = {field.key: field for field in FIELDS}
    for engine, profiles in value.get("profiles", {}).items():
        if engine not in ENGINES or not isinstance(profiles, dict):
            raise ValueError("قسم محرك غير صالح")
        clean["profiles"][engine] = {}
        for model_key, profile in profiles.items():
            if not isinstance(model_key, str) or not isinstance(profile, dict):
                raise ValueError("إعدادات نموذج غير صالحة")
            clean["profiles"][engine][model_key] = {key: validate_field(known[key], val) for key, val in profile.items() if key in known}
    for engine, model_id in value.get("selection", {}).items():
        if engine in ENGINES and isinstance(model_id, str):
            clean["selection"][engine] = model_id
    return clean


def migrate_legacy(old):
    result = initial_settings()
    profile = {field.key: old[field.key] for field in FIELDS if field.key in old}
    model = old.get("selected_model", "large-v3-turbo")
    aliases = {"large": "large-v3", "turbo": "large-v3-turbo"}
    model = aliases.get(model, model)
    repo = "Systran/faster-whisper-" + model
    if model in {"large-v3-turbo", "turbo"}:
        repo = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    if model.startswith("distil-"):
        repo = "Systran/faster-distil-whisper-" + model.removeprefix("distil-")
    result["selection"]["faster_whisper"] = "hf:" + repo
    result["profiles"]["faster_whisper"] = {"hf:" + repo: profile}
    result["general"]["formats"] = [fmt for fmt in ["srt", "txt", "vtt", "json", "ass", "csv"] if old.get("out_" + fmt)] or ["srt"]
    result["general"]["output_dir"] = old.get("custom_out_dir", "")
    result["general"]["output_suffix"] = old.get("output_suffix", "")
    result["general"]["split_stereo"] = bool(old.get("audio_split_stereo", False))
    return result


def load_settings():
    path = data_dir() / "settings.json"
    if path.exists():
        return validate_settings(read_json(path))
    legacy = read_json(ROOT / "whisper_config.json", {})
    return validate_settings(migrate_legacy(legacy)) if isinstance(legacy, dict) else initial_settings()


def save_settings(settings):
    clean = validate_settings(settings)
    write_json(data_dir() / "settings.json", clean)
    return clean


def export_settings(settings):
    return json.dumps(validate_settings(settings), ensure_ascii=False, indent=2, allow_nan=False)


def import_settings(text):
    return validate_settings(json.loads(text))
