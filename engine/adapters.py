"""Lazy, local inference adapters returning one common segment representation."""
import json
import os
import subprocess
import threading
import time
import wave
import itertools
from pathlib import Path

from core.media import audio_chunks
from core.storage import read_json


def timed_word(word, start, end, **extra):
    return {"word": str(word), "start": float(start), "end": float(end), **extra}


def segment(start, end, text, words=None, **extra):
    return {"start": float(start), "end": float(end), "text": str(text).strip(), "words": words or [], **extra}


def native_segments(data):
    """Crisp/whisper.cpp offsets are ms; legacy t0/t1 are centiseconds."""
    def times(item):
        offsets = item.get("offsets") or {}
        if "from" in offsets:
            return float(offsets["from"]) / 1000, float(offsets["to"]) / 1000
        if "start" in item:
            return float(item["start"]), float(item["end"])
        return float(item.get("t0", 0)) / 100, float(item.get("t1", 0)) / 100
    result = []
    for item in data.get("transcription", data.get("segments", [])):
        start, end = times(item)
        words = []
        for word in item.get("words", []):
            a, b = times(word)
            words.append(timed_word(word.get("word", word.get("text", "")), a, b))
        result.append(segment(start, end, item.get("text", ""), words, **({"speaker": str(item["speaker"])} if item.get("speaker") is not None else {})))
    return result


def native_command(binary, model_path, model, profile, general, wav_path, output):
    caps_data = read_json(Path(__file__).with_name("crisp_capabilities.json"), {})
    from core.paths import location
    installed_caps = read_json(location("catalog") / "crisp_capabilities.json", caps_data)
    family = model.get("family", "whisper")
    caps = next((set(b["caps"]) for b in installed_caps.get("backends", []) if b["name"] == family), set())
    args = [str(binary), "-m", str(model_path), "-f", str(wav_path), "-ojf", "-of", str(output), "-t", str(profile.get("threads", 4)), "-l", general.get("language", "auto"), "--chunk-seconds", str(profile.get("chunk_seconds", 30)), "--cache-dir", str(location("models/native"))]
    if family:
        args += ["--backend", family]
    gpu = profile.get("native_gpu", "cpu")
    args += ["--no-gpu"] if gpu == "cpu" else ["--gpu-backend", gpu]
    if "beam-search" in caps:
        args += ["-bs", str(profile.get("beam_size", 5))]
    if "temperature" in caps:
        args += ["-tp", str(profile.get("temperature", 0))]
    if profile.get("native_target_language"):
        if "src-tgt-language" not in caps:
            raise ValueError("هذا النموذج لا يعلن دعم اختيار لغة هدف مباشرة")
        args += ["-sl", general.get("language", "auto"), "-tl", profile["native_target_language"]]
    elif general.get("task") == "translate":
        if "translate" not in caps or "turbo" in model.get("name", "").lower():
            raise ValueError("النموذج المختار لا يدعم ترجمة الكلام إلى الإنجليزية")
        args += ["--translate"]
    if family == "whisper":
        args += ["-nth", str(profile.get("no_speech_threshold", 0.6)), "-lpt", str(profile.get("log_prob_threshold", -1))]
        if profile.get("initial_prompt"):
            args += ["--prompt", profile["initial_prompt"]]
    elif profile.get("initial_prompt") and family in {"qwen3", "qwen3-1.7b", "vibevoice"}:
        args += ["--context", profile["initial_prompt"]]
    if profile.get("hotwords") and family in {"whisper", "parakeet", "qwen3", "qwen3-1.7b", "voxtral"}:
        args += ["--hotwords", profile["hotwords"]]
    if profile.get("max_new_tokens"):
        args += ["--max-new-tokens", str(profile["max_new_tokens"])]
    if profile.get("vad_filter"):
        if profile.get("local_files_only") and not profile.get("vad_model_path"):
            raise ValueError("اختر ملف VAD محليًا عند منع التنزيل")
        args += ["--vad", "--vad-threshold", str(profile.get("vad_threshold", 0.5)), "--vad-min-speech-duration-ms", str(profile.get("vad_min_speech_duration_ms", 0)), "--vad-min-silence-duration-ms", str(profile.get("vad_min_silence_duration_ms", 1000)), "--vad-speech-pad-ms", str(profile.get("vad_speech_pad_ms", 400))]
        if profile.get("vad_model_path"):
            args += ["--vad-model", profile["vad_model_path"]]
        if profile.get("vad_max_speech_duration_s"):
            args += ["--vad-max-speech-duration-s", str(profile["vad_max_speech_duration_s"])]
    if profile.get("aligner_path"):
        if profile.get("local_files_only") and profile["aligner_path"] == "auto":
            raise ValueError("اختر ملف محاذاة محليًا عند منع التنزيل")
        args += ["-am", profile["aligner_path"]]
        if not profile.get("local_files_only"):
            args += ["--auto-download"]
    if profile.get("diarization"):
        if "diarize" not in caps:
            raise ValueError("بناء المحرك لا يعلن دعم تمييز المتحدثين لهذا النموذج")
        args += ["--diarize", "--diarize-method", "foxnose", "--diarize-max-speakers", str(profile.get("max_speakers", 8))]
        if profile.get("local_files_only") and not profile.get("diarization_model_path"):
            raise ValueError("اختر نموذج تضمين متحدثين محليًا عند منع التنزيل")
        if profile.get("diarization_model_path"):
            args += ["--diarize-embedder", profile["diarization_model_path"]]
    return args


def whisper_kwargs(profile):
    allowed = {"beam_size", "best_of", "patience", "length_penalty", "repetition_penalty", "no_repeat_ngram_size", "temperature", "compression_ratio_threshold", "log_prob_threshold", "no_speech_threshold", "condition_on_previous_text", "prompt_reset_on_temperature", "initial_prompt", "prefix", "suppress_blank", "max_initial_timestamp", "word_timestamps", "prepend_punctuations", "append_punctuations", "multilingual", "language_detection_threshold", "language_detection_segments", "without_timestamps"}
    kwargs = {key: value for key, value in profile.items() if key in allowed}
    for key in ("initial_prompt", "prefix"):
        if key in kwargs:
            kwargs[key] = kwargs[key].strip() or None
    if profile.get("temperature_list", "").strip():
        kwargs["temperature"] = [float(v.strip()) for v in profile["temperature_list"].split(",")]
    if profile.get("suppress_tokens", "").strip():
        kwargs["suppress_tokens"] = [int(v.strip()) for v in profile["suppress_tokens"].split(",")]
    for key in ("hallucination_silence_threshold", "max_new_tokens"):
        if profile.get(key):
            kwargs[key] = profile[key]
    return kwargs


def transcribe(engine, model_path, model, profile, general, wav_path, work_dir, emit, checkpoint, binary=None):
    from engine.specs import supports_translation
    language = general.get("language", "auto")
    language = None if language == "auto" else language
    task = general.get("task", "transcribe")
    duration = 0
    with wave.open(str(wav_path), "rb") as reader:
        duration = reader.getnframes() / reader.getframerate()
    if task == "translate" and not supports_translation(engine, model):
        raise ValueError("النموذج لا يدعم مهمة ترجمة الكلام إلى الإنجليزية")
    if task == "translate" and engine not in {"faster_whisper", "crisp", "openai_whisper", "mlx", "whisper_cpp", "whisperx", "transformers", "sherpa"}:
        raise ValueError("استخدم التفريغ ثم مرحلة الترجمة المستقلة لهذا المحرك")
    if profile.get("local_files_only"):
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ["OMP_NUM_THREADS"] = str(profile.get("threads", 4))
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    emit("log", message="تحميل النموذج " + model.get("name", ""))
    detected, confidence, quality = language or "unknown", None, "segment"
    segments = []
    if engine == "faster_whisper":
        import ctranslate2
        from faster_whisper import WhisperModel
        if profile.get("random_seed", -1) >= 0:
            ctranslate2.set_random_seed(profile["random_seed"])
        device = profile.get("device", "cpu")
        compute = profile.get("compute_type", "int8")
        if device != "auto" and compute not in {"default", *ctranslate2.get_supported_compute_types(device, profile.get("device_index", 0))}:
            raise ValueError("دقة الحساب غير مدعومة على الجهاز المختار؛ جرّب INT8 على CPU")
        instance = WhisperModel(str(model_path), device=device, compute_type=compute, device_index=profile.get("device_index", 0), cpu_threads=profile.get("threads", 4), num_workers=profile.get("workers", 1), local_files_only=True)
        kwargs = whisper_kwargs(profile)
        kwargs.update(vad_filter=bool(profile.get("vad_filter")), hotwords=profile.get("hotwords") or None, chunk_length=profile.get("chunk_seconds", 30), clip_timestamps=profile.get("clip_timestamps", "").strip() or "0")
        if kwargs["vad_filter"]:
            kwargs["vad_parameters"] = {key.removeprefix("vad_"): val for key, val in profile.items() if key in {"vad_threshold", "vad_neg_threshold", "vad_min_speech_duration_ms", "vad_max_speech_duration_s", "vad_min_silence_duration_ms", "vad_speech_pad_ms"}}
            if not kwargs["vad_parameters"].get("max_speech_duration_s"):
                kwargs["vad_parameters"]["max_speech_duration_s"] = float("inf")
        generated, info = instance.transcribe(str(wav_path), language=language, task=task, **kwargs)
        detected, confidence = info.language, float(info.language_probability)
        for item in generated:
            checkpoint()
            words = [timed_word(w.word, w.start, w.end, probability=float(w.probability)) for w in (item.words or [])]
            segments.append(segment(item.start, item.end, item.text, words))
            emit("progress", progress=min(0.95, item.end / max(duration, 0.01)), text=item.text)
    elif engine == "crisp":
        output = Path(work_dir) / "native"
        args = native_command(binary, model_path, model, profile, general, wav_path, output)
        env = os.environ.copy()
        env["PATH"] = str(Path(binary).parent) + os.pathsep + env.get("PATH", "")
        if profile.get("kv_quant", "default") != "default":
            env["CRISPASR_KV_QUANT"] = profile["kv_quant"]
        process = subprocess.Popen(args, cwd=Path(binary).parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        logs = []
        def read():
            for line in process.stdout:
                logs.append(line.rstrip())
                emit("log", message=line.rstrip())
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                checkpoint()
                time.sleep(0.15)
        finally:
            if process.poll() is None:
                process.kill()
            reader.join(timeout=3)
        if process.returncode:
            raise RuntimeError("\n".join(logs[-15:]))
        data = json.loads(output.with_suffix(".json").read_text(encoding="utf-8-sig"))
        segments = native_segments(data)
        detected = (data.get("result") or {}).get("language", detected)
        quality = "word" if any(s["words"] for s in segments) else "segment"
    elif engine == "vosk":
        import vosk
        vosk.SetLogLevel(-1)
        instance = vosk.Model(str(model_path))
        recognizer = vosk.KaldiRecognizer(instance, 16000, profile["grammar"]) if profile.get("grammar") else vosk.KaldiRecognizer(instance, 16000)
        recognizer.SetWords(bool(profile.get("word_timestamps", True)))
        def append(result, fallback_start, fallback_end):
            if not result.get("text", "").strip():
                return
            words = [timed_word(w["word"], w["start"], w["end"], probability=w.get("conf")) for w in result.get("result", [])]
            segments.append(segment(words[0]["start"] if words else fallback_start, words[-1]["end"] if words else fallback_end, result["text"], words))
        with wave.open(str(wav_path), "rb") as reader:
            start = 0
            while raw := reader.readframes(4000):
                checkpoint()
                end = reader.tell() / 16000
                if recognizer.AcceptWaveform(raw):
                    append(json.loads(recognizer.Result()), start, end)
                    start = end
                emit("progress", progress=end / max(duration, 0.01))
            append(json.loads(recognizer.FinalResult()), start, duration)
        quality = "word" if profile.get("word_timestamps", True) else "segment"
    elif engine == "whisperx":
        import whisperx
        import numpy as np
        device = profile.get("device", "cpu")
        import torch
        if profile.get("random_seed", -1) >= 0:
            torch.manual_seed(profile["random_seed"])
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        options = {key: value for key, value in whisper_kwargs(profile).items() if key in {"beam_size", "initial_prompt", "log_prob_threshold", "no_speech_threshold"}}
        options["temperatures"] = [profile.get("temperature", 0)]
        instance = whisperx.load_model(str(model_path), device, device_index=profile.get("device_index", 0), compute_type=profile.get("compute_type", "int8"), language=language, task=task, threads=profile.get("threads", 4), vad_method="silero", vad_options={"vad_onset": profile.get("vad_threshold", 0.5), "vad_offset": profile.get("vad_offset", 0.363)}, asr_options=options)
        align_device = device + ":" + str(profile.get("device_index", 0)) if device == "cuda" else device
        audio = np.concatenate([a for a, _, _ in audio_chunks(wav_path, 120, checkpoint)])
        data = instance.transcribe(audio, batch_size=profile.get("batch_size", 4), language=language, chunk_size=profile.get("chunk_seconds", 30))
        detected = data.get("language", detected)
        checkpoint()
        if profile.get("align_words", True):
            from core.paths import location
            aligner, metadata = whisperx.load_align_model(language_code=detected, device=align_device, model_dir=str(location("models/alignment")))
            data = whisperx.align(data["segments"], aligner, metadata, audio, align_device, return_char_alignments=False)
            quality = "word"
        if profile.get("diarization"):
            from whisperx.diarize import DiarizationPipeline
            pipeline = DiarizationPipeline(token=os.environ.get("HF_TOKEN"), device=align_device)
            speakers = pipeline(audio, min_speakers=profile.get("min_speakers", 1), max_speakers=profile.get("max_speakers", 8))
            data = whisperx.assign_word_speakers(speakers, data)
        segments = data["segments"]
        for item in segments:
            item["words"] = [timed_word(w.get("word", ""), w["start"], w["end"]) for w in item.get("words", []) if "start" in w and "end" in w]
    else:
        infer = make_chunk_adapter(engine, model_path, model, profile, language, task)
        quality = "chunk"
        chunks = iter(audio_chunks(wav_path, profile.get("chunk_seconds", 30), checkpoint))
        def batches():
            while batch := list(itertools.islice(chunks, profile.get("batch_size", 4) if hasattr(infer, "batch") else 1)):
                checkpoint()
                values = infer.batch([entry[0] for entry in batch]) if hasattr(infer, "batch") else [infer(batch[0][0])]
                if len(values) != len(batch):
                    raise RuntimeError("لم يرجع المحرك نتيجة لكل مقطع في الدفعة")
                for (_, offset, length), data in zip(batch, values):
                    yield offset, length, data
        for offset, length, data in batches():
            if isinstance(data, str):
                data = {"text": data}
            detected = data.get("language", detected)
            chunk_segments = data.get("segments")
            if chunk_segments is None and data.get("chunks"):
                if profile.get("word_timestamps"):
                    words = [timed_word(c.get("text", ""), c.get("timestamp", (None, None))[0] or 0, c.get("timestamp", (None, None))[1] or length) for c in data["chunks"]]
                    chunk_segments = [segment(0, length, data.get("text", ""), words)]
                else:
                    chunk_segments = [{"start": c.get("timestamp", (None, None))[0] or 0, "end": c.get("timestamp", (None, None))[1] or length, "text": c.get("text", "")} for c in data["chunks"]]
                quality = "word" if profile.get("word_timestamps") else "segment"
            if not chunk_segments:
                chunk_segments = [segment(0, length, data.get("text", ""))]
            elif quality == "chunk":
                quality = "segment"
            for item in chunk_segments:
                words = [timed_word(w.get("word", w.get("text", "")), offset + float(w["start"]), offset + float(w["end"])) for w in item.get("words", []) if w.get("start") is not None and w.get("end") is not None]
                segments.append(segment(offset + max(0, float(item.get("start", 0))), offset + min(length, float(item.get("end", length))), item.get("text", ""), words, **({"speaker": str(item["speaker"])} if item.get("speaker") is not None else {})))
                if words:
                    quality = "word"
            emit("progress", progress=(offset + length) / max(duration, 0.01), text=data.get("text", ""))
    if any(item.get("words") for item in segments):
        quality = "word"
    return {"segments": segments, "language": detected, "confidence": confidence, "duration": duration, "timestamp_quality": quality}


def make_chunk_adapter(engine, model_path, model, profile, language, task):
    device = profile.get("device", "cpu")
    if engine in {"openai_whisper", "transformers", "qwen", "funasr", "nemo"}:
        import torch
        torch.set_num_threads(profile.get("threads", 4))
        if profile.get("random_seed", -1) >= 0:
            torch.manual_seed(profile["random_seed"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA غير متاح في بيئة هذا المحرك؛ اختر CPU أو بيئة تسريع مخصصة")
        if device == "cuda":
            device += ":" + str(profile.get("device_index", 0))
        dtype_name = profile.get("dtype", "float32")
        if device == "cpu" and dtype_name == "float16":
            raise ValueError("استخدم float32 على CPU لهذا المحرك")
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
    if engine == "openai_whisper":
        import whisper
        instance = whisper.load_model(str(model_path), device=device)
        kwargs = whisper_kwargs(profile)
        for key in ("repetition_penalty", "no_repeat_ngram_size", "multilingual", "language_detection_threshold", "language_detection_segments"):
            kwargs.pop(key, None)
        return lambda samples: instance.transcribe(samples, language=language, task=task, verbose=False, fp16=device.startswith("cuda") and dtype_name != "float32", **kwargs)
    if engine == "mlx":
        import mlx_whisper
        kwargs = whisper_kwargs(profile)
        for key in ("beam_size", "patience", "repetition_penalty", "no_repeat_ngram_size", "multilingual", "language_detection_threshold", "language_detection_segments", "hallucination_silence_threshold"):
            kwargs.pop(key, None)
        return lambda samples: mlx_whisper.transcribe(samples, path_or_hf_repo=str(model_path), language=language, task=task, **kwargs)
    if engine == "whisper_cpp":
        from pywhispercpp.model import Model
        kwargs = {key: profile[key] for key in {"n_max_text_ctx", "no_context", "single_segment", "token_timestamps", "max_len", "split_on_word", "audio_ctx", "suppress_regex", "temperature_inc", "entropy_thold"} if key in profile}
        kwargs["max_tokens"] = profile.get("cpp_max_tokens", 0)
        strategy = profile.get("sampling_strategy", "beam")
        kwargs["beam_search"] = {"beam_size": profile.get("beam_size", 5), "patience": profile.get("patience", 1.0)}
        kwargs["greedy"] = {"best_of": profile.get("best_of", 5)}
        instance = Model(str(model_path), n_threads=profile.get("threads", 4), print_realtime=False, print_progress=False, language=language or "auto", translate=task == "translate", temperature=profile.get("temperature", 0), no_speech_thold=profile.get("no_speech_threshold", 0.6), logprob_thold=profile.get("log_prob_threshold", -1), suppress_blank=profile.get("suppress_blank", True), initial_prompt=profile.get("initial_prompt", ""), params_sampling_strategy=1 if strategy == "beam" else 0, **kwargs)
        return lambda samples: {"segments": [segment(s.t0 / 100, s.t1 / 100, s.text) for s in instance.transcribe(samples, n_processors=profile.get("cpp_processors", 1))]}
    if engine == "transformers":
        from transformers import pipeline
        instance = pipeline("automatic-speech-recognition", model=str(model_path), device=device, dtype=dtype, trust_remote_code=False)
        family = model.get("family", "whisper")
        generate = {}
        if family == "whisper":
            generate = {"task": task, "num_beams": profile.get("beam_size", 5)}
            generate.update(repetition_penalty=profile.get("repetition_penalty", 1), no_repeat_ngram_size=profile.get("no_repeat_ngram_size", 0))
            generate["early_stopping"] = profile.get("early_stopping", False)
            if profile.get("do_sample"):
                generate.update(do_sample=True, temperature=profile["temperature"], top_p=profile.get("top_p", 0.9), top_k=profile.get("top_k", 50))
            if language:
                generate["language"] = language
            if profile.get("max_new_tokens"):
                generate["max_new_tokens"] = profile["max_new_tokens"]
        def options():
            pipeline_type = getattr(instance, "type", "")
            kwargs = {}
            if pipeline_type == "seq2seq_whisper":
                kwargs["return_timestamps"] = "word" if profile.get("word_timestamps") else True
            elif pipeline_type in {"ctc", "ctc_with_lm"}:
                kwargs["return_timestamps"] = "word" if profile.get("word_timestamps") else "char"
            if generate:
                kwargs["generate_kwargs"] = generate
            return kwargs
        def infer(samples):
            return instance({"raw": samples, "sampling_rate": 16000}, **options())
        infer.batch = lambda arrays: instance([{"raw": a, "sampling_rate": 16000} for a in arrays], batch_size=profile.get("batch_size", 4), **options())
        return infer
    if engine == "qwen":
        from qwen_asr import Qwen3ASRModel
        instance = Qwen3ASRModel.from_pretrained(str(model_path), dtype=dtype, device_map=device, max_inference_batch_size=profile.get("batch_size", 4), max_new_tokens=profile.get("max_new_tokens") or 512)
        names = {"ar": "Arabic", "en": "English", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "de": "German", "fr": "French", "es": "Spanish", "ru": "Russian", "tr": "Turkish", "fa": "Persian", "pt": "Portuguese"}
        names.update(it="Italian", nl="Dutch", pl="Polish", uk="Ukrainian", id="Indonesian", vi="Vietnamese", th="Thai", ms="Malay", sv="Swedish", da="Danish", fi="Finnish", cs="Czech", tl="Filipino", el="Greek", hu="Hungarian", mk="Macedonian", ro="Romanian", hi="Hindi", yue="Cantonese")
        def result_data(result):
            reported = str(result.language or "unknown").lower()
            detected = next((key for key, name in names.items() if name.lower() == reported), reported)
            return {"text": result.text, "language": detected}
        def infer(samples):
            result = instance.transcribe(audio=(samples, 16000), language=names.get(language, language), context=profile.get("initial_prompt", ""))[0]
            return result_data(result)
        infer.batch = lambda arrays: [result_data(r) for r in instance.transcribe(audio=[(a, 16000) for a in arrays], language=names.get(language, language), context=profile.get("initial_prompt", ""))]
        return infer
    if engine == "funasr":
        from funasr import AutoModel
        instance = AutoModel(model=str(model_path), device=device, disable_update=True, trust_remote_code=False, hub="hf", ncpu=profile.get("threads", 4))
        if dtype != "auto" and hasattr(instance, "model"):
            instance.model.to(dtype=dtype)
        def infer(samples):
            data = instance.generate(input=samples, cache={}, language=language or "auto", use_itn=profile.get("use_itn", True), hotword=profile.get("hotwords", ""))[0]
            text = data.get("text", "")
            try:
                from funasr.utils.postprocess_utils import rich_transcription_postprocess
                text = rich_transcription_postprocess(text)
            except ImportError:
                pass
            return {"text": text}
        return infer
    if engine == "nemo":
        import nemo.collections.asr as nemo_asr
        instance = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location=device)
        if dtype != "auto":
            instance.to(dtype=dtype)
        instance.eval()
        def infer(samples):
            outputs = instance.transcribe(audio=[samples], batch_size=profile.get("batch_size", 4), return_hypotheses=True)
            item = outputs[0]
            return {"text": item.text if hasattr(item, "text") else str(item)}
        infer.batch = lambda arrays: [{"text": item.text if hasattr(item, "text") else str(item)} for item in instance.transcribe(audio=arrays, batch_size=profile.get("batch_size", 4), return_hypotheses=True)]
        return infer
    if engine == "sherpa":
        import sherpa_onnx
        folder = Path(model_path)
        def find(pattern):
            choices = list(folder.rglob(pattern))
            choices.sort(key=lambda p: ("int8" not in p.name.lower(), len(p.name)))
            if not choices:
                raise ValueError("ملف ONNX مفقود: " + pattern)
            return str(choices[0])
        family = model.get("family")
        online = family in {"transducer", "paraformer"} and (profile.get("sherpa_mode") == "online" or profile.get("sherpa_mode", "auto") == "auto" and "streaming" in model.get("name", "").lower())
        recognizer = sherpa_onnx.OnlineRecognizer if online else sherpa_onnx.OfflineRecognizer
        common = {"tokens": find("*tokens*.txt"), "num_threads": profile.get("threads", 4), "provider": profile.get("sherpa_provider", "cpu")}
        if family == "transducer":
            options = {key: profile[key] for key in {"decoding_method", "max_active_paths", "hotwords_file", "hotwords_score", "blank_penalty"} if key in profile}
            instance = recognizer.from_transducer(encoder=find("*encoder*.onnx"), decoder=find("*decoder*.onnx"), joiner=find("*joiner*.onnx"), **common, **options)
        elif family == "whisper":
            instance = sherpa_onnx.OfflineRecognizer.from_whisper(encoder=find("*encoder*.onnx"), decoder=find("*decoder*.onnx"), language=language or "", task=task, **common)
        elif family == "paraformer":
            instance = recognizer.from_paraformer(encoder=find("*encoder*.onnx"), decoder=find("*decoder*.onnx"), **common) if online else recognizer.from_paraformer(paraformer=find("*.onnx"), **common)
        elif family == "moonshine":
            instance = sherpa_onnx.OfflineRecognizer.from_moonshine(preprocessor=find("*preprocess*.onnx"), encoder=find("*encode*.onnx"), uncached_decoder=find("*uncached*.onnx"), cached_decoder=find("cached*.onnx"), **common)
        else:
            raise ValueError("اختر عائلة ONNX يدعمها المحول")
        def infer(samples):
            stream = instance.create_stream()
            stream.accept_waveform(16000, samples)
            if online:
                import numpy as np
                stream.accept_waveform(16000, np.zeros(8000, dtype="float32"))
                stream.input_finished()
                while instance.is_ready(stream):
                    instance.decode_stream(stream)
                return {"text": instance.get_result(stream)}
            instance.decode_stream(stream)
            return {"text": stream.result.text}
        return infer
    raise ValueError("محرك غير معروف: " + engine)
