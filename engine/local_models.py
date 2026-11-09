"""Find existing model data without loading model code or contacting a server."""
import hashlib
import json
import os
import re
import stat
import struct
import sys
import tarfile
import time
from pathlib import Path

from core.paths import ROOT, data_dir
from engine.catalog import artifact_id, relevant_files, ASR_FAMILIES

PRUNE = {".git", ".svn", "node_modules", "__pycache__", "venv", ".venv", "site-packages", "winsxs", "windows", "$recycle.bin", "system volume information", "temp", "gpuCache".lower(), "cache2", "code cache", "uv"}
WEIGHT_ENDINGS = (".safetensors", ".bin", ".npz", ".pt", ".onnx", ".model", ".tiktoken")
TEXT_ARCHITECTURES = {"llama", "qwen2", "qwen3", "qwen3moe", "gemma", "gemma2", "gemma3", "phi3", "mistral", "hunyuan", "hy-mt", "hy-mt2"}


def local_stat(path):
    info = Path(path).stat()
    # Do not hydrate cloud-only OneDrive/iCloud placeholders during a local scan.
    if getattr(info, "st_file_attributes", 0) & (0x1000 | 0x40000 | 0x400000):
        raise OSError("ملف سحابي غير متاح محليًا")
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        raise OSError("ملف غير مكتمل")
    return info


def small_json(path, limit=4 * 1024 * 1024):
    if local_stat(path).st_size > limit:
        raise ValueError("ملف تعريف أكبر من الحد المسموح")
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("ملف التعريف ليس كائن JSON")
    return value


def gguf_metadata(path):
    """Read a bounded GGUF metadata prefix; never map tensors or import gguf."""
    size = local_stat(path).st_size
    with open(path, "rb") as stream:
        def number(fmt):
            length = struct.calcsize(fmt)
            raw = stream.read(length)
            if len(raw) != length:
                raise ValueError("رأس GGUF ناقص")
            return struct.unpack(fmt, raw)[0]
        def skip(length):
            if length < 0 or stream.tell() + length > size:
                raise ValueError("بيانات GGUF ناقصة")
            stream.seek(length, 1)
        def string(read=True):
            length = number("<Q")
            if read:
                if length > 1024 * 1024:
                    raise ValueError("نص GGUF طويل")
                raw = stream.read(length)
                if len(raw) != length:
                    raise ValueError("نص GGUF ناقص")
                return raw.decode("utf-8")
            skip(length)
        formats = {0:"<B", 1:"<b", 2:"<H", 3:"<h", 4:"<I", 5:"<i", 6:"<f", 7:"<?", 10:"<Q", 11:"<q", 12:"<d"}
        def value(kind, read=False, depth=0):
            if depth > 2:
                raise ValueError("مصفوفة GGUF غير صالحة")
            if kind == 8:
                return string(read)
            if kind in formats:
                return number(formats[kind])
            if kind != 9:
                raise ValueError("نوع GGUF غير معروف")
            child, count = number("<I"), number("<Q")
            if count > 2_000_000:
                raise ValueError("مصفوفة GGUF كبيرة")
            if child in formats:
                skip(count * struct.calcsize(formats[child]))
            else:
                for _ in range(count):
                    value(child, False, depth + 1)
        if stream.read(4) != b"GGUF" or number("<I") not in {2, 3}:
            raise ValueError("هذا ليس ملف GGUF مدعومًا")
        tensors, count = number("<Q"), number("<Q")
        if not tensors or tensors > 2_000_000 or count > 50000:
            raise ValueError("رأس GGUF غير صالح")
        result = {}
        for _ in range(count):
            if stream.tell() > 64 * 1024 * 1024:
                break
            key, kind = string(), number("<I")
            wanted = key in {"general.architecture", "general.name", "general.source.repo_url"}
            if wanted and kind != 8:
                raise ValueError("حقل تعريف GGUF غير صالح")
            found = value(kind, wanted)
            if wanted:
                result[key] = found
            if "general.architecture" in result and "general.name" in result:
                break
        return result


def snapshot_identity(path):
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part.startswith("models--") and i + 2 < len(parts) and parts[i + 1] == "snapshots":
            repo = part[len("models--"):].replace("--", "/", 1)
            return repo, parts[i + 2]
    return "", ""


def default_roots(extra=(), broad=False):
    home = Path.home()
    cache = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
    hf_home = Path(os.environ.get("HF_HOME", cache / "huggingface"))
    roots = [os.environ.get(k) for k in ["HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "MODELSCOPE_CACHE", "OLLAMA_MODELS", "WHISPER_MODELS_DIR"]]
    roots += [hf_home / "hub", cache / "whisper", cache / "whisper.cpp", cache / "modelscope" / "hub", cache / "torch" / "NeMo", home / ".ollama/models", home / ".lmstudio/models", home / ".cache/lm-studio/models", home / ".local/share/pywhispercpp/models", home / ".local/share/whisper.cpp", data_dir() / "models", ROOT / "models"]
    for variable in ["LOCALAPPDATA", "APPDATA", "PROGRAMDATA"]:
        parent = os.environ.get(variable)
        if parent:
            roots += [Path(parent) / name for name in ["WhisperPro/models", "Subtitle Edit", "SubtitleEdit", "Buzz", "pywhispercpp/models", "whisper.cpp/models", "huggingface/hub", "LM Studio/models"]]
    roots += [home / name for name in ["Models", "models", "AI", "Downloads", "Desktop", "Documents"]]
    if sys.platform == "darwin":
        roots += [home / "Library/Caches/huggingface/hub", home / "Library/Application Support/Buzz", home / "Library/Application Support/WhisperPro/models"]
    if sys.platform == "linux":
        roots += [Path("/usr/share/ollama/.ollama/models"), Path("/var/lib/ollama/models")]
    roots += [Path(p).expanduser() for p in extra if str(p).strip()]
    if broad:
        if sys.platform == "win32":
            import ctypes
            roots += [Path(f"{chr(65+i)}:/") for i in range(26) if ctypes.windll.kernel32.GetLogicalDrives() & (1 << i) and ctypes.windll.kernel32.GetDriveTypeW(f"{chr(65+i)}:\\") == 3]
        else:
            roots += [home, Path("/mnt"), Path("/media"), Path("/Volumes")]
    unique = []
    for root in roots:
        if not root:
            continue
        path = Path(root).expanduser().absolute()
        try:
            if path.is_dir() and path not in unique:
                unique.append(path)
        except OSError:
            continue
    return unique


class ModelScanner:
    def __init__(self, models, installed=None):
        self.models = models
        self.by_repo = {m["id"]: m for m in models}
        self.known_paths = {}
        for model in models:
            path = model.get("path") or (installed or {}).get(model["id"], {}).get("path")
            paths = ([path] if path else []) + model.get("external_sources", []) + (installed or {}).get(model["id"], {}).get("external_sources", [])
            for saved in paths:
                self.known_paths[os.path.normcase(os.path.abspath(saved))] = model
        self.results, self.errors, self.seen_candidates = [], [], {}

    def candidate(self, path, fmt, family, engines, names, name=None, base=None, note=""):
        path = Path(path).absolute()
        base = Path(base or (path if path.is_dir() else path.parent)).absolute()
        files, signature = {}, []
        for relative in sorted(set(names)):
            target = base / relative
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise ValueError("مسار أوزان غير آمن")
            info = local_stat(target)
            files[relative] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            physical = (info.st_dev, info.st_ino) if info.st_ino else str(target.resolve())
            signature.append(("_primary" if path.is_file() and relative == path.name else relative, physical, info.st_size))
        repo, revision = snapshot_identity(path)
        identity = artifact_id(repo, path.name if path.is_file() else "") if repo else ""
        known = self.by_repo.get(identity) or self.known_paths.get(os.path.normcase(str(path)))
        if known:
            fmt, family, engines = known.get("format", fmt) if known.get("format") != "local" else fmt, known.get("family", family), known.get("engines", engines)
            missing = [c.get("name", c["filename"]) for c in known.get("companions", []) if not (base / c.get("name", c["filename"])).is_file()]
            if missing:
                note = "ملف مساعد مفقود: " + ", ".join(missing)
            for companion in known.get("companions", []):
                filename = companion.get("name", companion["filename"])
                if (base / filename).is_file():
                    info = local_stat(base / filename)
                    files[filename] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            for filename in known.get("companion_files", []):
                info = local_stat(base / filename)
                files[filename] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
        key = hashlib.sha256(repr((fmt, signature)).encode()).hexdigest()[:24]
        if key in self.seen_candidates:
            old = self.seen_candidates[key]
            if str(path) != old["path"] and str(path) not in old["alternatives"]:
                old["alternatives"].append(str(path))
            return
        item = {"key": key, "path": str(path), "base_path": str(base), "name": str(name or (known or {}).get("name", path.name)), "format": fmt, "family": family, "engines": engines, "files": files, "size": sum(f["size"] for f in files.values()), "repo": repo, "revision": revision, "catalog_id": (known or {}).get("id", ""), "note": note, "usable": not note.startswith("ملف مساعد مفقود"), "alternatives": []}
        self.results.append(item)
        self.seen_candidates[key] = item

    def folder(self, folder, names):
        if "final.mdl" in names and folder.name == "am" and (folder.parent / "conf/mfcc.conf").is_file():
            base = folder.parent
            if not (base / "graph").is_dir() or not any((base / "graph").glob("*.fst")):
                raise ValueError("نموذج Vosk ناقص؛ ملفات الرسم الصوتي غير موجودة")
            self.candidate(base, "vosk", "vosk", ["vosk"], [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file() and p.suffix not in {".zip", ".tmp"}])
        config = small_json(folder / "config.json") if "config.json" in names else {}
        model_type = str(config.get("model_type", "")).lower()
        if "model.bin" in names and "config.json" in names and ("alignment_heads" in config or "lang_ids" in config or model_type == "whisper") and ("tokenizer.json" in names or any(n.startswith("vocabulary.") for n in names)):
            chosen = relevant_files({"format": "ct2"}, [{"rfilename": n} for n in names])
            self.candidate(folder, "ct2", "whisper", ["faster_whisper", "whisperx"], chosen)
            return True
        if "config.json" in names and any(n.endswith(WEIGHT_ENDINGS) for n in names):
            config_text = json.dumps(config).lower()
            if "weights.npz" in names and ("n_audio_state" in config or model_type == "whisper"):
                fmt, family, engines = "mlx", "whisper", ["mlx"]
            elif model_type == "whisper":
                fmt, family, engines = "transformers", "whisper", ["transformers"]
            elif "qwen3_asr" in model_type or "qwen3asr" in config_text:
                fmt, family, engines = "qwen", "qwen3", ["qwen"]
            elif model_type in {"wav2vec2", "wav2vec2-bert", "hubert", "data2vec-audio", "moonshine"}:
                fmt, family, engines = "transformers", "asr", ["transformers"]
            elif model_type == "marian":
                fmt, family, engines = "translation", "seq2seq", ["translation"]
            elif "hy_mt" in config_text or "hunyuanmt" in config_text or "hy-mt" in folder.name.lower():
                fmt, family, engines = "translation", "causal", ["translation"]
            elif model_type.replace("_", "-") in TEXT_ARCHITECTURES and "causallm" in config_text:
                fmt, family, engines = "translation", "causal", ["translation"]
            else:
                return False
            chosen = relevant_files({"format": fmt}, [{"rfilename": n} for n in names])
            for index in [n for n in names if n.endswith(".index.json")]:
                if index not in chosen:
                    continue
                index_data = small_json(folder / index)
                mapping = index_data.get("weight_map", {})
                if not isinstance(mapping, dict):
                    raise ValueError("فهرس الأوزان غير صالح")
                if any(not isinstance(relative, str) for relative in mapping.values()):
                    raise ValueError("مسار في فهرس الأوزان غير صالح")
                for relative in set(mapping.values()):
                    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                        raise ValueError("فهرس أوزان غير آمن")
                    local_stat(folder / relative)
                    chosen[relative] = {}
            if not any(n.endswith(WEIGHT_ENDINGS) for n in chosen):
                raise ValueError("أوزان النموذج غير مكتملة")
            self.candidate(folder, fmt, family, engines, chosen)
            return True
        if "model.pt" in names and "config.yaml" in names:
            config_path = folder / "config.yaml"
            if local_stat(config_path).st_size > 1024 * 1024:
                return False
            text = config_path.read_text(encoding="utf-8").lower()
            family = next((f for f in ["sensevoice", "paraformer", "funasr"] if f in text or f in folder.name.lower()), None)
            if family:
                selected = [n for n in names if n.endswith((".json", ".yaml", ".yml", ".txt", ".model", ".pt"))]
                self.candidate(folder, "funasr", family, ["funasr"], selected)
                return True
        onnx = [n for n in names if n.endswith(".onnx")]
        if onnx and any("tokens" in n and n.endswith(".txt") for n in names):
            low = folder.name.lower()
            family = "moonshine" if "moonshine" in low else "whisper" if "whisper" in low else "transducer" if any("joiner" in n for n in onnx) else "paraformer" if "paraformer" in low else ""
            if family:
                if family in {"whisper", "transducer"} and not all(any(part in n for n in onnx) for part in (["encoder", "decoder", "joiner"] if family == "transducer" else ["encoder", "decoder"])):
                    raise ValueError("ملفات ONNX ناقصة")
                self.candidate(folder, "onnx", family, ["sherpa"], [n for n in names if n.endswith((".onnx", ".txt", ".json"))])
                return True
        return False

    def file(self, path, cancelled=None, name=None):
        info = local_stat(path)
        with open(path, "rb") as stream:
            head = stream.read(48)
        if head[:4] == b"lmgg" and len(head) == 48:
            values = struct.unpack("<11i", head[4:])
            if values[0] < 50000 or values[0] > 100000 or values[1] != 1500 or values[9] not in {80, 128} or info.st_size < 4096:
                return
            self.candidate(path, "ggml", "whisper", ["whisper_cpp", "crisp"], [path.name], name=name)
        elif head[:4] == b"GGUF":
            metadata = gguf_metadata(path)
            arch = str(metadata.get("general.architecture", "")).lower().replace("_", "-")
            repo, _ = snapshot_identity(path)
            known = self.by_repo.get(artifact_id(repo, path.name)) if repo else None
            if known:
                family, engines = known["family"], known["engines"]
            elif arch in (ASR_FAMILIES - TEXT_ARCHITECTURES - {"granite", "cohere", "qwen3-1.7b"}) | {"qwen3-asr", "granite-speech", "cohere-transcribe"}:
                family, engines = {"qwen3-asr": "qwen3"}.get(arch, arch), ["crisp"]
            elif arch in TEXT_ARCHITECTURES:
                family, engines = "causal", ["translation"]
            else:
                return
            shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path.name)
            names = [path.name]
            if shard:
                if int(shard[1]) != 1 or int(shard[2]) > 1024:
                    return
                names = [path.name[:shard.start()] + f"-{i:05d}-of-{int(shard[2]):05d}.gguf" for i in range(1, int(shard[2]) + 1)]
            note = "نموذج لغوي للترجمة النصية فقط؛ ليس محرك تفريغ صوت" if engines == ["translation"] else ""
            self.candidate(path, "gguf", family, engines, names, name=name or metadata.get("general.name"), note=note)
        elif path.suffix == ".pt":
            aliases = {"large.pt": "large-v3.pt", "turbo.pt": "large-v3-turbo.pt"}
            basename = aliases.get(path.name, path.name)
            known = next((m for m in self.models if m.get("format") == "pt" and m.get("filename") == basename and m.get("sha256")), None)
            if known:
                digest = hashlib.sha256()
                with open(path, "rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        if cancelled and cancelled():
                            raise InterruptedError("أُلغي البحث")
                        digest.update(block)
                if digest.hexdigest() == known["sha256"]:
                    after = local_stat(path)
                    if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise ValueError("تغير ملف الأوزان أثناء التحقق")
                    self.known_paths[os.path.normcase(str(path))] = known
                    self.candidate(path, "pt", "whisper", ["openai_whisper"], [path.name])
        elif path.suffix == ".nemo":
            with tarfile.open(path) as archive:
                for member_index, member in enumerate(archive):
                    if cancelled and cancelled():
                        raise InterruptedError("أُلغي البحث")
                    if member_index > 64 or member.size > 8 * 1024 * 1024:
                        break
                    if member.name.rsplit("/", 1)[-1] == "model_config.yaml" and member.isfile() and member.size < 1024 * 1024:
                        text = archive.extractfile(member).read().decode("utf-8").lower()
                        if any(s in text for s in ["encdec", "asr", "parakeet", "canary"]):
                            self.candidate(path, "nemo", "canary" if "canary" in text else "parakeet", ["nemo"], [path.name])
                        break

    def scan(self, roots, progress=None, cancelled=None, max_directories=100000, max_seconds=120):
        started, visited, warnings = time.monotonic(), set(), []
        stack = [(Path(p).absolute(), 0) for p in reversed(roots)]
        count, limited = 0, False
        while stack:
            if cancelled and cancelled():
                raise InterruptedError("أُلغي البحث")
            if count >= max_directories or time.monotonic() - started > max_seconds:
                limited = True
                break
            folder, depth = stack.pop()
            try:
                real = os.path.normcase(str(folder.resolve()))
                if real in visited:
                    continue
                visited.add(real)
                count += 1
                if progress and count % 25 == 1:
                    progress(0, 0, f"فحص {count} مجلدًا · عُثر على {len(self.results)} نموذجًا · {folder.name}")
                entries = list(os.scandir(folder))
                files = set()
                for entry in entries:
                    try:
                        if entry.is_file(follow_symlinks=True) and not entry.name.endswith((".incomplete", ".part", ".tmp", ".lock")):
                            files.add(entry.name)
                    except OSError as exc:
                        warnings.append(f"{entry.path}: {exc}")
                try:
                    recognized = self.folder(folder, files)
                except (OSError, ValueError, KeyError) as exc:
                    recognized = False
                    warnings.append(f"{folder}: {exc}")
                if "manifests" in {e.name for e in entries} and (folder / "blobs").is_dir():
                    for manifest in (folder / "manifests").rglob("*"):
                        if not manifest.is_file():
                            continue
                        try:
                            data = small_json(manifest)
                            layers = data.get("layers")
                            if not isinstance(layers, list):
                                continue
                            for layer in layers:
                                if not isinstance(layer, dict):
                                    continue
                                digest = layer.get("digest", "")
                                if layer.get("mediaType") == "application/vnd.ollama.image.model" and isinstance(digest, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", digest):
                                    self.file(folder / "blobs" / digest.replace(":", "-"), cancelled, "Ollama · " + "/".join(manifest.relative_to(folder / "manifests").parts[-2:]))
                        except (OSError, ValueError, KeyError):
                            continue
                for filename in sorted(files):
                    if cancelled and cancelled():
                        raise InterruptedError("أُلغي البحث")
                    if filename.endswith((".gguf", ".nemo")) or filename.endswith(".bin") and filename != "model.bin" or filename.endswith(".pt") and not recognized:
                        try:
                            self.file(folder / filename, cancelled)
                        except (OSError, ValueError, KeyError, struct.error, tarfile.TarError) as exc:
                            self.errors.append(f"{folder / filename}: {exc}")
                for entry in reversed(entries):
                    if depth >= 24 or entry.name.lower() in PRUNE or entry.name in {"blobs", "manifests"}:
                        continue
                    if entry.is_dir(follow_symlinks=False) and not entry.is_symlink() and not getattr(entry, "is_junction", lambda: False)():
                        if getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0) & (0x1000 | 0x40000 | 0x400000):
                            continue
                        stack.append((Path(entry.path), depth + 1))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                warnings.append(f"{folder}: {exc}")
        return {"candidates": self.results, "roots": [str(p) for p in roots], "directories": count, "limited": limited, "warnings": (warnings + self.errors)[:100], "elapsed": round(time.monotonic() - started, 2), "scanned_at": time.time()}
