"""One artifact catalog shared across engines, with content-aware update checks."""
import fnmatch
import hashlib
import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import ast
import shutil
import uuid
from pathlib import Path

from core.paths import location
from core.storage import read_json, write_json, fetch_json, download, extract_archive, process_lock

_LOCK = threading.RLock()
_DOWNLOAD_LOCKS = {}
SEEDS = Path(__file__).with_name("model_seeds.json")
TRUSTED_AUTHORS = {"openai", "Systran", "distil-whisper", "mobiuslabsgmbh", "ggerganov", "cstr", "Qwen", "FunAudioLLM", "iic", "nvidia", "facebook", "ibm-granite", "mlx-community", "csukuangfj", "k2-fsa", "Helsinki-NLP", "tencent"}
ASR_FAMILIES = {"whisper", "parakeet", "canary", "canary-qwen", "qwen3", "qwen3-1.7b", "voxtral", "voxtral4b", "granite", "granite-4.1", "granite-4.1-plus", "granite-4.1-nar", "cohere", "nemotron", "gigaam", "dolphin", "xasr", "phonon2", "reazonspeech", "quds", "quds-fa", "haas", "haas-ja", "sensevoice", "funasr", "paraformer", "moonshine", "moonshine-streaming", "kyutai-stt", "wav2vec2", "hubert", "data2vec", "firered-asr", "glm-asr", "omniasr", "moss-transcribe", "moss-diarize", "moss-audio", "higgs-stt", "ark-asr", "mimo-asr", "raon-speech", "hojo-asr", "index-echo", "mega-asr", "fastconformer-ctc", "tiron"}


def artifact_id(repo, filename=""):
    return "hf:" + repo + ("#" + filename if filename else "")


def original_whisper_models():
    url = "https://raw.githubusercontent.com/openai/whisper/main/whisper/__init__.py"
    with urllib.request.urlopen(url, timeout=30) as response:
        tree = ast.parse(response.read().decode("utf-8"))
    mapping = {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(target, ast.Name) and target.id == "_MODELS" for target in targets):
            mapping = ast.literal_eval(node.value)
    result = []
    for key, url in mapping.items():
        canonical = {"large": "large-v3", "turbo": "large-v3-turbo"}.get(key, key)
        parsed = urllib.parse.urlparse(url)
        if parsed.hostname != "openaipublic.azureedge.net" or not parsed.path.split("/")[-2].isalnum():
            continue
        checksum = parsed.path.split("/")[-2]
        result.append({"id": "openai:" + canonical, "name": canonical, "source": "url", "url": url, "sha256": checksum, "filename": canonical + ".pt", "format": "pt", "family": "whisper", "engines": ["openai_whisper"], "languages": ["en"] if canonical.endswith(".en") else [], "upstream_key": canonical})
    return list(merge_models(result).values())


def merge_models(models):
    merged = {}
    for model in models:
        item = dict(model)
        identity = item["id"]
        if identity in merged:
            old = merged[identity]
            item["engines"] = sorted(set(old.get("engines", [])) | set(item.get("engines", [])))
            item["aliases"] = sorted(set(old.get("aliases", [])) | set(item.get("aliases", [])))
            item = {**old, **item}
        merged[identity] = item
    return merged


def relevant_files(model, siblings):
    files = {s["rfilename"]: s for s in siblings}
    if any("\\" in name or ":" in name or any(part in {"", ".", ".."} for part in name.split("/")) for name in files):
        raise ValueError("مسار ملف غير صالح في مصدر النموذج")
    if model.get("filename"):
        names = [model["filename"], *model.get("companion_files", [])]
    elif model.get("format") == "ct2":
        names = [n for n in files if n in {"model.bin", "config.json", "tokenizer.json", "preprocessor_config.json"} or n.startswith("vocabulary.")]
        if "model.bin" not in names or "config.json" not in names:
            raise ValueError("المستودع لا يحتوي نموذج CTranslate2 مكتملًا")
    elif model.get("format") == "nemo":
        names = [n for n in files if n.endswith(".nemo")]
        if not names:
            raise ValueError("لم يوجد ملف NeMo")
        names = names[:1]
    else:
        patterns = ("*.json", "*.safetensors", "*.bin", "*.model", "*.txt", "*.yaml", "*.yml", "*.npz", "*.pt", "*.onnx", "*.tiktoken", "*.jinja", "*.jinja2")
        names = [n for n in files if any(fnmatch.fnmatch(n, p) for p in patterns) and not any(x in n.lower() for x in ["optimizer", "training_args", "scheduler", "rng_state"])]
        if any(n.endswith(".safetensors") for n in names):
            names = [n for n in names if not n.startswith("pytorch_model")]
        if model.get("format") in {"transformers", "translation", "qwen"}:
            if "model.safetensors" in files:
                weights = {"model.safetensors"}
            elif "model.safetensors.index.json" in files:
                weights = {n for n in names if re.fullmatch(r"model-\d+-of-\d+\.safetensors", n)}
            elif "pytorch_model.bin" in files:
                weights = {"pytorch_model.bin"}
            elif "pytorch_model.bin.index.json" in files:
                weights = {n for n in names if re.fullmatch(r"pytorch_model-\d+-of-\d+\.bin", n)}
            else:
                weights = None
            if weights:
                names = list(dict.fromkeys(names + sorted(weights)))
                names = [n for n in names if not n.endswith((".safetensors", ".bin", ".pt", ".onnx", ".npz")) or n in weights]
    if not names or any(n not in files for n in names):
        raise ValueError("ملفات تشغيل النموذج ناقصة في المصدر")
    return {n: files[n] for n in sorted(names)}


def fingerprint(files):
    values = [(name, value.get("blobId"), (value.get("lfs") or {}).get("sha256"), value.get("size") or (value.get("lfs") or {}).get("size")) for name, value in sorted(files.items())]
    if not values or any(not blob and not sha for _, blob, sha, _ in values):
        return None
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


class Catalog:
    def __init__(self):
        self.root = location("catalog")

    def models(self, engine=None):
        seeds = read_json(SEEDS, [])
        discovered = read_json(self.root / "models.json", [])
        values = merge_models(seeds + discovered).values()
        return sorted((m for m in values if engine is None or engine in m.get("engines", [])), key=lambda m: (not m.get("recommended", False), m.get("name", "").lower()))

    def get(self, identity):
        return next((m for m in self.models() if m["id"] == identity), None)

    def installed(self):
        return read_json(self.root / "installed.json", {})

    def record(self, identity, data):
        with _LOCK, process_lock(self.root / "locks/installed.lock"):
            items = self.installed()
            items[identity] = data
            write_json(self.root / "installed.json", items)

    def adopt(self, candidate, copy_files=False, progress=None, cancelled=None):
        """Register compatible existing weights; never edit their source files."""
        from engine.local_models import local_stat
        if not candidate.get("usable"):
            raise ValueError(candidate.get("note") or "النموذج غير مكتمل")
        source = Path(candidate["path"]).absolute()
        base = Path(candidate["base_path"]).absolute()
        files = candidate["files"]
        if not source.exists() or not files:
            raise ValueError("ملفات النموذج غير موجودة")
        for name, expected in files.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("مسار أوزان غير صالح")
            info = local_stat(base / name)
            if info.st_size != expected["size"] or expected.get("mtime_ns") and info.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError("تغير ملف بعد البحث؛ أعد فحص هذا المجلد")
        models = self.models()
        known = next((m for m in models if m["id"] == candidate.get("catalog_id")), None)
        identity = known["id"] if known else "local:" + str(source)
        sources = list(dict.fromkeys([str(source)] + candidate.get("alternatives", []) + (known or {}).get("external_sources", [])))
        previous = self.installed().get(identity)
        if previous and self.valid_install(previous) and str(source) in previous.get("external_sources", []) and previous.get("source_files") == files and (not copy_files or previous.get("external_copied")):
            return {"adopted_model": identity, "path": previous["path"], "copied": previous.get("external_copied", False)}
        path, actual_base = source, base
        destination = None
        if copy_files:
            parent = location("models/imported")
            destination = (parent / (candidate["key"] + "-" + uuid.uuid4().hex[:8])).resolve()
            if destination.parent != parent.resolve():
                raise ValueError("مسار نسخ غير صالح")
            total, done = sum(v["size"] for v in files.values()), 0
            try:
                destination.mkdir()
                for name, expected in files.items():
                    target = destination / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(base / name, "rb") as reader, open(target, "xb") as writer:
                        while block := reader.read(4 * 1024 * 1024):
                            if cancelled and cancelled():
                                raise InterruptedError("أُلغي نسخ النموذج")
                            writer.write(block)
                            done += len(block)
                            if progress:
                                progress(done, total, "نسخ " + name)
                    after = local_stat(base / name)
                    if target.stat().st_size != expected["size"] or after.st_size != expected["size"] or after.st_mtime_ns != expected.get("mtime_ns", after.st_mtime_ns):
                        raise ValueError("تغير المصدر أثناء النسخ؛ لم تُعتمد النسخة")
                if cancelled and cancelled():
                    raise InterruptedError("أُلغي نسخ النموذج")
                actual_base = destination
                path = destination if source.is_dir() else destination / source.name
            except BaseException:
                if destination.exists() and destination.parent == parent.resolve():
                    shutil.rmtree(destination)
                raise
        if cancelled and cancelled():
            raise InterruptedError("أُلغي اعتماد النموذج")
        metadata = {"path": str(path), "base_path": str(actual_base), "revision": candidate.get("revision", ""), "fingerprint": None, "files": {name: {"size": meta["size"]} for name, meta in files.items()}, "external": True, "external_sources": sources, "external_copied": copy_files, "source_files": files, "installed_at": time.time()}
        if known and known.get("format") == "pt" and known.get("sha256"):
            metadata["fingerprint"] = known["sha256"]
        if known and known.get("source") != "local":
            old = self.installed().get(identity)
            if old and self.valid_install(old) and Path(old["path"]).resolve() == path.resolve():
                return {"adopted_model": identity, "path": old["path"], "copied": False}
            self.record(identity, metadata)
        else:
            item = {**(known or {}), "id": identity, "name": candidate["name"], "source": "local", "path": str(path), "engines": candidate["engines"], "format": candidate["format"], "family": candidate["family"], "size": candidate["size"], "external_sources": sources, "license": (known or {}).get("license", "راجع ترخيص النموذج في مصدره"), "experimental_translation": candidate.get("engines") == ["translation"] and not known}
            with _LOCK, process_lock(self.root / "locks/models.lock"):
                saved = list(merge_models(read_json(self.root / "models.json", []) + [item]).values())
                write_json(self.root / "models.json", saved)
            self.record(identity, metadata)
        return {"adopted_model": identity, "path": str(path), "copied": copy_files}

    def status(self, model):
        if model.get("source") == "local":
            return ("local" if Path(model.get("path", "")).exists() else "repair"), {}
        local = self.installed().get(model["id"])
        check = read_json(self.root / "checks.json", {}).get(model["id"], {})
        if not local:
            return "download", check
        if not self.valid_install(local):
            return "repair", check
        if check.get("error") or not check.get("fingerprint") or not local.get("fingerprint"):
            return "unknown", check
        if check["fingerprint"] != local["fingerprint"]:
            return "update", check
        return "current", check

    def inventory(self, engine):
        installed = self.installed()
        checks = read_json(self.root / "checks.json", {})
        result = []
        for model in self.models(engine):
            local = installed.get(model["id"])
            check = checks.get(model["id"], {})
            if model.get("source") == "local":
                state = "local" if Path(model["path"]).exists() else "repair"
            elif not local:
                state = "download"
            elif not self.valid_install(local):
                state = "repair"
            elif check.get("error") or not check.get("fingerprint") or not local.get("fingerprint"):
                state = "unknown"
            else:
                state = "update" if check["fingerprint"] != local["fingerprint"] else "current"
            size = check.get("size") or model.get("size") or sum(v.get("size", 0) for v in (local or {}).get("files", {}).values()) or None
            result.append({"model": model, "state": state, "checked": check, "installed": local, "size": size})
        return result

    def check_many(self, engine, token=None, progress=None, models=None):
        models = self.models(engine) if models is None else models
        cache, errors = {}, []
        for index, model in enumerate(models):
            if progress:
                progress(index, len(models), "فحص الحجم والإصدار: " + model["name"])
            result = self.check(model, token, metadata_cache=cache, verify_progress=(lambda done, total, message: progress(index, len(models), message)) if progress else None)
            if result.get("error"):
                errors.append(model["name"] + ": " + result["error"])
        return {"count": len(models), "errors": errors}

    @staticmethod
    def valid_install(local):
        path = Path(local.get("path", ""))
        if not path.exists():
            return False
        if not local.get("files"):
            return path.is_file() and path.stat().st_size > 0
        base = Path(local.get("base_path")) if local.get("base_path") else path if path.is_dir() else path.parent
        return all((base / name).is_file() and (not meta.get("size") or (base / name).stat().st_size == meta["size"]) for name, meta in local["files"].items())

    def verify_external(self, model, remote, progress=None):
        local = self.installed().get(model["id"])
        if not local or not local.get("external") or not remote.get("files"):
            return
        base = Path(local.get("base_path", Path(local["path"]).parent))
        files = remote.get("verification_files", remote["files"])
        facts = []
        for name in files:
            path = base / name
            info = path.stat() if path.is_file() else None
            facts.append((name, info.st_size if info else None, info.st_mtime_ns if info else None))
        signature = hashlib.sha256(repr(facts).encode()).hexdigest()
        if local.get("verified_source") == remote.get("fingerprint") and local.get("verified_files") == signature:
            return
        matching, digests = True, []
        for name, meta in files.items():
            path = base / name
            if not path.is_file():
                matching = False
                digests.append((name, "missing"))
                continue
            before = path.stat()
            expected = (meta.get("lfs") or {}).get("sha256")
            if not expected and not meta.get("blobId"):
                return  # Unknown remains unknown if the source lacks a digest.
            digest = hashlib.sha256() if expected else hashlib.sha1()
            if not expected:
                digest.update(f"blob {before.st_size}\0".encode())
            done = 0
            with open(path, "rb") as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    done += len(block)
                    if progress:
                        progress(done, before.st_size, f"التحقق من الملف الموجود: {name} · {done // 1048576} MB")
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("تغير النموذج أثناء التحقق؛ أعد الفحص بعد انتهاء البرنامج الآخر")
            actual = digest.hexdigest()
            matching = matching and actual == (expected or meta["blobId"])
            digests.append((name, actual))
        local.update(fingerprint=remote["fingerprint"] if matching else "external:" + hashlib.sha256(repr(digests).encode()).hexdigest(), verified_source=remote["fingerprint"], verified_files=signature)
        self.record(model["id"], local)

    def check(self, model, token=None, revision=None, metadata_cache=None, verify_progress=None):
        cache = metadata_cache if metadata_cache is not None else {}
        try:
            if model.get("source") == "local":
                path = Path(model["path"]).expanduser()
                if not path.exists():
                    raise ValueError("مسار النموذج المحلي غير موجود")
                size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
                result = {"checked_at": time.time(), "size": size}
            elif model.get("format") == "pt" and model.get("upstream_key"):
                if "original_whisper" not in cache:
                    cache["original_whisper"] = original_whisper_models()
                latest = next((m for m in cache["original_whisper"] if m["id"] == model["id"]), None)
                if not latest:
                    raise ValueError("لم يعد المصدر يعلن هذا النموذج")
                result = {"checked_at": time.time(), "fingerprint": latest["sha256"], "url": latest["url"], "sha": latest["sha256"]}
            elif model.get("source") != "hf":
                result = {"checked_at": time.time(), "error": "المصدر لا يوفر بصمة إصدار قابلة للمقارنة؛ يمكن إعادة التحقق من المصدر"}
            else:
                repo = urllib.parse.quote(model["repo"], safe="/")
                endpoint = f"https://huggingface.co/api/models/{repo}"
                if revision:
                    endpoint += "/revision/" + urllib.parse.quote(revision, safe="")
                if endpoint not in cache:
                    cache[endpoint] = fetch_json(endpoint + "?blobs=true", token)
                info = cache[endpoint]
                files = relevant_files(model, info.get("siblings", []))
                result = {"checked_at": time.time(), "sha": info["sha"], "fingerprint": fingerprint(files), "files": files, "size": sum(v.get("size") or (v.get("lfs") or {}).get("size", 0) for v in files.values()), "license": (info.get("cardData") or {}).get("license", model.get("license", ""))}
                if model.get("companions"):
                    result["verification_files"] = dict(files)
                    prints = [result["fingerprint"]]
                    for companion in model["companions"]:
                        child = {"id": artifact_id(companion["repo"], companion["filename"]), "source": "hf", "repo": companion["repo"], "filename": companion["filename"]}
                        child_revision = companion.get("revision")
                        sub = self.check(child, token, child_revision if child_revision != "main" else None, cache)
                        if sub.get("error"):
                            raise ValueError(sub["error"])
                        prints.append(sub["fingerprint"])
                        result["size"] += sub.get("size", 0)
                        result["verification_files"][companion.get("name", companion["filename"])] = sub["files"][companion["filename"]]
                    result["fingerprint"] = hashlib.sha256(json.dumps(prints).encode()).hexdigest() if all(prints) else None
                self.verify_external(model, result, verify_progress)
        except InterruptedError:
            raise
        except Exception as exc:
            result = {"checked_at": time.time(), "error": str(exc)}
        with _LOCK, process_lock(self.root / "locks/checks.lock"):
            checks = read_json(self.root / "checks.json", {})
            # Checks for a pinned revision must not change the global latest status.
            if not revision:
                checks[model["id"]] = result
                write_json(self.root / "checks.json", checks)
        return result

    def ensure(self, model, profile, progress=None, cancelled=None, token=None, models_dir="", force=False, repair=False):
        key = (str(models_dir), model["id"])
        with _LOCK:
            lock = _DOWNLOAD_LOCKS.setdefault(key, threading.RLock())
        lock_path = self.root / "locks" / (hashlib.sha256(repr(key).encode()).hexdigest() + ".lock")
        with lock, process_lock(lock_path, cancelled):
            return self._ensure(model, profile, progress, cancelled, token, models_dir, force, repair)

    def _ensure(self, model, profile, progress=None, cancelled=None, token=None, models_dir="", force=False, repair=False):
        if model.get("source") == "local":
            path = Path(model["path"]).expanduser()
            if not path.exists():
                raise ValueError("مسار النموذج المحلي غير موجود")
            return str(path)
        local = self.installed().get(model["id"])
        revision = profile.get("revision", "").strip()
        if not force and local and self.valid_install(local) and (not revision or local.get("revision") == revision):
            return local["path"]
        if profile.get("local_files_only"):
            raise ValueError("النموذج غير مثبت أو ناقص؛ أوقف وضع دون تنزيل لتجهيزه")
        root = Path(models_dir).expanduser() if models_dir else location("models")
        root.mkdir(parents=True, exist_ok=True)
        if model["source"] == "hf":
            remote = self.check(model, token, revision)
            if remote.get("error"):
                raise ValueError(remote["error"])
            downloaded = {}
            files = remote["files"]
            revision_id = remote["sha"]
            if not isinstance(revision_id, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision_id):
                raise ValueError("مسار مراجعة النموذج غير صالح")
            snapshot_key = revision_id + ("-" + remote["fingerprint"][:16] if model.get("companions") else "")
            snapshots = (root / "snapshots" / hashlib.sha256(model["repo"].encode()).hexdigest()[:20]).resolve()
            snapshot = (snapshots / snapshot_key).resolve()
            if snapshot.parent != snapshots:
                raise ValueError("مسار لقطة النموذج غير صالح")
            snapshot.mkdir(parents=True, exist_ok=True)
            blobs = (root / "blobs").resolve()
            completed_bytes = 0
            total_bytes = sum(meta.get("size") or (meta.get("lfs") or {}).get("size", 0) for meta in files.values())
            for index, (name, meta) in enumerate(files.items()):
                if cancelled and cancelled():
                    raise InterruptedError("أُلغي تنزيل النموذج")
                expected = (meta.get("lfs") or {}).get("sha256")
                blob_id = expected or meta.get("blobId") or hashlib.sha256((model["repo"] + remote["sha"] + name).encode()).hexdigest()
                if not isinstance(blob_id, str) or not re.fullmatch(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?", blob_id):
                    raise ValueError("مسار ملف النموذج في المخزن غير صالح")
                blob = (blobs / blob_id).resolve()
                if blob.parent != blobs:
                    raise ValueError("مسار ملف النموذج خارج المخزن")
                expected_size = meta.get("size") or (meta.get("lfs") or {}).get("size", 0)
                file = (snapshot / name).resolve()
                if not file.is_relative_to(snapshot.resolve()):
                    raise ValueError("مسار ملف غير صالح في مصدر النموذج")
                if repair or not blob.is_file() or expected_size and blob.stat().st_size != expected_size:
                    url = f"https://huggingface.co/{model['repo']}/resolve/{remote['sha']}/" + urllib.parse.quote(name, safe="/")
                    download(url, blob, sha256=expected, progress=(lambda a, b, done=completed_bytes, label=name: progress(done + a, total_bytes or b, "تنزيل " + label)) if progress else None, cancelled=cancelled, headers={"Authorization": "Bearer " + token} if token else None)
                    if not expected and meta.get("blobId"):
                        git = hashlib.sha1()
                        git.update(f"blob {blob.stat().st_size}\0".encode())
                        with open(blob, "rb") as stream:
                            for block in iter(lambda: stream.read(1048576), b""):
                                git.update(block)
                        if git.hexdigest() != meta["blobId"]:
                            raise ValueError("فشلت مطابقة بصمة ملف النموذج")
                file.parent.mkdir(parents=True, exist_ok=True)
                if file.exists() and repair:
                    file.unlink()
                if not file.exists():
                    try:
                        os.link(blob, file)
                    except FileExistsError:
                        pass
                    except OSError:
                        import shutil
                        shutil.copy2(blob, file)
                downloaded[name] = {"size": file.stat().st_size, "sha256": expected}
                completed_bytes += file.stat().st_size
            for companion in model.get("companions", []):
                child = {"id": artifact_id(companion["repo"], companion["filename"]), "source": "hf", "repo": companion["repo"], "filename": companion["filename"], "format": "gguf"}
                child_profile = {**profile, "revision": companion.get("revision", "") if companion.get("revision") != "main" else ""}
                child_path = Path(self.ensure(child, child_profile, progress, cancelled, token, models_dir, force=force, repair=repair))
                name = companion.get("name", child_path.name)
                companion_path = (snapshot / name).resolve()
                if not companion_path.is_relative_to(snapshot.resolve()):
                    raise ValueError("اسم ملف مساعد غير صالح")
                if not companion_path.exists():
                    try:
                        os.link(child_path, companion_path)
                    except OSError:
                        import shutil
                        shutil.copy2(child_path, companion_path)
                downloaded[name] = {"size": child_path.stat().st_size}
            path = snapshot / model["filename"] if model.get("filename") else snapshot
            if model.get("format") == "nemo":
                path = snapshot / next(iter(files))
            if local:
                write_json(self.root / ("previous-" + hashlib.sha256(model["id"].encode()).hexdigest()[:16] + ".json"), local)
            self.record(model["id"], {"path": str(path), "base_path": str(snapshot), "revision": remote["sha"], "fingerprint": remote["fingerprint"], "files": downloaded, "installed_at": time.time()})
            return str(path)
        target = root / hashlib.sha256(model["id"].encode()).hexdigest()[:20]
        target.mkdir(parents=True, exist_ok=True)
        filename = model.get("filename", "model.zip")
        source_url, checksum = model["url"], model.get("sha256")
        if model.get("format") == "pt" and model.get("upstream_key"):
            remote = self.check(model, token)
            if remote.get("error"):
                raise ValueError(remote["error"])
            source_url, checksum = remote["url"], remote["sha"]
            target = root / "url" / checksum
            target.mkdir(parents=True, exist_ok=True)
        archive = download(source_url, target / filename, sha256=checksum, progress=(lambda a, b: progress(a, b, "تنزيل النموذج")) if progress else None, cancelled=cancelled)
        if model.get("format") == "vosk":
            extract_archive(archive, target)
            path = next((p.parent for p in target.rglob("mfcc.conf")), None)
            if path is None:
                raise ValueError("حزمة Vosk لا تحتوي إعدادات النموذج")
            path = path.parent
            files = {str(p.relative_to(path)): {"size": p.stat().st_size} for p in path.rglob("*") if p.is_file()}
        else:
            path = archive
            files = {path.name: {"size": path.stat().st_size}}
        self.record(model["id"], {"path": str(path), "files": files, "revision": checksum or "", "fingerprint": checksum, "installed_at": time.time()})
        return str(path)

    def invalidate(self, model):
        # Retain the previous snapshot for rollback; next ensure selects the latest SHA.
        with _LOCK:
            installed = self.installed()
            previous = installed.pop(model["id"], None)
            if previous:
                write_json(self.root / ("previous-" + hashlib.sha256(model["id"].encode()).hexdigest()[:16] + ".json"), previous)
            write_json(self.root / "installed.json", installed)

    def register_local(self, engine, path, family="whisper"):
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise ValueError("ملف النموذج أو مجلده غير موجود")
        if engine in {"faster_whisper", "whisperx"} and not (path.is_dir() and (path / "model.bin").is_file() and (path / "config.json").is_file()):
            raise ValueError("اختر مجلد CTranslate2 الذي يحتوي model.bin وconfig.json")
        item = {"id": "local:" + str(path), "name": path.name, "source": "local", "path": str(path), "engines": [engine], "family": family, "format": "local", "license": "راجع ترخيص النموذج المحلي"}
        metadata = self.check(item)
        if metadata.get("size"):
            item["size"] = metadata["size"]
        with _LOCK:
            models = list(merge_models(read_json(self.root / "models.json", []) + [item]).values())
            write_json(self.root / "models.json", models)
        return item

    def add_hf(self, repo, engine, token=None, filename="", family="whisper"):
        if "/" not in repo or repo.split("/", 1)[0] not in TRUSTED_AUTHORS:
            raise ValueError("اختر مستودعًا من الناشرين المعتمدين المعروضين في الدليل، أو أضف نموذجًا محليًا")
        formats = {"faster_whisper": "ct2", "whisperx": "ct2", "nemo": "nemo", "crisp": "gguf", "whisper_cpp": "ggml", "qwen": "qwen", "funasr": "funasr", "sherpa": "onnx", "mlx": "mlx"}
        item = {"id": artifact_id(repo, filename), "name": repo.split("/")[-1], "source": "hf", "repo": repo, "filename": filename, "family": family, "format": formats.get(engine, "transformers"), "engines": [engine]}
        result = self.check(item, token)
        if result.get("error"):
            raise ValueError(result["error"])
        item["size"] = result["size"]
        item["license"] = result.get("license", "")
        with _LOCK:
            models = list(merge_models(read_json(self.root / "models.json", []) + [item]).values())
            write_json(self.root / "models.json", models)
        return item

    def refresh(self, engine=None, token=None, progress=None):
        """Discover from vetted publishers; a failed source never erases the old list."""
        if engine in {"vosk", "openai_whisper"}:
            if engine == "openai_whisper":
                discovered = original_whisper_models()
            else:
                discovered = []
                for item in fetch_json("https://alphacephei.com/vosk/models/model-list.json"):
                    url = item.get("url", "")
                    if item.get("obsolete") in (True, "true") or not url.startswith("https://alphacephei.com/vosk/models/"):
                        continue
                    discovered.append({"id": "vosk:" + item["name"], "name": item["name"], "source": "url", "url": url, "filename": url.rsplit("/", 1)[-1], "format": "vosk", "family": "vosk", "engines": ["vosk"], "languages": [item.get("lang", "en").split("-")[0]], "size": item.get("size", 0)})
            with _LOCK:
                combined = list(merge_models(read_json(self.root / "models.json", []) + discovered).values())
                write_json(self.root / "models.json", combined)
            return {"count": len(discovered), "errors": []}
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        sources = [
            ("Systran", "faster_whisper", "ct2"), ("distil-whisper", "faster_whisper", "ct2"),
            ("mobiuslabsgmbh", "faster_whisper", "ct2"), ("openai", "transformers", "transformers"),
            ("ggerganov", "whisper_cpp", "ggml"), ("facebook", "transformers", "transformers"),
            ("distil-whisper", "transformers", "transformers"), ("Qwen", "qwen", "qwen"),
            ("FunAudioLLM", "funasr", "funasr"), ("nvidia", "nemo", "nemo"),
            ("iic", "funasr", "funasr"),
            ("mlx-community", "mlx", "mlx"), ("cstr", "crisp", "gguf"),
            ("csukuangfj", "sherpa", "onnx"), ("Helsinki-NLP", "translation", "translation"),
            ("tencent", "translation", "translation"), ("ibm-granite", "transformers", "transformers"),
        ]
        discovered, errors = [], []
        for index, (author, target, fmt) in enumerate(sources):
            if engine and engine != target and not (engine == "whisperx" and target == "faster_whisper") and not (engine == "crisp" and target == "whisper_cpp"):
                continue
            if progress:
                progress(index, len(sources), author)
            try:
                for info in api.list_models(author=author, sort="last_modified", full=True, cardData=True, **({"filter": "automatic-speech-recognition"} if target == "transformers" else {})):
                    name = info.id.split("/", 1)[1]
                    low = name.lower()
                    siblings = [s.rfilename for s in (info.siblings or [])]
                    task = info.pipeline_tag
                    card = info.card_data.to_dict() if hasattr(info.card_data, "to_dict") else info.card_data or {}
                    family = "whisper"
                    if fmt == "ct2" and not ("model.bin" in siblings and "config.json" in siblings):
                        continue
                    if target == "qwen" and "asr" not in low:
                        continue
                    if target == "whisper_cpp" and "whisper" not in low:
                        continue
                    if target == "funasr" and not any(x in low for x in ("sensevoice", "paraformer", "fun-asr")):
                        continue
                    if target == "nemo" and not any(s.endswith(".nemo") for s in siblings):
                        continue
                    if target == "mlx" and ("whisper" not in low or not any(s.endswith((".npz", ".safetensors")) for s in siblings)):
                        continue
                    if target == "transformers" and (task != "automatic-speech-recognition" or "config.json" not in siblings or info.library_name == "ctranslate2"):
                        continue
                    if target == "transformers" and "whisper" not in low:
                        family = "asr"
                    if target == "translation" and not ("opus-mt-" in low or "hy-mt" in low):
                        continue
                    if target == "translation":
                        family = "causal" if "hy-mt" in low else "seq2seq"
                    if target == "crisp":
                        family = next((f for f in sorted(ASR_FAMILIES, key=len, reverse=True) if f in low), "")
                        if not family or task not in (None, "automatic-speech-recognition"):
                            continue
                    if target == "sherpa":
                        family = next((f for f in ("transducer", "paraformer", "whisper", "moonshine") if f in low), "")
                        if "sherpa-onnx" not in low or not family:
                            continue
                    if target in {"qwen", "funasr", "nemo"}:
                        family = "qwen3" if target == "qwen" else ("sensevoice" if "sensevoice" in low else "paraformer" if "paraformer" in low else "funasr") if target == "funasr" else "canary" if "canary" in low else "parakeet"
                    if target == "crisp" or (target == "translation" and "gguf" in low):
                        variants = [s for s in siblings if s.endswith(".gguf")]
                    elif target == "whisper_cpp":
                        variants = [s for s in siblings if Path(s).name.startswith("ggml-") and s.endswith(".bin") and not any(tag in s.lower() for tag in ("silero", "vad"))]
                    else:
                        variants = [""]
                    for filename in variants:
                        engines = [target, "whisperx"] if target == "faster_whisper" else [target, "crisp"] if target == "whisper_cpp" else [target]
                        shard = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", filename)
                        companion_files = []
                        if shard:
                            if int(shard[1]) != 1:
                                continue
                            prefix = filename[:shard.start()]
                            companion_files = [s for s in siblings if s.startswith(prefix + "-") and s != filename and s.endswith(f"-of-{shard[2]}.gguf")]
                        discovered.append({"id": artifact_id(info.id, filename), "name": name + (" · " + Path(filename).stem if filename else ""), "repo": info.id, "filename": filename, "companion_files": companion_files, "source": "hf", "engines": engines, "format": fmt if target == "whisper_cpp" else "gguf" if filename else fmt, "family": family, "languages": card.get("language", []), "license": card.get("license", ""), "discovered_at": time.time()})
            except Exception as exc:
                errors.append(author + ": " + str(exc))
        with _LOCK:
            combined = list(merge_models(read_json(self.root / "models.json", []) + discovered).values())
            write_json(self.root / "models.json", combined)
            write_json(self.root / "refresh.json", {"time": time.time(), "errors": errors, "new_count": len(discovered)})
        return {"count": len(discovered), "errors": errors}
