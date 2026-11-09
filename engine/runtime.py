"""Install only the selected engine, in its own reusable Python environment."""
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from core.paths import ROOT, location
from core.storage import fetch_json, read_json, write_json, download, extract_archive
from engine.specs import ENGINES

_LOCKS = {}
_LOCK = threading.Lock()
VERSIONS = read_json(Path(__file__).with_name("runtime_versions.json"), {})
MODULES = {"faster_whisper": "faster_whisper", "whisper_cpp": "pywhispercpp.model", "openai_whisper": "whisper", "transformers": "transformers", "qwen": "qwen_asr", "funasr": "funasr", "nemo": "nemo.collections.asr", "mlx": "mlx_whisper", "vosk": "vosk", "whisperx": "whisperx", "sherpa": "sherpa_onnx", "bfa": "bournemouth_aligner", "translation": "transformers"}


def child_env():
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", UV_CACHE_DIR=str(location("cache/uv")))
    env["PYTHONPATH"] = str(ROOT)
    return env


def run_logged(command, emit=None, cancelled=None, cwd=None, env=None, timeout=1800):
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen([str(v) for v in command], cwd=cwd or ROOT, env=env or child_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=flags)
    lines = []
    def read():
        for line in process.stdout:
            lines.append(line.rstrip())
            if emit:
                emit(line.rstrip())
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    started = time.monotonic()
    try:
        while process.poll() is None:
            if cancelled and cancelled():
                raise InterruptedError("أُلغي تجهيز المحرك")
            if time.monotonic() - started > timeout:
                raise TimeoutError("استغرق تجهيز المحرك أكثر من المهلة؛ أعد المحاولة من الإصلاح")
            time.sleep(0.15)
    except BaseException:
        stop_tree(process)
        raise
    finally:
        reader.join(timeout=2)
    if process.returncode:
        raise RuntimeError("\n".join(lines[-25:]) or f"فشل تشغيل الأداة ({process.returncode})")
    return "\n".join(lines)


def stop_tree(process):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children[::-1]:
            try:
                child.kill()
            except psutil.Error:
                pass
        try:
            parent.kill()
        except psutil.Error:
            pass
        process.wait(timeout=5)
    except (psutil.Error, subprocess.TimeoutExpired):
        pass


def uv_binary():
    existing = shutil.which("uv")
    if existing:
        return Path(existing)
    root = location("tools/uv-0.8.14")
    found = list(root.rglob("uv.exe" if sys.platform == "win32" else "uv"))
    if found:
        return found[0]
    machine = platform.machine().lower()
    arch = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
    target = f"uv-{arch}-" + {"win32": "pc-windows-msvc.zip", "darwin": "apple-darwin.tar.gz", "linux": "unknown-linux-gnu.tar.gz"}[sys.platform]
    info = fetch_json("https://api.github.com/repos/astral-sh/uv/releases/tags/0.8.14")
    asset = next((a for a in info["assets"] if a["name"] == target), None)
    if not asset or not asset.get("digest"):
        raise RuntimeError("لم توجد حزمة uv موثقة لهذه المنصة")
    archive = download(asset["browser_download_url"], root / target, sha256=asset["digest"])
    extract_archive(archive, root)
    binary = next(root.rglob("uv.exe" if sys.platform == "win32" else "uv"))
    if sys.platform != "win32":
        binary.chmod(0o755)
    return binary


class Runtimes:
    def __init__(self):
        self.root = location("runtimes")

    def python_environment(self, engine, profile):
        package = "bournemouth-forced-aligner" if engine == "bfa" else "transformers" if engine == "translation" else ENGINES[engine].package
        name = package.split("[")[0]
        extras = "[" + package.split("[", 1)[1] if "[" in package else ""
        package = name + extras + ("==" + VERSIONS[name] if name in VERSIONS else "")
        signature = hashlib.sha256((package + "|" + str(profile.get("device", "cpu")) + "|py312|v2").encode()).hexdigest()[:12]
        root = self.root / (engine + "-" + signature)
        return package, root, root / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def native_folder(self, engine, profile):
        arch = "arm64" if platform.machine().lower() in {"aarch64", "arm64"} else "x86_64"
        gpu = profile.get("native_gpu", "cpu") if engine == "crisp" else "cpu"
        return self.root / f"{engine}-{sys.platform}-{arch}-{gpu}"

    def status(self, engine, profile=None):
        """Read installation facts without installing, importing ML or networking."""
        import importlib.util
        import importlib.metadata
        profile = profile or {}
        reason = ENGINES[engine].unavailable_reason() if engine in ENGINES else ""
        if reason:
            return {"state": "unavailable", "message": reason}
        custom = profile.get("native_binary" if engine == "crisp" else "llama_binary" if engine == "llama" else "runtime_python", "")
        if custom:
            path = Path(custom).expanduser().resolve()
            return {"state": "custom" if path.is_file() else "broken", "path": str(path), "message": "ملف مخصص موجود؛ استخدم فحص المحرك للتحقق من تشغيله" if path.is_file() else "مسار المحرك المخصص غير موجود"}
        _, private_root, private_binary = self.python_environment(engine, profile) if engine not in {"crisp", "llama"} else (None, self.root, self.root / "missing")
        private_ready = (private_root / "whisper-ready.json").is_file() and private_binary.is_file()
        if engine == "faster_whisper" and not private_ready and importlib.util.find_spec("faster_whisper"):
            return {"state": "installed", "path": sys.executable, "version": importlib.metadata.version("faster-whisper"), "message": "مثبت ضمن بيئة البرنامج الأساسية"}
        if engine in {"crisp", "llama"}:
            folder = self.native_folder(engine, profile)
            ready = read_json(folder / "ready.json", {})
            path = Path(ready.get("binary", folder / "missing"))
            version = ready.get("version", "")
        else:
            package, folder, path = self.python_environment(engine, profile)
            ready = read_json(folder / "whisper-ready.json", {})
            version = ready.get("package", package)
        installed = bool(ready) and path.is_file()
        return {"state": "installed" if installed else "broken" if ready else "missing", "path": str(path), "version": version, "message": "مثبت وجاهز وفق آخر فحص تثبيت" if installed else "ملفات التثبيت ناقصة" if ready else "لم يثبت هذا المحرك بهذه الإعدادات بعد"}

    def probe(self, engine, profile=None):
        status = self.status(engine, profile)
        if status["state"] not in {"installed", "custom"}:
            raise RuntimeError(status["message"])
        binary = status["path"]
        command = [binary, "--version"] if engine in {"crisp", "llama"} else [binary, "-c", "import " + MODULES[engine] + "; print('Engine import OK')"]
        details = run_logged(command, timeout=60)
        return {"engine_state": "installed", "details": details, "verified_at": time.time()}

    def ensure(self, engine, profile=None, emit=None, cancelled=None, repair=False):
        profile = profile or {}
        if engine in ENGINES and ENGINES[engine].unavailable_reason():
            raise RuntimeError(ENGINES[engine].unavailable_reason())
        if engine == "crisp" or engine == "llama":
            return self.native(engine, profile, emit, cancelled, repair)
        if profile.get("runtime_python"):
            binary = Path(profile["runtime_python"]).expanduser().resolve()
            if not binary.is_file():
                raise ValueError("مسار Python المخصص غير موجود")
            return str(binary)
        package, root, binary = self.python_environment(engine, profile)
        ready = root / "whisper-ready.json"
        if ready.exists() and binary.is_file() and not repair:
            return str(binary)
        if engine == "faster_whisper" and not repair:
            try:
                import importlib.util
                if importlib.util.find_spec("faster_whisper"):
                    return sys.executable
            except ImportError:
                pass
        with _LOCK:
            lock = _LOCKS.setdefault(str(root), threading.Lock())
        with lock:
            if ready.exists() and binary.exists() and not repair:
                return str(binary)
            if profile.get("local_files_only"):
                raise RuntimeError("هذا المحرك لم يُجهّز بعد، ووضع دون تنزيل مفعل")
            uv = uv_binary()
            if emit:
                emit("إعداد بيئة " + engine)
            if not binary.exists():
                run_logged([uv, "venv", "--python", "3.12", root], emit, cancelled)
            args = [uv, "pip", "install", "--quiet", "--python", binary, package, "numpy", "av", "psutil", "soundfile"]
            if repair:
                args.append("--reinstall")
            if engine in {"openai_whisper", "transformers", "qwen", "funasr", "nemo", "whisperx", "translation"}:
                help_text = subprocess.run([str(uv), "pip", "install", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
                if "--torch-backend" in help_text:
                    args += ["--torch-backend", "cpu" if profile.get("device", "cpu") in {"cpu", "mps"} else "auto"]
            run_logged(args, emit, cancelled)
            run_logged([binary, "-c", "import " + MODULES[engine]], emit, cancelled, timeout=240)
            freeze = run_logged([uv, "pip", "freeze", "--python", binary], cancelled=cancelled)
            (root / "installed.lock.txt").write_text(freeze + "\n", encoding="utf-8")
            write_json(ready, {"package": package, "python": "3.12", "created_at": time.time()})
        return str(binary)

    def native(self, engine, profile, emit=None, cancelled=None, repair=False):
        custom = profile.get("native_binary", "") if engine == "crisp" else profile.get("llama_binary", "")
        if custom:
            binary = Path(custom).expanduser().resolve()
            if not binary.is_file():
                raise ValueError("ملف المحرك المخصص غير موجود")
            return str(binary)
        machine = platform.machine().lower()
        arch = "arm64" if machine in {"aarch64", "arm64"} else "x86_64"
        gpu = profile.get("native_gpu", "cpu") if engine == "crisp" else "cpu"
        folder = self.native_folder(engine, profile)
        ready = read_json(folder / "ready.json", {})
        if not repair and ready.get("binary") and Path(ready["binary"]).is_file():
            return ready["binary"]
        if profile.get("local_files_only"):
            raise RuntimeError("ملفات المحرك غير مثبتة ووضع دون تنزيل مفعل")
        with _LOCK:
            lock = _LOCKS.setdefault(str(folder), threading.Lock())
        with lock:
            ready = read_json(folder / "ready.json", {})
            if not repair and ready.get("binary") and Path(ready["binary"]).is_file():
                return ready["binary"]
            repo = "CrispStrobe/CrispASR" if engine == "crisp" else "ggml-org/llama.cpp"
            info = fetch_json(f"https://api.github.com/repos/{repo}/releases/" + ("tags/v0.8.51" if engine == "crisp" else "latest"))
            if engine == "llama" and not any(a["name"].endswith((".zip", ".tar.gz")) for a in info["assets"]):
                nightly = next((a for a in info["assets"] if a["name"] == "nightly-tag.txt"), None)
                if nightly:
                    with urllib.request.urlopen(nightly["browser_download_url"], timeout=30) as response:
                        tag = response.read().decode().strip()
                    if not re.fullmatch(r"[A-Za-z0-9._-]+", tag):
                        raise ValueError("وسم إصدار غير صالح")
                    info = fetch_json(f"https://api.github.com/repos/{repo}/releases/tags/{tag}")
            if engine == "crisp":
                osname = {"win32": "windows", "linux": "linux", "darwin": "macos"}[sys.platform]
                suffix = "-cpu" if sys.platform == "win32" and gpu == "cpu" else "" if gpu in {"cpu", "metal"} else "-" + gpu
                expected = f"crispasr-{osname}-{arch}{suffix}" + (".zip" if sys.platform == "win32" else ".tar.gz")
                assets = [a for a in info["assets"] if a["name"] == expected]
            else:
                platform_words = {"win32": ("win", "x64" if arch == "x86_64" else "arm64"), "linux": ("ubuntu", "x64" if arch == "x86_64" else "arm64"), "darwin": ("macos", "arm64" if arch == "arm64" else "x64")}[sys.platform]
                assets = [a for a in info["assets"] if all(w in a["name"].lower() for w in platform_words) and a["name"].endswith((".zip", ".tar.gz")) and not any(w in a["name"].lower() for w in ("cuda", "rocm", "sycl", "vulkan", "cudart", "rpc"))]
            if not assets:
                raise RuntimeError("لا توجد حزمة رسمية جاهزة لهذا النظام والتسريع؛ اختر CPU أو مسار محرك جاهز")
            asset = sorted(assets, key=lambda a: a["size"])[0]
            if not asset.get("digest"):
                raise RuntimeError("المصدر لا يوفر بصمة الحزمة؛ اختر ملف محرك موثق يدويًا")
            folder.mkdir(parents=True, exist_ok=True)
            archive = download(asset["browser_download_url"], folder / asset["name"], asset["digest"], progress=(lambda a, b: emit(f"تنزيل المحرك: {a // 1048576}/{b // 1048576} MB")) if emit else None, cancelled=cancelled)
            extract_archive(archive, folder)
            names = ["crispasr"] if engine == "crisp" else ["llama-server", "llama"]
            binary = next((p for name in names for p in folder.rglob(name + (".exe" if sys.platform == "win32" else "")) if p.is_file()), None)
            if not binary:
                raise RuntimeError("لم يوجد الملف التنفيذي داخل الحزمة")
            if sys.platform != "win32":
                binary.chmod(0o755)
            write_json(folder / "ready.json", {"binary": str(binary), "version": info["tag_name"], "sha256": asset["digest"]})
            if engine == "crisp":
                text = run_logged([binary, "--list-backends-json"], cancelled=cancelled, timeout=30)
                try:
                    capabilities = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
                    write_json(location("catalog") / "crisp_capabilities.json", capabilities)
                except (ValueError, KeyError):
                    pass
            return str(binary)
