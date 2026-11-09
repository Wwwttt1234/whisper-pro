"""Small frozen launcher: Python need not be installed by the end user."""
import os
import platform
import queue
import re
import signal
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk
from pathlib import Path
from launcher import setup_ready


VC_REDIST = "https://aka.ms/vc14/vc_redist.x64.exe"


def preflight():
    machine = platform.machine().lower()
    if sys.platform == "win32":
        if machine not in {"amd64", "x86_64"} or os.environ.get("PROCESSOR_ARCHITEW6432") == "ARM64":
            return "تحتاج هذه الحزمة Windows بمعالج x64.", None
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        if any(not (system / dll).is_file() for dll in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")):
            return "ثبّت Microsoft Visual C++ Redistributable (x64)، ثم اضغط زر الإعداد والتشغيل مرة أخرى.", VC_REDIST
    elif machine not in {"amd64", "x86_64", "arm64", "aarch64"}:
        return "تحتاج الحزمة جهازًا بمعالج x64 أو ARM64.", None
    elif sys.platform == "darwin":
        if int(platform.mac_ver()[0].split(".")[0]) < 14:
            return "حدّث نظامك إلى macOS 14 أو أحدث ثم أعد فتح البرنامج.", None
    elif sys.platform == "linux":
        try:
            glibc = os.confstr("CS_GNU_LIBC_VERSION") or ""
        except (OSError, ValueError):
            glibc = ""
        version = re.fullmatch(r"glibc (\d+)\.(\d+)", glibc)
        if not version or tuple(map(int, version.groups())) < (2, 28):
            return "تحتاج الحزمة توزيعة Linux فيها glibc 2.28 أو أحدث.", None
    else:
        return "اختر حزمة Windows أو macOS أو Linux المطابقة لجهازك.", None
    return "", None


def source_root():
    source = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return source / "app" if (source / "app/pyproject.toml").exists() else source


def self_test(target):
    """Packaging check without opening a window or installing anything."""
    import json
    import hashlib
    import tomllib
    source = source_root()
    manifest = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    models = json.loads((source / "engine/model_seeds.json").read_text(encoding="utf-8"))
    assert (source / "launcher.py").is_file() and (source / "ui/application.py").is_file()
    assert len(models) == len({m["id"] for m in models})
    Path(target).write_text(json.dumps({"version": manifest["project"]["version"], "models": len(models), "lock_sha256": hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest()}), encoding="utf-8")


def main():
    from core.paths import data_dir
    from engine.runtime import uv_binary
    source = source_root()
    data = data_dir()
    environment = data / "app-environment-v2"
    messages = queue.Queue()
    process = None
    current_url = None
    closing = threading.Event()
    process_guard = threading.Lock()
    window = tk.Tk()
    window.title("Whisper Pro")
    window.geometry("620x330")
    ttk.Label(window, text="Whisper Pro", font=("Arial", 22, "bold")).pack(pady=12)
    label = ttk.Label(window, wraplength=580, justify="right")
    label.pack(padx=15, pady=5)
    progress = ttk.Progressbar(window, mode="indeterminate")
    progress.pack(fill="x", padx=20, pady=12)
    details = tk.Text(window, height=8, wrap="word", state="disabled")
    details.pack(fill="both", expand=True, padx=15, pady=8)

    def run():
        nonlocal process
        try:
            uv = uv_binary()
            env = os.environ.copy()
            env.setdefault("UV_HTTP_TIMEOUT", "120")
            env.setdefault("UV_CACHE_DIR", str(data / "cache/uv"))
            env.update(WHISPER_PRO_DATA_DIR=str(data), UV_PROJECT_ENVIRONMENT=str(environment), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
            command = [str(uv), "run", "--project", str(source), "--locked", "--no-dev", "--python", "3.12", "python", str(source / "launcher.py")]
            with process_guard:
                if closing.is_set():
                    return
                process = subprocess.Popen(command, env=env, cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0, start_new_session=os.name != "nt")
            with process.stdout:
                for line in process.stdout:
                    messages.put(line.rstrip())
            if process.wait():
                messages.put("ERROR: تعذر إكمال التشغيل. راجع التفاصيل ثم اضغط الزر لإعادة المحاولة.")
            else:
                messages.put("CLOSED")
        except Exception as exc:
            messages.put("ERROR: " + str(exc))

    def refresh_button():
        ready = setup_ready(source, environment)
        action.config(text="تشغيل" if ready else "الإعداد والتشغيل", state="normal")

    def start():
        if current_url:
            webbrowser.open(current_url)
            return
        problem, link = preflight()
        if problem:
            label.config(text=problem)
            if link:
                if messagebox.askyesno("متطلب قبل الإعداد", problem + "\n\nهل تفتح رابط التنزيل من Microsoft؟", parent=window):
                    webbrowser.open(link)
            else:
                messagebox.showerror("متطلب قبل الإعداد", problem, parent=window)
            return
        ready = setup_ready(source, environment)
        if not ready and not messagebox.askokcancel(
            "تأكيد الإعداد",
            "سيجهّز Whisper Pro بيئة Python 3.12 ومكتبات التطبيق في مجلد البيانات. "
            "اترك الاتصال بالإنترنت متاحًا حتى يكتمل الإعداد. سيفتح البرنامج تلقائيًا عند الانتهاء.\n\n"
            + str(data),
            parent=window,
        ):
            return
        action.config(state="disabled")
        label.config(text="جارٍ فتح البرنامج…" if ready else "جارٍ تجهيز البرنامج…")
        progress.start()
        threading.Thread(target=run, daemon=True).start()

    action = ttk.Button(window, command=start)
    action.pack(pady=8, before=progress)
    refresh_button()
    label.config(text="اضغط تشغيل لفتح البرنامج." if setup_ready(source, environment) else "اضغط الإعداد والتشغيل لفحص المتطلبات والبدء.")

    def poll():
        nonlocal current_url, poll_id
        try:
            while True:
                line = messages.get_nowait()
                if line.startswith("WHISPER_PRO_URL="):
                    current_url = line.removeprefix("WHISPER_PRO_URL=")
                    progress.stop()
                    refresh_button()
                    label.config(text="البرنامج يعمل في المتصفح. إغلاق هذه النافذة يوقفه.")
                elif line.startswith("ERROR:"):
                    current_url = None
                    progress.stop()
                    label.config(text=line.removeprefix("ERROR:"))
                    refresh_button()
                elif line == "CLOSED":
                    current_url = None
                    progress.stop()
                    label.config(text="توقف البرنامج. اضغط تشغيل لفتحه مرة أخرى.")
                    refresh_button()
                details.config(state="normal")
                details.insert("end", re.sub(r"\x1b\[[0-9;]*m", "", line) + "\n")
                details.see("end")
                details.config(state="disabled")
        except queue.Empty:
            pass
        poll_id = window.after(150, poll)

    def close():
        closing.set()
        window.after_cancel(poll_id)
        with process_guard:
            active = process
        if active and active.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(active.pid), "/T", "/F"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                try:
                    os.killpg(active.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    poll_id = window.after(150, poll)
    window.mainloop()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--self-test":
        self_test(sys.argv[2])
    else:
        main()
