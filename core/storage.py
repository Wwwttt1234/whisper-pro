"""Atomic metadata writes and downloads; never execute a remote manifest."""
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def process_lock(path, cancelled=None, timeout=1800):
    """OS lock is released even if a downloader is killed or the app crashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            if cancelled and cancelled():
                raise InterruptedError("أُلغي انتظار التنزيل")
            try:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("توجد عملية أخرى تستخدم الملفات؛ أعد المحاولة بعد انتهائها")
                time.sleep(0.1)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != "https":
            raise ValueError("إعادة التوجيه تتطلب رابط HTTPS")
        result = super().redirect_request(request, fp, code, msg, headers, newurl)
        if result is not None and urllib.parse.urlsplit(request.full_url).netloc.lower() != target.netloc.lower():
            result.remove_header("Authorization")
        return result


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.02 * (attempt + 1))
    finally:
        Path(tmp).unlink(missing_ok=True)


def fetch_json(url, token=None, timeout=30):
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("طلب بيانات المصدر يتطلب رابط HTTPS")
    headers = {"User-Agent": "WhisperPro/2.0", "Accept": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    opener = urllib.request.build_opener(_SafeRedirect())
    with opener.open(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def digest_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(url, destination, sha256=None, progress=None, cancelled=None, headers=None):
    """Resume only when the server confirms the exact byte range and entity."""
    if urllib.parse.urlparse(url).scheme != "https":
        raise ValueError("التنزيل يتطلب رابط HTTPS")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".incomplete")
    metadata = partial.with_name(partial.name + ".json")
    for attempt in range(3):
        if cancelled and cancelled():
            raise InterruptedError("أُلغي التنزيل")
        saved = read_json(metadata, {})
        offset = partial.stat().st_size if partial.exists() and saved.get("url") == url else 0
        request_headers = {"User-Agent": "WhisperPro/2.0", **(headers or {})}
        if offset and saved.get("etag"):
            request_headers.update({"Range": f"bytes={offset}-", "If-Range": saved["etag"]})
        else:
            offset = 0
        try:
            opener = urllib.request.build_opener(_SafeRedirect())
            with opener.open(urllib.request.Request(url, headers=request_headers), timeout=45) as response:
                resumed = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                if not resumed:
                    offset = 0
                write_json(metadata, {"url": url, "etag": response.headers.get("ETag", "")})
                total = offset + int(response.headers.get("Content-Length", 0))
                done = offset
                with open(partial, "ab" if resumed else "wb") as stream:
                    while True:
                        if cancelled and cancelled():
                            raise InterruptedError("أُلغي التنزيل")
                        chunk = response.read(1024 * 512)
                        if not chunk:
                            break
                        stream.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(done, total)
                if total and done != total:
                    raise OSError("التنزيل غير مكتمل")
            if sha256 and digest_file(partial).lower() != sha256.removeprefix("sha256:").lower():
                partial.unlink(missing_ok=True)
                metadata.unlink(missing_ok=True)
                raise ValueError("بصمة الملف لا تطابق المصدر")
            os.replace(partial, destination)
            metadata.unlink(missing_ok=True)
            return destination
        except InterruptedError:
            raise
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def extract_archive(archive, target):
    """Reject escaping paths, symlinks and archive device entries."""
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    def checked(name):
        normalized = name.replace("\\", "/")
        if ":" in normalized or normalized.startswith("/"):
            raise ValueError("مسار غير آمن في الحزمة")
        result = (target / normalized).resolve()
        if not result.is_relative_to(target):
            raise ValueError("مسار خارج مجلد الحزمة")
        return result

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                path = checked(item.filename)
                if (item.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("الحزمة تحتوي رابطًا رمزيًا")
                if item.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(item) as src, open(path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as source:
            for item in source:
                path = checked(item.name)
                if item.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif item.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(item) as src, open(path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    path.chmod(item.mode & 0o755)
                else:
                    raise ValueError("عنصر غير عادي في الحزمة")
