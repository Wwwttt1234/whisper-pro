import hashlib
import json
import os
import struct
from pathlib import Path

import pytest

from core.storage import write_json
from engine.catalog import Catalog
from engine.local_models import ModelScanner, default_roots, gguf_metadata


def ggml(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"lmgg" + struct.pack("<11i", 51864, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1) + b"\0" * 4096)
    return path


def gguf(path, architecture="whisper"):
    def string(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 2)
    for key, value in [("general.architecture", architecture), ("general.name", "Test model")]:
        data += string(key) + struct.pack("<I", 8) + string(value)
    path.write_bytes(data + b"\0" * 4096)
    return path


def ct2(folder):
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", {"alignment_heads": [[0, 0]], "lang_ids": [1]})
    (folder / "model.bin").write_bytes(b"fixture weights")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    return folder


def test_hf_cache_reused_under_original_catalog_identity_without_network(tmp_path, monkeypatch):
    import engine.catalog as module
    folder = ct2(tmp_path / "other-tool/hub/models--Systran--faster-whisper-tiny.en/snapshots/abcdef")
    catalog = Catalog()
    found = ModelScanner(catalog.models()).scan([tmp_path / "other-tool"])
    item = found["candidates"][0]
    assert item["catalog_id"] == "hf:Systran/faster-whisper-tiny.en"
    assert set(item["engines"]) == {"faster_whisper", "whisperx"}
    monkeypatch.setattr(module, "download", lambda *a, **k: pytest.fail("Existing weights must not be downloaded"))
    adopted = catalog.adopt(item)
    model = catalog.get(item["catalog_id"])
    assert catalog.ensure(model, {"local_files_only": True}) == str(folder)
    assert adopted["copied"] is False
    assert catalog.status(model)[0] == "unknown"


def test_ggml_header_and_shared_engines_ignoring_arbitrary_bin(tmp_path):
    good = ggml(tmp_path / "models/ggml-tiny.en.bin")
    (good.parent / "unrelated.bin").write_bytes(b"not a model")
    results = ModelScanner([]).scan([tmp_path])["candidates"]
    assert len(results) == 1
    assert results[0]["format"] == "ggml" and set(results[0]["engines"]) == {"crisp", "whisper_cpp"}


def test_same_file_aliases_not_duplicated(tmp_path):
    source = ggml(tmp_path / "a.bin")
    os.link(source, tmp_path / "b.bin")
    report = ModelScanner([]).scan([tmp_path, tmp_path])
    assert len(report["candidates"]) == 1 and len(report["candidates"][0]["alternatives"]) == 1


def test_unrelated_or_invalid_parent_config_does_not_hide_children(tmp_path):
    write_json(tmp_path / "config.json", ["unrelated config"])
    ggml(tmp_path / "nested/model.bin")
    # model.bin is a folder-format name, so use a standalone GGML name here.
    (tmp_path / "nested/model.bin").rename(tmp_path / "nested/ggml-tiny.bin")
    report = ModelScanner([]).scan([tmp_path])
    assert len(report["candidates"]) == 1 and report["warnings"]


def test_missing_transformer_shard_is_not_adoptable(tmp_path):
    folder = tmp_path / "partial"
    folder.mkdir()
    write_json(folder / "config.json", {"model_type": "whisper"})
    write_json(folder / "model.safetensors.index.json", {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}})
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"partial")
    report = ModelScanner([]).scan([tmp_path])
    assert not report["candidates"] and report["warnings"]


def test_gguf_architecture_separates_transcription_and_translation(tmp_path):
    gguf(tmp_path / "asr.gguf", "whisper")
    gguf(tmp_path / "language.gguf", "qwen3")
    gguf(tmp_path / "vision.gguf", "clip")
    found = ModelScanner([]).scan([tmp_path])["candidates"]
    assert len(found) == 2
    assert next(c for c in found if c["path"].endswith("asr.gguf"))["engines"] == ["crisp"]
    assert next(c for c in found if c["path"].endswith("language.gguf"))["engines"] == ["translation"]


def test_gguf_shards_and_invalid_header_are_rejected(tmp_path):
    gguf(tmp_path / "weights-00001-of-00002.gguf")
    report = ModelScanner([]).scan([tmp_path])
    assert not report["candidates"]
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1000000))
    with pytest.raises(ValueError):
        gguf_metadata(bad)


def test_ollama_model_manifest_uses_existing_blob(tmp_path):
    root = tmp_path / "ollama"
    digest = "a" * 64
    blob = gguf(root / "blobs" / ("sha256-" + digest), "llama")
    write_json(root / "manifests/registry.ollama.ai/library/demo/latest", {"layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + digest}]})
    found = ModelScanner([]).scan([root])["candidates"]
    assert len(found) == 1 and found[0]["path"] == str(blob)
    assert found[0]["name"].startswith("Ollama") and found[0]["engines"] == ["translation"]


def test_malformed_ollama_manifests_do_not_abort_other_models(tmp_path):
    root = tmp_path / "ollama"
    (root / "blobs").mkdir(parents=True)
    write_json(root / "manifests/empty", {"layers": None})
    write_json(root / "manifests/wrong", {"layers": [None, {"mediaType": "application/vnd.ollama.image.model", "digest": None}]})
    ggml(root / "other/ggml-model.bin")
    assert len(ModelScanner([]).scan([root])["candidates"]) == 1


def test_copy_keeps_source_and_repeat_import_does_not_duplicate(tmp_path):
    source = ggml(tmp_path / "outside/model.bin")
    source.rename(source.with_name("ggml-model.bin"))
    source = source.with_name("ggml-model.bin")
    catalog = Catalog()
    item = ModelScanner(catalog.models()).scan([source.parent])["candidates"][0]
    before = source.read_bytes()
    result = catalog.adopt(item, copy_files=True)
    copied = Path(result["path"])
    assert copied != source and copied.read_bytes() == source.read_bytes() == before
    assert catalog.adopt(item, copy_files=True)["path"] == str(copied)
    assert len([m for m in catalog.models() if m["id"] == result["adopted_model"]]) == 1


def test_copy_can_follow_reuse_and_cancellation_removes_partial_copy(tmp_path):
    source = ggml(tmp_path / "models/ggml-model.bin")
    catalog = Catalog()
    candidate = ModelScanner(catalog.models()).scan([source.parent])["candidates"][0]
    assert catalog.adopt(candidate)["path"] == str(source)
    copied = catalog.adopt(candidate, True)
    assert copied["copied"] and Path(copied["path"]) != source
    # A separate model is cancelled after copying begins.
    other = ggml(tmp_path / "other/another.bin")
    other.write_bytes(other.read_bytes() + b"more data")
    candidate = ModelScanner(catalog.models()).scan([other.parent])["candidates"][0]
    before = set((Path(copied["path"]).parents[1]).iterdir())
    with pytest.raises(InterruptedError):
        catalog.adopt(candidate, True, cancelled=lambda: True)
    assert set((Path(copied["path"]).parents[1]).iterdir()) == before
    assert other.is_file()


def test_source_changes_after_scan_require_rescan(tmp_path):
    source = ggml(tmp_path / "model.bin")
    source.rename(tmp_path / "ggml-model.bin")
    source = tmp_path / "ggml-model.bin"
    catalog = Catalog()
    candidate = ModelScanner([]).scan([tmp_path])["candidates"][0]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="تغير"):
        catalog.adopt(candidate)


def test_scan_limits_cancellation_and_env_cache_roots(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    assert tmp_path in default_roots()
    assert ModelScanner([]).scan([tmp_path], max_directories=0)["limited"]
    with pytest.raises(InterruptedError):
        ModelScanner([]).scan([tmp_path], cancelled=lambda: True)


def test_cloud_placeholder_not_opened_or_imported(tmp_path, monkeypatch):
    from types import SimpleNamespace
    cloud = gguf(tmp_path / "cloud.gguf")
    original = Path.stat
    def metadata(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path == cloud:
            return SimpleNamespace(st_file_attributes=0x400000, st_mode=result.st_mode, st_size=result.st_size)
        return result
    monkeypatch.setattr(Path, "stat", metadata)
    with pytest.raises(OSError):
        gguf_metadata(cloud)
    assert not ModelScanner([]).scan([tmp_path])["candidates"]


def test_imported_hf_weights_checked_by_content_not_assumed_current(tmp_path, monkeypatch):
    import engine.catalog as module
    folder = ct2(tmp_path / "hub/models--Systran--faster-whisper-tiny.en/snapshots/revision")
    catalog = Catalog()
    candidate = ModelScanner(catalog.models()).scan([folder])["candidates"][0]
    catalog.adopt(candidate)
    def meta(path):
        raw = path.read_bytes()
        return {"rfilename": path.name, "size": len(raw), "blobId": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()}
    remote = {"sha": "revision", "siblings": [meta(p) for p in folder.iterdir()]}
    monkeypatch.setattr(module, "fetch_json", lambda *a, **k: remote)
    model = catalog.get(candidate["catalog_id"])
    assert not catalog.check(model).get("error")
    assert catalog.status(model)[0] == "current"
    raw = (folder / "model.bin").read_bytes()
    (folder / "model.bin").write_bytes(b"X" + raw[1:])
    assert not catalog.check(model).get("error")
    assert catalog.status(model)[0] == "update"
