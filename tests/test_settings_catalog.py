import copy
import pytest

from core.settings import initial_settings, import_settings, export_settings, save_settings, load_settings, validate_profile
from engine.catalog import Catalog, relevant_files, fingerprint
from engine.specs import ENGINES, fields_for


def test_settings_roundtrip_and_secret_exclusion():
    settings = initial_settings()
    settings["use_auth_token"] = "SECRET_SHOULD_NOT_EXPORT"
    settings["profiles"]["faster_whisper"] = {"example": {"beam_size": 7, "use_auth_token": "ALSO_SECRET"}}
    exported = export_settings(settings)
    assert "SECRET" not in exported
    save_settings(settings)
    assert load_settings()["profiles"]["faster_whisper"]["example"]["beam_size"] == 7
    assert import_settings(exported) == load_settings()


@pytest.mark.parametrize("key,value", [("beam_size", "seven"), ("threads", 0), ("temperature", float("nan")), ("suppress_tokens", "1,word"), ("clip_timestamps", "30,10"), ("temperature_list", "0,2")])
def test_invalid_profile_rejected(key, value):
    with pytest.raises(ValueError):
        validate_profile("faster_whisper", {}, {key: value})


def test_invalid_import_does_not_overwrite_saved_file():
    settings = initial_settings()
    save_settings(settings)
    invalid = copy.deepcopy(settings)
    invalid["profiles"] = {"faster_whisper": {"model": {"beam_size": "bad"}}}
    with pytest.raises(ValueError):
        save_settings(invalid)
    assert load_settings() == settings


def test_every_engine_has_seed_and_unique_artifacts():
    catalog = Catalog()
    models = catalog.models()
    assert len({m["id"] for m in models}) == len(models)
    for engine in ENGINES:
        assert catalog.models(engine), engine
    fw = catalog.models("faster_whisper")
    assert len({m["repo"] for m in fw}) == len(fw)


def test_documentation_change_is_not_a_model_update():
    model = {"format": "ct2"}
    files = [{"rfilename": "model.bin", "blobId": "weights", "size": 12}, {"rfilename": "config.json", "blobId": "config", "size": 10}, {"rfilename": "README.md", "blobId": "old"}]
    old = fingerprint(relevant_files(model, files))
    files[-1]["blobId"] = "new-docs"
    assert old == fingerprint(relevant_files(model, files))
    files[0]["blobId"] = "new-weights"
    assert old != fingerprint(relevant_files(model, files))


def test_ct2_rejects_unconverted_safetensors():
    with pytest.raises(ValueError):
        relevant_files({"format": "ct2"}, [{"rfilename": "model.safetensors"}])


@pytest.mark.parametrize("unsafe_field", ["sha", "blobId"])
def test_model_source_cannot_escape_cache(tmp_path, monkeypatch, unsafe_field):
    import engine.catalog as module

    catalog = Catalog()
    model = {"id": "fixture", "source": "hf", "repo": "Systran/fixture", "format": "ct2"}
    remote = {"sha": "a" * 40, "files": {"model.bin": {"blobId": "b" * 40, "size": 10}}}
    if unsafe_field == "sha":
        remote["sha"] = str(tmp_path / "outside")
    else:
        remote["files"]["model.bin"]["blobId"] = str(tmp_path / "outside")
    monkeypatch.setattr(catalog, "check", lambda *args: remote)
    monkeypatch.setattr(module, "download", lambda *args, **kwargs: pytest.fail("Unsafe path reached downloader"))
    with pytest.raises(ValueError, match="مسار"):
        catalog.ensure(model, {}, models_dir=str(tmp_path / "cache"))
    assert not (tmp_path / "outside").exists()


def test_capability_fields_change_with_model():
    whisper = {f.key for f in fields_for("crisp", {"family": "whisper"})}
    parakeet = {f.key for f in fields_for("crisp", {"family": "parakeet"})}
    qwen = {f.key for f in fields_for("crisp", {"family": "qwen3"})}
    assert "no_speech_threshold" in whisper and "no_speech_threshold" not in parakeet
    assert "kv_quant" in qwen and "kv_quant" not in parakeet
    assert "max_new_tokens" in qwen and "max_new_tokens" not in parakeet
    for engine in ENGINES:
        for field in fields_for(engine):
            assert field.explanation and field.recommendation
    assert not ({"threads", "chunk_seconds"} & {f.key for f in fields_for("vosk")})
    assert next(f for f in fields_for("openai_whisper") if f.key == "dtype").choices == ("float32", "float16")


def test_model_status_unknown_current_update_and_damage(tmp_path, monkeypatch):
    catalog = Catalog()
    path = tmp_path / "model.bin"
    path.write_bytes(b"test")
    model = {"id": "test", "source": "hf", "repo": "Systran/test", "format": "ct2"}
    catalog.record("test", {"path": str(path), "files": {"model.bin": {"size": 4}}, "fingerprint": "local"})
    assert catalog.status(model)[0] == "unknown"
    from core.storage import write_json
    write_json(catalog.root / "checks.json", {"test": {"fingerprint": "local"}})
    assert catalog.status(model)[0] == "current"
    write_json(catalog.root / "checks.json", {"test": {"fingerprint": "remote"}})
    assert catalog.status(model)[0] == "update"
    path.write_bytes(b"bad")
    assert catalog.status(model)[0] == "repair"


def test_discovery_adds_shared_cpp_files_and_translation_family(monkeypatch):
    import types
    import huggingface_hub
    import inspect
    signature = inspect.signature(huggingface_hub.HfApi.list_models)
    calls = []
    class FakeAPI:
        def __init__(self, **kwargs):
            pass
        def list_models(self, **kwargs):
            signature.bind(self, **kwargs)
            calls.append(kwargs)
            repo, files = ("ggerganov/whisper.cpp", ["ggml-new-test.bin", "ggml-silero-v6.bin"]) if kwargs["author"] == "ggerganov" else ("tencent/Hy-MT-test", ["config.json", "model.safetensors"]) if kwargs["author"] == "tencent" else (None, [])
            return [] if repo is None else [types.SimpleNamespace(id=repo, siblings=[types.SimpleNamespace(rfilename=f) for f in files], pipeline_tag=None, card_data={}, library_name=None)]
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeAPI)
    catalog = Catalog()
    catalog.refresh("whisper_cpp")
    item = catalog.get("hf:ggerganov/whisper.cpp#ggml-new-test.bin")
    assert item["format"] == "ggml" and set(item["engines"]) == {"crisp", "whisper_cpp"}
    assert catalog.get("hf:ggerganov/whisper.cpp#ggml-silero-v6.bin") is None
    catalog.refresh("translation")
    assert catalog.get("hf:tencent/Hy-MT-test")["family"] == "causal"
    assert all("limit" not in call for call in calls)
