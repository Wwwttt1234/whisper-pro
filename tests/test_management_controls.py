import sys
import types
from pathlib import Path

import pytest

from core.outputs import ass_style, write_ass
from core.settings import GENERAL, validate_profile
from core.storage import write_json
from engine.adapters import make_chunk_adapter
from engine.catalog import Catalog, relevant_files
from engine.runtime import Runtimes


def test_runtime_status_is_read_only_and_matches_reused_environment(monkeypatch):
    import engine.runtime as module
    monkeypatch.setattr(module, "run_logged", lambda *a, **k: pytest.fail("status must not execute or install anything"))
    runtimes = Runtimes()
    assert runtimes.status("whisper_cpp")["state"] == "missing"
    package, folder, binary = runtimes.python_environment("whisper_cpp", {})
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test interpreter")
    write_json(folder / "whisper-ready.json", {"package": package})
    assert runtimes.status("whisper_cpp")["state"] == "installed"
    assert runtimes.ensure("whisper_cpp") == str(binary)
    binary.unlink()
    assert runtimes.status("whisper_cpp")["state"] == "broken"


def test_repaired_faster_environment_takes_precedence_over_base():
    runtimes = Runtimes()
    package, folder, binary = runtimes.python_environment("faster_whisper", {})
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"test interpreter")
    write_json(folder / "whisper-ready.json", {"package": package})
    assert runtimes.status("faster_whisper")["path"] == str(binary)
    assert runtimes.ensure("faster_whisper") == str(binary)


def test_cpp_additional_controls_reach_real_adapter_interface(monkeypatch):
    seen = {}
    class FakeModel:
        def __init__(self, path, **kwargs):
            seen.update(kwargs)
        def transcribe(self, samples, **kwargs):
            seen.update(kwargs)
            return [types.SimpleNamespace(t0=0, t1=100, text="hello")]
    monkeypatch.setitem(sys.modules, "pywhispercpp", types.ModuleType("pywhispercpp"))
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", types.SimpleNamespace(Model=FakeModel))
    profile = validate_profile("whisper_cpp", {}, {"sampling_strategy": "greedy", "cpp_processors": 2, "token_timestamps": True, "max_len": 42, "cpp_max_tokens": 64, "suppress_regex": "noise"})
    infer = make_chunk_adapter("whisper_cpp", "test.bin", {}, profile, "en", "transcribe")
    assert infer([0])["segments"][0]["text"] == "hello"
    assert seen["params_sampling_strategy"] == 0 and seen["n_processors"] == 2
    assert seen["max_len"] == 42 and seen["max_tokens"] == 64 and seen["suppress_regex"] == "noise"


def test_invalid_control_combinations_fail_before_model_load():
    with pytest.raises(ValueError):
        validate_profile("whisper_cpp", {}, {"max_len": 42, "token_timestamps": False})
    with pytest.raises(ValueError):
        validate_profile("transformers", {"family": "whisper"}, {"do_sample": True, "temperature": 0})
    with pytest.raises(ValueError):
        validate_profile("sherpa", {"family": "transducer"}, {"hotwords_file": "terms.txt"})


def test_ass_customization_changes_export(tmp_path):
    config = {**GENERAL, "ass_width": 1920, "ass_height": 1080, "ass_font_name": "TestFont", "ass_font_size": 48, "ass_text_color": "#112233", "ass_bold": True, "ass_alignment": 8}
    path = tmp_path / "custom.ass"
    write_ass(path, [{"start": 0, "end": 1, "text": "hello"}], ass_style(config))
    text = path.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in text and "PlayResY: 1080" in text
    assert "TestFont, 48, &H00332211" in text and ", -1, 0, 0, 0," in text
    assert ", 8, 10, 10, 10," in text


def test_bulk_metadata_checks_reuse_repo_and_include_all_sizes(monkeypatch):
    import engine.catalog as module
    catalog = Catalog()
    models = [{"id": "one", "name": "one", "source": "hf", "repo": "cstr/shared", "filename": "one.gguf"}, {"id": "two", "name": "two", "source": "hf", "repo": "cstr/shared", "filename": "two.gguf"}]
    monkeypatch.setattr(catalog, "models", lambda engine: models)
    calls = []
    def fetch(*args):
        calls.append(args)
        return {"sha": "revision", "siblings": [{"rfilename": "one.gguf", "size": 12, "blobId": "a"}, {"rfilename": "two.gguf", "size": 24, "blobId": "b"}]}
    monkeypatch.setattr(module, "fetch_json", fetch)
    assert catalog.check_many("crisp")["errors"] == []
    assert len(calls) == 1
    assert [entry["size"] for entry in catalog.inventory("crisp")] == [12, 24]


def test_transformers_download_chooses_one_weight_variant():
    siblings = [{"rfilename": name} for name in ["config.json", "model.safetensors", "model.fp16.safetensors", "pytorch_model.bin", "onnx/model.onnx"]]
    chosen = relevant_files({"format": "transformers"}, siblings)
    assert set(chosen) == {"config.json", "model.safetensors"}


def test_transformers_keeps_standard_bin_when_safetensors_are_auxiliary():
    files = [{"rfilename": n} for n in ["config.json", "pytorch_model.bin", "onnx/auxiliary.safetensors"]]
    assert set(relevant_files({"format": "transformers"}, files)) == {"config.json", "pytorch_model.bin"}


def test_local_model_size_and_presence_have_no_download_action(tmp_path):
    model = tmp_path / "local.bin"
    model.write_bytes(b"model data")
    catalog = Catalog()
    item = catalog.register_local("whisper_cpp", model, "whisper")
    entry = next(e for e in catalog.inventory("whisper_cpp") if e["model"]["id"] == item["id"])
    assert entry["state"] == "local" and entry["size"] == 10


@pytest.mark.parametrize("family", ["transducer", "paraformer"])
def test_streaming_sherpa_uses_online_factory_and_flushes_input(tmp_path, monkeypatch, family):
    seen = {"decoded": 0}
    for name in ["tokens.txt", "encoder.onnx", "decoder.onnx", "joiner.onnx"]:
        (tmp_path / name).write_bytes(b"fixture")
    class Stream:
        def accept_waveform(self, rate, samples):
            pass
        def input_finished(self):
            seen["finished"] = True
    class Recognizer:
        @classmethod
        def from_transducer(cls, **kwargs):
            seen.update(kwargs)
            return cls()
        from_paraformer = from_transducer
        def create_stream(self):
            return Stream()
        def is_ready(self, stream):
            return seen["decoded"] == 0
        def decode_stream(self, stream):
            seen["decoded"] += 1
        def get_result(self, stream):
            return "hello"
    monkeypatch.setitem(sys.modules, "sherpa_onnx", types.SimpleNamespace(OnlineRecognizer=Recognizer))
    infer = make_chunk_adapter("sherpa", tmp_path, {"name": "sherpa-onnx-streaming-model", "family": family}, {}, "en", "transcribe")
    assert infer([0])["text"] == "hello"
    assert seen["finished"] and seen["decoded"] == 1
    assert Path(seen["encoder"]).name == "encoder.onnx"
