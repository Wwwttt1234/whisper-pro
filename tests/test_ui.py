import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


def test_engine_switching_help_save_and_invalid_export(tmp_path):
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    key = next(n.key for n in app.number_input if n.key and n.key.endswith(":beam_size"))
    app.number_input(key=key).set_value(7).run()
    next(b for b in app.sidebar.button if b.label == "حفظ الإعدادات").click().run()
    assert not app.exception
    import os
    saved = json.loads((Path(os.environ["WHISPER_PRO_DATA_DIR"]) / "settings.json").read_text(encoding="utf-8"))
    assert any(profile.get("beam_size") == 7 for profile in saved["profiles"]["faster_whisper"].values())
    for engine in ["crisp", "qwen", "vosk", "sherpa", "transformers", "whisper_cpp", "funasr", "mlx", "nemo", "whisperx", "openai_whisper"]:
        app.button(key="engine_choice:" + engine).click().run()
        assert not app.exception, engine
        assert len(app.tabs) == 6, engine
        keys = [n.key for n in app.number_input if n.key and n.key.startswith("setting:")]
        assert all(k.startswith("setting:" + engine + ":") for k in keys)
    app.button(key="engine_choice:vosk").click().run()
    assert not any(n.key and n.key.endswith(":beam_size") for n in app.number_input)
    next(m for m in app.multiselect if m.label == "صيغ الإخراج").set_value([]).run()
    assert not app.exception

def test_model_inventory_actions_selection_and_all_setting_hints(tmp_path, monkeypatch):
    from core.storage import write_json
    from engine.catalog import Catalog
    catalog = Catalog()
    models = catalog.models("faster_whisper")[:3]
    original_models = Catalog.models
    monkeypatch.setattr(Catalog, "models", lambda self, engine=None: models if engine == "faster_whisper" else original_models(self, engine))
    installed = tmp_path / "model.bin"
    installed.write_bytes(b"real")
    for model in models[:2]:
        catalog.record(model["id"], {"path": str(installed), "files": {"model.bin": {"size": 4}}, "fingerprint": "local"})
    write_json(catalog.root / "checks.json", {models[0]["id"]: {"fingerprint": "local"}, models[1]["id"]: {"fingerprint": "new"}})
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception
    keys = {b.key: b for b in app.button if b.key}
    prefix = "model_action:faster_whisper:"
    assert prefix + models[0]["id"] not in keys
    assert keys[prefix + models[1]["id"]].label == "تحديث"
    assert keys[prefix + models[2]["id"]].label == "تحميل"
    app.button(key="choose_model:faster_whisper:" + models[2]["id"]).click().run()
    assert app.selectbox(key="model:faster_whisper").value == models[2]["id"]
    hints = [str(item.proto) for item in app.get("html")]
    assert any('tooltip' in hint and "صيغ الإخراج" in hint for hint in hints)
    assert any('tooltip' in hint and "لغة الصوت" in hint for hint in hints)
    app.multiselect[0].set_value(["srt", "ass"]).run()
    assert any(n.label == "حجم خط ASS" for n in app.number_input)


def test_local_search_reuses_one_scan_across_compatible_engines(tmp_path, monkeypatch):
    import time
    import struct
    import ui.local_models as ui_local
    from engine.catalog import Catalog
    folder = tmp_path / "external-tool"
    folder.mkdir()
    path = folder / "ggml-existing.bin"
    path.write_bytes(b"lmgg" + struct.pack("<11i", 51864, 1500, 384, 6, 4, 448, 384, 6, 4, 80, 1) + b"\0" * 4096)
    ct2 = folder / "local-ct2"
    ct2.mkdir()
    (ct2 / "config.json").write_text('{"alignment_heads": []}', encoding="utf-8")
    (ct2 / "model.bin").write_bytes(b"ct2 weights")
    (ct2 / "tokenizer.json").write_text("{}", encoding="utf-8")
    marian = folder / "local-marian"
    marian.mkdir()
    (marian / "config.json").write_text('{"model_type": "marian"}', encoding="utf-8")
    (marian / "model.safetensors").write_bytes(b"translation weights")
    monkeypatch.setattr(ui_local, "default_roots", lambda *a, **k: [folder])
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.button(key="scan_local_models").click().run()
    catalog = Catalog()
    deadline = time.monotonic() + 5
    report_path = catalog.root / "local-scan.json"
    while not report_path.exists() and time.monotonic() < deadline:
        time.sleep(.02)
    original_report = report_path.read_bytes()
    app.run()
    assert not app.exception
    def visible_paths():
        return {item.value for item in app.get("code")} & {str(path), str(ct2), str(marian)}

    assert visible_paths() == {str(ct2)}
    app.button(key="engine_choice:whisper_cpp").click().run()
    assert visible_paths() == {str(path)}
    app.button(key="engine_choice:crisp").click().run()
    assert visible_paths() == {str(path)}
    app.button(key="engine_choice:whisper_cpp").click().run()
    button = next(b for b in app.button if b.key and b.key.startswith("adopt_local:") and b.key.endswith(":False"))
    button.click().run()
    identity = "local:" + str(path)
    deadline = time.monotonic() + 5
    while not catalog.get(identity) and time.monotonic() < deadline:
        time.sleep(.02)
    app.run()
    assert not app.exception and catalog.get(identity)["path"] == str(path)
    assert any("لا يحتاج إعادة تنزيل" in m.value for m in app.success)
    translation = next(s for s in app.selectbox if s.label == "محرك الترجمة النصية")
    translation.select("transformers").run()
    assert visible_paths() == {str(path), str(marian)}
    next(s for s in app.selectbox if s.label == "محرك الترجمة النصية").select("llama").run()
    assert visible_paths() == {str(path)}
    assert report_path.read_bytes() == original_report
