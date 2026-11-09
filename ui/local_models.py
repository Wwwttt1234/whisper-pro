"""Discovery and adoption controls for existing models from other applications."""
import time
from pathlib import Path

import streamlit as st

from core.paths import location
from core.storage import read_json, write_json
from engine.local_models import ModelScanner, default_roots
from engine.specs import ENGINES
from ui.controls import controls


def render_local_models(catalog, ops, general, engine):
    busy = ops.snapshot()["running"]
    with st.expander("خيارات البحث عن نماذج موجودة على الجهاز", expanded=False):
        st.caption("يبحث في مخازن Hugging Face وWhisper وModelScope وOllama وLM Studio ومجلدات المستخدم الشائعة. لا يشغّل ملفات النماذج ولا ينزّل أوزانًا أثناء البحث.")
        extra = controls.text_area("مجلدات إضافية للبحث اختياريًا", key="local_scan_extra", help="مجلد في كل سطر، مثل D:/Models. أضف مخزن برنامج آخر إذا كان في مكان مخصص. فارغ يستخدم الأماكن الشائعة.")
        broad = controls.checkbox("بحث موسع في الأقراص المحلية", key="local_scan_broad", help="أبطأ؛ يضيف الأقراص المحلية على Windows ومجلدات الأقراص المركبة على Linux وMac. يستثني النظام والبيئات البرمجية، ويتوقف عند بلوغ حد الوقت أو المجلدات.")
    if st.button("البحث عن نماذج موجودة في الجهاز", key="scan_local_models", disabled=busy, help="يبحث مرة واحدة لكل المحركات. يعرض النتائج الموافقة لاختيارك، ويمكنك تبديل المحرك دون إعادة البحث.", width="stretch"):
        extra_roots = [line.strip().strip('"') for line in extra.splitlines() if line.strip()]
        if general.get("models_dir"):
            extra_roots.append(general["models_dir"])

        def scan(progress):
            scanner = ModelScanner(catalog.models(), catalog.installed())
            result = scanner.scan(default_roots(extra_roots, broad), progress, ops.cancel_event.is_set, max_seconds=300 if broad else 120)
            write_json(location("catalog") / "local-scan.json", result)
            return {"local_scan": True, "count": len(result["candidates"])}
        ops.start("البحث عن النماذج الموجودة محليًا", scan, cancellable=True)
    render_local_results(catalog, ops, engine)


def translation_matches(model, backend):
    return (model.get("format") == "gguf") == (backend == "llama")


def render_local_results(catalog, ops, engine, translation_backend=None):
    report = read_json(location("catalog") / "local-scan.json", {})
    if not report:
        return
    busy = ops.snapshot()["running"]
    name = ENGINES[engine].name if engine in ENGINES else ("llama.cpp" if translation_backend == "llama" else "Transformers")
    candidates = [
        candidate for candidate in report.get("candidates", [])
        if engine in candidate.get("engines", []) and (
            translation_backend is None or translation_matches(candidate, translation_backend)
        )
    ]
    with st.expander("النماذج المحلية المكتشفة · " + name, expanded=True):
        st.caption(f"آخر بحث: {time.strftime('%Y-%m-%d %H:%M', time.localtime(report['scanned_at']))} · {report.get('directories', 0)} مجلدًا · {len(candidates)} نتيجة لهذا المحرك")
        if report.get("limited"):
            st.warning("توقف البحث عند الحد المحدد؛ لم يُفحص الجهاز كاملًا. أضف مجلدًا محددًا أو استخدم البحث الموسع للمتابعة.")
        if not candidates:
            st.info("لم يُعثر على نموذج متوافق مع المحرك المختار. يمكنك إضافة مجلد النماذج إلى خيارات البحث وإعادة الفحص.")
        installed = catalog.installed()
        for candidate in candidates:
            identity = candidate.get("catalog_id") or "local:" + candidate["path"]
            current = installed.get(identity, {})
            same_source = candidate["path"] in current.get("external_sources", []) and current.get("source_files") == candidate["files"]
            used = bool(current and catalog.valid_install(current) and (same_source or Path(current["path"]).absolute() == Path(candidate["path"]).absolute()))
            with st.container(border=True):
                st.write(candidate["name"])
                st.caption(" · ".join(ENGINES[e].name if e in ENGINES else "الترجمة النصية" for e in candidate["engines"]) + f" · {candidate['size'] / 1048576:,.1f} MiB")
                st.code(candidate["path"], language=None)
                if candidate.get("note"):
                    st.info(candidate["note"])
                if candidate.get("alternatives"):
                    st.caption(f"نفس الملفات موجودة في {len(candidate['alternatives'])} مسارات أخرى؛ لم تكرر في القائمة.")
                if used:
                    st.success("مضاف إلى البرنامج؛ لا يحتاج إعادة تنزيل")
                columns = st.columns(2)
                for index, (copy_files, label) in enumerate([(False, "استخدام من مكانه"), (True, "نسخ واستخدام")]):
                    if columns[index].button(label, key=f"adopt_local:{candidate['key']}:{copy_files}", disabled=busy or (used and (not copy_files or current.get("external_copied"))) or not candidate["usable"], help="يسجل الملفات في الدليل دون نقلها أو تعديلها؛ أبق مجلد المصدر في مكانه." if not copy_files else "ينسخ ملفات النموذج إلى مخزن البرنامج ويترك ملفات البرنامج الآخر كما هي. يحتاج مساحة بحجم النموذج.", width="stretch"):
                        def adopt(progress, chosen=dict(candidate), make_copy=copy_files):
                            return catalog.adopt(chosen, make_copy, progress, ops.cancel_event.is_set)
                        ops.start(label + " · " + candidate["name"], adopt, cancellable=True)
        if report.get("warnings"):
            with st.expander("مجلدات أو ملفات تعذر فحصها"):
                st.caption("بعض الملفات قد تكون غير مكتملة أو سحابية أو غير مسموح بقراءتها؛ لا يعد ذلك إثباتًا لعدم وجود نماذج أخرى.")
                st.code("\n".join(report["warnings"][:40]), language=None)
