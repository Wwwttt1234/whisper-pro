"""Arabic desktop-local UI; all engine settings are capability-driven."""
import copy
import math
import os
import threading
import time
import uuid
from pathlib import Path

import streamlit as st

from core.paths import data_dir, location
from core.storage import read_json, write_json, download
from core.settings import load_settings, initial_settings, save_settings, import_settings, export_settings, validate_profile
from core.jobs import JobManager, TERMINAL
from core.database import Database
from core.media import inspect_media, MEDIA_EXTENSIONS
from core.outputs import write_all_outputs, ass_style
from engine.catalog import Catalog, TRUSTED_AUTHORS
from engine.specs import ENGINES, fields_for, defaults_for, supports_translation
from engine.runtime import Runtimes
from ui.controls import controls, tooltip_label, CSS
from ui.local_models import render_local_models, render_local_results, translation_matches

LANGUAGES = {"auto": "كشف تلقائي", "ar": "العربية", "en": "الإنجليزية", "fr": "الفرنسية", "de": "الألمانية", "es": "الإسبانية", "tr": "التركية", "ru": "الروسية", "ja": "اليابانية", "ko": "الكورية", "zh": "الصينية", "yue": "الكانتونية", "pt": "البرتغالية", "hi": "الهندية", "it": "الإيطالية", "nl": "الهولندية", "pl": "البولندية", "uk": "الأوكرانية", "fa": "الفارسية", "id": "الإندونيسية", "vi": "الفيتنامية", "th": "التايلاندية", "ms": "الملايوية", "sv": "السويدية", "da": "الدنماركية", "fi": "الفنلندية", "cs": "التشيكية", "tl": "الفلبينية", "el": "اليونانية", "hu": "المجرية", "mk": "المقدونية", "ro": "الرومانية"}
STATUS = {"queued": "في الانتظار", "preparing": "تجهيز المحرك والنموذج", "running": "قيد المعالجة", "paused": "متوقفة مؤقتًا", "cancelling": "جارٍ الإلغاء", "completed": "مكتملة", "no_speech": "لم يُكتشف كلام", "cancelled": "ملغاة", "skipped": "متخطاة", "error": "فشلت"}
MODEL_STATUS = {"download": "غير محمّل", "repair": "ملفات ناقصة", "unknown": "محمّل · التحديث غير متحقق", "update": "محمّل · تحديث متاح", "current": "محمّل ومحدَّث", "local": "نموذج محلي موجود"}


class Operations:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"running": False, "id": "", "message": "", "progress": 0.0}
        self.cancel_event = threading.Event()

    def start(self, label, action, cancellable=False):
        with self.lock:
            if self.state["running"]:
                return
            self.cancel_event.clear()
            self.state = {"running": True, "id": uuid.uuid4().hex, "message": label, "progress": 0.0, "cancellable": cancellable}
        def progress(a, b, message=""):
            if cancellable and self.cancel_event.is_set():
                raise InterruptedError("أُوقفت العملية")
            with self.lock:
                self.state.update(progress=max(0, min(1, a / b if b else 0)), message=message or label)
        def run():
            try:
                result = action(progress)
                with self.lock:
                    self.state.update(running=False, progress=1.0, result=result)
            except InterruptedError:
                with self.lock:
                    self.state.update(running=False, cancelled=True)
            except Exception as exc:
                with self.lock:
                    self.state.update(running=False, error=str(exc))
        threading.Thread(target=run, daemon=True).start()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.state)


@st.cache_resource
def resources(session):
    return JobManager(), Operations()


def choose_engine(engine):
    st.session_state.settings_v2["engine"] = engine


def choose_model(engine, identity):
    st.session_state.settings_v2["selection"][engine] = identity
    st.session_state["model:" + engine] = identity
    st.session_state["family:" + engine] = "all"


def engine_help(engine):
    spec = ENGINES[engine]
    return spec.note + "\n\n" + (spec.unavailable_reason() or "متاح لهذا النظام؛ تثبيت المحرك منفصل عن تحميل أوزان النموذج.")


def engine_picker(engine):
    tooltip_label("محرك التعرف على الكلام", "افتح القائمة ثم مرّر الماوس فوق اسم أي محرك لشرح استخدامه قبل اختياره. ابدأ بـFaster Whisper علىCPU وINT8 للتوافق.")
    with st.popover(ENGINES[engine].name, width="stretch", help="اعرض المحركات واقرأ شرح كل محرك قبل الاختيار."):
        st.caption("مرّر الماوس فوق اسم المحرك لشرح عمله، ثم اضغط لاختياره.")
        for identity, spec in ENGINES.items():
            st.button(spec.name, key="engine_choice:" + identity, help=engine_help(identity), type="primary" if identity == engine else "secondary", width="stretch", on_click=choose_engine, args=(identity,))


def engine_state(engine, profile):
    status = Runtimes().status(engine, profile)
    labels = {"installed": "مثبت", "missing": "غير مثبت", "broken": "يحتاج إصلاحًا", "custom": "محرك مخصص موجود", "unavailable": "غير متاح لهذا النظام"}
    return status, labels[status["state"]]


def render_fields(engine, model, profile):
    groups = {}
    for field in fields_for(engine, model):
        groups.setdefault(field.group, []).append(field)
    values = {**defaults_for(engine, model), **profile}
    shown = []
    for group_name, fields in groups.items():
        with st.expander(group_name, expanded=group_name in {"الأداء", "جودة التفريغ"}):
            columns = st.columns(2)
            for index, field in enumerate(fields):
                if field.key.startswith("vad_") and field.key != "vad_filter" and not values.get("vad_filter") and engine != "whisperx":
                    continue
                if field.key in {"min_speakers", "max_speakers"} and not values.get("diarization"):
                    continue
                if field.key in {"top_p", "top_k"} and not values.get("do_sample"):
                    continue
                if engine == "whisper_cpp" and field.key in {"beam_size", "best_of", "patience"} and ((field.key == "best_of") != (values.get("sampling_strategy") == "greedy")):
                    continue
                if engine == "sherpa" and field.key in {"max_active_paths", "hotwords_file", "hotwords_score"} and values.get("decoding_method") != "modified_beam_search":
                    continue
                key = f"setting:{engine}:{model['id']}:{field.key}"
                value = values.get(field.key, field.default)
                with columns[index % 2]:
                    tooltip_label(field.label, field.help)
                    options = list(field.choices)
                    if field.key == "device":
                        options = ["cpu", "cuda", "auto"] + (["mps"] if os.sys.platform == "darwin" and engine not in {"faster_whisper", "whisperx"} else [])
                    if options:
                        if value not in options:
                            value = field.default if field.default in options else options[0]
                        selected = controls.selectbox(field.label, options, index=options.index(value), key=key, help=field.help, label_visibility="collapsed")
                    elif isinstance(field.default, bool):
                        selected = controls.checkbox(field.label, value=bool(value), key=key, help=field.help, label_visibility="collapsed")
                    elif isinstance(field.default, (int, float)):
                        kind = type(field.default)
                        selected = controls.number_input(field.label, value=kind(value), min_value=kind(field.minimum) if field.minimum is not None else None, max_value=kind(field.maximum) if field.maximum is not None else None, step=kind(field.step or 1), key=key, help=field.help, label_visibility="collapsed")
                    else:
                        widget = controls.text_area if field.multiline else controls.text_input
                        selected = widget(field.label, value=str(value), key=key, help=field.help, label_visibility="collapsed")
                    values[field.key] = selected
                    shown.append(field.key)
    st.caption(f"{len(shown)} إعدادًا ظاهرًا لهذا المحرك والنموذج. مرّر المؤشر فوق اسم الإعداد أو علامة المساعدة لقراءة عمله ونقطة البداية المقترحة.")
    return values


@st.fragment(run_every=1)
def render_operations(ops):
    state = ops.snapshot()
    if not state.get("id"):
        return
    if state["running"]:
        st.info(state["message"])
        st.progress(state["progress"])
        if state.get("cancellable") and st.button("إيقاف البحث أو النسخ", key="cancel_operation:" + state["id"], help="يوقف العملية الحالية ويحافظ على النماذج الأصلية."):
            ops.cancel_event.set()
    elif state.get("cancelled"):
        st.info("أُوقفت العملية؛ لم تُعدّل ملفات النماذج الأصلية.")
    elif state.get("error"):
        st.error(state["error"])
    else:
        result = state.get("result")
        st.success("اكتملت العملية")
        if isinstance(result, dict) and result.get("errors"):
            st.warning("بعض المصادر لم تستجب؛ احتُفظ بالقائمة السابقة.")
        if isinstance(result, dict) and result.get("details"):
            with st.expander("نتيجة فحص المحرك"):
                st.code(result["details"])
        if st.session_state.get("applied_operation") != state["id"]:
            st.session_state.applied_operation = state["id"]
            if isinstance(result, dict) and result.get("media_path"):
                st.session_state.queue.append({"path": result["media_path"], "name": Path(result["media_path"]).name, "uploaded": True})
            st.rerun()


def queue_add(paths, uploaded=False):
    for path in paths:
        path = Path(path).expanduser()
        if path.is_dir():
            queue_add(sorted(p for p in path.iterdir() if p.suffix.lower() in MEDIA_EXTENSIONS))
        elif path.is_file() and path.stat().st_size > 0:
            resolved = str(path.resolve())
            if not any(item["path"] == resolved for item in st.session_state.queue):
                st.session_state.queue.append({"path": resolved, "name": path.name, "uploaded": uploaded})
        else:
            st.warning("ملف غير موجود أو فارغ: " + str(path))


def render_inputs(ops):
    left, right = st.columns([1, 2])
    with left:
        if st.button("اختيار ملفات من الجهاز", help="نافذة اختيار محلية؛ إن لم تتوفر، استخدم رفع الملفات أو مساراتها.", width="stretch"):
            from utils.files import select_files_dialog
            queue_add(select_files_dialog())
        if st.button("مسح قائمة الملفات", help="يزيل الملفات من القائمة فقط؛ لا يحذف ملفاتك.", width="stretch"):
            st.session_state.queue = []
            st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
            st.rerun()
    with right:
        uploaded = controls.file_uploader("أضف فيديو أو صوتًا", type=[ext[1:] for ext in sorted(MEDIA_EXTENSIONS)], accept_multiple_files=True, key=f"upload:{st.session_state.get('upload_generation', 0)}", help="اسحب الملفات هنا. تُحفظ نسخة عمل مرة واحدة؛ للملفات الكبيرة يمكن اختيارها مباشرة من الجهاز.")
        handled = st.session_state.setdefault("handled_uploads", set())
        for item in uploaded or []:
            identity = item.file_id
            if identity not in handled:
                if not item.size:
                    st.warning("تجاهل ملف فارغ: " + item.name)
                    handled.add(identity)
                    continue
                folder = location("uploads") / uuid.uuid4().hex
                folder.mkdir()
                target = folder / ("media_" + Path(item.name).name)
                with open(target, "wb") as stream:
                    stream.write(item.getbuffer())
                st.session_state.queue.append({"path": str(target), "name": item.name, "uploaded": True})
                handled.add(identity)
    with st.expander("إضافة مسارات أو رابط وسائط"):
        paths = controls.text_area("مسار ملف أو مجلد في كل سطر", help="الصق مسارًا من مدير الملفات؛ إضافة مجلد تجمع ملفات الوسائط الموجودة داخله دون المجلدات الفرعية.")
        if st.button("إضافة المسارات", help="يضيف الملفات الموجودة وغير الفارغة إلى القائمة."):
            queue_add([p.strip().strip('"') for p in paths.splitlines() if p.strip()])
        url = controls.text_input("رابط HTTPS أو YouTube", help="ينزّل الملف أولًا محليًا. بعض المواقع تحتاج تسجيل الدخول أو متطلبات إضافية يفرضها المصدر.")
        if st.button("تنزيل الوسائط", disabled=ops.snapshot()["running"], help="تنزيل نسخة عمل محلية؛ لا يبدأ التفريغ حتى تضغط بدء المعالجة.") and url:
            def fetch(progress):
                import urllib.parse
                parsed = urllib.parse.urlparse(url)
                if parsed.scheme != "https":
                    raise ValueError("استخدم رابط HTTPS صالحًا")
                folder = location("uploads") / uuid.uuid4().hex
                folder.mkdir()
                if parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}:
                    import yt_dlp
                    options = {"format": "bestaudio/best", "outtmpl": str(folder / "%(id)s.%(ext)s"), "noplaylist": True, "quiet": True, "progress_hooks": [lambda event: progress(event.get("downloaded_bytes", 0), event.get("total_bytes") or event.get("total_bytes_estimate") or 0, "تنزيل الوسائط")]}
                    with yt_dlp.YoutubeDL(options) as ydl:
                        info = ydl.extract_info(url, download=True)
                        path = ydl.prepare_filename(info)
                else:
                    filename = Path(urllib.parse.unquote(parsed.path)).name or "media.bin"
                    path = str(download(url, folder / ("media_" + filename), progress=lambda a, b: progress(a, b, "تنزيل الوسائط")))
                return {"media_path": path}
            ops.start("تنزيل الوسائط", fetch)
    if st.session_state.queue:
        st.dataframe([{"الملف": item["name"], "الحجم MB": round(Path(item["path"]).stat().st_size / 1048576, 1) if Path(item["path"]).exists() else 0, "المصدر": "نسخة عمل" if item["uploaded"] else "ملف محلي"} for item in st.session_state.queue], hide_index=True, width="stretch")


def render_models(catalog, engine, model, profile, general, ops, token):
    busy = ops.snapshot()["running"]
    status, status_label = engine_state(engine, profile)
    st.subheader(ENGINES[engine].name)
    st.info("حالة المحرك: " + status_label + " · " + status["message"])
    if status.get("version"):
        st.caption("الإصدار المثبت أو المطلوب: " + status["version"])
    runtime_cols = st.columns(2)
    if status["state"] in {"missing", "broken"}:
        if runtime_cols[0].button("تثبيت المحرك" if status["state"] == "missing" else "إصلاح المحرك", disabled=busy, help="يجهز أدوات هذا المحرك وحده من المصدر الموثق؛ لا ينزّل نماذج الكلام.", width="stretch"):
            ops.start("تجهيز المحرك", lambda progress: Runtimes().ensure(engine, {**profile, "local_files_only": False}, lambda message: progress(0, 1, message), repair=status["state"] == "broken"))
    if runtime_cols[1].button("فحص المحرك", disabled=busy or status["state"] not in {"installed", "custom"}, help="يختبر تحميل مكتبة المحرك أو استعلام نسخته؛ لا يحمّل نموذجًا ولا يثبت دعم كلGPU.", width="stretch"):
        ops.start("فحص المحرك", lambda progress: Runtimes().probe(engine, profile))
    if status["state"] == "unavailable":
        st.caption("يمكن عرض نماذجه ومعلوماتها؛ تشغيل المحرك يتطلب النظام المذكور في شرحه.")
    cols = st.columns(2)
    if cols[0].button("فحص النماذج الجديدة", disabled=busy, help="يبحث لدى ناشري هذا المحرك المعتمدين ويضيف النماذج المتوافقة دون تكرار هويتها.", width="stretch"):
        ops.start("فحص المصادر", lambda progress: discover_models(catalog, engine, token, progress))
    if cols[1].button("فحص الأحجام والتحديثات", disabled=busy, help="يفحص بيانات كل نماذج المحرك دون تنزيل أوزانها، ويقارن الملفات المثبتة بالمصدر؛ قد تستغرق القائمة الكبيرة وقتًا.", width="stretch"):
        ops.start("فحص أحجام وإصدارات النماذج", lambda progress: catalog.check_many(engine, token, progress), cancellable=True)
    render_local_models(catalog, ops, general, engine)
    entries = catalog.inventory(engine)
    ready = sum(e["state"] in {"unknown", "current", "update", "local"} for e in entries)
    updates = sum(e["state"] == "update" for e in entries)
    st.caption(f"{len(entries)} نموذجًا في الدليل · {ready} محمّلًا أو محليًا · {updates} تحديثات مثبتة بالفحص")
    search_cols = st.columns(2)
    query = controls.text_input("البحث في نماذج المحرك", key="model_search:" + engine, help="ابحث بالاسم أو العائلة أو المصدر. فارغ يعرض جميع نماذج المحرك.", _container=search_cols[0]).strip().lower()
    view = controls.selectbox("حالة نماذج العرض", ["all", "installed", "download", "update", "repair"], key="model_view:" + engine, format_func=lambda v: {"all": "كل النماذج", "installed": "المحمّلة والمحلية", "download": "غير المحمّلة", "update": "تحديث متاح", "repair": "تحتاج إصلاحًا"}[v], help="الافتراضي كل النماذج؛ التصفية لا تحذف أي عنصر من الدليل.", _container=search_cols[1])
    shown = [e for e in entries if (not query or query in " ".join(str(e["model"].get(k, "")) for k in ["name", "family", "repo"]).lower()) and (view == "all" or view == "installed" and e["state"] in {"unknown", "current", "update", "local"} or view == e["state"])]
    st.caption("الحجم يشمل ملفات تشغيل النموذج، ولا يمثل استهلاك RAM أوVRAM. لا يظهر زر تحديث عند تعذر التحقق من المصدر.")
    with st.container(height=600, border=True):
        if not shown:
            st.info("لا توجد نماذج مطابقة لهذا البحث.")
        for entry in shown:
            item, state, checked = entry["model"], entry["state"], entry["checked"]
            identity = item["id"]
            item_profile = profile if identity == model["id"] else {**defaults_for(engine, item), **st.session_state.settings_v2["profiles"].get(engine, {}).get(identity, {})}
            with st.container(border=True):
                row = st.columns([3.3, 1.4, 1.8, 1.1, 1.2])
                tooltip_label(("✓ " if identity == model["id"] else "") + item["name"], "العائلة: " + item.get("family", "غير معلنة") + "\nالمصدر: " + item.get("repo", item.get("path", item.get("url", "محلي"))) + "\nالترخيص: " + str(item.get("license") or "راجع المصدر"), row[0])
                row[0].caption(item.get("family", "") + " · " + str(item.get("format", "")))
                row[1].write(format_size(entry["size"]))
                row[2].write(MODEL_STATUS[state])
                if checked.get("checked_at"):
                    row[2].caption(time.strftime("%Y-%m-%d %H:%M", time.localtime(checked["checked_at"])))
                row[3].button("محدد" if identity == model["id"] else "اختيار", key="choose_model:" + engine + ":" + identity, disabled=identity == model["id"], help="يختار هذا النموذج للتفريغ ويعرض إعداداته؛ لا يبدأ التنزيل.", width="stretch", on_click=choose_model, args=(engine, identity))
                action = {"download": "تحميل", "repair": "إصلاح", "update": "تحديث"}.get(state) if item.get("source") != "local" else None
                pinned = state == "update" and bool(item_profile.get("revision"))
                if action and row[4].button(action, key="model_action:" + engine + ":" + identity, disabled=busy or pinned, help="الإصدار مثبت في الإعدادات؛ ألغ تثبيته للتحديث." if pinned else "ينزّل هذا النموذج ويتحقق من ملفاته. لا يشغل التفريغ ولا ينزّل باقي النماذج.", width="stretch"):
                    def install(progress, chosen=dict(item), options=dict(item_profile), chosen_state=state):
                        return catalog.ensure(chosen, {**options, "local_files_only": False}, progress=progress, token=token, models_dir=general.get("models_dir", ""), force=chosen_state in {"repair", "update"}, repair=chosen_state == "repair")
                    ops.start(action + " " + item["name"], install)
                if checked.get("error"):
                    row[2].caption("التحديث غير متحقق")
    with st.expander("تفاصيل النموذج المحدد وإدارة المحرك"):
        st.write("المصدر:", model.get("repo", model.get("path", model.get("url", ""))))
        st.caption("الترخيص: " + str(model.get("license") or "راجع المصدر"))
        if st.button("فحص إصدار النموذج", disabled=busy or model.get("source") == "local", help="يفحص النموذج المحدد فقط، بدل القائمة كاملة."):
            ops.start("فحص الإصدار", lambda progress: catalog.check(model, token))
        if st.button("إعادة تجهيز المحرك", disabled=busy or bool(ENGINES[engine].unavailable_reason()), help="يعيد تثبيت حزم هذا المحرك لإصلاح تلفها. انتظر انتهاء المهام التي تستخدمه قبل ذلك."):
            ops.start("إعادة تجهيز المحرك", lambda progress: Runtimes().ensure(engine, {**profile, "local_files_only": False}, lambda message: progress(0, 1, message), repair=True))
    with st.expander("إضافة نموذج محلي أو مستودع معتمد"):
        path = controls.text_input("مسار النموذج المحلي", help="مجلد CTranslate2 أو Transformers أو ONNX، أو ملف GGUF/GGML/PyTorch وفق المحرك المختار.")
        family = controls.text_input("عائلة النموذج", value=model.get("family", "whisper"), help="مثلwhisper أوparakeet أوqwen3. يلزم أن تتوافق العائلة مع المحرك والملفات.")
        if st.button("إضافة النموذج المحلي", help="يتحقق من المسار ويضيفه للقائمة دون نسخ ملفاته."):
            try:
                catalog.register_local(engine, path, family)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        repo = controls.text_input("هوية مستودع Hugging Face", placeholder="publisher/model", help="الناشر/المستودع. لا يجري تشغيل كود Python من المستودع افتراضيًا.")
        filename = controls.text_input("اسم ملف الأوزان عند الحاجة", help="لـGGUF وGGML اختر اسم الملف داخل المستودع؛ اتركه فارغًا للنموذج الذي يستخدم مجلدًا كاملًا.")
        st.caption("الناشرون المعتمدون: " + ", ".join(sorted(TRUSTED_AUTHORS)))
        if st.button("فحص وإضافة المستودع", disabled=busy, help="يفحص وجود ملفات التشغيل في المصدر قبل إضافتها."):
            ops.start("فحص المستودع", lambda progress: catalog.add_hf(repo.strip(), engine, token, filename.strip(), family.strip()))


def format_size(size):
    if not size:
        return "غير معلن · افحص الحجم"
    return f"{size / 1073741824:,.2f} GiB" if size >= 1073741824 else f"{size / 1048576:,.1f} MiB"


def discover_models(catalog, engine, token=None, progress=None):
    result = catalog.refresh(engine, token, progress)
    needed = [e["model"] for e in catalog.inventory(engine) if not e["size"] or e["installed"]]
    checked = catalog.check_many(engine, token, progress, models=needed)
    return {"count": result["count"], "errors": result["errors"] + checked["errors"]}


def render_general(general, model, engine):
    languages = model.get("languages") or []
    if isinstance(languages, str):
        languages = [languages]
    options = list(dict.fromkeys(["auto"] + (languages if languages else list(LANGUAGES))))
    selected = general["language"] if general["language"] in options else "auto"
    general["language"] = controls.selectbox("لغة الصوت", options, index=options.index(selected), format_func=lambda v: LANGUAGES.get(v, v), help="اختر لغة الصوت، وليست لغة الترجمة المطلوبة. الكشف التلقائي يعتمد على قدرة النموذج.")
    tasks = ["transcribe", "translate"] if supports_translation(engine, model) else ["transcribe"]
    general["task"] = controls.selectbox("مهمة التفريغ", tasks, index=tasks.index(general["task"]) if general["task"] in tasks else 0, format_func=lambda v: "النص بنفس لغة الصوت" if v == "transcribe" else "ترجمة الكلام إلى الإنجليزية · للنماذج الداعمة", help="Whisper translate يخرج الإنجليزية. للترجمة إلى العربية استخدم مرحلة الترجمة النصية أدناه.")
    with st.expander("تنسيق الترجمة ومكان حفظها", expanded=True):
        general["formats"] = controls.multiselect("صيغ الإخراج", ["srt", "vtt", "ass", "txt", "json", "csv"], default=general["formats"], help="SRT مناسب لمعظم المشغلات، وJSON يحتفظ ببيانات المقاطع والكلمات.")
        general["beside_source"] = controls.checkbox("الحفظ بجوار الملف المحلي", value=general["beside_source"], help="للملفات المختارة مباشرة من الجهاز. الملفات المرفوعة عبر المتصفح تحفظ في مجلد النتائج وتتاح للتنزيل.")
        general["output_dir"] = controls.text_input("مجلد نتائج مخصص", value=general["output_dir"], help="يستخدم عند إيقاف الحفظ بجوار المصدر. فارغ يعني مجلد نتائج البرنامج.")
        general["output_suffix"] = controls.text_input("لاحقة اسم الترجمة", value=general["output_suffix"], help="مثل_ar أو.en. الاسم نفسه لا يغيّر لغة المحتوى.")
        general["output_subfolder"] = controls.checkbox("مجلد نتائج مستقل لكل ملف", value=general["output_subfolder"], help="يجمع صيغ الإخراج الخاصة بكل فيديو في مجلد مستقل.")
        if general["output_subfolder"]:
            general["output_subfolder_name"] = controls.text_input("قالب اسم المجلد", value=general["output_subfolder_name"], help="استخدم{original_name} لاسم الفيديو؛ لا تستخدم فواصل المسارات في الاسم.")
        general["overwrite"] = controls.checkbox("استبدال ملفات الإخراج الموجودة", value=general["overwrite"], help="معطل افتراضيًا لحماية ملفاتك؛ ينشئ البرنامج أسماء مرقمة عند التعارض.")
        c1, c2 = st.columns(2)
        general["max_line_length"] = controls.number_input("أقصى طول للسطر", 10, 200, int(general["max_line_length"]), help="42 حرفًا نقطة بداية شائعة؛ الطول المناسب يعتمد على الشاشة والخط واللغة.", _container=c1)
        general["max_lines"] = controls.number_input("أسطر المقطع", 1, 5, int(general["max_lines"]), help="ابدأ بسطرين. تُستخدم توقيتات الكلمات عند توفرها لتقسيم المقاطع.", _container=c2)
        general["max_cue_seconds"] = controls.number_input("مدة المقطع المستهدفة القصوى", 1.0, 60.0, float(general["max_cue_seconds"]), help="7 ثوانٍ نقطة بداية؛ لا تُختلق حدود كلمات دقيقة إذا لم يوفر المحرك توقيتها.", _container=c1)
        general["min_cue_seconds"] = controls.number_input("أقل مدة عرض", 0.0, 10.0, float(general["min_cue_seconds"]), step=0.1, help="0.5 ثانية بداية مناسبة؛ لا يتم تمديد المقطع فوق بداية المقطع التالي.", _container=c2)
        general["time_offset"] = controls.number_input("إزاحة التوقيت بالثواني", value=float(general["time_offset"]), step=0.1, help="الموجب يؤخر ظهور النص والسالب يقدمه. ابدأ بصفر.", _container=c1)
    if "ass" in general["formats"]:
        with st.expander("خط وألوان وموضع ترجمة ASS"):
            general["ass_font_name"] = controls.text_input("اسم خط ASS", value=general["ass_font_name"], help="يجب توفر الخط على جهاز العرض. Arial نقطة بداية؛ ملف ASS لا يضم الخط نفسه.")
            general["ass_font_size"] = controls.number_input("حجم خط ASS", 8, 200, int(general["ass_font_size"]), help="16 مناسب لدقة384×288 الافتراضية؛ استخدم نحو48 مع1080p ثم راجع الفيديو.")
            general["ass_text_color"] = controls.color_picker("لون نص ASS", general["ass_text_color"], help="الأبيض مع حد أسود بداية واضحة على معظم الخلفيات.")
            general["ass_outline_color"] = controls.color_picker("لون حد النص", general["ass_outline_color"], help="الأسود يفصل النص الأبيض عن الخلفية.")
            for key, label in [("bold", "خط عريض"), ("italic", "خط مائل")]:
                general["ass_" + key] = controls.checkbox(label, value=general["ass_" + key], help="يطبق على ASS فقط. ابدأ معطلًا ثم راجع القراءة.")
            general["ass_outline"] = controls.number_input("سماكة حد النص", 0.0, 20.0, float(general["ass_outline"]), step=0.5, help="2 نقطة بداية؛ صفر بلا حد.")
            general["ass_shadow"] = controls.number_input("حجم ظل النص", 0.0, 20.0, float(general["ass_shadow"]), step=0.5, help="1 نقطة بداية؛ صفر بلا ظل.")
            positions = {1:"أسفل يسار",2:"أسفل وسط",3:"أسفل يمين",4:"وسط يسار",5:"وسط الشاشة",6:"وسط يمين",7:"أعلى يسار",8:"أعلى وسط",9:"أعلى يمين"}
            general["ass_alignment"] = controls.selectbox("موضع نص ASS", list(positions), index=general["ass_alignment"]-1, format_func=positions.get, help="أسفل وسط نقطة البداية للترجمة المعتادة.")
            for key, label in [("margin_l", "الهامش الأيسر"), ("margin_r", "الهامش الأيمن"), ("margin_v", "الهامش الرأسي")]:
                general["ass_" + key] = controls.number_input(label, 0, 2000, int(general["ass_" + key]), help="بوحدات دقة ASS المرجعية؛10 نقطة بداية للدقة الافتراضية.")
            general["ass_width"] = controls.number_input("عرض ASS المرجعي", 128, 7680, int(general["ass_width"]), help="384 افتراضي. غيّره مع الارتفاع وحجم الخط إذا أردت تصميم1920×1080.")
            general["ass_height"] = controls.number_input("ارتفاع ASS المرجعي", 128, 4320, int(general["ass_height"]), help="288 افتراضي؛1080 مناسب لتصميم1080p.")
    with st.expander("المعالجة والتخزين"):
        general["parallel_files"] = controls.number_input("الملفات المتزامنة", 1, 8, int(general["parallel_files"]), help="ابدأ بملف واحد. كل ملف قد يحمل نسخة مستقلة من النموذج ويستهلك ذاكرة إضافية.")
        general["audio_stream"] = controls.number_input("ترتيب مسار الصوت", 0, 100, int(general["audio_stream"]), help="صفر هو أول مسار صوت، و1 الثاني. افحص مسارات الفيديو من تبويب الملفات.")
        general["split_stereo"] = controls.checkbox("معالجة اليسار واليمين منفصلين", value=general["split_stereo"], help="للمقابلات المسجلة بقناة لكل متحدث. ينشئ مهمتين ونتيجتين مستقلتين؛ ليس كشف متحدثين تلقائيًا.")
        general["models_dir"] = controls.text_input("مخزن نماذج مخصص", value=general["models_dir"], help="فارغ يستخدم بيانات المستخدم. تغيير المسار لا يحذف النماذج المثبتة السابقة.")
        general["auto_check_models"] = controls.checkbox("فحص النماذج الجديدة تلقائيًا", value=general["auto_check_models"], help="فحص خلفي للمحرك المختار عند انتهاء المدة؛ لا يثبت أو يحدّث الأوزان تلقائيًا.")
        general["catalog_interval_hours"] = controls.number_input("الفاصل بين الفحوص بالساعات", 1, 720, int(general["catalog_interval_hours"]), help="24 ساعة كافية عادةً وتقلل طلبات الشبكة.")


def render_postprocessing(general, catalog):
    general["alignment"] = controls.selectbox("محاذاة إضافية", ["none", "whisperx", "bfa"], index=["none", "whisperx", "bfa"].index(general["alignment"]), format_func=lambda v: {"none": "بدون مرحلة إضافية", "whisperx": "WhisperX · توقيت الكلمات", "bfa": "Bournemouth Forced Aligner"}[v], help="تحدد موضع النص الموجود في الصوت ولا تصلح الكلمات الخاطئة. أوقفها إذا كان توقيت المحرك كافيًا.")
    if general["alignment"] == "bfa":
        general["alignment_language"] = controls.text_input("لغة محاذاة BFA", value=general["alignment_language"], help="مثلen-us أوar. يجب أن تطابق لغة النص والصوت. أول استخدام ينزل نماذج إضافية.")
    options = ["none", "transformers", "llama"]
    general["translation_engine"] = controls.selectbox("محرك الترجمة النصية", options, index=options.index(general["translation_engine"]), key="translation_engine", format_func=lambda v: {"none": "بدون ترجمة إضافية", "transformers": "Transformers · محلي", "llama": "llama.cpp · GGUF محلي"}[v], help="يترجم النص بعد التفريغ مع الاحتفاظ بأزمنة المقاطع. لا يرسل النص لخدمة سحابية.")
    if general["alignment"] != "none" or general["translation_engine"] == "transformers":
        general["post_device"] = controls.selectbox("جهاز المحاذاة والترجمة الإضافيتين", ["cpu", "cuda"], index=["cpu", "cuda"].index(general["post_device"]), help="مستقل عن محرك الصوت. ابدأ بـCPU، واخترCUDA عند وجود NVIDIA وبيئة تسريع متوافقة.")
    if general["translation_engine"] == "transformers":
        general["post_dtype"] = controls.selectbox("دقة ترجمة PyTorch", ["float32", "float16", "bfloat16"], index=["float32", "float16", "bfloat16"].index(general["post_dtype"]), help="float32 علىCPU؛ بقية الدقات تحتاج دعم الجهاز والنموذج.")
    if general["translation_engine"] != "none":
        models = [m for m in catalog.models("translation") if translation_matches(m, general["translation_engine"])]
        if models:
            ids = [m["id"] for m in models]
            current = general["translation_model"] if general["translation_model"] in ids else ids[0]
            general["translation_model"] = controls.selectbox("نموذج الترجمة", ids, index=ids.index(current), format_func=lambda v: next(m["name"] for m in models if m["id"] == v), help="اختر نموذجًا يدعم لغة المصدر والهدف؛ Hy-MT2 يحتاج محركًا متوافقًا، وOPUS en-ar مخصص للإنجليزية إلى العربية.")
            chosen = next(m for m in models if m["id"] == general["translation_model"])
            if chosen.get("experimental_translation"):
                st.caption("هذا نموذج لغوي عام مكتشف على الجهاز؛ يمكن تجربته للترجمة النصية، وجودة الترجمة تعتمد على قدرته وليست مضمونة باكتشاف الملف.")
        else:
            st.warning("افحص نماذج الترجمة لإضافتها من المصادر المعتمدة.")
        _, ops = resources(st.session_state.session_id)
        render_local_results(catalog, ops, "translation", general["translation_engine"])
        if st.button("فحص نماذج الترجمة", help="يحدّث كتالوج نماذج الترجمة من ناشريها المعتمدين."):
            ops.start("فحص نماذج الترجمة", lambda progress: discover_models(catalog, "translation", progress=progress))
        general["target_language"] = controls.text_input("لغة الترجمة المطلوبة", value=general["target_language"], help="مثلArabic أوEnglish. النموذج المتخصص باتجاه واحد لا يصبح متعدد اللغات بتغيير هذا النص.")
        if models and (chosen.get("family") == "causal" or chosen.get("format") == "gguf"):
            general["translation_context"] = controls.text_area("سياق المشهد", value=general["translation_context"], help="ملخص قصير للمشهد يحسن فهم الضمائر والمصطلحات في النماذج التي تتبع التعليمات.")
            general["glossary"] = controls.text_area("قاموس الأسماء والمصطلحات", value=general["glossary"], help="مثال: John = جون. يمرر للنموذج الحواري كتعليمات ترجمة.")
            general["translation_batch"] = controls.number_input("مقاطع دفعة الترجمة", 1, 64, int(general["translation_batch"]), help="8 نقطة بداية للنماذج الحوارية؛ الزيادة تمنح سياقًا أكثر وتحتاج مساحة سياق أكبر.")
        if general["translation_engine"] == "llama":
            general["llama_binary"] = controls.text_input("مسار llama-server مخصص اختياريًا", value=general["llama_binary"], help="فارغ للتجهيز الرسمي التلقائي. اختر بناءً متوافقًا إذا كان النموذج يتطلب قدرات أحدث مثلSTQ.")
            general["llama_context"] = controls.number_input("سعة سياق llama.cpp", 512, 131072, int(general["llama_context"]), step=512, help="8192 نقطة بداية؛ تشمل تعليمات الترجمة والنص والنتيجة، وزيادتها تستهلك ذاكرة.")
            general["llama_gpu_layers"] = controls.number_input("طبقات ترجمة محملة على GPU", 0, 999, int(general["llama_gpu_layers"]), help="0 للتوافق مع حزمةCPU. أي قيمة أعلى تحتاج بناء محرك يدعم بطاقة الجهاز وذاكرة مناسبة.")
            general["translation_threads"] = controls.number_input("خيوط معالج الترجمة", 1, 256, int(general["translation_threads"]), help="4 نقطة بداية؛ إعداد مستقل عن خيوط محرك الصوت.")
        general["translation_max_tokens"] = controls.number_input("حد رموز نتيجة الترجمة", 64, 8192, int(general["translation_max_tokens"]), step=64, help="1024 نقطة بداية. إذا نقص عدد المقاطع، لا يعتمد البرنامج النتيجة الناقصة.")
        general["translation_temperature"] = controls.number_input("حرارة الترجمة", 0.0, 1.0, float(general["translation_temperature"]), step=0.05, help="ابدأ بصفر للحصول على ترجمة ثابتة وأقل عشوائية.")


def clock_text(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


@st.fragment(run_every=1)
def render_jobs(manager, general):
    st.html('<style>div[class*="st-key-transcript-"] {overscroll-behavior-y: contain;}</style>')
    for job in manager.snapshots()[::-1]:
        with st.expander(job["out_name"], expanded=True, key="job-" + job["id"]):
            status = STATUS.get(job["status"], job["status"])
            stage = job.get("stage") if job["status"] not in TERMINAL else None
            st.caption(f"{status} · {stage}" if stage and stage != status else status)
            elapsed, remaining, percent = st.columns(3)
            estimate = job["remaining"]
            waiting = "—" if job["status"] in TERMINAL or job["status"] == "queued" else "غير متاح حاليًا"
            elapsed.metric("الوقت المستغرق", clock_text(job["elapsed"]), help="وقت العمل منذ بدء التجهيز، من دون انتظار الدور أو مدة الإيقاف المؤقت.")
            remaining.metric("الوقت المتبقي تقديريًا", clock_text(math.ceil(estimate)) if estimate is not None else waiting, help="يُحسب أثناء التفريغ من سرعة تقدّم المحرك. قد يتغيّر مع اختلاف المقاطع، ولا يشمل مراحل المحاذاة والترجمة والحفظ.")
            percent.metric("تقدّم المعالجة", f"{int(job['progress'] * 100)}%", help="تقدّم تقريبي عبر التجهيز والتفريغ والحفظ. يكتمل عند تسجيل النتيجة.")
            st.progress(job["progress"])
            result = job.get("result", {})
            segments = job["live_segments"] or result.get("segments", [])
            st.caption(f"المقاطع · {len(segments)}")
            follow = st.toggle("متابعة آخر مقطع", key="follow:" + job["id"], help="فعّلها للانتقال إلى أحدث مقطع. أوقفها لتتصفح المقاطع القديمة داخل اللوحة.")
            with st.container(height=320, autoscroll=follow, key="transcript-" + job["id"]):
                if segments:
                    text = "\n\n".join(f"{i}.  {clock_text(s['start'])} – {clock_text(s['end'])}\n{s['text'].strip()}" for i, s in enumerate(segments, 1))
                    st.text(text)
                else:
                    message = "لا توجد مقاطع لعرضها." if job["status"] in TERMINAL else "ستظهر المقاطع هنا حين يرسلها المحرك. بعض المحركات ترسل دفعة كاملة."
                    st.caption(message)
            if job["status"] in {"running", "paused"}:
                c1, c2, c3 = st.columns(3)
                if c1.button("استئناف" if job["status"] == "paused" else "إيقاف مؤقت", key="pause:" + job["id"], help="يوقف عملية هذا الملف وعملياتها التابعة؛ استأنف من النقطة نفسها."):
                    manager.pause(job["id"], job["status"] != "paused")
                if c2.button("تخطي هذا الملف", key="skip:" + job["id"], help="يلغي هذه المهمة فقط ويترك بقية الملفات في القائمة."):
                    manager.cancel(job["id"], skipped=True)
                if c3.button("إلغاء", key="cancel:" + job["id"], help="يوقف المهمة دون تسجيلها كمكتملة."):
                    manager.cancel(job["id"])
            elif job["status"] in {"queued", "preparing"} and st.button("إلغاء التجهيز", key="cancelprep:" + job["id"], help="يطلب إيقاف التجهيز؛ قد يكتمل نقل الملف الجاري قبل توقف تنزيله."):
                manager.cancel(job["id"])
            if job.get("error"):
                st.error(job["error"])
            if job["status"] in {"completed", "no_speech"}:
                if result.get("timestamp_quality") == "chunk" or result.get("estimated_splits"):
                    st.info("بعض حدود المقاطع تقديرية. استخدم المحاذاة ومراجعة الفيديو إذا احتجت توقيت كلمات أدق.")
                if result.get("alignment_partial"):
                    st.warning("لم تكتمل محاذاة بعض المقاطع؛ احتُفظ بنصها وتوقيتها السابقين للمراجعة.")
                paths = [Path(p) for p in result.get("written_files", []) if Path(p).is_file()]
                for path in paths:
                    st.download_button("تنزيل " + path.name, path.read_bytes(), file_name=path.name, key="dl:" + job["id"] + ":" + path.suffix, help="ينزل ملف الإخراج المكتمل إلى جهازك.")
                if result.get("segments") and controls.checkbox("تحرير المقاطع وإعادة تصديرها", key="edit:" + job["id"], help="يمكن تعديل النص والبداية والنهاية؛ تُحفظ نسخة جديدة افتراضيًا."):
                    rows = [{"start": s["start"], "end": s["end"], "text": s["text"], "speaker": s.get("speaker", "")} for s in result["segments"]]
                    edited = st.data_editor(rows, key="editor:" + job["id"], hide_index=True, width="stretch", column_config={"start": st.column_config.NumberColumn("البداية", min_value=0, step=0.01), "end": st.column_config.NumberColumn("النهاية", min_value=0, step=0.01), "text": "النص", "speaker": "المتحدث"})
                    if st.button("حفظ التعديلات وتصديرها", key="saveedit:" + job["id"], help="يتحقق من التوقيت ويصدر الملفات دون تغيير الفيديو."):
                        try:
                            if any(not isinstance(r["text"], str) or r["start"] < 0 or r["end"] <= r["start"] for r in edited):
                                raise ValueError("يلزم نص وتوقيت نهاية أكبر من البداية لكل مقطع")
                            config = {"out_" + fmt: True for fmt in general["formats"]}
                            config.update(overwrite=False, output_suffix="_edited")
                            new_paths = write_all_outputs(job["out_dir"], job["out_name"], edited, config, ass_style(general))
                            st.success("حُفظت النسخة المعدّلة: " + ", ".join(new_paths))
                            for path in new_paths:
                                st.download_button("تنزيل " + Path(path).name, Path(path).read_bytes(), file_name=Path(path).name, key="edited-dl:" + path)
                        except Exception as exc:
                            st.error(str(exc))
            if job.get("logs"):
                st.code("\n".join(job["logs"][-35:]), language=None)


@st.fragment(run_every=3)
def render_history():
    history = Database().get_file_history()
    if not history:
        st.info("سيظهر هنا سجل الملفات بعد معالجة أول ملف.")
        return
    rows = []
    for record in history:
        engine = record["engine"]
        rows.append({
            "الملف": record["file_name"],
            "الوقت المستغرق (ثانية)": record["elapsed"],
            "المحرك": ENGINES[engine].name if engine in ENGINES else "استخراج مسار الترجمة" if engine == "extract" else engine or "—",
            "النموذج": record["model"] or "—",
            "تاريخ ووقت المعالجة": record["created_at"],
        })
    st.caption("التاريخ والوقت بحسب ساعة الجهاز عند تسجيل النتيجة.")
    st.dataframe(rows, hide_index=True, width="stretch", column_config={"الوقت المستغرق (ثانية)": st.column_config.NumberColumn(format="%.1f")})


def main():
    st.set_page_config(page_title="Whisper Pro", page_icon="🎙️", layout="wide")
    st.html("""<style>html,body,[data-testid=stAppViewContainer]{direction:rtl}.stCode{direction:ltr} [data-testid=stSidebar]{min-width:20rem} </style>""" + CSS)
    if "settings_v2" not in st.session_state:
        try:
            st.session_state.settings_v2 = load_settings()
        except Exception as exc:
            st.session_state.settings_v2 = initial_settings()
            st.warning("لم يُطبّق ملف الإعدادات غير الصالح: " + str(exc))
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.queue = []
    settings = st.session_state.settings_v2
    general = settings["general"]
    catalog = Catalog()
    manager, ops = resources(st.session_state.session_id)
    st.title("🎙️ Whisper Pro")
    st.caption("تفريغ وترجمة محليان · محركات متعددة · إعدادات بحسب النموذج")
    with st.sidebar:
        engine_picker(settings["engine"])
        engine = settings["engine"]
        st.caption(ENGINES[engine].note)
        runtime_status = st.empty()
        unavailable = ENGINES[engine].unavailable_reason()
        if unavailable:
            st.warning(unavailable)
        models = catalog.models(engine)
        if not models:
            st.info("لم تُضف نماذج لهذا المحرك بعد. استخدم الفحص أو أضف نموذجًا محليًا.")
            render_local_models(catalog, ops, general, engine)
            if st.button("البحث عن نماذج المحرك", help="يجلب نماذج متوافقة من الناشرين المعتمدين."):
                ops.start("فحص النماذج", lambda progress: discover_models(catalog, engine, progress=progress))
            local_path = controls.text_input("أو أضف نموذجًا محليًا", help="مسار مجلد أو ملف نموذج متوافق مع المحرك المختار.")
            local_family = controls.text_input("عائلة النموذج المحلي", value="whisper", help="مثلwhisper أوtransducer أوparaformer بحسب المحرك.")
            if st.button("إضافة النموذج", help="يتحقق من المسار ويضيفه دون نسخ الأوزان."):
                try:
                    catalog.register_local(engine, local_path, local_family)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
            render_operations(ops)
            return
        if engine in {"crisp", "transformers", "sherpa", "funasr", "nemo"}:
            families = ["all"] + sorted(set(m.get("family", "unknown") for m in models))
            family = controls.selectbox("عائلة النموذج", families, format_func=lambda v: "كل العائلات" if v == "all" else v, key="family:" + engine, help="يرشح النماذج حسب بنيتها؛ قد تختلف اللغات والتوقيت وإعدادات البحث بين العائلات.")
            if family != "all":
                models = [m for m in models if m.get("family") == family]
        ids = [m["id"] for m in models]
        selected = settings["selection"].get(engine)
        selected = selected if selected in ids else ids[0]
        widget_key = "model:" + engine
        if st.session_state.get(widget_key) not in ids:
            st.session_state[widget_key] = selected
        selected = controls.selectbox("النموذج", ids, format_func=lambda v: next(m["name"] for m in models if m["id"] == v), key=widget_key, help="اختر الحجم واللغة المناسبين لذاكرة جهازك. الدقة والسرعة تختلف حسب التسجيل؛ الأحدث ليس الأفضل دائمًا.")
        settings["selection"][engine] = selected
        model = next(m for m in models if m["id"] == selected)
        token = controls.text_input("رمز Hugging Face اختياري", type="password", key="session_hf_token", help="للنماذج المقيدة فقط؛ يجب قبول شروطها لدى المصدر. يبقى الرمز في الجلسة ولا يحفظ أو يصدر مع الإعدادات.")
        save = st.button("حفظ الإعدادات", type="primary", help="يحفظ اختيارات المحركات والنماذج والإعدادات بعد التحقق منها، دون رمز المصادقة.", width="stretch")
        if st.button("استعادة الإعدادات الافتراضية", help="يعيد إعدادات البرنامج فقط؛ لا يحذف النماذج أو نتائجك.", width="stretch"):
            st.session_state.settings_v2 = initial_settings()
            for key in list(st.session_state):
                if key.startswith(("setting:", "model:", "family:")) or key in {"active_engine", "translation_engine"}:
                    del st.session_state[key]
            st.rerun()
        st.caption("مجلد البيانات: " + str(data_dir()))
    profile = settings["profiles"].setdefault(engine, {}).setdefault(selected, defaults_for(engine, model))
    tabs = st.tabs(["الملفات والمهام", "المحرك والنموذج", "إعدادات التفريغ", "الإخراج والترجمة", "السجل", "الدليل والإعدادات المحفوظة"])
    with tabs[2]:
        values = render_fields(engine, model, profile)
        settings["profiles"][engine][selected] = values
        profile = values
    _, runtime_label = engine_state(engine, profile)
    runtime_status.info("المحرك: " + runtime_label)
    with tabs[3]:
        render_general(general, model, engine)
        render_postprocessing(general, catalog)
    with tabs[1]:
        render_models(catalog, engine, model, profile, general, ops, token)
    with tabs[0]:
        render_inputs(ops)
        mode = controls.radio("نوع المهمة", ["transcribe", "extract"], format_func=lambda v: "إنشاء ترجمة من الصوت" if v == "transcribe" else "استخراج مسار ترجمة نصية موجود", horizontal=True, help="استخراج ترجمة مضمنة لا يحتاج نموذج كلام. النص المحروق على الصورة يحتاج OCR مستقلًا.")
        subtitle_order = 0
        if st.session_state.queue:
            with st.expander("فحص مسارات أول ملف"):
                if st.button("فحص الوسائط", help="يعرض مسارات الصوت والفيديو والترجمة ولغاتها دون تشغيل نموذج."):
                    try:
                        st.json(inspect_media(st.session_state.queue[0]["path"]))
                    except Exception as exc:
                        st.error(str(exc))
        if mode == "extract":
            subtitle_order = controls.number_input("ترتيب مسار الترجمة", 0, 100, 0, help="صفر أول مسار ترجمة نصية داخل كل فيديو. مسارات الصور تحتاج OCR.")
        start = st.button("بدء المعالجة", type="primary", disabled=not st.session_state.queue or bool(unavailable and mode == "transcribe"), help="يتحقق من الإعدادات ويجهز المحرك والنموذج الناقصين، ثم يعالج الملفات بالخلفية.")
        if start:
            try:
                from core.settings import validate_settings
                validate_settings(settings)
                validate_profile(engine, model, profile)
                active = any(j["status"] not in TERMINAL for j in manager.snapshots())
                if manager.parallel != general["parallel_files"]:
                    if active:
                        raise ValueError("انتظر اكتمال المهام قبل تغيير عدد الملفات المتزامنة")
                    from concurrent.futures import ThreadPoolExecutor
                    manager.pool.shutdown(wait=False)
                    manager.pool = ThreadPoolExecutor(max_workers=general["parallel_files"])
                    manager.parallel = general["parallel_files"]
                for item in st.session_state.queue:
                    info = inspect_media(item["path"])
                    options = copy.deepcopy(general)
                    if item["uploaded"]:
                        options["beside_source"] = False
                    if mode == "extract":
                        tracks = [s for s in info["streams"] if s["type"] == "subtitle"]
                        if subtitle_order >= len(tracks):
                            raise ValueError("مسار الترجمة غير موجود في " + item["name"])
                        manager.submit(item["path"], engine, model, profile, options, token, mode, tracks[subtitle_order]["index"], display_name=item["name"])
                    else:
                        channels = [0, 1] if general["split_stereo"] else [None]
                        for channel in channels:
                            manager.submit(item["path"], engine, model, profile, options, token, channel=channel, display_name=item["name"])
                st.success("أُضيفت الملفات إلى المهام")
            except Exception as exc:
                st.error(str(exc))
        if any(j["status"] not in TERMINAL for j in manager.snapshots()) and st.button("إلغاء جميع المهام", help="يلغي المهام الجارية والمنتظرة دون حذف الفيديوهات أو نتائج سابقة."):
            for job in manager.snapshots():
                manager.cancel(job["id"])
        render_jobs(manager, general)
    with tabs[4]:
        render_history()
    with tabs[5]:
        st.write("ابدأ بمحرك Faster Whisper وCPU ودقةINT8 أو CrispASR مع نموذج صغير. اختر لغة الصوت، ثم صيغ الإخراج، وجرّب مقطعًا قصيرًا قبل الفيديو الطويل.")
        st.write("لكل محرك ملفات وأدوات مختلفة. تجهيز محرك اختياري لا يغير بيئة المحركات الأخرى. تتطلب بعض النماذج حسابًا وقبول شروط المصدر.")
        st.write("الإعدادات المتقدمة تظهر بحسب المحرك والنموذج. مرر المؤشر على اسم الإعداد لشرح عمله والقيمة المقترحة؛ لا توجد قيمة واحدة أفضل لكل التسجيلات.")
        st.markdown("\n".join(f"- [{e.name}]({e.source})" for e in ENGINES.values()))
        try:
            exported = export_settings(settings)
        except ValueError as exc:
            exported = ""
            st.warning(str(exc))
        st.download_button("تصدير الإعدادات", data=exported, disabled=not bool(exported), file_name="whisper-pro-settings.json", mime="application/json", help="ملف قابل للاستيراد؛ لا يحتوي رمز Hugging Face.")
        imported = controls.file_uploader("استيراد إعدادات البرنامج", type=["json"], key="settings_import", help="يتحقق من الأنواع والحدود قبل تطبيق أي تغيير.")
        if imported and st.button("تطبيق الإعدادات المستوردة", help="يطبق الملف كاملًا بعد نجاح التحقق؛ لا يثبت النماذج تلقائيًا."):
            try:
                clean = import_settings(imported.getvalue().decode("utf-8-sig"))
                st.session_state.settings_v2 = clean
                for key in list(st.session_state):
                    if key.startswith(("setting:", "model:", "family:")) or key in {"active_engine", "translation_engine"}:
                        del st.session_state[key]
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    if save:
        try:
            save_settings(settings)
            st.sidebar.success("حُفظت الإعدادات")
        except Exception as exc:
            st.sidebar.error(str(exc))
    render_operations(ops)
    if general["auto_check_models"] and not profile.get("local_files_only") and not os.environ.get("WHISPER_PRO_DISABLE_AUTOCHECK"):
        marker = read_json(location("catalog") / ("auto-" + engine + ".json"), {})
        if time.time() - marker.get("time", 0) > general["catalog_interval_hours"] * 3600 and not ops.snapshot()["running"]:
            write_json(location("catalog") / ("auto-" + engine + ".json"), {"time": time.time()})
            ops.start("فحص كتالوج النماذج بالخلفية", lambda progress: discover_models(catalog, engine, token, progress))
