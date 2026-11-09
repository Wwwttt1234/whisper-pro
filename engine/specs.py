"""Declarative engine capabilities and setting descriptions used by UI and validation."""
from dataclasses import dataclass, replace
import os
import platform
import sys


@dataclass(frozen=True)
class Engine:
    id: str
    name: str
    source: str
    package: str = ""
    note: str = ""
    platforms: tuple = ("win32", "linux", "darwin")

    def unavailable_reason(self):
        if sys.platform not in self.platforms:
            return "هذا المحرك غير متاح لهذا النظام؛ اختر محركًا آخر."
        if self.id == "mlx" and platform.machine().lower() not in ("arm64", "aarch64"):
            return "MLX يحتاج جهاز Mac بمعالج Apple Silicon."
        return ""


ENGINES = {e.id: e for e in [
    Engine("faster_whisper", "Faster Whisper · CTranslate2", "https://github.com/SYSTRAN/faster-whisper", "faster-whisper", "ابدأ بـ INT8 على CPU. يدعم NVIDIA عند توفر مكتبات التسريع."),
    Engine("crisp", "CrispASR · نماذج متعددة", "https://github.com/CrispStrobe/CrispASR", note="Parakeet وQwen وCanary وعائلات أخرى؛ تتغير القدرات بحسب النموذج وبناء المحرك."),
    Engine("whisper_cpp", "Whisper.cpp", "https://github.com/absadiki/pywhispercpp", "pywhispercpp", "ربط pywhispercpp؛ الحزمة التلقائية تستخدم بناء CPU، ويمكن اختيار بناء مخصص للتسريع."),
    Engine("openai_whisper", "Whisper · PyTorch", "https://github.com/openai/whisper", "openai-whisper", "التنفيذ المرجعي المحلي. لا يستخدم خدمة سحابية."),
    Engine("transformers", "Transformers · ASR", "https://huggingface.co/docs/transformers/en/tasks/asr", "transformers", "Whisper وWav2Vec2 وعائلات ASR المتوافقة مع واجهة pipeline، دون تشغيل كود مستودعات افتراضيًا."),
    Engine("qwen", "Qwen ASR", "https://github.com/QwenLM/Qwen3-ASR", "qwen-asr", "إعدادات Qwen الأصلية. توقيت المقاطع تقريبي دون محاذاة مستقلة."),
    Engine("funasr", "FunASR · SenseVoice / Paraformer", "https://github.com/modelscope/FunASR", "funasr", "اختيار اللغة والتوقيت يعتمد على النموذج؛ SenseVoiceSmall ليس نموذجًا عربيًا."),
    Engine("nemo", "NVIDIA NeMo · Speech", "https://github.com/NVIDIA-NeMo/Speech", "nemo_toolkit[asr]", "يشمل Parakeet وCanary. بيئة Linux موصى بها للمحرك الأصلي؛ تتوفر نماذج بديلة عبر CrispASR على Windows.", ("linux",)),
    Engine("mlx", "MLX Whisper · Apple Silicon", "https://github.com/ml-explore/mlx-examples/tree/main/whisper", "mlx-whisper", "تشغيل محلي على Apple Silicon.", ("darwin",)),
    Engine("vosk", "Vosk · Kaldi", "https://alphacephei.com/vosk/", "vosk", "نماذج خفيفة محلية؛ اختر نموذج اللغة المطابق للصوت."),
    Engine("whisperx", "WhisperX", "https://github.com/m-bain/whisperX", "whisperx", "تفريغ دفعي ومحاذاة اختيارية. المحاذاة وكشف المتحدثين قد يحتاجان تنزيل نماذج إضافية."),
    Engine("sherpa", "Sherpa ONNX", "https://github.com/k2-fsa/sherpa-onnx", "sherpa-onnx", "نماذج ONNX رسمية من مشروع Sherpa: Transducer وParaformer وWhisper وMoonshine."),
]}

WHISPER = {"faster_whisper", "openai_whisper", "mlx", "whisperx", "whisper_cpp"}


def supports_translation(engine, model):
    if "turbo" in model.get("name", "").lower() or model.get("languages") == ["en"]:
        return False
    return (engine in WHISPER or engine in {"crisp", "transformers", "sherpa"} and model.get("family") == "whisper")

TORCH = {"openai_whisper", "transformers", "qwen", "funasr", "nemo", "whisperx"}
LLM_BACKENDS = {"qwen3", "voxtral", "voxtral4b", "granite", "granite-speech", "canary-qwen", "cohere", "glm-asr", "moss-transcribe", "moss-diarize"}


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    default: object
    explanation: str
    recommendation: str
    group: str = "جودة التفريغ"
    engines: frozenset = frozenset()
    minimum: float = None
    maximum: float = None
    step: float = None
    choices: tuple = ()
    multiline: bool = False

    @property
    def help(self):
        return f"{self.explanation}\n\nالبدء المقترح: {self.recommendation}"


def F(key, label, default, explanation, recommendation, group="جودة التفريغ", engines=(), minimum=None, maximum=None, step=None, choices=(), multiline=False):
    return Field(key, label, default, explanation, recommendation, group, frozenset(engines), minimum, maximum, step, tuple(choices), multiline)


FIELDS = [
    F("random_seed", "بذرة العشوائية", 42, "يثبت بداية مولد العشوائية؛ لا يضمن تطابق النتائج بين الأجهزة.", "42 للمقارنة، و-1 لعدم تثبيتها.", "متقدم", TORCH | {"faster_whisper"}, minimum=-1, maximum=2147483647, step=1),
    F("without_timestamps", "فك النص دون رموز التوقيت", False, "يعطل رموز التوقيت في Faster Whisper؛ قد تصبح حدود المقاطع تقريبية.", "معطل لملفات SRT وVTT؛ يفيد لتجارب إخراج النص.", "متقدم", {"faster_whisper"}),
    F("sampling_strategy", "طريقة بحث whisper.cpp", "beam", "الشعاعي يقارن عدة مسارات؛ الجشع يختار الأقرب في كل خطوة.", "beam للجودة أوgreedy للسرعة؛ قارن تسجيلًا ثابتًا.", engines={"whisper_cpp"}, choices=("beam", "greedy")),
    F("cpp_processors", "عمليات whisper.cpp المتوازية", 1, "يقسم الاستدلال داخل المحرك؛ الزيادة قد تضعف السياق وتستهلك موارد أكثر.", "1 ثم القياس على ملف طويل.", "الأداء", {"whisper_cpp"}, minimum=1, maximum=16, step=1),
    F("n_max_text_ctx", "أقصى رموز سياق النص", 16384, "حد النص السابق المستخدم كسياق لفك الترميز في whisper.cpp.", "16384 الافتراضية.", "متقدم", {"whisper_cpp"}, minimum=0, maximum=32768, step=128),
    F("no_context", "تجاهل سياق النص السابق", True, "يمنع استخدام التفريغ السابق كسياق في whisper.cpp.", "مفعل إذا ظهر تكرار غير حقيقي.", "متقدم", {"whisper_cpp"}),
    F("single_segment", "مقطع محرك واحد", False, "يدمج نتيجة مقطع العمل في قطعة واحدة؛ يختلف عن تنسيق أسطر الإخراج.", "معطل لتقسيم أفضل.", "التوقيت", {"whisper_cpp"}),
    F("token_timestamps", "توقيت الرموز في whisper.cpp", False, "يحسب توقيت الرموز داخل المحرك؛ الرمز قد يكون جزءًا من كلمة.", "فعّله عند تحديد طول مقطع المحرك.", "التوقيت", {"whisper_cpp"}),
    F("max_len", "طول مقطع whisper.cpp بالحروف", 0, "صفر دون حد إضافي؛ فرض حد يحتاج تفعيل توقيت الرموز.", "0، أو42 مع تفعيل توقيت الرموز.", "التوقيت", {"whisper_cpp"}, minimum=0, maximum=500, step=1),
    F("split_on_word", "تقسيم whisper.cpp عند الكلمات", True, "يحاول الفصل عند الكلمات بدل الرموز عند تحديد طول المقطع.", "مفعل مع حد الطول.", "التوقيت", {"whisper_cpp"}),
    F("cpp_max_tokens", "أقصى رموز مقطع whisper.cpp", 0, "حد الرموز داخل المقطع؛ صفر دون حد إضافي.", "0 حتى لا يقطع الكلام.", "متقدم", {"whisper_cpp"}, minimum=0, maximum=1024, step=16),
    F("audio_ctx", "سياق صوت whisper.cpp", 0, "يتجاوز حجم السياق الصوتي المعتاد؛ قيمة غير مناسبة قد تضعف التفريغ.", "0 لاستخدام الافتراضي.", "متقدم", {"whisper_cpp"}, minimum=0, maximum=1500, step=10),
    F("suppress_regex", "منع نص بنمط منتظم", "", "نمط Regex لمنع نصوص أثناء فك الترميز؛ النمط الواسع قد يمنع كلمات صحيحة.", "فارغ للتفريغ العام.", "متقدم", {"whisper_cpp"}),
    F("temperature_inc", "زيادة حرارة إعادة المحاولة", 0.2, "زيادة الحرارة عند تعثر فك الترميز في whisper.cpp.", "0.2؛ وصفر لإيقاف الزيادة.", engines={"whisper_cpp"}, minimum=0.0, maximum=1.0, step=0.05),
    F("entropy_thold", "عتبة إنتروبيا whisper.cpp", 2.4, "عتبة لاكتشاف مخرجات غير موثوقة وإعادة المحاولة.", "2.4.", engines={"whisper_cpp"}, minimum=0.0, maximum=10.0, step=0.1),
    F("do_sample", "أخذ عينات في Transformers", False, "يستخدم أخذ العينات بدل فك الترميز الحتمي؛ يحتاج حرارة موجبة.", "معطل أولًا؛ جرّبه بحرارة0.2.", engines={"transformers"}),
    F("top_p", "كتلة احتمالات أخذ العينات", 0.9, "يختار من أصغر مجموعة رموز تجمع هذه الكتلة الاحتمالية.", "0.9 عند تفعيل أخذ العينات.", engines={"transformers"}, minimum=0.05, maximum=1.0, step=0.05),
    F("top_k", "مرشحو أخذ العينات", 50, "يقصر أخذ العينات على أعلى الرموز احتمالًا؛ صفر يلغي هذا القيد.", "50 عند تفعيل أخذ العينات.", engines={"transformers"}, minimum=0, maximum=500, step=1),
    F("early_stopping", "إنهاء البحث مبكرًا", False, "ينهي البحث الشعاعي بعد اكتمال عدد كافٍ من المسارات.", "معطل أولًا ثم قارن.", engines={"transformers"}),
    F("use_itn", "تطبيع الأرقام والنص", True, "يحاول FunASR كتابة الأرقام والتعبيرات بصيغتها المقروءة وفق دعم النموذج.", "مفعل؛ أوقفه لمقارنة النص الخام.", engines={"funasr"}),
    F("decoding_method", "طريقة بحث Sherpa Transducer", "greedy_search", "البحث المعدل يحتفظ بمسارات أكثر وقد يحسن المصطلحات بكلفة أعلى.", "greedy_search للسرعة، وmodified_beam_search للتجربة.", engines={"sherpa"}, choices=("greedy_search", "modified_beam_search")),
    F("sherpa_mode", "واجهة نموذج Sherpa", "auto", "التلقائي يختار واجهة التدفق للنماذج التي تحملstreaming في اسمها. حددها يدويًا للنموذج المحلي عند الحاجة.", "auto للنماذج الموجودة في الدليل.", "متقدم", {"sherpa"}, choices=("auto", "offline", "online")),
    F("max_active_paths", "مسارات بحث Sherpa", 4, "المسارات النشطة عند استخدام البحث المعدل.", "4.", engines={"sherpa"}, minimum=1, maximum=32, step=1),
    F("hotwords_file", "ملف مصطلحات Sherpa", "", "ملف مصطلحات بصيغة نموذج Transducer؛ يحتاج البحث المعدل.", "فارغ أو ملف محلي موثق.", "متقدم", {"sherpa"}),
    F("hotwords_score", "ترجيح مصطلحات Sherpa", 1.5, "ترجيح كلمات ملف المصطلحات؛ القيمة العالية قد تفرض كلمات خاطئة.", "1.5.", "متقدم", {"sherpa"}, minimum=0.0, maximum=10.0, step=0.1),
    F("blank_penalty", "عقوبة فراغ Sherpa", 0.0, "تغير ترجيح رمز الفراغ في Transducer، وقد تؤثر في الكلمات المحذوفة أو الزائدة.", "0.0 أولًا.", "متقدم", {"sherpa"}, minimum=-10.0, maximum=10.0, step=0.1),
    F("device", "جهاز المعالجة", "cpu", "يحدد المعالج الذي ينفذ النموذج؛ الدعم يتغير بحسب المحرك وبنائه.", "CPU للتوافق، وCUDA عند التأكد من توفر NVIDIA ومتطلباتها.", "الأداء", choices=("cpu", "cuda", "auto", "mps")),
    F("compute_type", "دقة الحساب", "int8", "تقليل الدقة يقلل الذاكرة وقد يغير النتيجة؛ الدقات المتاحة تعتمد على الجهاز.", "INT8 على CPU في Faster Whisper، وfloat32 على CPU في PyTorch.", "الأداء", {"faster_whisper", "whisperx"}, choices=("int8", "int8_float32", "int8_float16", "float16", "float32", "default")),
    F("dtype", "دقة أوزان PyTorch", "float32", "نوع الأرقام المستخدمة أثناء الاستدلال.", "float32 على CPU؛ float16 أو bfloat16 على بطاقة تدعمهما.", "الأداء", TORCH - {"whisperx"}, choices=("float32", "float16", "bfloat16", "auto")),
    F("threads", "خيوط المعالج", min(4, os.cpu_count() or 1), "عدد خيوط CPU المخصصة لكل مهمة، وليس عدد الملفات المتزامنة.", "ابدأ بـ4 أو عدد الأنوية الفعلية، وخفّضه إذا كنت تعالج ملفات بالتوازي.", "الأداء", minimum=1, maximum=max(256, os.cpu_count() or 1), step=1),
    F("workers", "عمال المحرك", 1, "عدد العمال الداخليين للنموذج في CTranslate2.", "1؛ زد العدد بعد قياس الذاكرة والأداء.", "الأداء", {"faster_whisper"}, minimum=1, maximum=32, step=1),
    F("device_index", "رقم البطاقة", 0, "ترقيم البطاقات يبدأ من صفر.", "0 إذا كانت لديك بطاقة واحدة.", "الأداء", minimum=0, maximum=32, step=1),
    F("batch_size", "حجم الدفعة", 4, "عدد مقاطع الصوت التي يعالجها المحرك معًا؛ الزيادة تستهلك ذاكرة أكثر.", "4؛ خفّض إلى1 عند نفاد الذاكرة.", "الأداء", {"transformers", "nemo", "qwen", "whisperx", "funasr"}, minimum=1, maximum=128, step=1),
    F("chunk_seconds", "مدة مقطع العمل بالثواني", 30, "تقسيم المعالجة يتيح التقدم والإلغاء ويحد من استهلاك الذاكرة. قد تفقد المقاطع القصيرة بعض السياق.", "30 ثانية، ثم استخدم المحاذاة إذا احتجت توقيت كلمات أدق.", "التوقيت", minimum=5, maximum=120, step=1),
    F("beam_size", "عرض البحث", 5, "عدد مسارات البحث عن النص. زيادة القيمة قد ترفع وقت المعالجة دون تحسن مضمون.", "5 للتوازن، و1 للسرعة.", engines=WHISPER | {"crisp", "transformers", "sherpa"}, minimum=1, maximum=30, step=1),
    F("best_of", "عدد الترشيحات", 5, "يستخدم مع أخذ العينات عند حرارة أكبر من صفر.", "5؛ لا تتوقع تأثيره عند حرارة صفر.", engines={"faster_whisper", "openai_whisper", "mlx", "whisper_cpp"}, minimum=1, maximum=30, step=1),
    F("temperature", "حرارة أخذ العينات", 0.0, "زيادة العشوائية قد تساعد عند تعثر فك الترميز، لكنها قد تزيد عدم ثبات النص.", "0 للتفريغ الثابت؛ ليست أداة لتحسين الدقة دائمًا.", engines=WHISPER | {"crisp", "transformers"}, minimum=0.0, maximum=1.0, step=0.05),
    F("temperature_list", "درجات حرارة إعادة المحاولة", "", "قائمة مفصولة بفاصلة لتجربة درجات حرارة عند فشل معايير الجودة.", "اتركها فارغة أو استخدم 0,0.2,0.4,0.6 عند الحاجة.", engines={"faster_whisper", "openai_whisper", "mlx"}),
    F("initial_prompt", "سياق وأسماء متوقعة", "", "نص قصير يساعد في التعرف على الأسماء والمصطلحات؛ لا يضمن فرض الكلمات.", "أسماء ومصطلحات متوقعة فعلًا؛ تجنب الجمل الطويلة غير الموجودة بالصوت.", engines=WHISPER | {"crisp", "qwen"}, multiline=True),
    F("hotwords", "كلمات ومصطلحات مرجّحة", "", "ترجيح كلمات متوقعة عند دعم النموذج لذلك.", "أسماء الأشخاص والمصطلحات الصعبة فقط.", engines={"faster_whisper", "crisp", "funasr"}),
    F("condition_on_previous_text", "الاعتماد على النص السابق", False, "يحافظ على السياق بين مقاطع Whisper، وقد يزيد تكرار النص في بعض التسجيلات.", "معطل عند التكرار؛ فعّله للمقارنة في الحوار المترابط.", engines={"faster_whisper", "openai_whisper", "mlx"}),
    F("patience", "صبر البحث", 1.0, "يوسع البحث قبل قبول النتيجة عند استخدام البحث الشعاعي.", "1.0.", engines={"faster_whisper", "openai_whisper", "mlx", "whisper_cpp"}, minimum=0.1, maximum=10.0, step=0.1),
    F("length_penalty", "عامل طول النص", 1.0, "يؤثر في مقارنة مسارات البحث مختلفة الأطوال.", "1.0؛ غيّره فقط بعد مقارنة تسجيل ثابت.", engines={"faster_whisper", "openai_whisper", "mlx"}, minimum=0.1, maximum=5.0, step=0.1),
    F("repetition_penalty", "عقوبة تكرار الرموز", 1.0, "يخفض ترجيح الرموز المتكررة؛ قد يضر بالتكرار الحقيقي في الكلام.", "1.0، ثم زيادة صغيرة مثل1.1 عند تكرار غير حقيقي.", engines={"faster_whisper", "transformers"}, minimum=1.0, maximum=2.0, step=0.05),
    F("no_repeat_ngram_size", "منع تكرار مجموعات الرموز", 0, "يمنع تكرار مجموعات بطول محدد، وقد يمنع كلامًا صحيحًا أيضًا.", "0 لتعطيله.", engines={"faster_whisper", "transformers"}, minimum=0, maximum=10, step=1),
    F("compression_ratio_threshold", "عتبة كشف التكرار", 2.4, "ارتفاع قابلية ضغط النص قد يشير إلى تكرار آلي؛ ليست درجة دقة للكلمات.", "2.4.", engines={"faster_whisper", "openai_whisper", "mlx"}, minimum=0.0, maximum=10.0, step=0.1),
    F("log_prob_threshold", "عتبة احتمال النص", -1.0, "انخفاض احتمال النص قد يؤدي لإعادة المحاولة أو رفض مقطع بحسب المحرك.", "-1.0؛ تخفيفها قد يحتفظ بكلام ضعيف لكنه قد يزيد الهلوسة.", engines=WHISPER | {"crisp"}, minimum=-100.0, maximum=0.0, step=0.1),
    F("no_speech_threshold", "عتبة احتمال الصمت", 0.6, "تساعد في تجاهل المقاطع التي يرجح النموذج أنها بلا كلام.", "0.6؛ رفعها يحتفظ بمقاطع أكثر مع احتمال نص وهمي في الصمت.", engines=WHISPER | {"crisp"}, minimum=0.0, maximum=1.0, step=0.05),
    F("word_timestamps", "توقيت الكلمات", True, "يحسب زمن كل كلمة عندما يدعم النموذج ذلك؛ لا يعني دقة مضمونة.", "مفعل لتحسين تقسيم الترجمة، مع مراجعة النتائج.", "التوقيت", {"faster_whisper", "openai_whisper", "mlx", "transformers", "vosk", "whisperx"}),
    F("max_new_tokens", "أقصى رموز مولدة", 0, "حد للرموز المولدة، وهي ليست مساوية لعدد الكلمات. صفر يستخدم الإعداد المعتاد للمحرك.", "0 أو256 للمقاطع القصيرة؛ ارفعه عند انقطاع النص.", engines={"faster_whisper", "transformers", "qwen", "crisp"}, minimum=0, maximum=8192, step=16),
    F("clip_timestamps", "نطاقات زمنية محددة", "", "أزواج بداية ونهاية بالثواني، مثل0,30,60,90.", "فارغ لمعالجة الملف كاملًا.", "التوقيت", {"faster_whisper"}),
    F("max_initial_timestamp", "أقصى توقيت مبدئي", 1.0, "قيد على اختيار Whisper لتوقيت البداية الأولي.", "1.0.", "التوقيت", {"faster_whisper", "openai_whisper", "mlx"}, minimum=0.0, maximum=60.0, step=0.5),
    F("hallucination_silence_threshold", "تجاوز الصمت عند الاشتباه بالهلوسة", 0.0, "مدة الصمت المستخدمة مع توقيت الكلمات لتجاوز مقاطع مشتبه بها. صفر يعطله.", "معطل أولًا؛ جرّبه عند نص متكرر في الصمت.", "التوقيت", {"faster_whisper", "openai_whisper"}, minimum=0.0, maximum=60.0, step=0.5),
    F("suppress_tokens", "رموز ممنوعة", "-1", "معرّفات رموز مفصولة بفاصلة؛ -1 لقائمة المنع المعتادة للمحرك.", "-1؛ لا تضف أرقامًا دون معرفة معاني الرموز.", "متقدم", {"faster_whisper", "openai_whisper", "mlx"}),
    F("suppress_blank", "منع رمز الفراغ", True, "يمنع بعض المخرجات الفارغة في بداية فك الترميز.", "مفعل.", "متقدم", {"faster_whisper", "openai_whisper", "mlx", "whisper_cpp"}),
    F("prompt_reset_on_temperature", "حرارة إعادة ضبط السياق", 0.5, "يمسح السياق السابق عند إعادة المحاولة بحرارة أعلى من هذه القيمة.", "0.5.", "متقدم", {"faster_whisper", "openai_whisper", "mlx"}, minimum=0.0, maximum=1.0, step=0.05),
    F("prefix", "بداية مفروضة للنص", "", "نص بداية لفك الترميز؛ يختلف عن السياق التلميحي.", "فارغ إلا في حالات متخصصة.", "متقدم", {"faster_whisper", "openai_whisper", "mlx"}),
    F("prepend_punctuations", "علامات تلحق بالكلمة التالية", "\"'¿([{-", "ترقيم يدمج مع بداية الكلمة عند حساب توقيت الكلمات.", "القيمة الافتراضية.", "متقدم", {"faster_whisper", "openai_whisper", "mlx"}),
    F("append_punctuations", "علامات تلحق بالكلمة السابقة", "\"'.،!؟?)]}،", "ترقيم يدمج مع نهاية الكلمة عند حساب توقيت الكلمات.", "القيمة الافتراضية.", "متقدم", {"faster_whisper", "openai_whisper", "mlx"}),
    F("language_detection_threshold", "ثقة كشف اللغة", 0.5, "الثقة المطلوبة أثناء كشف لغة الصوت.", "0.5.", "متقدم", {"faster_whisper"}, minimum=0.0, maximum=1.0, step=0.05),
    F("language_detection_segments", "مقاطع كشف اللغة", 1, "زيادة المقاطع قد تحسن كشف اللغة وتزيد وقت البدء.", "1، أو3 إذا أخطأ الكشف.", "متقدم", {"faster_whisper"}, minimum=1, maximum=20, step=1),
    F("multilingual", "كشف اللغة لكل مقطع", False, "يسمح بتغير لغة المقاطع عند دعم نموذج Whisper لها.", "معطل؛ فعّله للصوت المختلط اللغات.", "متقدم", {"faster_whisper"}),
    F("vad_filter", "كشف مقاطع الكلام", False, "يحدد مقاطع الكلام؛ لا يفصل الموسيقى ولا يضمن إزالة الأخطاء.", "معطل للمقارنة أولًا، أو مفعل لتسجيلات الصمت الطويل.", "كشف الكلام", {"faster_whisper", "whisperx", "crisp"}),
    F("vad_threshold", "حساسية وجود الكلام", 0.5, "القيمة الأقل تلتقط كلامًا أهدأ وقد تلتقط ضجيجًا.", "0.5؛ جرّب0.3 إذا اختفت كلمات هادئة.", "كشف الكلام", {"faster_whisper", "whisperx", "crisp"}, minimum=0.05, maximum=0.95, step=0.05),
    F("vad_offset", "عتبة نهاية الكلام في WhisperX", 0.363, "عتبة إنهاء مقطع الكلام بعد انخفاض احتمال الصوت؛ يفسرها كاشف VAD المختار.", "0.363 مع عتبة بداية0.5.", "كشف الكلام", {"whisperx"}, minimum=0.0, maximum=0.95, step=0.01),
    F("vad_neg_threshold", "عتبة نهاية الكلام", 0.35, "يجب أن تقل عن عتبة وجود الكلام لتقليل التقطع.", "0.35 عند حساسية0.5.", "كشف الكلام", {"faster_whisper"}, minimum=0.0, maximum=0.9, step=0.05),
    F("vad_min_speech_duration_ms", "أقل مدة كلام بالمللي ثانية", 0, "تجاهل مقاطع الكلام الأقصر من هذه المدة.", "0 للحفاظ على الكلمات القصيرة.", "كشف الكلام", {"faster_whisper", "crisp"}, minimum=0, maximum=5000, step=50),
    F("vad_min_silence_duration_ms", "مدة الصمت للفصل", 1000, "أقل فترة صمت تفصل مقطعين.", "1000؛ جرّب500 للحوار السريع.", "كشف الكلام", {"faster_whisper", "crisp"}, minimum=50, maximum=10000, step=50),
    F("vad_speech_pad_ms", "هامش حول الكلام", 400, "يحافظ على صوت قبل المقطع وبعده حتى لا تُقص بدايات الكلمات.", "400 مللي ثانية.", "كشف الكلام", {"faster_whisper", "crisp"}, minimum=0, maximum=2000, step=50),
    F("vad_max_speech_duration_s", "أقصى مقطع كلام", 0, "يفصل الكلام الطويل؛ صفر يعني دون حد إضافي.", "0 أو30 حسب الذاكرة.", "كشف الكلام", {"faster_whisper", "crisp"}, minimum=0, maximum=300, step=5),
    F("align_words", "محاذاة الكلمات بعد التفريغ", True, "ينزّل نموذج محاذاة للغة إن كان مدعومًا ثم يحسن توقيت الكلمات.", "مفعل في WhisperX، مع مراجعة التوقيت.", "المحاذاة والمتحدثون", {"whisperx"}),
    F("diarization", "تمييز المتحدثين", False, "يُسند رموزًا للمتحدثين؛ لا يتعرف على هوياتهم الحقيقية.", "فعّله للمقابلات، ويتطلب نماذج إضافية وشروطها.", "المحاذاة والمتحدثون", {"whisperx", "crisp"}),
    F("min_speakers", "أقل عدد متحدثين", 1, "قيد يساعد خوارزمية التمييز؛ لا يثبت العدد الحقيقي.", "1 إن لم تكن متأكدًا.", "المحاذاة والمتحدثون", {"whisperx", "crisp"}, minimum=1, maximum=32, step=1),
    F("max_speakers", "أقصى عدد متحدثين", 8, "حد أعلى للتجميع بين المتحدثين.", "عدد قريب من المتوقع.", "المحاذاة والمتحدثون", {"whisperx", "crisp"}, minimum=1, maximum=32, step=1),
    F("native_binary", "مسار محرك جاهز اختياري", "", "يمكنك اختيار ملف CrispASR الذي ثبته بنفسك بدل النسخة التي يوفرها البرنامج.", "فارغ للتجهيز التلقائي من الإصدار الرسمي.", "متقدم", {"crisp"}),
    F("native_gpu", "بناء وتسريع CrispASR", "cpu", "يحدد حزمة المحرك وخيار التسريع؛ يتطلب GPU تعريفًا متوافقًا.", "CPU للتوافق، ثم جرّب بناء Vulkan أوCUDA المتاح لنظامك.", "الأداء", {"crisp"}, choices=("cpu", "vulkan", "cuda", "metal")),
    F("aligner_path", "نموذج محاذاة CrispASR اختياري", "", "مسار ملف محاذاة جاهز أوauto للاستعانة بمحلل المحرك. قد ينزّل نموذجًا إضافيًا.", "فارغ أولًا؛ auto عند الحاجة لتوقيت كلمات نموذج لا يوفرها.", "المحاذاة والمتحدثون", {"crisp"}),
    F("kv_quant", "دقة ذاكرة السياق", "default", "تقليل ذاكرة السياق للنماذج اللغوية المدعومة؛ الدعم مرتبط ببناء CrispASR.", "default، ثمq8_0 إذا ضاقت الذاكرة.", "متقدم", {"crisp"}, choices=("default", "q8_0", "q4_0", "f16")),
    F("grammar", "قاموس Vosk المقيد", "", "قائمة JSON لعبارات مسموحة في نماذج Vosk التي تدعم القاموس المتغير.", "فارغ للتفريغ العام؛ لا تستخدمه للنصوص الحرة.", "متقدم", {"vosk"}, multiline=True),
    F("sherpa_provider", "مزود ONNX", "cpu", "مزود التنفيذ في حزمة ONNX؛ المزود المتاح يعتمد على بناء Sherpa.", "cpu للحزمة التلقائية.", "الأداء", {"sherpa"}, choices=("cpu",)),
    F("revision", "تثبيت إصدار النموذج", "", "وسم أو بصمة إصدار من المصدر؛ فارغ يعني الإصدار الحالي عند التنزيل.", "فارغ عادةً؛ ثبّت بصمة لإعادة نتائج تجربة سابقة.", "متقدم"),
    F("runtime_python", "بيئة محرك مخصصة اختياريًا", "", "مسار Python داخل بيئة جهزتها بنفسك؛ يتيح استخدام بناء تسريع مخصص.", "فارغ للتجهيز التلقائي. لا يلزم هذا الإعداد للاستخدام المعتاد.", "متقدم", set(ENGINES) - {"crisp"}),
    F("native_target_language", "لغة هدف الترجمة المباشرة", "", "رمز لغة الهدف للمحركات التي تعلن دعم ترجمة الكلام بين لغتين.", "فارغ للتفريغ المعتاد؛ استخدم مرحلة الترجمة النصية إذا لم يدعم النموذج اللغة المطلوبة.", "جودة التفريغ", {"crisp"}),
    F("vad_model_path", "ملف كشف كلام محلي اختياري", "", "مسار نموذج VAD الذي سيستخدمه CrispASR بدل التنزيل الافتراضي.", "فارغ للتجهيز الافتراضي؛ اختر ملفًا موجودًا عند منع التنزيل.", "كشف الكلام", {"crisp"}),
    F("diarization_model_path", "ملف تضمين متحدثين محلي", "", "نموذج تضمين صوتي متوافق مع طريقة التمييز في CrispASR.", "فارغ للتجهيز المعتاد؛ اختر نموذجًا محليًا عند منع التنزيل.", "المحاذاة والمتحدثون", {"crisp"}),
    F("local_files_only", "العمل دون تنزيل", False, "يمنع تجهيز نموذج أو محرك ناقص عبر الإنترنت، ويستخدم المثبت السليم فقط.", "فعّله بعد اكتمال تجهيز الأدوات والنماذج.", "متقدم"),
]
FIELD_MAP = {f.key: f for f in FIELDS}


def fields_for(engine_id, model=None):
    model = model or {}
    family = model.get("family", "")
    caps = set()
    if engine_id == "crisp":
        from pathlib import Path
        from core.paths import data_dir
        from core.storage import read_json
        fallback = read_json(Path(__file__).with_name("crisp_capabilities.json"), {})
        matrix = read_json(data_dir() / "catalog/crisp_capabilities.json", fallback)
        caps = next((set(b["caps"]) for b in matrix.get("backends", []) if b["name"] == family), set())
    for field in FIELDS:
        if field.engines and engine_id not in field.engines:
            continue
        if engine_id == "crisp":
            if field.key in {"device", "dtype", "device_index"}:
                continue
            if field.key == "kv_quant" and family not in LLM_BACKENDS:
                continue
            if field.key == "beam_size" and "beam-search" not in caps or field.key == "temperature" and "temperature" not in caps or field.key == "diarization" and "diarize" not in caps:
                continue
            if field.key in {"min_speakers", "max_new_tokens"} and (field.key == "min_speakers" or family not in LLM_BACKENDS):
                continue
            if field.key == "initial_prompt" and family not in {"whisper", "qwen3", "qwen3-1.7b", "vibevoice"}:
                continue
            if field.key == "hotwords" and family not in {"whisper", "parakeet", "qwen3", "qwen3-1.7b", "voxtral"}:
                continue
            if field.key == "native_target_language" and "src-tgt-language" not in caps:
                continue
            if field.key in {"log_prob_threshold", "no_speech_threshold"} and family != "whisper":
                continue
        if engine_id in {"mlx", "vosk", "whisper_cpp", "sherpa"} and field.key in {"device", "device_index"}:
            continue
        if engine_id == "mlx" and field.key == "threads":
            continue
        if engine_id == "vosk" and field.key in {"threads", "chunk_seconds"}:
            continue
        if field.key == "revision" and (model.get("source") == "local" or model.get("format") in {"pt", "vosk"}):
            continue
        if engine_id == "openai_whisper" and field.key == "dtype":
            field = replace(field, choices=("float32", "float16"))
        if engine_id == "mlx" and field.key in {"beam_size", "patience"}:
            continue
        if engine_id == "whisperx" and field.key in {"word_timestamps", "vad_filter"}:
            continue
        if engine_id == "funasr" and field.key == "batch_size":
            continue
        if engine_id == "sherpa" and field.key == "beam_size":
            continue
        if engine_id == "sherpa" and family not in {"transducer", "paraformer"} and field.key == "sherpa_mode":
            continue
        if engine_id == "sherpa" and family != "transducer" and field.key in {"decoding_method", "max_active_paths", "hotwords_file", "hotwords_score", "blank_penalty"}:
            continue
        if engine_id == "transformers" and family != "whisper" and field.key in {"temperature", "beam_size", "repetition_penalty", "no_repeat_ngram_size", "max_new_tokens", "do_sample", "top_p", "top_k", "early_stopping"}:
            continue
        yield field


def defaults_for(engine_id, model=None):
    values = {field.key: field.default for field in fields_for(engine_id, model)}
    if engine_id == "crisp" and (model or {}).get("family") != "whisper" and "beam_size" in values:
        values["beam_size"] = 1
    return values
