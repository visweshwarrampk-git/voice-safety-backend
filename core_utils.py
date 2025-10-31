# core_utils.py
# --------------------------------------------------------
# Shared logic for Safety Incident Follow-up
# Used by both GUI (safety_core.py) and API (app_backend.py)
# --------------------------------------------------------

import os
import time
import uuid
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound
import whisper
import sounddevice as sd
from scipy.io.wavfile import write

# ---------------- SETTINGS ----------------
MODEL_SIZE = "small"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
TMP_DIR = "tmp_audio"
os.makedirs(TMP_DIR, exist_ok=True)

EMB_KEYWORD_WEIGHT = 0.18
CONF_THRESHOLD = 0.48
MIN_GAP = 0.06
TRANSLATION_FALLBACK = True

SUPPORTED_LANGS = {
    "en": "en",
    "ta": "ta", 
    "hi": "hi",
    "English": "en",
    "Tamil": "ta",
    "Hindi": "hi"
}

DetectorFactory.seed = 0

# Global models
whisper_model = None
embedding_model = None
scenario_embeddings = {}

# ---------------- SCENARIOS ----------------
SCENARIOS = {
    "fire_near_electrical_panel": {
        "description": "Fire or smoke near electrical panel or cables",
        "questions": [
            "Where exactly was the fire or smoke observed?",
            "What was the source (panel, cable, equipment)?",
            "When did you first notice the fire or smoke?"
        ],
        "examples": [
            "There was fire near the electrical panel",
            "Smoke came from the control panel",
            "I saw sparks and fire near the cables",
            "மின்சார பேனலில் தீப்பொறி அல்லது புகை வந்தது",
            "इलेक्ट्रिक पैनल के पास आग लग गई"
        ]
    },
    "machine_malfunction": {
        "description": "Machine stopped, jammed, noisy or malfunctioned during operation",
        "questions": [
            "Which machine malfunctioned (machine name or location)?",
            "What happened (stopped, jammed, loud noise)?",
            "When did the malfunction occur during the operation?"
        ],
        "examples": [
            "The cutting machine stopped suddenly while I was working",
            "The machine jammed and made loud noise",
            "Machine malfunctioned and would not restart",
            "பணியின் போது இயந்திரம் திடீரென நின்று விட்டது",
            "मशीन अचानक काम करना बंद कर दी"
        ]
    },
    "unauthorized_entry": {
        "description": "Someone entered a restricted/no-entry area without permission",
        "questions": [
            "Where did the unauthorized person enter (which area)?",
            "Was the person wearing any identification or uniform?",
            "When did you notice the person entering the area?"
        ],
        "examples": [
            "Someone entered the restricted area without permission",
            "An unauthorized person crossed into the no-entry zone",
            "I saw a person in the prohibited area without ID",
            "தடை செய்யப்பட்ட பகுதியில் ஒரு நபர் அனுமதி இல்லாமல் நுழைந்தார்",
            "कोई बिना अनुमति के प्रतिबंधित क्षेत्र में गया"
        ]
    }
}

# ---------------- KEYWORDS ----------------
SCENARIO_KEYWORDS = {
    "fire_near_electrical_panel": [
        "fire", "smoke", "spark", "burn", "electrical", "panel", "cable",
        "தீ", "புகை", "மின்பலகை",
        "आग", "धुआँ", "सर्किट"
    ],
    "machine_malfunction": [
        "machine", "stopped", "jam", "noise", "malfunction", "broken",
        "இயந்திரம்", "சத்தம்", "நிறுத்தியது",
        "மஷின்",
        "मशीन", "रुक", "जाम"
    ],
    "unauthorized_entry": [
        "restricted", "no entry", "unauthorized", "prohibited", "entered",
        "தடை", "நுழைந்தார்", "அனுமதி",
        "प्रतिबंधित", "प्रवेश"
    ]
}

# ---------------- AUDIO HELPERS ----------------
def safe_filename(prefix="audio", ext=".wav"):
    return os.path.join(TMP_DIR, f"{prefix}_{uuid.uuid4().hex}{ext}")

def record_to_file(seconds=RECORD_SECONDS):
    """Record microphone audio for given seconds."""
    fname = safe_filename("rec", ".wav")
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(fname, SAMPLE_RATE, audio)
    return fname

# ---------------- MODEL LOADING ----------------
def load_models():
    """Initialize Whisper and embedding models."""
    global whisper_model, embedding_model, scenario_embeddings
    if whisper_model is None:
        print("Loading Whisper model...")
        whisper_model = whisper.load_model(MODEL_SIZE)
    if embedding_model is None:
        print("Loading embedding model...")
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    if not scenario_embeddings:
        print("Precomputing scenario embeddings...")
        for key, data in SCENARIOS.items():
            examples = data.get("examples", [])
            scenario_embeddings[key] = embedding_model.encode(examples, convert_to_tensor=True)
    print("✅ Models ready.")

# Load in background when imported
load_models()

# ---------------- LANGUAGE DETECTION ----------------
def detect_language_improved(text, whisper_hint=None):
    DetectorFactory.seed = 0
    counts = {
        'ta': sum(1 for c in text if '\u0B80' <= c <= '\u0BFF'),
        'hi': sum(1 for c in text if '\u0900' <= c <= '\u097F'),
        'en': sum(1 for c in text if 'a' <= c.lower() <= 'z' or c.isdigit())
    }
    total = len(text) if text else 1
    ratios = {k: v / total for k, v in counts.items()}
    script_best = max(ratios, key=ratios.get)
    script_conf = ratios[script_best]

    try:
        ld = detect(text)
        if ld.lower() in ["ur", "ar"]:
            ld = "hi"
    except:
        ld = "en"

    if script_best == 'hi' and script_conf > 0.25:
        final = 'hi'
    else:
        final = script_best if script_conf > 0.25 else ld

    if whisper_hint:
        if whisper_hint.lower() in ["ur", "ar"]:
            whisper_hint = "hi"
        if whisper_hint in ["ta", "hi", "en"]:
            if script_conf < 0.10 or whisper_hint == script_best:
                final = whisper_hint
            elif whisper_hint == "hi" and final not in ["ta"]:
                final = "hi"

    if final == "ur":
        final = "hi"

    return final

# ---------------- TRANSCRIPTION ----------------
def transcribe_with_whisper(file_path):
    """Transcribe an audio file with Whisper and detect language."""
    global whisper_model
    if whisper_model is None:
        load_models()
    res = whisper_model.transcribe(file_path, language=None, fp16=False)
    text = res.get("text", "").strip()
    whisper_lang = res.get("language", "en")
    detected = detect_language_improved(text, whisper_hint=whisper_lang)
    return text, detected, whisper_lang

# ---------------- TRANSLATION ----------------
def translate_text(src_text, src_lang, tgt_lang):
    """Translate text using GoogleTranslator."""
    try:
        translator = GoogleTranslator(source=src_lang if src_lang else "auto", target=tgt_lang)
        return translator.translate(src_text)
    except Exception as e:
        print("Translate error:", e)
        return src_text

# ---------------- TEXT-TO-SPEECH ----------------
def tts_and_play(text, lang_code):
    """Convert text to speech and play it."""
    try:
        out = safe_filename("tts", ".mp3")
        tts = gTTS(text=text, lang=lang_code)
        tts.save(out)
        playsound(out)
    except Exception as e:
        print("TTS error:", e)

# ---------------- SCENARIO DETECTION ----------------
def compute_keyword_boost(text, scenario_key):
    txt = text.lower()
    keywords = SCENARIO_KEYWORDS.get(scenario_key, [])
    if not keywords:
        return 0.0
    found = 0
    for kw in keywords:
        if kw.lower() in txt:
            found += 1
    return min(1.0, found / max(1, len(keywords)))

def detect_scenario_hybrid(text, source_lang=None):
    """Detect scenario using embeddings + keywords + fallback translation."""
    global embedding_model, scenario_embeddings
    if embedding_model is None or not scenario_embeddings:
        load_models()

    input_emb = embedding_model.encode(text, convert_to_tensor=True)
    emb_scores = {}
    final_scores = {}

    for key, emb_set in scenario_embeddings.items():
        sims = util.cos_sim(input_emb, emb_set)
        emb_scores[key] = float(sims.max())

    kw_boosts = {k: compute_keyword_boost(text, k) for k in SCENARIOS.keys()}
    for k in SCENARIOS.keys():
        final_scores[k] = emb_scores.get(k, 0.0) + EMB_KEYWORD_WEIGHT * kw_boosts.get(k, 0.0)

    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best, best_val = sorted_scores[0]
    second_val = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

    debug = {
        "emb_scores": emb_scores,
        "keyword_boosts": kw_boosts,
        "final_scores": final_scores,
        "method": "direct"
    }

    if best_val >= CONF_THRESHOLD and (best_val - second_val) >= MIN_GAP:
        return best, debug

    # fallback: translate to English and retry
    if TRANSLATION_FALLBACK:
        try:
            translated = translate_text(text, source_lang or "auto", "en")
            input_emb2 = embedding_model.encode(translated, convert_to_tensor=True)
            emb_scores2 = {k: float(util.cos_sim(input_emb2, v).max()) for k, v in scenario_embeddings.items()}
            kw_boosts2 = {k: compute_keyword_boost(translated, k) for k in SCENARIOS.keys()}
            final_scores2 = {k: emb_scores2.get(k, 0.0) + EMB_KEYWORD_WEIGHT * kw_boosts2.get(k, 0.0) for k in SCENARIOS.keys()}
            sorted2 = sorted(final_scores2.items(), key=lambda x: x[1], reverse=True)
            best2, best2_val = sorted2[0]
            debug.update({
                "fallback_emb_scores": emb_scores2,
                "fallback_final_scores": final_scores2,
                "fallback_translated_text": translated,
                "method": "translated"
            })
            if best2_val >= CONF_THRESHOLD:
                return best2, debug
        except Exception as e:
            debug["fallback_error"] = str(e)

    return None, debug
