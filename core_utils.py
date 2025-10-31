# core_utils.py
# --------------------------------------------------------
# Optimized for Render Free Plan (512MB)
# Lightweight, no sounddevice, lazy embedding load, tiny Whisper
# --------------------------------------------------------

import os
import time
import uuid
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from gtts import gTTS
import whisper

# ---------------- SETTINGS ----------------
MODEL_SIZE = "tiny"  # ✅ small → tiny (fits under 512MB)
SAMPLE_RATE = 16000
TMP_DIR = "tmp_audio"
os.makedirs(TMP_DIR, exist_ok=True)

EMB_KEYWORD_WEIGHT = 0.18
CONF_THRESHOLD = 0.48
MIN_GAP = 0.06
TRANSLATION_FALLBACK = True

SUPPORTED_LANGS = {"en": "en", "ta": "ta", "hi": "hi"}

DetectorFactory.seed = 0

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
        "தீ", "புகை", "மின்பலகை", "आग", "धुआँ", "सर्किट"
    ],
    "machine_malfunction": [
        "machine", "stopped", "jam", "noise", "malfunction", "broken",
        "இயந்திரம்", "சத்தம்", "நிறுத்தியது", "மஷின்",
        "मशीन", "रुक", "जाम"
    ],
    "unauthorized_entry": [
        "restricted", "no entry", "unauthorized", "prohibited", "entered",
        "தடை", "நுழைந்தார்", "அனுமதி", "प्रतिबंधित", "प्रवेश"
    ]
}

# ---------------- HELPERS ----------------
def safe_filename(prefix="audio", ext=".wav"):
    return os.path.join(TMP_DIR, f"{prefix}_{uuid.uuid4().hex}{ext}")

# ---------------- MODEL LOADING ----------------
def load_models():
    """Load Whisper (tiny) model only."""
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper model (tiny)...")
        whisper_model = whisper.load_model(MODEL_SIZE)
        print("✅ Whisper ready (tiny).")

# Lazy-load embeddings only when needed
def load_embeddings():
    """Load sentence transformer only when required."""
    global embedding_model, scenario_embeddings
    if embedding_model is None:
        from sentence_transformers import SentenceTransformer, util
        print("⚙️ Loading lightweight embedding model (MiniLM-L3-v2)...")
        embedding_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
        for key, data in SCENARIOS.items():
            examples = data.get("examples", [])
            scenario_embeddings[key] = embedding_model.encode(examples, convert_to_tensor=True)
    return embedding_model

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

    final = script_best if script_conf > 0.25 else ld
    if whisper_hint and whisper_hint in ["ta", "hi", "en"]:
        final = whisper_hint
    return final

# ---------------- TRANSCRIPTION ----------------
def transcribe_with_whisper(file_path):
    """Transcribe audio file using Whisper tiny model."""
    global whisper_model
    if whisper_model is None:
        load_models()
    res = whisper_model.transcribe(file_path, fp16=False)
    text = res.get("text", "").strip()
    whisper_lang = res.get("language", "en")
    detected = detect_language_improved(text, whisper_hint=whisper_lang)
    return text, detected, whisper_lang

# ---------------- TRANSLATION ----------------
def translate_text(src_text, src_lang, tgt_lang):
    try:
        return GoogleTranslator(source=src_lang or "auto", target=tgt_lang).translate(src_text)
    except Exception as e:
        print("Translate error:", e)
        return src_text

# ---------------- TTS ----------------
def tts_and_play(text, lang_code):
    try:
        out = safe_filename("tts", ".mp3")
        gTTS(text=text, lang=lang_code).save(out)
        print(f"TTS generated ({lang_code}): {out}")
    except Exception as e:
        print("TTS error:", e)

# ---------------- SCENARIO DETECTION ----------------
def compute_keyword_boost(text, scenario_key):
    txt = text.lower()
    keywords = SCENARIO_KEYWORDS.get(scenario_key, [])
    found = sum(1 for kw in keywords if kw.lower() in txt)
    return min(1.0, found / max(1, len(keywords)))

def detect_scenario_hybrid(text, source_lang=None):
    """Detect scenario using small embeddings and keywords."""
    from sentence_transformers import util
    emb_model = load_embeddings()
    input_emb = emb_model.encode(text, convert_to_tensor=True)
    emb_scores = {k: float(util.cos_sim(input_emb, v).max()) for k, v in scenario_embeddings.items()}
    kw_boosts = {k: compute_keyword_boost(text, k) for k in SCENARIOS.keys()}
    final_scores = {k: emb_scores[k] + EMB_KEYWORD_WEIGHT * kw_boosts[k] for k in SCENARIOS.keys()}
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best, best_val = sorted_scores[0]
    if best_val < CONF_THRESHOLD:
        translated = translate_text(text, source_lang or "auto", "en")
        input_emb2 = emb_model.encode(translated, convert_to_tensor=True)
        emb_scores2 = {k: float(util.cos_sim(input_emb2, v).max()) for k, v in scenario_embeddings.items()}
        sorted2 = sorted(emb_scores2.items(), key=lambda x: x[1], reverse=True)
        best, best_val = sorted2[0]
    return best, {"final_scores": final_scores}
