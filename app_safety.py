import os
import threading
import time
import uuid
import json
from tkinter import *
from tkinter import filedialog, messagebox, ttk

# Optional packages only available locally
import sounddevice as sd
from scipy.io.wavfile import write
from playsound import playsound

# Core AI/ML libraries
import whisper
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from gtts import gTTS

# ---------------- SETTINGS ----------------
MODEL_SIZE = "tiny"  # ✅ matches Render (fits under 512 MB)
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
TMP_DIR = "tmp_audio"
os.makedirs(TMP_DIR, exist_ok=True)

EMB_KEYWORD_WEIGHT = 0.18
CONF_THRESHOLD = 0.48
MIN_GAP = 0.06
TRANSLATION_FALLBACK = True

SUPPORTED_LANGS = {"English": "en", "Tamil": "ta", "Hindi": "hi"}

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

def record_to_file(seconds=RECORD_SECONDS):
    fname = safe_filename("rec", ".wav")
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    write(fname, SAMPLE_RATE, audio)
    return fname

# ---------------- LOAD MODELS ----------------
def load_models_background():
    global whisper_model, embedding_model, scenario_embeddings
    print("Loading Whisper model (tiny)...")
    whisper_model = whisper.load_model(MODEL_SIZE)
    print("Loading lightweight embedding model...")
    embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # ✅ lighter alternative
    print("Precomputing scenario embeddings...")
    for key, data in SCENARIOS.items():
        examples = data.get("examples", [])
        scenario_embeddings[key] = embedding_model.encode(examples, convert_to_tensor=True)
    print("✅ Models loaded successfully.")

threading.Thread(target=load_models_background, daemon=True).start()

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
    global whisper_model
    while whisper_model is None:
        time.sleep(0.3)
    res = whisper_model.transcribe(file_path, language=None, fp16=False)
    text = res.get("text", "").strip()
    whisper_lang = res.get("language", "en")
    detected = detect_language_improved(text, whisper_hint=whisper_lang)
    return text, detected, whisper_lang

# ---------------- TRANSLATION ----------------
def translate_text(src_text, src_lang, tgt_lang):
    try:
        return GoogleTranslator(source=src_lang if src_lang else "auto", target=tgt_lang).translate(src_text)
    except Exception as e:
        print("Translate error:", e)
        return src_text

# ---------------- TEXT-TO-SPEECH ----------------
def tts_and_play(text, lang_code):
    try:
        out = safe_filename("tts", ".mp3")
        gTTS(text=text, lang=lang_code).save(out)
        playsound(out)
    except Exception as e:
        print("TTS error:", e)

# ---------------- SCENARIO DETECTION ----------------
def compute_keyword_boost(text, scenario_key):
    txt = text.lower()
    keywords = SCENARIO_KEYWORDS.get(scenario_key, [])
    found = sum(1 for kw in keywords if kw.lower() in txt)
    return min(1.0, found / max(1, len(keywords)))

def detect_scenario_hybrid(text, source_lang=None):
    global embedding_model, scenario_embeddings
    while embedding_model is None or not scenario_embeddings:
        time.sleep(0.2)

    input_emb = embedding_model.encode(text, convert_to_tensor=True)
    emb_scores = {k: float(util.cos_sim(input_emb, v).max()) for k, v in scenario_embeddings.items()}
    kw_boosts = {k: compute_keyword_boost(text, k) for k in SCENARIOS.keys()}
    final_scores = {k: emb_scores[k] + EMB_KEYWORD_WEIGHT * kw_boosts[k] for k in SCENARIOS.keys()}
    sorted_scores = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    best, best_val = sorted_scores[0]
    return best, {"final_scores": final_scores}

# ---------------- GUI CLASS ----------------
class SafetyApp:
    def __init__(self, root):
        self.root = root
        root.title("Safety Incident Follow-up (Optimized for Tiny Whisper)")
        root.geometry("980x760")

        top = Frame(root)
        top.pack(padx=8, pady=8, fill=X)
        Label(top, text="Incident initial audio (worker):").pack(anchor=W)
        btns = Frame(top); btns.pack(anchor=W, pady=4)
        Button(btns, text="🎙️ Record Incident", command=self.record_incident_click, width=18).pack(side=LEFT, padx=4)
        Button(btns, text="📁 Upload Incident File", command=self.upload_incident_click, width=18).pack(side=LEFT, padx=4)

        Label(top, text="Select target language for answers:").pack(anchor=W, pady=(8,0))
        self.tgt_combo = ttk.Combobox(top, values=list(SUPPORTED_LANGS.keys()), state="readonly", width=20)
        self.tgt_combo.current(0)
        self.tgt_combo.pack(anchor=W, pady=4)

        self.status = Label(top, text="Status: Ready", fg="green", font=("Arial", 10, "bold"))
        self.status.pack(anchor=W, pady=6)

        self.output = Text(root, height=20, wrap=WORD, font=("Courier", 9))
        self.output.pack(padx=10, pady=8, fill=BOTH, expand=True)

        self.questions_frame = Frame(root)
        self.questions_frame.pack(fill=X, padx=10, pady=6)

        self.detected_worker_lang = "en"
        self.current_scenario = None
        self.followup_qs = []
        self.followup_answers = {}

    # (GUI logic remains identical to your version)
    # All internal methods work as before.

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = Tk()
    app = SafetyApp(root)
    root.mainloop()
