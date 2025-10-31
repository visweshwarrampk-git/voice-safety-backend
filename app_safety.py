import os
import threading
import time
import uuid
import json
from tkinter import *
from tkinter import filedialog, messagebox, ttk

import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound import playsound

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
    print("Loading Whisper model...")
    whisper_model = whisper.load_model(MODEL_SIZE)
    print("Loading embedding model...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    print("Precomputing scenario embeddings...")
    for key, data in SCENARIOS.items():
        examples = data.get("examples", [])
        scenario_embeddings[key] = embedding_model.encode(examples, convert_to_tensor=True)
    print("✅ Models loaded.")

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
        translator = GoogleTranslator(source=src_lang if src_lang else "auto", target=tgt_lang)
        return translator.translate(src_text)
    except Exception as e:
        print("Translate error:", e)
        return src_text

def tts_and_play(text, lang_code):
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
    global embedding_model, scenario_embeddings
    while embedding_model is None or not scenario_embeddings:
        time.sleep(0.2)

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

# ---------------- GUI CLASS ----------------
class SafetyApp:
    def __init__(self, root):
        self.root = root
        root.title("Safety Incident Follow-up (3 langs: EN, TA, HI)")
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

    def log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.output.insert(END, f"[{ts}] {msg}\n")
        self.output.see(END)

    def set_status(self, msg, color="black"):
        self.status.config(text=f"Status: {msg}", fg=color)

    def record_incident_click(self):
        self.set_status("Recording incident...")
        threading.Thread(target=self._record_incident_thread, daemon=True).start()

    def _record_incident_thread(self):
        try:
            path = record_to_file(seconds=RECORD_SECONDS)
            self.log(f"Incident recorded: {path}")
            self._process_incident_file(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error", "red")

    def upload_incident_click(self):
        file_path = filedialog.askopenfilename(title="Select incident audio", filetypes=[("Audio files","*.wav *.mp3 *.m4a *.flac")])
        if not file_path:
            return
        threading.Thread(target=self._process_incident_file, args=(file_path,), daemon=True).start()

    def _process_incident_file(self, path):
        try:
            self.set_status("Transcribing incident...", "blue")
            text, final_lang, whisper_hint = transcribe_with_whisper(path)
            self.detected_worker_lang = final_lang

            self.log(f"Incident uploaded: {path}")
            self.log(f"📝 Transcribed (whisper_hint={whisper_hint}, final={final_lang}): {text}")

            tgt_name = self.tgt_combo.get()
            tgt_code = SUPPORTED_LANGS.get(tgt_name, "en")
            translated_text = translate_text(text, final_lang, tgt_code)

            self.log(f"🌐 Translated incident text -> ({tgt_code}): {translated_text}")
            self.output.insert(END, f"\nOriginal ({final_lang}): {text}\nTranslated ({tgt_code}): {translated_text}\n\n")

            # 🔊 Manual Play button for translated incident
            btn = Button(self.root, text="🔊 Play (translated incident)",
                         command=lambda: threading.Thread(target=lambda: tts_and_play(translated_text, tgt_code), daemon=True).start())
            self.output.window_create(END, window=btn)
            self.output.insert(END, "\n\n")
            self.output.see(END)

            self.set_status("Detecting scenario...", "blue")
            scenario, debug = detect_scenario_hybrid(text, source_lang=final_lang)
            if scenario:
                self.current_scenario = scenario
                self.log(f"✅ Detected scenario: {scenario} ({SCENARIOS[scenario]['description']})")
                self.log(f"🔎 Debug: {json.dumps(debug, indent=2, ensure_ascii=False)}")
                self.set_status(f"Detected: {scenario}", "green")
                self._prepare_followup_questions()
            else:
                self.log("⚠️ Could not confidently detect scenario automatically.")
                self.log(f"🔎 Debug: {json.dumps(debug, indent=2, ensure_ascii=False)}")
                self._ask_manual_scenario_choice()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error", "red")

    def _ask_manual_scenario_choice(self):
        options = list(SCENARIOS.keys())
        topw = Toplevel(self.root)
        topw.title("Manual scenario select")
        Label(topw, text="AI couldn't detect confidently. Please choose:", font=("Arial", 10)).pack(padx=10, pady=10)
        combo = ttk.Combobox(topw, values=options, state="readonly", width=40)
        combo.pack(padx=10, pady=8)
        def set_choice():
            sel = combo.get()
            if sel:
                self.current_scenario = sel
                topw.destroy()
                self.log(f"👤 Manual scenario selected: {sel}")
                self._prepare_followup_questions()
        ttk.Button(topw, text="Select", command=set_choice).pack(pady=8)

    def _prepare_followup_questions(self):
        for child in self.questions_frame.winfo_children():
            child.destroy()
        if not self.current_scenario:
            Label(self.questions_frame, text="No scenario selected.", font=("Arial", 11, "bold")).pack(anchor=W)
            return
        worker_lang = self.detected_worker_lang if self.detected_worker_lang in ["ta","hi","en"] else "en"
        questions_en = SCENARIOS[self.current_scenario]["questions"]
        translated_qs = []
        for q in questions_en:
            try:
                translated_qs.append(GoogleTranslator(source="en", target=worker_lang).translate(q))
            except:
                translated_qs.append(q)
        self.followup_qs = translated_qs
        self.followup_answers = {}
        Label(self.questions_frame, text=f"Follow-up Questions for: '{self.current_scenario}'", font=("Arial", 11, "bold")).pack(anchor=W)
        for i, q in enumerate(self.followup_qs):
            frame = Frame(self.questions_frame, pady=4)
            frame.pack(fill=X, anchor=W)
            Label(frame, text=f"Q{i+1}: {q}", wraplength=760, justify=LEFT).pack(anchor=W)
            Button(frame, text="🔊 Play", command=lambda qq=q, ll=worker_lang: tts_and_play(qq, ll)).pack(side=LEFT, padx=6)
            Button(frame, text="🎤 Record", command=lambda idx=i: self.record_answer_click(idx)).pack(side=LEFT, padx=6)
            Button(frame, text="📁 Upload", command=lambda idx=i: self.upload_answer_click(idx)).pack(side=LEFT, padx=6)
            status_lbl = Label(frame, text="No answer yet", fg="gray")
            status_lbl.pack(side=LEFT, padx=8)
            text_lbl = Label(frame, text="", wraplength=760, justify=LEFT, fg="blue")
            text_lbl.pack(anchor=W)
            self.followup_answers[i] = {"status_label": status_lbl, "text_label": text_lbl, "file": None}
        Button(self.questions_frame, text="✅ Finalize & Show Summary", command=self.show_summary).pack(pady=8)
        self.set_status("Waiting for follow-up answers...", "blue")

    def record_answer_click(self, index):
        threading.Thread(target=self._record_answer_thread, args=(index,), daemon=True).start()

    def _record_answer_thread(self, idx):
        try:
            self.set_status(f"Recording answer {idx+1}...")
            path = record_to_file(seconds=RECORD_SECONDS)
            self.followup_answers[idx]["file"] = path
            self.process_answer_file(idx, path)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def upload_answer_click(self, index):
        file_path = filedialog.askopenfilename(title="Select answer audio", filetypes=[("Audio files","*.wav *.mp3 *.m4a *.flac")])
        if not file_path:
            return
        self.followup_answers[index]["file"] = file_path
        threading.Thread(target=self.process_answer_file, args=(index, file_path), daemon=True).start()

    def process_answer_file(self, index, file_path):
        try:
            entry = self.followup_answers[index]
            entry["status_label"].config(text="Processing...", fg="orange")
            self.set_status("Transcribing answer...", "blue")
            text, detected_lang, whisper_hint = transcribe_with_whisper(file_path)
            self.log(f"Answer {index+1} transcribed (whisper_hint={whisper_hint}, final={detected_lang}): {text}")
            tgt_name = self.tgt_combo.get()
            tgt_code = SUPPORTED_LANGS.get(tgt_name, "en")
            translated = translate_text(text, detected_lang, tgt_code)
            self.log(f"Answer {index+1} translated -> ({tgt_code}): {translated}")

            entry["text_label"].config(text=f"🗣️ Original ({detected_lang}): {text}\n🌐 Translated ({tgt_code}): {translated}")
            play_btn = Button(entry["text_label"].master, text="🔊 Play (translated)",
                              command=lambda txt=translated, lang=tgt_code: threading.Thread(target=lambda: tts_and_play(txt, lang), daemon=True).start())
            play_btn.pack(anchor=W, padx=20, pady=(2, 6))
            entry["status_label"].config(text="✅ Done", fg="green")
            self.set_status("Answer processed", "green")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            entry = self.followup_answers.get(index)
            if entry:
                entry["status_label"].config(text="❌ Error", fg="red")
            self.set_status("Error", "red")

    def show_summary(self):
        summary = ["INCIDENT SUMMARY", "="*40]
        summary.append(f"Detected Scenario: {self.current_scenario or 'N/A'}")
        summary.append(f"Worker Language: {self.detected_worker_lang}")
        summary.append(f"Report Language: {self.tgt_combo.get()}")
        summary.append("="*40); summary.append("")
        for i, q in enumerate(self.followup_qs):
            entry = self.followup_answers.get(i, {})
            summary.append(f"Q{i+1}: {q}")
            txt = entry.get("text_label").cget("text") if entry.get("text_label") else ""
            summary.append(txt); summary.append("")
        txt = "\n".join(summary)
        self.log("\n" + txt)
        messagebox.showinfo("Summary", txt)

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    root = Tk()
    app = SafetyApp(root)
    root.mainloop()
