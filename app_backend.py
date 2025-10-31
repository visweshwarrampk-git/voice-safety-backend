# app_backend.py
# --------------------------------------------------------
# Render-optimized FastAPI backend
# Includes all APIs: Phase 1, Phase 2, Batch Translation
# --------------------------------------------------------

from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import os
import asyncio

# Import core utilities (optimized tiny Whisper + lazy embeddings)
from core_utils import (
    transcribe_with_whisper,
    detect_scenario_hybrid,
    translate_text,
    SCENARIOS,
)

# --------------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------------
app = FastAPI(title="Safety Incident Follow-up API (Render Optimized)", version="2.0")

# CORS (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for flexibility
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS = {}  # session_id -> dict

# --------------------------------------------------------
# HELPER: Normalize Language Codes
# --------------------------------------------------------
def normalize_language(lang_input: str) -> str:
    """Normalize language to en/ta/hi."""
    if not lang_input:
        return "en"
    lang_map = {
        "en": "en", "english": "en",
        "ta": "ta", "tamil": "ta",
        "hi": "hi", "hindi": "hi"
    }
    return lang_map.get(lang_input.lower(), "en")

# --------------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend is running on Render 🚀"}

# --------------------------------------------------------
# PHASE 1 — INCIDENT AUDIO PROCESSING
# --------------------------------------------------------
@app.post("/pipeline_phase1")
async def pipeline_phase1(
    incident_audio: UploadFile = File(...),
    target_lang: str = Form("en"),
):
    """Accept an incident audio, transcribe, translate, and detect scenario."""
    try:
        if not incident_audio:
            return JSONResponse(status_code=400, content={"error": "incident_audio file is required"})

        session_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{incident_audio.filename}")

        # Save audio file
        file_bytes = await incident_audio.read()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        await asyncio.sleep(0.05)

        print(f"✅ Audio received: {incident_audio.filename}")
        tgt_code = normalize_language(target_lang)
        print(f"🌍 Target language normalized: {target_lang} -> {tgt_code}")

        # Transcribe and detect
        text, detected_lang, whisper_hint = transcribe_with_whisper(file_path)
        print(f"📝 Transcribed ({detected_lang}): {text}")

        translated_text = translate_text(text, detected_lang, tgt_code)
        scenario, debug = detect_scenario_hybrid(text, source_lang=detected_lang)

        # If scenario not confident
        if not scenario:
            print("⚠️ Scenario not confidently detected.")
            return JSONResponse(
                status_code=200,
                content={
                    "session_id": session_id,
                    "phase": "incident_processed",
                    "detected_lang": detected_lang,
                    "translated_text": translated_text,
                    "scenario": None,
                    "message": "Scenario not confidently detected.",
                    "debug": debug,
                },
            )

        # Prepare translated follow-up questions
        worker_lang = detected_lang if detected_lang in ["ta", "hi", "en"] else "en"
        questions_en = SCENARIOS[scenario]["questions"]
        translated_questions = []
        for q in questions_en:
            try:
                translated_questions.append(translate_text(q, "en", worker_lang))
            except Exception as e:
                print("Translation error:", e)
                translated_questions.append(q)

        # Save session
        SESSIONS[session_id] = {
            "incident_file": file_path,
            "incident_text": text,
            "incident_lang": detected_lang,
            "translated_incident": translated_text,
            "target_lang": tgt_code,
            "scenario": scenario,
            "answers": [],
        }

        print(f"✅ Scenario detected: {scenario}")
        return {
            "session_id": session_id,
            "phase": "incident_processed",
            "detected_lang": detected_lang,
            "translated_text": translated_text,
            "scenario": scenario,
            "scenario_description": SCENARIOS.get(scenario, {}).get("description", ""),
            "followup_questions": translated_questions,
        }

    except Exception as e:
        print("❌ Error in Phase 1:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --------------------------------------------------------
# PHASE 2 — FOLLOW-UP ANSWERS
# --------------------------------------------------------
@app.post("/pipeline_phase2")
async def pipeline_phase2(
    session_id: str = Form(...),
    followup_audios: List[UploadFile] = File(...),
    target_lang: str = Form(None),
):
    """Accept follow-up answers, transcribe, translate, and summarize."""
    try:
        if session_id not in SESSIONS:
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        sess = SESSIONS[session_id]
        tgt_code = normalize_language(target_lang) if target_lang else sess.get("target_lang", "en")

        print(f"🎯 Target language for answers: {tgt_code}")

        answers = []
        for idx, upload in enumerate(followup_audios):
            ans_path = os.path.join(UPLOAD_DIR, f"{session_id}_ans{idx}_{upload.filename}")
            with open(ans_path, "wb") as f:
                f.write(await upload.read())
            await asyncio.sleep(0.05)

            # Transcribe and translate
            ans_text, ans_lang, whisper_hint = transcribe_with_whisper(ans_path)
            ans_translated = translate_text(ans_text, ans_lang, tgt_code)

            print(f"🗣️ Answer {idx}: {ans_text} ({ans_lang}) -> {ans_translated} ({tgt_code})")

            answers.append({
                "index": idx,
                "original_text": ans_text,
                "original_lang": ans_lang,
                "translated_text": ans_translated,
                "target_lang": tgt_code,
            })

        sess["answers"] = answers

        summary = {
            "scenario": sess.get("scenario"),
            "incident_text": sess.get("incident_text"),
            "translated_incident": sess.get("translated_incident"),
            "target_lang": tgt_code,
            "answers": answers,
        }

        print(f"✅ Phase 2 complete — {len(answers)} answers processed.")
        return {
            "session_id": session_id,
            "phase": "followup_processed",
            "scenario": sess.get("scenario"),
            "target_lang": tgt_code,
            "answers": answers,
            "summary": summary,
        }

    except Exception as e:
        print("❌ Error in Phase 2:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --------------------------------------------------------
# BATCH TRANSLATION (TEXT ONLY)
# --------------------------------------------------------
@app.post("/translate_text_batch")
async def translate_text_batch(data: dict = Body(...)):
    """Translate a batch of texts between languages."""
    try:
        texts: List[str] = data.get("texts", [])
        source_lang = normalize_language(data.get("source_lang", "en"))
        target_lang = normalize_language(data.get("target_lang", "en"))

        translated = [translate_text(t, source_lang, target_lang) for t in texts]
        print(f"🌐 Batch translated {len(texts)} texts from {source_lang} -> {target_lang}")
        return {"translated": translated}

    except Exception as e:
        print("❌ Batch translation error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

# --------------------------------------------------------
# MAIN (Render Deployment)
# --------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app_backend:app", host="0.0.0.0", port=port)
