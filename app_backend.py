# app_backend.py
from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid
import os
import asyncio

# Import shared core utilities
from core_utils import (
    transcribe_with_whisper,
    detect_scenario_hybrid,
    translate_text,
    SCENARIOS,
)

app = FastAPI(title="Safety Incident Follow-up API (Pipeline v2)", version="1.0")

# -----------------------------
# CORS CONFIGURATION
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
SESSIONS = {}  # session_id -> dict


# -----------------------------
# ✅ Language Normalization Helper
# -----------------------------
def normalize_language(lang_input: str) -> str:
    """Convert any language input to standard code (en/ta/hi)."""
    if not lang_input:
        return "en"

    lang_map = {
        "en": "en",
        "english": "en",
        "ta": "ta",
        "tamil": "ta",
        "hi": "hi",
        "hindi": "hi",
    }

    normalized = lang_map.get(lang_input.lower())
    if normalized:
        return normalized

    # Default fallback
    return "en"


# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok", "message": "Backend running"}


# -----------------------------
# Phase 1: Incident Processing
# -----------------------------
@app.post("/pipeline_phase1")
async def pipeline_phase1(
    incident_audio: UploadFile = File(...),
    target_lang: str = Form("en"),
):
    try:
        if not incident_audio:
            return JSONResponse(
                status_code=400, content={"error": "incident_audio file is required"}
            )

        session_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{incident_audio.filename}")

        file_bytes = await incident_audio.read()
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        await asyncio.sleep(0.1)

        print(f"✅ Received audio file: {incident_audio.filename}")
        print(f"🎯 Target language (raw): {target_lang}")

        # ✅ Normalize the target language
        tgt_code = normalize_language(target_lang)
        print(f"🌍 Normalized target language: {tgt_code}")

        # Transcribe
        text, detected_lang, whisper_hint = transcribe_with_whisper(file_path)
        print(f"📝 Transcribed ({detected_lang}): {text}")

        translated_text = translate_text(text, detected_lang, tgt_code)

        # Detect scenario
        scenario, debug = detect_scenario_hybrid(text, source_lang=detected_lang)
        if not scenario:
            print("⚠️ Could not confidently detect scenario.")
            return JSONResponse(
                status_code=200,
                content={
                    "session_id": session_id,
                    "phase": "incident_processed",
                    "detected_lang": detected_lang,
                    "translated_text": translated_text,
                    "scenario": None,
                    "message": "Could not confidently detect scenario automatically.",
                    "debug": debug,
                },
            )

        # Translate follow-up questions
        worker_lang = detected_lang if detected_lang in ["ta", "hi", "en"] else "en"
        questions_en = SCENARIOS[scenario]["questions"]
        translated_questions = []
        for q in questions_en:
            try:
                translated_questions.append(translate_text(q, "en", worker_lang))
            except Exception as e:
                print("Translation error:", e)
                translated_questions.append(q)

        # Save session details
        SESSIONS[session_id] = {
            "incident_file": file_path,
            "incident_text": text,
            "incident_lang": detected_lang,
            "translated_incident": translated_text,
            "target_lang": tgt_code,  # ✅ Store normalized code
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
        print("❌ Error in pipeline_phase1:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -----------------------------
# Phase 2: Follow-up Answers
# -----------------------------
@app.post("/pipeline_phase2")
async def pipeline_phase2(
    session_id: str = Form(...),
    followup_audios: List[UploadFile] = File(...),
    target_lang: str = Form(None),
):
    try:
        if session_id not in SESSIONS:
            return JSONResponse(status_code=404, content={"error": "Session not found"})

        sess = SESSIONS[session_id]

        # ✅ Use target_lang from request if available
        if target_lang:
            tgt_code = normalize_language(target_lang)
            print(f"🎯 Using target_lang from request: {target_lang} -> {tgt_code}")
            sess["target_lang"] = tgt_code
        else:
            print(f"⚠️ No target_lang provided in request. Using session default.")
            tgt_code = normalize_language(sess.get("target_lang", "en"))

        print(f"🌍 Translating answers to: {tgt_code}")

        answers = []
        for idx, upload in enumerate(followup_audios):
            ans_path = os.path.join(UPLOAD_DIR, f"{session_id}_ans{idx}_{upload.filename}")
            file_bytes = await upload.read()
            with open(ans_path, "wb") as f:
                f.write(file_bytes)
            await asyncio.sleep(0.1)

            # Ensure file is completely written
            if not os.path.exists(ans_path) or os.path.getsize(ans_path) == 0:
                print(f"⚠️ Skipping empty upload: {upload.filename}")
                continue

            # Transcribe and translate
            ans_text, ans_lang, whisper_hint = transcribe_with_whisper(ans_path)
            print(f"📝 Answer {idx} transcribed ({ans_lang}): {ans_text}")

            # ✅ Translate to the correct target language
            ans_translated = translate_text(ans_text, ans_lang, tgt_code)
            print(f"🌐 Answer {idx} translated to {tgt_code}: {ans_translated}")

            answers.append(
                {
                    "index": idx,
                    "original_text": ans_text,
                    "original_lang": ans_lang,
                    "translated_text": ans_translated,
                    "target_lang": tgt_code,
                }
            )

        sess["answers"] = answers

        summary = {
            "scenario": sess.get("scenario"),
            "incident_text": sess.get("incident_text"),
            "translated_incident": sess.get("translated_incident"),
            "target_lang": tgt_code,
            "answers": answers,
        }

        print(f"✅ Processed {len(answers)} follow-up answers in {tgt_code}")
        return {
            "session_id": session_id,
            "phase": "followup_processed",
            "scenario": sess.get("scenario"),
            "target_lang": tgt_code,
            "answers": answers,
            "summary": summary,
        }

    except Exception as e:
        print("❌ Error in pipeline_phase2:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


# -------------------------------------------------------------
# Batch Translation
# -------------------------------------------------------------
@app.post("/translate_text_batch")
async def translate_text_batch(data: dict = Body(...)):
    texts: List[str] = data.get("texts", [])
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "en")

    # ✅ Normalize language codes
    source_lang = normalize_language(source_lang)
    target_lang = normalize_language(target_lang)

    try:
        results = [translate_text(t, source_lang, target_lang) for t in texts]
        return {"translated": results}
    except Exception as e:
        print("❌ Batch translation error:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})
