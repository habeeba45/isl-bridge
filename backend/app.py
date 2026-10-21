from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import db
import verbx_client
from catalog import load_catalog, match_speech_to_sign
from recognize import recognize_frames

load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="ISL Bridge",
    description="Indian Sign Language demo subset + VerbX STT/TTS/translate. VerbX does not recognize signs.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://isl-bridge-xivercel.app",
        "https://isl-bridge-main-habeeba45.vercel.app",
        "https://isl-bridge-fp2ykuqg6-habeeba45.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsentIn(BaseModel):
    subject_name: str = Field(min_length=1, max_length=120)
    operator_name: str = Field(min_length=1, max_length=120)
    purpose: str = Field(default="Hackathon demo of ISL Bridge (camera and microphone).")
    granted: bool


class RecognizeIn(BaseModel):
    frames: list
    consent_id: int | None = None


class TranslateIn(BaseModel):
    text: str
    source: str = "en"
    target: str = "hi"
    consent_id: int | None = None
    approved: bool = False


class SpeakSignIn(BaseModel):
    sign_id: str
    consent_id: int | None = None
    approved: bool = False


class ReverseMatchIn(BaseModel):
    transcript: str
    corrected_transcript: str | None = None
    consent_id: int | None = None


class RunUpdateIn(BaseModel):
    case_id: str
    actual: str
    status: str
    evidence: dict | None = None
    input_extra: dict | None = None



def _require_consent(consent_id: int | None, *, camera: bool) -> dict:
    if consent_id is None:
        raise HTTPException(
            status_code=403,
            detail="Consent is required before camera or microphone use. Record it on the Safety tab.",
        )

    # The requested record must be the most recent consent record.
    latest = db.latest_consent()

    if (
        not latest
        or not latest.get("granted")
        or latest.get("id") != consent_id
    ):
        raise HTTPException(
            status_code=403,
            detail="Consent is missing, refused, or revoked. Please provide consent again.",
        )

    return latest



@app.get("/api/health")
def health():
    catalog = load_catalog()
    return {
        "ok": True,
        "sign_language": catalog["sign_language"],
        "supported_count": len(catalog["signs"]),
        "verbx_configured": verbx_client.configured(),
        "verbx_fallback_allowed": verbx_client.allow_fallback(),
        "verbx_does_not_recognize_signs": True,
        "recognition": "MediaPipe Hands landmarks + local geometric classifier",
        "data_origin": catalog["data_origin"],
    }


@app.get("/api/signs")
def signs():
    catalog = load_catalog()
    return {
        "sign_language": catalog["sign_language"],
        "sign_language_code": catalog["sign_language_code"],
        "scope": catalog["scope"],
        "data_origin": catalog["data_origin"],
        "languages": catalog["languages"],
        "confidence_threshold": catalog["confidence_threshold"],
        "unrecognized_state": "UNRECOGNIZED",
        "signs": catalog["signs"],
        "verbx_role": "STT, TTS, and translation only. No native sign-gesture vision.",
    }


@app.post("/api/consent")
def create_consent(body: ConsentIn):
    record = db.add_consent(body.subject_name, body.operator_name, body.purpose, body.granted)
    db.add_event("consent", record)
    return record


@app.get("/api/consent/latest")
def latest_consent():
    return {"consent": db.latest_consent()}


@app.post("/api/recognize")
def recognize(body: RecognizeIn, x_consent_id: int | None = Header(default=None)):
    consent_id = body.consent_id or x_consent_id
    consent = _require_consent(consent_id, camera=True)
    result = recognize_frames(body.frames)
    result["consent_id"] = consent["id"]
    db.add_event("recognize", {"consent_id": consent["id"], "result": {k: result[k] for k in result if k != "sign"}})
    return result


@app.post("/api/translate")
async def translate(body: TranslateIn):
    if not body.approved:
        raise HTTPException(status_code=412, detail="Human approval is required before translation/speech.")
    result = await verbx_client.translate(body.text, body.source, body.target)
    db.add_event("translate", result)
    return result


@app.post("/api/speak-sign")
async def speak_sign(body: SpeakSignIn, x_consent_id: int | None = Header(default=None)):
    if not body.approved:
        raise HTTPException(
            status_code=412,
            detail="Approve before sending text to translation/TTS. This has real audio output.",
        )
    consent_id = body.consent_id or x_consent_id
    _require_consent(consent_id, camera=True)
    catalog = load_catalog()
    sign = next((s for s in catalog["signs"] if s["id"] == body.sign_id), None)
    if sign is None:
        raise HTTPException(status_code=400, detail="Unknown sign_id. Refusing to speak an undocumented gloss.")

    hi = await verbx_client.translate(sign["en"], "en", "hi")
    hindi_text = hi.get("text") or sign["hi"]
    en_audio = await verbx_client.tts(sign["en"], "en")
    hi_audio = await verbx_client.tts(hindi_text, "hi")
    payload = {
        "sign_language": catalog["sign_language"],
        "sign": sign,
        "english": {"text": sign["en"], "tts": en_audio},
        "hindi": {"text": hindi_text, "tts": hi_audio, "translate": hi},
        "verbx_does_not_recognize_signs": True,
    }
    db.add_event("speak-sign", {"sign_id": sign["id"], "providers": [en_audio.get("provider"), hi_audio.get("provider"), hi.get("provider")]})
    return payload


@app.post("/api/stt")
async def stt(
    audio: UploadFile = File(...),
    language: str = Form(default="en"),
    consent_id: int = Form(...),
    approved: bool = Form(default=False),
):
    if not approved:
        raise HTTPException(status_code=412, detail="Approve before uploading speech to STT.")
    _require_consent(consent_id, camera=False)
    raw = await audio.read()
    result = await verbx_client.stt(raw, audio.filename or "speech.webm", audio.content_type or "audio/webm", language)
    match = match_speech_to_sign(result.get("text") or "")
    envelope = {"stt": result, "reverse_route": match, "sign_language": "Indian Sign Language (ISL)"}
    db.add_event("stt", {"provider": result.get("provider"), "text": result.get("text"), "match": match.get("state")})
    return envelope


@app.post("/api/reverse")
def reverse_match(body: ReverseMatchIn):
    text = (body.corrected_transcript or body.transcript or "").strip()
    match = match_speech_to_sign(text)
    db.add_event("reverse", {"text": text, "state": match["state"]})
    return {"transcript": text, "sign_language": "Indian Sign Language (ISL)", "reverse_route": match}


@app.get("/api/events")
def events(limit: int = 40):
    return {"events": db.list_events(limit)}


@app.get("/api/runs")
def runs():
    return {"sign_language": "Indian Sign Language (ISL)", "runs": db.list_runs()}


@app.post("/api/runs")
def update_run(body: RunUpdateIn):
    if body.status not in {"pending", "pass", "fail", "blocked", "difference"}:
        raise HTTPException(status_code=400, detail="status must be pending|pass|fail|blocked|difference")
    try:
        updated = db.update_run(body.case_id, body.actual, body.status, body.evidence, body.input_extra)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown case_id")
    db.add_event("test-run", {"case_id": body.case_id, "status": body.status})
    return updated


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
