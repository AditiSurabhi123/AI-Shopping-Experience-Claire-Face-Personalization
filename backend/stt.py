#!/usr/bin/env python3
"""
Lenskart Claire AI — Speech-to-Text via Gemini audio understanding

The browser's SpeechRecognition API is fast but often mistranscribes
multi-word answers, code-mixed Hinglish, and domain-specific vocabulary
("adventurous" → "professional" etc.). This module forwards the raw audio
to Gemini 2.5 Flash, which handles audio + multilingual transcription far
more accurately.

Public API:
    transcribe(audio_b64, mime, lang='en') -> dict
      → { "success": True, "text": "<transcript>", "source": "gemini" }
      or { "success": False, "error": "..." }
"""
import base64
import http.client
import json
import os
import re

from ssl_ctx import SSL_CTX

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
)
GEMINI_HOST  = "generativelanguage.googleapis.com"
# Audio understanding works on the same multimodal text model.
STT_MODEL    = os.environ.get("GEMINI_STT_MODEL", "gemini-2.5-flash")
_STT_FALLBACKS = ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash")

# Guide the model specifically to an eyewear shopping context so it resolves
# likely-misheard words ("adventurous", "progressive", "wayfarer") correctly.
_PROMPT_EN = (
    "Transcribe the audio verbatim in English. The speaker is shopping for "
    "eyewear at Lenskart and is answering a quiz question about their frame "
    "preferences. Typical vocabulary includes: adventurous, classic, timeless, "
    "bold, trendy, minimal, wayfarer, aviator, round, cat-eye, clubmaster, "
    "rimless, square, butterfly, single vision, progressive, zero power, "
    "prescription, sunglasses, eyeglasses, contact lens, bestseller, under 1000, "
    "around 2000, premium. Prefer these terms when audio is ambiguous. "
    "Return ONLY the raw transcript — no quotes, no prose."
)

_PROMPT_HI = (
    "Transcribe the audio verbatim. The speaker is shopping for eyewear at "
    "Lenskart and may speak in Hindi, Hinglish, or English. Typical terms: "
    "चश्मा, सनग्लास, कॉन्टैक्ट लेंस, क्लासिक, बोल्ड, एडवेंचरस, मिनिमल, "
    "wayfarer, aviator, round, square, cat-eye, single vision, progressive, "
    "zero power, prescription, bestseller, hazaar se kam, do hazaar, "
    "2000 ke aaspaas, premium, pehla, dusra, teesra. Keep the speaker's "
    "original language (English words in English, Hindi in Devanagari or Roman "
    "as spoken). Return ONLY the raw transcript."
)


def _post(model: str, payload: bytes) -> tuple:
    path = f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=30, context=SSL_CTX)
    try:
        conn.request("POST", path, payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw  = resp.read().decode("utf-8")
        return resp.status, raw
    finally:
        conn.close()


def transcribe(audio_b64: str, mime: str = "audio/webm", lang: str = "en") -> dict:
    """Send audio to Gemini for transcription. Falls back across models on 404."""
    if not audio_b64:
        return {"success": False, "error": "empty audio"}
    if "," in audio_b64:  # tolerate accidental data-URI prefix
        audio_b64 = audio_b64.split(",", 1)[1]

    prompt = _PROMPT_HI if (lang or "").lower().startswith("hi") else _PROMPT_EN

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime or "audio/webm", "data": audio_b64}},
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 300,
            "temperature": 0.0,
        },
    }).encode("utf-8")

    models_to_try = [STT_MODEL] + [m for m in _STT_FALLBACKS if m != STT_MODEL]
    last_err = ""
    for model in models_to_try:
        try:
            status, raw = _post(model, payload)
        except Exception as exc:
            last_err = f"{model}: network {exc}"
            continue
        if status == 200:
            try:
                data  = json.loads(raw)
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text  = "".join(p.get("text", "") for p in parts).strip()
                # Strip accidental quote wrapping or markdown fences
                text = re.sub(r"^```[a-z]*\s*|```$", "", text, flags=re.IGNORECASE).strip()
                text = text.strip('"\'`').strip()
                if not text:
                    last_err = f"{model}: empty transcript"
                    continue
                return {"success": True, "text": text, "source": "gemini",
                         "model": model}
            except Exception as exc:
                last_err = f"{model}: parse {exc}"
                continue
        if status == 404 or "NOT_FOUND" in raw:
            last_err = f"{model}: 404"
            continue
        last_err = f"{model}: HTTP {status}"
        break

    return {"success": False, "error": last_err or "unknown"}
