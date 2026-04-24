#!/usr/bin/env python3
"""
Lenskart Claire AI — Natural Text-to-Speech

Primary : Gemini 2.5 Flash TTS (`gemini-2.5-flash-preview-tts`) — LLM-grade
          natural voice, Hindi + English, 30 prebuilt voices. Output is 24 kHz
          mono PCM → we wrap it in a WAV header for browser playback.

Fallback: Google Translate's free TTS endpoint (basic neural, reliable, no key).
          Chunks long text on sentence boundaries — the frontend plays chunks
          sequentially so there's no first-word cutoff.

Endpoint: GET /api/tts?text=...&lang=en|hi
Returns : JSON { "chunks": [ "<base64 audio>", ... ], "mime": "audio/wav"|"audio/mpeg" }
"""
import base64
import hashlib
import http.client
import json
import os
import re
import ssl
import struct
from pathlib import Path
from urllib.parse import quote

# Disk cache so precaching survives restarts
_DISK_CACHE_DIR = Path(__file__).parent / "data" / "tts_cache"
_DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

from ssl_ctx import SSL_CTX as _SSL_CTX

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
)
GEMINI_HOST      = "generativelanguage.googleapis.com"
# Do NOT change to gemini-3 / gemini-3.1 — those IDs 404 on the public API.
# `gemini-2.5-flash-preview-tts` is the working LLM-grade TTS model.
GEMINI_TTS_MODEL = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
GEMINI_VOICE_EN  = os.environ.get("GEMINI_VOICE_EN", "Aoede")
GEMINI_VOICE_HI  = os.environ.get("GEMINI_VOICE_HI", "Leda")

FALLBACK_HOST  = "translate.google.com"
USER_AGENT     = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

CHUNK_MAX_CHARS = 180
CACHE_MAX       = 256   # large enough to hold precached phrases + session churn

_CACHE: dict = {}
# Track Gemini reliability: allow N consecutive failures before falling back for
# the next request (instead of disabling forever).
_GEMINI_FAILS = 0
_GEMINI_FAIL_THRESHOLD = 3


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _cache_key(text: str, lang: str) -> str:
    return hashlib.md5(f"{lang}::{text}".encode("utf-8")).hexdigest()


# Strip EVERYTHING that isn't a letter / digit / whitespace / essential punctuation.
# This robustly removes emojis, bullet dots, arrows, stars, etc. that TTS engines
# either skip or choke on.
_ALLOWED_RE = re.compile(r"[^\w\s₹.,?!:;'\"\-()–—।]", re.UNICODE)


def _clean_for_speech(text: str) -> str:
    # Strip markdown tokens first so the inner text survives
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*",     r"\1", text)
    text = re.sub(r"`([^`]+)`",     r"\1", text)
    text = re.sub(r"[#>_~]",         " ", text)
    # Drop anything non-essential
    text = _ALLOWED_RE.sub(" ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str, limit: int = CHUNK_MAX_CHARS) -> list:
    """Split on sentence / clause boundaries so each chunk is self-contained."""
    if len(text) <= limit:
        return [text]

    # Split on ., !, ?, । (Devanagari danda), and commas as secondary breaks
    parts = re.split(r"(?<=[.!?।])\s+", text)
    out, cur = [], ""
    for p in parts:
        if not p:
            continue
        if len(cur) + len(p) + 1 > limit:
            if cur:
                out.append(cur.strip())
            # Hard-split any overlong sentence by commas, else by chars
            while len(p) > limit:
                soft = re.split(r"(?<=,)\s+", p, maxsplit=1)
                if len(soft) == 2 and len(soft[0]) <= limit:
                    out.append(soft[0].strip())
                    p = soft[1]
                else:
                    out.append(p[:limit].strip())
                    p = p[limit:]
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        out.append(cur.strip())
    return out


def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate    = sample_rate * channels * bits // 8
    block_align  = channels * bits // 8
    data_size    = len(pcm)
    riff_size    = 36 + data_size
    header  = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits)
    header += b"data" + struct.pack("<I", data_size)
    return header + pcm


# ─── Gemini 2.5 Flash TTS (primary) ──────────────────────────────────────────

_STYLE_PROMPT_EN = ("Speak in a warm, upbeat, friendly salesperson voice, like a "
                    "helpful Lenskart store stylist welcoming a customer. Keep it "
                    "natural and conversational: ")
_STYLE_PROMPT_HI = ("एक गर्मजोशी भरी, मददगार और दोस्ताना सेल्सपर्सन की आवाज़ में, "
                    "Lenskart की एक स्टाइलिस्ट की तरह बोलें। स्वाभाविक और बातचीत के "
                    "लहज़े में: ")


# Known-working Gemini TTS model IDs. Kept here as hardcoded fallbacks so the
# feature stays alive even if the primary constant gets changed to a bad ID.
_TTS_KNOWN_WORKING = (
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)


def _gemini_tts_once(text: str, lang: str, plain: bool = False) -> bytes:
    """Single Gemini TTS call. On any model-level error (404, or 400 'only
    supports text output' for non-TTS models), retries with known-working TTS
    models so playback keeps working."""
    voice  = GEMINI_VOICE_HI if lang == "hi" else GEMINI_VOICE_EN
    style  = "" if plain else (_STYLE_PROMPT_HI if lang == "hi" else _STYLE_PROMPT_EN)
    prompt = (style + text) if style else text

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            },
        },
    })

    # Try configured first, then the always-working hardcoded list.
    models_to_try = [GEMINI_TTS_MODEL] + [m for m in _TTS_KNOWN_WORKING if m != GEMINI_TTS_MODEL]
    raw = b""; status = 0
    for m in models_to_try:
        path = f"/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=30, context=_SSL_CTX)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw  = resp.read()
            status = resp.status
        finally:
            conn.close()
        if status == 200:
            break
        text_err = raw.decode("utf-8", "replace")
        # Any model-level rejection (404 missing, or 400 "only supports text
        # output" when the caller's model isn't a TTS model) → try the next.
        looks_like_model_issue = (
            status == 404 or "NOT_FOUND" in text_err
            or (status == 400 and "only supports text" in text_err.lower())
            or (status == 400 and "does not support" in text_err.lower())
        )
        if looks_like_model_issue:
            print(f"  [tts] model {m} unusable ({status}), trying next")
            continue
        try:
            err = json.loads(text_err).get("error", {}).get("message", "")
        except Exception:
            err = text_err[:200]
        raise RuntimeError(f"Gemini TTS {status}: {err}")

    if status != 200:
        raise RuntimeError(f"Gemini TTS: no working model found (last status {status})")

    data = json.loads(raw.decode("utf-8"))
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for p in parts:
        inline = p.get("inlineData") or p.get("inline_data")
        if not inline:
            continue
        b64  = inline.get("data")
        mime = (inline.get("mimeType") or inline.get("mime_type") or "").lower()
        if not b64:
            continue
        pcm = base64.b64decode(b64)
        m    = re.search(r"rate=(\d+)", mime)
        rate = int(m.group(1)) if m else 24000
        return _wrap_pcm_as_wav(pcm, sample_rate=rate)

    raise RuntimeError(f"Gemini TTS: no audio in response — {raw[:200]}")


# ─── Google Translate TTS (fallback, chunked) ────────────────────────────────

def _translate_tts_chunk(chunk: str, lang: str) -> bytes:
    tl = "hi" if lang == "hi" else "en-in"
    path = f"/translate_tts?ie=UTF-8&q={quote(chunk, safe='')}&tl={tl}&client=tw-ob&ttsspeed=1"
    conn = http.client.HTTPSConnection(FALLBACK_HOST, timeout=10, context=_SSL_CTX)
    try:
        conn.request("GET", path, headers={
            "User-Agent": USER_AGENT,
            "Accept":     "audio/mpeg, */*",
            "Referer":    "https://translate.google.com/",
        })
        resp = conn.getresponse()
        data = resp.read()
    finally:
        conn.close()
    if resp.status != 200 or not data:
        raise RuntimeError(f"Translate TTS {resp.status}")
    return data


# ─── Public API ──────────────────────────────────────────────────────────────

def synthesise(text: str, lang: str = "en") -> dict:
    """
    Returns { "chunks": [base64, ...], "mime": "audio/wav"|"audio/mpeg",
              "source": "gemini"|"translate" }.
    """
    global _GEMINI_FAILS

    if not text or not text.strip():
        return {"chunks": [], "mime": "audio/wav", "source": "none"}

    cleaned = _clean_for_speech(text)[:1800]
    if not cleaned:
        return {"chunks": [], "mime": "audio/wav", "source": "none"}

    lang = (lang or "en").lower()
    if lang not in ("en", "hi"):
        lang = "en"

    key = _cache_key(cleaned, lang)
    if key in _CACHE:
        return _CACHE[key]

    # Disk cache lookup (survives restarts)
    disk_path = _DISK_CACHE_DIR / f"{key}.json"
    if disk_path.exists():
        try:
            with open(disk_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("chunks"):
                _CACHE[key] = cached
                return cached
        except Exception:
            pass

    result = None

    # 1) Gemini natural voice (single call, handles long text).
    # Try every request unless we've just had several consecutive failures —
    # then skip Gemini for this one request to keep latency low, but give it
    # another shot next time.
    if _GEMINI_FAILS < _GEMINI_FAIL_THRESHOLD:
        wav = None
        for plain in (False, True):   # retry with plain text on first failure
            try:
                wav = _gemini_tts_once(cleaned, lang, plain=plain)
                break
            except Exception as exc:
                msg = str(exc)
                if plain or "no audio" not in msg.lower():
                    # Not a content-filter refusal — stop retrying
                    print(f"  [tts] gemini failed ({msg[:120]})")
                    break
                print(f"  [tts] gemini refused styled prompt — retrying plain")
        if wav:
            result = {
                "chunks": [base64.b64encode(wav).decode("ascii")],
                "mime":   "audio/wav",
                "source": "gemini",
            }
            _GEMINI_FAILS = 0
            print(f"  [tts] gemini OK — {len(wav)} bytes for {len(cleaned)} chars ({lang})")
        else:
            _GEMINI_FAILS += 1
            print(f"  [tts] gemini fallback engaged [{_GEMINI_FAILS}/{_GEMINI_FAIL_THRESHOLD}]")
    else:
        # After threshold, skip ONE request then retry next time
        _GEMINI_FAILS = 0
        print(f"  [tts] skipping gemini for this request, will retry next time")

    # 2) Google Translate fallback (chunked, frontend plays sequentially)
    if result is None:
        chunks = _chunk_text(cleaned, CHUNK_MAX_CHARS)
        b64_chunks = []
        for i, ch in enumerate(chunks):
            try:
                mp3 = _translate_tts_chunk(ch, lang)
                b64_chunks.append(base64.b64encode(mp3).decode("ascii"))
            except Exception as exc:
                print(f"  [tts] translate chunk {i+1}/{len(chunks)} failed: {exc}")
        if not b64_chunks:
            return {"chunks": [], "mime": "audio/wav", "source": "none"}
        result = {"chunks": b64_chunks, "mime": "audio/mpeg", "source": "translate"}
        print(f"  [tts] translate OK — {len(b64_chunks)} chunks for {len(cleaned)} chars")

    # Cache (non-empty only)
    if result and result.get("chunks"):
        if len(_CACHE) >= CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)), None)
        _CACHE[key] = result
        # Persist to disk for the next restart
        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump(result, f)
        except Exception as exc:
            print(f"  [tts] disk cache write failed: {exc}")

    return result


# ─── Precache: warm the TTS cache for common phrases ─────────────────────────

def _build_common_phrases() -> list:
    """
    All static lines Claire says during the happy path: greeting, quiz question
    introductions, and the 5 quiz questions — in both English and Hindi.
    Listed here (not imported from agent.py) so we don't create an import cycle
    and the set stays in sync with what run_demo_agent returns.
    """
    en_greet = ("Hey, welcome to Lenskart! I'm Claire — think of me as your personal "
                "frames stylist. I'll ask you a few quick questions and find you frames "
                "that actually suit your face and lifestyle. Sound good? Let's start — "
                "tell me, what do you do most of the day?")
    hi_greet = ("नमस्ते! Lenskart में आपका स्वागत है। मैं Claire हूँ — आपकी अपनी फ्रेम "
                "स्टाइलिस्ट। बस कुछ छोटे-छोटे सवाल पूछूँगी और आपके चेहरे और लाइफस्टाइल "
                "के हिसाब से परफेक्ट फ्रेम ढूँढ दूँगी। चलिए शुरू करते हैं — बताइए, आप "
                "दिन भर क्या करते हैं?")

    en_quiz = [
        ("", "What best describes your daily lifestyle?"),
        ("Nice — I get it. Now tell me, which style feels most like you?",
         "Which style feels most like you?"),
        ("Ooh, lovely pick! One more — let's talk color.",
         "What is your frame color vibe?"),
        ("Got it. Now the practical bit — what's your budget?",
         "What is your budget for frames in rupees?"),
        ("Almost done — just one last thing —",
         "What are your vision needs?"),
    ]
    hi_quiz = [
        ("", "आपका रोज़ का काम या लाइफस्टाइल कैसा है?"),
        ("बढ़िया, समझ गई। अब बताइए — आपको कौन सा स्टाइल अच्छा लगता है?",
         "आपको कौन सा स्टाइल सबसे अच्छा लगता है?"),
        ("वाह, बहुत बढ़िया चुनाव! अब एक और सवाल — रंग की बात करें?",
         "फ्रेम का कौन सा रंग पसंद है?"),
        ("अच्छा! अब ज़रूरी बात — आपका बजट कितना है?",
         "आपका बजट कितना है?"),
        ("बस एक आख़िरी सवाल बचा है —",
         "आपकी नज़र की ज़रूरत क्या है?"),
    ]

    # Short agent introductions (spoken instantly after language pick).
    en_intro = "Hi! I'm Claire, your personal eyewear stylist at Lenskart. Let's get started."
    hi_intro = "नमस्ते! मैं Claire हूँ, Lenskart की आपकी personal eyewear stylist। चलिए शुरू करते हैं।"

    phrases = [
        (en_intro, "en"), (hi_intro, "hi"),
        (en_greet, "en"), (hi_greet, "hi"),
    ]
    for intro, q in en_quiz:
        phrases.append(((intro + " " + q).strip(), "en"))
        phrases.append((intro, "en") if intro else (q, "en"))
    for intro, q in hi_quiz:
        phrases.append(((intro + " " + q).strip(), "hi"))
        phrases.append((intro, "hi") if intro else (q, "hi"))

    # Dedup while preserving order
    seen = set()
    out  = []
    for text, lang in phrases:
        t = (text or "").strip()
        if not t:
            continue
        key = f"{lang}::{t}"
        if key in seen:
            continue
        seen.add(key)
        out.append((t, lang))
    return out


def preload_disk_cache() -> int:
    """
    On boot, load every previously-synthesised TTS clip from disk into the
    in-memory LRU. Zero-latency for all common phrases on the very first
    request after startup.
    """
    files = sorted(_DISK_CACHE_DIR.glob("*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    loaded = skipped = 0
    for f in files:
        if len(_CACHE) >= CACHE_MAX:
            skipped += 1
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("chunks"):
                _CACHE[f.stem] = data
                loaded += 1
        except Exception as exc:
            print(f"  [tts] skipped corrupt cache file {f.name}: {exc}")
    total_bytes = sum((_DISK_CACHE_DIR / f"{k}.json").stat().st_size
                      for k in _CACHE if (_DISK_CACHE_DIR / f"{k}.json").exists())
    print(f"  [tts] preloaded {loaded} cached clips into memory "
          f"({total_bytes/1024:.0f} KB on disk, {skipped} skipped)")
    return loaded


def precache_common(async_mode: bool = True) -> None:
    """
    Pre-fetch and cache TTS audio for every common phrase Claire says during
    the first-turn happy path, in both languages. Runs in a background thread
    by default so it never blocks server startup.
    """
    phrases = _build_common_phrases()

    def _run():
        ok = fail = 0
        for text, lang in phrases:
            try:
                r = synthesise(text, lang)
                if r.get("chunks"):
                    ok += 1
                else:
                    fail += 1
            except Exception as exc:
                fail += 1
                print(f"  [tts] precache error ({lang}): {exc}")
        print(f"  [tts] precache done — {ok} cached, {fail} failed "
              f"({len(_CACHE)} entries in LRU)")

    if async_mode:
        import threading
        threading.Thread(target=_run, name="tts-precache", daemon=True).start()
        print(f"  [tts] precaching {len(phrases)} common phrases in background…")
    else:
        _run()
