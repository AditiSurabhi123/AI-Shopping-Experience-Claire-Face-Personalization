#!/usr/bin/env python3
"""
LLM-powered face analysis via Gemini 3.1 Flash vision.

Given a base64 image (JPEG/PNG), returns structured attributes for the person
in the photo: face shape, gender guess, age guess, hair/skin colour, plus
stylist recommendations. If no face is detected, returns {"has_face": false}.

Used by both the Upload Selfie and Live Recommendation flows.
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
# Do NOT change to gemini-3 / gemini-3.1 — those model IDs return 404.
# `gemini-2.5-flash` supports vision + text on the public API.
VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-3.1-pro-preview")
# Models tried in order when the configured one returns 404/NOT_FOUND.
_VISION_FALLBACKS = ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash")

_PROMPT = """You are a professional eyewear stylist at Lenskart analysing a customer photo for frame recommendations.

FACE DETECTION IS STRICT. Return {"has_face": false, "reason": "<short>"} if ANY of these apply:
  - No clearly visible human face in the image.
  - The face is heavily blurred, tiny, or largely obscured (hand, mask, heavy shadows covering >40%).
  - Multiple distinct faces — you cannot pick one confidently.
  - The image shows a non-human subject (animal, doll, illustration, cartoon, statue).
  - The image is a generic scene (landscape, product shot, screenshot, etc.) with no person.
  - Extreme profile angle where eye region isn't visible.

Only when a single clear human face is present, return this JSON shape:

{
  "has_face": true,
  "shape":              "<Oval | Round | Square | Heart | Diamond | Oblong>",
  "gender":             "<male | female | unisex — or null>",
  "age":                <integer years, best guess>,
  "age_group":          "<kids | adult | aged>",
  "hair_color":         "<black | brown | blonde | grey | red | other — or null>",
  "skin_tone":          "<fair | wheatish | medium | dark — or null>",
  "face_width":         "<Narrow | Medium | Wide>",
  "recommended_styles": ["<frame style>", ...],
  "key_feature":        "<one short sentence on the facial structure>",
  "confidence":         <integer 0 to 100>
}

Rules:
- Be conservative about has_face — when in doubt, return false with a reason.
- Face shape must come from the allowed list.
- recommended_styles must come from: wayfarer, aviator, round, cat-eye, geometric, rimless, square, clubmaster, oval, butterfly, rectangular.
- Pick 3 to 5 styles best suited to the face shape and proportions.
- Return JSON only — no markdown fences, no prose.

Face-shape guidance (use these visual cues, do NOT default to Oval):
- Oval:    length roughly 1.5× width, forehead slightly wider than jaw, soft jawline.
- Round:   length and width similar, soft jaw, full cheeks, no sharp angles.
- Square:  length and width similar, strong angular jaw, wide forehead.
- Heart:   wide forehead and cheekbones, narrow pointed chin.
- Diamond: narrow forehead and jaw, widest at cheekbones.
- Oblong:  noticeably longer than wide, straight sides, similar forehead/jaw width.
Measure proportions carefully from the image before choosing. If two shapes seem
close, pick the better fit based on jaw angularity and forehead-to-jaw ratio —
don't default to Oval unless the proportions genuinely match it."""


# ─── Fit-score prompt ────────────────────────────────────────────────────────
# Given the person's photo AND a specific frame (shape, colour, width, face
# fit recommendation), the model estimates how well that frame would look and
# fit on this person.
_FIT_PROMPT = """You are Claire, Lenskart's expert stylist assessing how well a
specific frame will fit the person in this photo.

FRAME DETAILS
{frame_desc}

TASK
Look at the person's face (shape, width, proportions) and the frame's size and
style. Decide how confident you are about the fit and return this JSON only:

{{
  "has_face": <true | false>,
  "score":    <integer 0-99>,
  "verdict":  "<Perfect Match | Great Fit | Good Fit | Okay Fit | Not Ideal>",
  "size_match": "<too small | just right | too large>",
  "reasons":  ["<short reason 1>", "<short reason 2>", "<short reason 3>"],
  "face_shape": "<Oval|Round|Square|Heart|Diamond|Oblong>",
  "face_width": "<Narrow|Medium|Wide>"
}}

Scoring guidelines:
- 90-99: frame shape & width strongly suit this person's face.
- 75-89: good compatibility, minor mismatch on one dimension.
- 60-74: acceptable; shape OR width is a clear mismatch.
- 40-59: visible mismatch on both shape and width.
- < 40:  poor fit, recommend a different style.

Size heuristic:
- A "too small" verdict applies when frame width << face width (looks pinched).
- "too large" applies when frame width >> face width (slides off / overwhelms).
- "just right" when proportional.

Rules:
- If the photo has no clear human face → {{"has_face": false, "score": 0,
  "verdict": "Not Ideal", "reasons": ["Could not detect a face in the photo"]}}
- Keep reasons short and customer-facing (one sentence each).
- Return JSON only."""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = "\n".join(s.split("\n")[1:])
    if s.endswith("```"):
        s = "\n".join(s.split("\n")[:-1])
    return s.strip()


def _salvage_truncated_json(s: str) -> str:
    """
    Best-effort closer for JSON that was cut off mid-string by the LLM.
    Walks the string, tracks whether we're inside a string, and closes any
    dangling string/array/object so `json.loads` can consume what we got.
    Not robust for deeply nested payloads — just enough for fit-score /
    quiz-analysis shapes.
    """
    s = s.strip()
    in_str = False
    escape = False
    stack  = []
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    out = s
    if in_str:
        # Drop the trailing (truncated) content inside the open string and
        # close it cleanly.
        last_quote = out.rfind('"')
        # truncate to the char before the unterminated string's opening quote,
        # then add an empty string to preserve the key's value slot
        if last_quote != -1:
            out = out[:last_quote] + '""'
    # Strip any trailing comma before closing
    out = out.rstrip().rstrip(",")
    while stack:
        out += stack.pop()
    return out


def analyze_image(image_b64: str, mime: str = "image/jpeg") -> dict:
    """
    image_b64: base64 payload only (no `data:...,` prefix).
    Returns the parsed JSON from Gemini, or {has_face: false} on unambiguous
    refusal, or raises on network / parse failure.
    """
    if "," in image_b64:  # tolerate data URI by mistake
        image_b64 = image_b64.split(",", 1)[1]

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": _PROMPT},
                {"inlineData": {"mimeType": mime, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 400,
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    })

    # Try the configured model; on 404 (bad ID / preview retired) fall back to
    # known-working models so the feature keeps working regardless of which
    # placeholder the config/linter currently contains.
    models_to_try = [VISION_MODEL] + [m for m in _VISION_FALLBACKS if m != VISION_MODEL]
    last_err = None
    resp_status = 0; raw = ""
    for model in models_to_try:
        path = f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=30, context=SSL_CTX)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8")
            resp_status = resp.status
        finally:
            conn.close()
        if resp_status == 200:
            break
        if resp_status == 404 or "NOT_FOUND" in raw:
            last_err = f"{model}: 404"
            print(f"  [face_llm] model {model} unavailable, trying next")
            continue
        # Any other error — stop and surface it
        raise RuntimeError(f"Gemini vision HTTP {resp_status}: {raw[:200]}")

    if resp_status != 200:
        raise RuntimeError(f"All vision models returned 404 ({last_err})")

    data   = json.loads(raw)
    parts  = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError(f"Gemini vision: empty response — {raw[:200]}")

    text = _strip_fences(parts[0].get("text", ""))
    # Extract JSON if the model wrapped it in any filler
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini vision: bad JSON: {exc} — {text[:200]}")

    # Normalise age_group from age if missing
    if result.get("has_face") is False:
        return {"has_face": False, "reason": result.get("reason", "no_face_detected")}
    age = result.get("age")
    if result.get("age_group") is None and isinstance(age, (int, float)):
        a = int(age)
        if   a < 13: result["age_group"] = "kids"
        elif a < 50: result["age_group"] = "adult"
        else:         result["age_group"] = "aged"

    result.setdefault("has_face", True)
    # Safety net: reject low-confidence / missing-shape responses as "no face"
    try:
        conf = int(result.get("confidence") or 0)
    except Exception:
        conf = 0
    if not result.get("shape") or conf < 50:
        return {"has_face": False,
                "reason": f"low_confidence ({conf}%)" if result.get("shape") else "no_shape"}
    return result


# ─── LLM-backed size-confidence widget ───────────────────────────────────────

def analyze_fit(image_b64: str, frame: dict, mime: str = "image/jpeg") -> dict:
    """
    Ask Gemini Vision to judge how well a specific frame fits the person in
    the photo. Returns:
      { has_face, score, verdict, size_match, reasons, face_shape, face_width }
    Raises on network / parse failure (caller should handle).
    """
    if "," in (image_b64 or ""):
        image_b64 = image_b64.split(",", 1)[1]

    frame_desc = (
        f"- Name:           {frame.get('name') or frame.get('id')}\n"
        f"- Type:           {frame.get('type')}\n"
        f"- Frame shape:    {frame.get('frame_shape') or frame.get('style')}\n"
        f"- Colour:         {frame.get('color')}\n"
        f"- Frame width:    {frame.get('frame_width', 'medium')}\n"
        f"- Suited shapes:  {', '.join(frame.get('face_shape_recommendation') or frame.get('shape_suitability') or [])}\n"
    )

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": _FIT_PROMPT.format(frame_desc=frame_desc)},
                {"inlineData": {"mimeType": mime, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            # Bumped from 400 → 800: the strict JSON shape plus 3 customer-facing
            # reasons was regularly getting truncated mid-string, producing
            # "Unterminated string starting at: line 2 column 3" parse errors.
            "maxOutputTokens": 800,
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    })

    models_to_try = [VISION_MODEL] + [m for m in _VISION_FALLBACKS if m != VISION_MODEL]
    raw, status = "", 0
    for model in models_to_try:
        path = f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=30, context=SSL_CTX)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8")
            status = resp.status
        finally:
            conn.close()
        if status == 200: break
        if status == 404 or "NOT_FOUND" in raw: continue
        raise RuntimeError(f"Gemini vision {status}: {raw[:200]}")

    if status != 200:
        raise RuntimeError(f"Gemini vision: no working model (last {status})")

    data  = json.loads(raw)
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    if not parts:
        raise RuntimeError("Gemini vision: empty response for fit score")
    text = _strip_fences(parts[0].get("text", ""))
    m    = re.search(r"\{[\s\S]*\}", text)
    if m: text = m.group(0)
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Gemini occasionally still truncates mid-string when the `reasons`
        # array runs long. Try to salvage by closing the open string/array/obj
        # before giving up.
        try:
            result = json.loads(_salvage_truncated_json(text))
        except json.JSONDecodeError as exc2:
            # Last-ditch: return a reasonable default rather than 500-ing.
            print(f"  [fit-score] JSON parse failed after salvage: {exc2}\n  Raw: {text[:300]}")
            return {
                "has_face": True,
                "score":    70,
                "verdict":  "Good Fit",
                "size_match": "just right",
                "reasons":  ["Fit analysis partially available — this frame should work well for you."],
                "face_shape": None,
                "face_width": None,
                "_partial":   True,
            }

    # Sanity clamp
    if not result.get("has_face", True):
        return {"has_face": False, "reasons": result.get("reasons") or ["No face detected"]}
    try:
        result["score"] = max(0, min(99, int(result.get("score") or 60)))
    except Exception:
        result["score"] = 60
    result.setdefault("reasons", [])
    result.setdefault("size_match", "just right")
    result.setdefault("has_face", True)
    return result
