#!/usr/bin/env python3
"""
Lenskart Claire AI — AWS Bedrock Integration
AI-powered quiz response analysis: language detection, translation, tag extraction.
Uses AWS Bedrock Runtime via Bearer token auth.
"""
import json
import http.client
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BEDROCK_BEARER_TOKEN = os.environ.get(
    "BEDROCK_BEARER_TOKEN", "")

BEDROCK_REGION = os.environ.get("BEDROCK_REGION", "ap-south-1")
# Anthropic models on Bedrock require a cross-region *inference profile* ID
# (not the bare model ARN) for on-demand invocation. Profile prefix depends on
# region: `apac.` for ap-*, `us.` for us-*, `eu.` for eu-*.
# Claude Opus 4.x isn't enabled for this account/region — Sonnet 4 is the
# highest-tier Claude verified working in ap-south-1 (May-2025 release).
# Override via BEDROCK_MODEL if you have Opus access enabled.
BEDROCK_MODEL  = os.environ.get(
    "BEDROCK_MODEL",
    "apac.anthropic.claude-sonnet-4-20250514-v1:0",
)

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────

QUIZ_ANALYSIS_PROMPT = """\
You are a multilingual AI assistant for Lenskart, an Indian eyewear brand.

A customer answered a quiz step about eyewear preferences. Analyse their response and extract structured product recommendation tags.

QUIZ STEP CONTEXT: {context}
CUSTOMER RESPONSE: {response}

Return ONLY a valid JSON object — no markdown fences, no commentary:
{{
  "original_response": "<exact user response>",
  "detected_language": "<ISO 639-1 code: en/hi/te/ta/bn/mr/gu/kn/ml/pa/other>",
  "language_name": "<human-readable name e.g. English / Hindi / Telugu>",
  "english_translation": "<English translation; identical to original if already English>",
  "tags": {{
    "price":        "<budget | mid-range | premium | ultra-premium — or null>",
    "lifestyle":    "<professional | creative | active | fashion — or null>",
    "trend":        "<classic | trendy | minimal | bold — or null>",
    "color":        "<neutral | warm | cool | statement — or null>",
    "color_type":   "<light | dark — or null>",
    "budget":       "<under_500 | under_1000 | under_1500 | under_2000 | under_2500 | under_3000 | above_3000 — or null>",
    "vision_need":  "<zero_power | single_vision | progressive | not_sure — or null>",
    "product_type": "<eyeglasses | sunglasses | contact_lens — or null>",
    "age_group":    "<kids | adult | aged — or null>",
    "gender_pref":  "<male | female | unisex — or null>",
    "frame_shape_pref": "<wayfarer | aviator | round | rectangular | square | geometric | rimless | clubmaster | oval | cat-eye | butterfly — or null>"
  }},
  "confidence": <integer 0-100 extraction confidence>
}}

Extraction rules:
- Detect language accurately. Roman-script Hindi / Hinglish → "hi".
- Budget: "500 se niche"→under_500, "sasta"/"cheap"→under_1000, "2000 tak"→under_2000, "premium"→above_3000.
- Vision: "power wala"/"number wali"→single_vision, "zero power"/"bina power"→zero_power, "progressive"/"bifocal"→progressive.
- Color type: "light color"/"halka rang"→light, "dark color"/"gehra rang"→dark.
- Product: "sunglass"/"goggle"→sunglasses, "contact"→contact_lens, "chashma"/"specs"→eyeglasses.
- Age: "kids"/"bachon ke liye"→kids, "elderly"/"uncle"/"aunty"→aged.
- Gender: "ladies"/"women"→female, "gents"/"men"→male.
- Extract ALL applicable tags. Use null only when truly undetermined.
"""


# ─────────────────────────────────────────────
# BEDROCK CALLER
# ─────────────────────────────────────────────

def _call_bedrock(prompt: str, max_tokens: int = 1024) -> str:
    """
    Invoke an Anthropic Claude model on AWS Bedrock using Bearer-token auth.
    Returns the assistant's text response.
    Raises RuntimeError on non-200 HTTP status.
    """
    endpoint = f"bedrock-runtime.{BEDROCK_REGION}.amazonaws.com"
    path     = f"/model/{BEDROCK_MODEL}/invoke"

    payload = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    })

    headers = {
        "Content-Type":  "application/json",
        "Accept":        "application/json",
        "Authorization": f"Bearer {BEDROCK_BEARER_TOKEN}",
    }

    from ssl_ctx import SSL_CTX
    conn = http.client.HTTPSConnection(endpoint, timeout=30, context=SSL_CTX)
    try:
        conn.request("POST", path, payload, headers)
        resp = conn.getresponse()
        raw  = resp.read().decode("utf-8")
    finally:
        conn.close()

    if resp.status != 200:
        raise RuntimeError(f"Bedrock API error {resp.status}: {raw[:400]}")

    data = json.loads(raw)
    # Standard Bedrock / Anthropic Messages response shape
    content = data.get("content", [])
    parts   = [b.get("text", "") for b in content if b.get("type") == "text"]
    return "".join(parts).strip()


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────

def analyze_quiz_response(user_response: str, quiz_context: str = "") -> dict:
    """
    Analyse a single quiz answer using AWS Bedrock.

    Returns:
        {
          success: bool,
          original_response: str,
          detected_language: str,
          language_name: str,
          english_translation: str,
          tags: { price, lifestyle, trend, color, budget, vision_need },
          confidence: int,
          error: str  # only present on failure
        }
    """
    if not user_response or not user_response.strip():
        return _fallback_result(user_response or "", "Empty response")

    prompt = QUIZ_ANALYSIS_PROMPT.format(
        context=quiz_context or "General eyewear preference question",
        response=user_response.strip()
    )

    try:
        raw_text = _call_bedrock(prompt)
    except Exception as exc:
        print(f"  [Bedrock] API call failed: {exc}")
        return _fallback_result(user_response, str(exc))

    # Parse JSON — Claude should return clean JSON but strip any stray fences
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"  [Bedrock] JSON parse error: {exc}\n  Raw: {cleaned[:300]}")
        return _fallback_result(user_response, f"JSON parse error: {exc}")

    result["success"] = True
    tags = result.setdefault("tags", {})
    for key in _ALL_TAG_KEYS:
        tags.setdefault(key, None)

    return result


_ALL_TAG_KEYS = (
    "price", "lifestyle", "trend", "color", "color_type",
    "budget", "vision_need", "product_type", "age_group",
    "gender_pref", "frame_shape_pref",
)


def _fallback_result(original: str, error: str) -> dict:
    return {
        "success":             False,
        "original_response":   original,
        "detected_language":   "en",
        "language_name":       "English",
        "english_translation": original,
        "tags":                _heuristic_tags(original.lower()),
        "confidence":          40,
        "error":               error,
    }


def _heuristic_tags(text: str) -> dict:
    """Keyword fallback — mirrors gemini._heuristic_tags, keeps Bedrock in sync."""
    from gemini import _heuristic_tags as _g_heuristic
    return _g_heuristic(text)
