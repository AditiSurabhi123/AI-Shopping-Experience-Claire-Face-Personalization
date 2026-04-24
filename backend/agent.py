"""
Lenskart Claire AI — Agent Loop & System Prompt
Integrates with Claude API for intelligent eyewear assistance.
Falls back to rich demo mode when no API key is set.
"""
import json
import os
import re
import sys
import http.client
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tools import execute_tool, search_frames, search_products, FACE_SHAPES
from gemini import _heuristic_tags as _extract_tags
from product_db import BUDGET_PRICE_MAP

try:
    from recommend import recommend as _llm_recommend
    _HAS_LLM_RECOMMEND = True
except Exception as _e:
    print(f"  [agent] LLM recommender unavailable: {_e}")
    _HAS_LLM_RECOMMEND = False


def _fmt_tag(key: str, val) -> str:
    """Human-friendly chip label for the filters summary line."""
    if key == "min_rating":
        try: return f"{float(val)}★ & up"
        except Exception: return f"{val}★"
    if key == "bestseller_only":  return "bestsellers"
    if key == "trending_only":    return "trending"
    if key == "new_arrival_only": return "new arrivals"
    if key == "sort_by":
        return {
            "rating":     "by rating",
            "price_asc":  "price ↑",
            "price_desc": "price ↓",
            "newest":     "newest first",
        }.get(str(val), f"sort: {val}")
    # Budget: render around_N as a ±band range, above_N / under_N clearly
    if key == "budget":
        import re as _rb
        m = _rb.match(r"^(under|above|around)_(\d+)$", str(val))
        if m:
            kind, n = m.group(1), int(m.group(2))
            if kind == "around":
                band = max(200, int(n * 0.10))
                return f"around ₹{n:,} (₹{n-band:,}–₹{n+band:,})"
            if kind == "under":
                return f"under ₹{n:,}"
            return f"₹{n:,}+"
    return str(val).replace("_", " ")


def _enrich_for_frontend(p: dict) -> dict:
    """Add legacy-compat fields the frontend expects."""
    out = dict(p)
    out.setdefault("shape_suitability", p.get("face_shape_recommendation") or [])
    out.setdefault("rating", p.get("avg_rating") or 4.0)
    out.setdefault("reviews", int(round(float(out["rating"]) * 250)))
    out.setdefault("original_price", p.get("strikeout_price") or p.get("price"))
    out.setdefault("bestseller", "bestseller" in (p.get("tags") or []))
    out.setdefault("new_arrival", "new-arrival" in (p.get("tags") or []))
    out.setdefault("style", p.get("frame_shape") or "wayfarer")
    out.setdefault("color_hex", p.get("color_hex") or "#00C3C3")
    out.setdefault("frame_width", "medium")
    return out


def _llm_pick(session_data: dict, tags: dict, face_data: dict, chat_hint: str = "", limit: int = 6):
    """Build a context dict and delegate to the LLM recommender. Returns list of enriched products + reasoning."""
    if not _HAS_LLM_RECOMMEND:
        return None

    budget_key = tags.get("budget") or ""
    bmap       = BUDGET_PRICE_MAP.get(budget_key, None)
    if bmap is None:
        # Dynamic parse for arbitrary "under_N" / "above_N" / "around_N"
        import re as _re
        m = _re.match(r"^(under|above|around)_(\d+)$", budget_key)
        if m:
            kind, n = m.group(1), int(m.group(2))
            if kind == "around":
                band = max(200, int(n * 0.10))   # ±10%, min ₹200 window
                bmap = {"min": max(0, n - band), "max": n + band}
            elif kind == "under":
                bmap = {"max": n}
            else:
                bmap = {"min": n}
        else:
            bmap = {}

    # Resolve defaults: prefer eyeglasses + adult unless the user signalled otherwise
    lifestyle = (tags.get("lifestyle") or "").lower()
    product_type = tags.get("product_type")
    age_group    = tags.get("age_group")
    if not product_type and lifestyle in {"professional", "fashion", "creative"}:
        product_type = "eyeglasses"
    if not age_group and lifestyle in {"professional", "fashion", "creative", "active"}:
        age_group = "adult"

    ctx = {
        "quiz_answers":   (session_data or {}).get("quiz_answers") or {},
        "active_filters": tags,
        "face_data":      face_data,
        "chat_hint":      chat_hint,
        "product_type":   product_type,
        "budget_max":     bmap.get("max"),
        "budget_min":     bmap.get("min"),
        "age_group":      age_group,
        "frame_shape":    tags.get("frame_shape_pref"),
        "color_type":     tags.get("color_type"),
        "specific_color": tags.get("specific_color"),
        "gender":         tags.get("gender_pref"),
        "trend":          tags.get("trend"),
        # Chat-driven refinements:
        "min_rating":        tags.get("min_rating"),
        "bestseller_only":   tags.get("bestseller_only"),
        "new_arrival_only":  tags.get("new_arrival_only"),
        "trending_only":     tags.get("trending_only"),
        "sort_by":           tags.get("sort_by"),
    }
    vision = tags.get("vision_need")
    if vision == "zero_power":
        ctx["has_power"] = False
    elif vision in ("single_vision", "progressive"):
        ctx["has_power"] = True

    try:
        # RAG-first: embedding retrieval + optional LLM re-rank.
        # Set rerank_with_llm=True for richer reasoning (adds ~400ms).
        result = _llm_recommend(ctx, limit=limit, rerank_with_llm=True)
    except Exception as exc:
        print(f"  [agent] RAG recommend failed: {exc}")
        return None

    if not result.get("success"):
        return None
    return result

# Primary agent runtime — Google Gemini 2.5 Flash with native function calling.
# Bedrock Opus wasn't available on the project's AWS account, so we switched
# back to Gemini for the main tool-use loop. The Anthropic-flavoured `TOOLS`
# list below is translated to Gemini `functionDeclarations` at first call.
GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
)
GEMINI_HOST  = "generativelanguage.googleapis.com"
# Don't change to `gemini-3*` — those IDs 404. 2.5-flash is the current
# public-API model that supports text + function calling reliably.
GEMINI_MODEL = os.environ.get("GEMINI_AGENT_MODEL", "gemini-2.5-flash")
_GEMINI_FALLBACKS = ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash")
# Demo mode kicks in only if we have no Gemini key configured at all.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are Claire, Lenskart's expert AI optometrist and personal eyewear stylist. You combine the warmth of a trusted friend with the knowledge of a professional eye care specialist.

## Your Personality
- Warm, encouraging, and genuinely excited about helping people find their perfect frames
- Conversational and natural — never robotic or sales-y
- Expert but accessible — explain optometry concepts in simple, relatable terms
- Celebratory of choices: "That style suits your personality perfectly!"
- Honest: if something won't suit a face shape, gently suggest better alternatives

## Your Tools
You have four specialist tools. Use them proactively:

1. **analyze_face(image_url)** — IMMEDIATELY call this when a user uploads a photo. Determines face shape, dimensions, and ideal frame styles. Reference these results throughout the entire conversation.

2. **search_frames(style, shape, color, price_range, gender)** — Call before showing any product recommendations. Use face shape from analysis + quiz preferences to populate parameters. ALWAYS call this before outputting a [CAROUSEL].

3. **get_lens_recommendation(prescription_type, screen_time_hours)** — Call when users mention prescriptions, screen time, or lens questions. Combine with lifestyle quiz data.

4. **calculate_fit_confidence(frame_id, face_shape, face_width)** — Call for each highlighted frame after face analysis. Shows users the science behind why a frame suits them.

## Response Format — CRITICAL

Use these exact tags to render rich UI components. The JSON after each tag must be valid:

**Product Carousel** — Use after calling search_frames:
[CAROUSEL]{"title": "Recommended for You", "frames": [...array of frame objects from search_frames...]}

**Quiz Step** — For multi-step discovery:
[QUIZ]{"question": "Question text?", "options": ["Option A", "Option B", "Option C", "Option D"], "step": 1, "total": 5, "key": "lifestyle"}

**Face Analysis** — After analyze_face tool returns:
[FACE_ANALYSIS]{"shape": "Oval", "description": "...", "recommended_styles": [...], "face_width": "Medium", "key_feature": "..."}

**Lens Recommendation** — After get_lens_recommendation:
[LENS_REC]{"packages": [...lens package objects...], "reasoning": ["reason 1", "reason 2"]}

**Fit Score** — After calculate_fit_confidence for top picks:
[FIT_SCORE]{"frame_id": "LK-001", "frame_name": "Name", "score": 87, "verdict": "Great Fit", "reasons": ["reason1"]}

Place plain text BEFORE or AFTER these tags. Never nest tags. Keep tag JSON on one line.

## Conversation Flow

### First Message (Greeting)
Introduce yourself warmly. Offer the Frame Discovery Quiz and photo upload. Use [QUIZ] for step 1.

### Frame Quiz (5 Steps)
Step 1 — Lifestyle: "Office Professional", "Creative & Artistic", "Active & Outdoorsy", "Fashion-Forward"
Step 2 — Style Preference: "Classic & Timeless", "Bold & Trendy", "Minimal & Clean", "Adventurous"
Step 3 — Color Preference: "Neutral (Black/Grey/Brown)", "Warm (Gold/Tortoise)", "Cool (Blue/Silver)", "Statement Color"
Step 4 — Budget (INR): "Under ₹1,000", "₹1,000–₹2,000", "₹2,000–₹3,500", "Premium ₹3,500+"
Step 5 — Vision Needs: "Zero power (fashion)", "Single vision", "Progressive/Bifocal", "Not sure yet"

After step 5, call search_frames() with the gathered preferences and show a [CAROUSEL].

### Photo Upload Flow
When a photo is uploaded: acknowledge it excitedly → call analyze_face → show [FACE_ANALYSIS] result → call search_frames with face shape → show [CAROUSEL] with personalized recommendations.

### Lens Guidance
After quiz step 5 or when prescriptions are mentioned, call get_lens_recommendation and show [LENS_REC].

### Fit Confidence
When a user shows interest in a specific frame, call calculate_fit_confidence and show [FIT_SCORE].

## Context Management
- Reference face shape data throughout: "Since your Oval face suits almost any style..."
- Recall quiz answers: "Given your preference for bold, trendy styles..."
- If user returns to a topic, recall previous answers without asking again
- Track: face_data, quiz_answers (all 5), selected_frames, prescription_info

## Tone Guidelines
✅ "Those titanium aviators would look stunning on you!"
✅ "Your heart-shaped face has such beautiful proportions — lightweight frames will highlight your features perfectly."
✅ "At ₹1,499, the Horizon Classic is incredible value — 50% off today!"
❌ Never be pushy: "You MUST buy this now!"
❌ Never be dismissive of any budget
❌ If unsure about a medical question, recommend an in-store eye test at Lenskart

End each response with a natural follow-up question or action prompt to keep the conversation flowing."""

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────

TOOLS = [
    {
        "name": "analyze_face",
        "description": "Analyze a user's uploaded face photo to determine face shape (Oval, Round, Square, Heart, Diamond, Oblong) and facial dimensions. Returns recommended frame styles tailored to their features. ALWAYS call this immediately when a photo is uploaded.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "URL or path to the uploaded face photo"
                }
            },
            "required": ["image_url"]
        }
    },
    {
        "name": "search_frames",
        "description": "Search Lenskart's frame catalog with optional filters. Returns matching frames with full product details. Use face shape from analysis and quiz answers to populate parameters.",
        "input_schema": {
            "type": "object",
            "properties": {
                "style": {"type": "string", "description": "Frame style: wayfarer, aviator, round, cat-eye, geometric, rimless, square, sports, clubmaster, oval, butterfly"},
                "shape": {"type": "string", "description": "Best for face shape: oval, round, square, heart, diamond, oblong"},
                "color": {"type": "string", "description": "Color preference: black, gold, silver, tortoise, rose gold, clear"},
                "price_range": {"type": "string", "description": "Price range in INR, e.g. '500-1500', 'under 2000', '1000-3000'"},
                "gender": {"type": "string", "description": "Filter by gender: male, female, unisex"},
                "limit": {"type": "integer", "description": "Max frames to return (default 6)"}
            }
        }
    },
    {
        "name": "get_lens_recommendation",
        "description": "Get personalized lens package recommendations based on the user's prescription type and screen time habits. Always call this after the quiz step 5 or when lens needs are mentioned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prescription_type": {
                    "type": "string",
                    "description": "Vision correction type: zero_power, single_vision, progressive, bifocal"
                },
                "screen_time_hours": {
                    "type": "number",
                    "description": "Average daily screen time in hours (0-16)"
                }
            },
            "required": ["prescription_type"]
        }
    },
    {
        "name": "calculate_fit_confidence",
        "description": "Calculate how well a specific frame fits the user's face. Returns a confidence score (0-100) with detailed reasoning. Call this for top 2 frame recommendations after face analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "frame_id": {"type": "string", "description": "Product ID of the frame (e.g. LK-HC-001)"},
                "face_shape": {"type": "string", "description": "User's face shape from analysis"},
                "face_width": {"type": "string", "description": "User's face width from analysis"}
            },
            "required": ["frame_id"]
        }
    }
]


# ─────────────────────────────────────────────
# RESPONSE PARSER
# ─────────────────────────────────────────────

def parse_response(text: str) -> dict:
    """
    Extract structured components from Claude's tagged response text.
    Returns {text: str, components: list}
    """
    components = []
    clean_text = text

    tag_patterns = {
        "CAROUSEL": "carousel",
        "QUIZ": "quiz",
        "FACE_ANALYSIS": "face_analysis",
        "LENS_REC": "lens_rec",
        "FIT_SCORE": "fit_score"
    }

    for tag, comp_type in tag_patterns.items():
        # Match [TAG]{...} where {...} is JSON (handles nested braces)
        pattern = rf'\[{tag}\](\{{.*?\}})'
        matches = list(re.finditer(pattern, clean_text, re.DOTALL))

        for match in reversed(matches):  # Reverse to preserve indices
            try:
                data = json.loads(match.group(1))
                components.insert(0, {"type": comp_type, **data})
                clean_text = clean_text[:match.start()] + clean_text[match.end():]
            except json.JSONDecodeError:
                pass

        # Also try array format [TAG][...]
        pattern2 = rf'\[{tag}\](\[.*?\])'
        matches2 = list(re.finditer(pattern2, clean_text, re.DOTALL))
        for match in reversed(matches2):
            try:
                data = json.loads(match.group(1))
                components.insert(0, {"type": comp_type, "data": data})
                clean_text = clean_text[:match.start()] + clean_text[match.end():]
            except json.JSONDecodeError:
                pass

    return {
        "text": clean_text.strip(),
        "components": components
    }


# ─────────────────────────────────────────────
# GEMINI AGENT LOOP (native function calling)
# ─────────────────────────────────────────────

def _tools_to_gemini(tools: list) -> list:
    """
    Convert the Anthropic-style `TOOLS` list to Gemini's `functionDeclarations`.
    Gemini uses `parameters` (OpenAPI-flavoured JSON-schema) instead of
    `input_schema` but the schema shape is otherwise identical.
    """
    decls = []
    for t in tools:
        decls.append({
            "name":        t["name"],
            "description": t.get("description", ""),
            "parameters":  t.get("input_schema") or {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}]


_GEMINI_TOOLS = _tools_to_gemini(TOOLS)


def _post_gemini(model: str, body: str) -> tuple[int, str]:
    from ssl_ctx import SSL_CTX
    path = f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=60, context=SSL_CTX)
    try:
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        raw  = resp.read().decode("utf-8")
    finally:
        conn.close()
    return resp.status, raw


# Cached model id once a successful call lands. If GEMINI_AGENT_MODEL isn't
# available, the loop walks _GEMINI_FALLBACKS and sticks with whichever works.
_RESOLVED_GEMINI_MODEL: str = GEMINI_MODEL


def call_gemini_agent(contents: list) -> dict:
    """
    Invoke Gemini with the Claire system prompt + tool declarations and the
    running `contents` array (Gemini's equivalent of `messages`).
    Returns the parsed JSON response. Raises RuntimeError on transport /
    non-200 / bad JSON.
    """
    global _RESOLVED_GEMINI_MODEL

    body = json.dumps({
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents":          contents,
        "tools":             _GEMINI_TOOLS,
        "toolConfig":        {"functionCallingConfig": {"mode": "AUTO"}},
        "generationConfig":  {
            "maxOutputTokens": 4096,
            "temperature":     0.7,
        },
    })

    tried = []
    for model in [_RESOLVED_GEMINI_MODEL, *(_GEMINI_FALLBACKS)]:
        if model in tried:
            continue
        tried.append(model)
        status, raw = _post_gemini(model, body)
        if status == 200:
            _RESOLVED_GEMINI_MODEL = model
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise RuntimeError(f"Gemini non-JSON response: {raw[:400]}")
        # Try next model on 404 / NOT_FOUND / model-not-supported
        low = raw.lower()
        if status == 404 or "not_found" in low or "not supported" in low or "does not support" in low:
            print(f"  [gemini-agent] {model} unavailable ({status}); trying next")
            continue
        # Any other error → surface immediately
        raise RuntimeError(f"Gemini API error {status}: {raw[:400]}")

    raise RuntimeError(f"Gemini API: no working model from {tried}")


def _parse_gemini_parts(parts: list) -> tuple[list, list]:
    """Split Gemini response parts into (text_segments, function_calls)."""
    texts, calls = [], []
    for p in parts or []:
        if "text" in p and p["text"]:
            texts.append(p["text"])
        fc = p.get("functionCall") or p.get("function_call")
        if fc:
            calls.append({
                "name": fc.get("name"),
                "args": fc.get("args") or {},
            })
    return texts, calls


def run_agent(messages: list, session_data: dict = None) -> dict:
    """
    Run the Claire agent.

    Routing is scripted-first, Gemini-second:
      1. `run_demo_agent` drives every documented flow — greeting, the 8-step
         quiz, face analysis, post-quiz frame carousel, lens recommendation,
         fit-score, add-to-cart. This guarantees the user sees the listed
         questions in order rather than Gemini improvising its own.
      2. Only when the scripted agent marks its answer as `_fallback=True`
         (i.e. no scripted path matched — typically free-form Q&A after the
         quiz is complete) do we hand the turn to Gemini for a conversational
         reply. Gemini can still invoke tools in that path.

    Returns: {role, text, components}
    """
    session_data = session_data or {}
    scripted = run_demo_agent(messages, session_data)
    if not scripted.get("_fallback"):
        return scripted

    # Off-script turn — let Gemini take over (if we have a key). Strip the
    # internal marker from the scripted fallback before returning it as a
    # last-resort answer when Gemini is unavailable.
    scripted.pop("_fallback", None)
    if not GEMINI_API_KEY:
        return scripted

    # Translate Claire's conversation into Gemini's `contents` shape:
    # {role: "user"|"model", parts: [{text: "..."}]}
    contents = []
    for m in messages:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        text = m.get("content") or m.get("text") or ""
        if isinstance(text, list):
            # Tolerate legacy Anthropic-shaped content blocks
            text = " ".join(b.get("text", "") for b in text if isinstance(b, dict))
        if not text:
            continue
        contents.append({
            "role":  "user" if role == "user" else "model",
            "parts": [{"text": str(text)}],
        })

    if not contents:
        return run_demo_agent(messages, session_data or {})

    for _ in range(8):
        try:
            resp = call_gemini_agent(contents)
        except Exception as exc:
            print(f"  [gemini-agent] call failed: {exc}")
            return scripted

        candidates = resp.get("candidates") or []
        if not candidates:
            break
        content = candidates[0].get("content") or {}
        parts   = content.get("parts") or []

        text_segments, tool_calls = _parse_gemini_parts(parts)

        if not tool_calls:
            full_text = "\n".join(text_segments).strip()
            parsed = parse_response(full_text)
            return {
                "role":       "assistant",
                "text":       parsed["text"],
                "components": parsed["components"],
            }

        # Append the model's tool-call turn verbatim, then run the tools and
        # feed the results back as a `user` turn with functionResponse parts.
        contents.append({"role": "model", "parts": parts})
        resp_parts = []
        for call in tool_calls:
            name = call["name"]
            args = call["args"] if isinstance(call["args"], dict) else {}
            try:
                result = execute_tool(name, args)
            except Exception as exc:
                result = {"success": False, "error": str(exc)}
            resp_parts.append({
                "functionResponse": {
                    "name":     name,
                    "response": result if isinstance(result, dict) else {"result": result},
                }
            })
        contents.append({"role": "user", "parts": resp_parts})

    # Loop exhausted without a clean answer — fall back to the scripted reply.
    return scripted


# ─────────────────────────────────────────────
# DEMO MODE (No API Key)
# ─────────────────────────────────────────────

def run_demo_agent(messages: list, session_data: dict) -> dict:
    """
    Scripted demo agent that simulates the full Claire conversation flow
    without requiring an API key. Uses session_data to track quiz progress.
    """
    quiz_step = session_data.get("quiz_step", 0)
    quiz_answers = session_data.get("quiz_answers", {})
    face_data = session_data.get("face_data", None)
    last_shown = session_data.get("last_shown", None)
    lang = (session_data.get("lang") or "en").lower()
    if lang not in ("en", "hi"):
        lang = "en"

    # Get last user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    last_user = user_messages[-1] if user_messages else {}
    last_text = str(last_user.get("content", last_user.get("text", ""))).lower().strip()

    # ── RESTART INTENT — user wants to start the whole journey over ──
    # e.g. "start over", "वापस शुरू करो", "phir se suru kro"
    from gemini import parse_restart_intent, parse_change_gender
    msg_type = last_user.get("type", "text")
    if msg_type != "greet" and parse_restart_intent(last_text):
        # Wipe session state and send a fresh greeting + Q1
        if session_data is not None:
            session_data["quiz_step"]      = 0
            session_data["quiz_answers"]   = {}
            session_data["active_filters"] = {}
            session_data["face_data"]      = None
            session_data["last_shown"]     = None
        ack_en = "Sure, let's start fresh. "
        ack_hi = "ठीक है, नए से शुरू करते हैं। "
        g = _demo_greeting(lang)
        g["text"] = (ack_hi if lang == "hi" else ack_en) + (g.get("text") or "")
        return g

    # ── GENDER SWITCH MID-JOURNEY — e.g. "show for my wife" ──
    # Update active_filters.gender_pref and re-run the product search with
    # existing tags so the user doesn't have to redo the whole quiz.
    if msg_type != "greet" and msg_type != "quiz_answer":
        new_gender = parse_change_gender(last_text)
        if new_gender:
            tags = dict((session_data or {}).get("active_filters") or {})
            tags["gender_pref"] = new_gender
            if session_data is not None:
                session_data["active_filters"] = tags
            # Reuse the existing filter-search path by calling _llm_pick
            llm_result = _llm_pick(session_data, tags, face_data, chat_hint=last_text, limit=6)
            if llm_result and llm_result.get("products"):
                products  = [_enrich_for_frontend(p) for p in llm_result["products"]]
                label = {"male":"पुरुष","female":"महिला","unisex":"सबके लिए"}.get(new_gender, new_gender)
                text  = (f"समझ गई — अब {label} के लिए फ्रेम दिखा रही हूँ।"
                         if lang == "hi"
                         else f"Got it — switching to frames for {new_gender}.")
                return {"role":"assistant","text":text,
                        "components":[{"type":"carousel","title":"Updated picks","frames":products}]}
        # Ambiguous "for someone else" (no gender specified) → ask for clarity
        _AMBIG_SOMEONE = (
            "someone else", "kisi aur", "किसी और", "किसी और के लिए",
            "for another", "for somebody", "kisi aur ke", "kisi aur ke lia",
        )
        if any(w in last_text for w in _AMBIG_SOMEONE):
            msg = ("बिलकुल — बताइए किसके लिए? पुरुष, महिला, या कोई और?"
                   if lang == "hi"
                   else "Sure — who are we shopping for? A man, a woman, or should I show everything?")
            return {"role":"assistant","text":msg,"components":[]}

    # ── FILTER QUERY — detect explicit filter intent before quiz flow ──
    # e.g. "under 500", "dark color frames", "kids sunglasses", "500 se niche"
    _FILTER_SIGNALS = [
        # Budget / price
        "under", "below", "above", "budget", "niche", "kam", "sasta", "mehnga",
        "500", "1000", "1500", "2000", "2500", "3000", "3500",
        "luxury", "premium", "affordable", "cheap",
        # Colour / style / shape
        "dark", "light", "rang", "color", "colour", "classic", "trendy",
        "stylish", "fashion", "bold", "minimal",
        "gold", "silver", "black", "tortoise", "blue", "red", "white", "green",
        "brown", "pink", "grey", "navy", "rose",
        "aviator", "wayfarer", "round", "square", "rectangular", "rimless",
        "cat-eye", "butterfly", "oval", "shape", "style",
        # Product type / audience
        "sunglass", "goggle", "contact", "chashma", "specs", "eyeglass",
        "lens", "frame",
        "sports", "office", "kids", "bachon",
        "women", "men", "ladies", "gents",
        # Rating / bestseller / newness / sort intents
        "rating", "stars", "star", "highly", "top", "rated",
        "bestseller", "best-seller", "popular",
        "new arrival", "latest", "new",
        "trending", "hot",
        "sort", "order", "ascending", "descending", "cheapest", "costliest",
        "low to high", "high to low", "highest",
        # Remove / reset intents
        "remove", "drop", "clear", "cancel", "reset", "without",
        "हटा", "निकाल", "साफ", "बिना", "रेटिंग", "बेस्टसेलर",
    ]
    msg_type = last_user.get("type", "text")
    if msg_type not in ("greet", "quiz_answer") and any(w in last_text for w in _FILTER_SIGNALS):
        # Cumulative tags: quiz answers + previously active filters + current text
        new_tags   = _extract_tags(last_text)
        quiz_blob  = " ".join(str(v) for v in (quiz_answers or {}).values() if v)
        quiz_tags  = _extract_tags(quiz_blob) if quiz_blob else {}
        active     = dict(session_data.get("active_filters") or {}) if session_data else {}
        tags = dict(active)
        for k, v in quiz_tags.items():
            if v is not None and not tags.get(k): tags[k] = v
        # ── REMOVE intent: "remove color", "budget हटा दो", "clear all" ──
        from gemini import parse_remove_intent
        remove_keys = parse_remove_intent(last_text)
        if remove_keys == ["__ALL__"]:
            tags = {}
            # A "clear all" message should NOT leak any other tags the heuristic
            # happened to extract from the same phrase (e.g. "clear" as a colour).
            new_tags = {k: None for k in new_tags}
        elif remove_keys:
            for k in remove_keys:
                tags.pop(k, None)

        # Group-aware merge: a new color/budget/vision/type/shape message
        # replaces any older value from the same group first.
        GROUPS = {
            "color":   ("color", "color_type", "specific_color"),
            "budget":  ("budget", "price"),
            "vision":  ("vision_need", "has_power"),
            "type":    ("product_type",),
            "shape":   ("frame_shape_pref",),
            "sort":    ("sort_by",),
            "rating":  ("min_rating",),
        }
        new_non_null = {k: v for k, v in new_tags.items() if v is not None}
        for group_keys in GROUPS.values():
            if any(k in new_non_null for k in group_keys):
                for k in group_keys:
                    tags.pop(k, None)
        for k, v in new_non_null.items():
            tags[k] = v
        non_null_tags = {k: v for k, v in tags.items() if v is not None}

        # If the user just said "clear all" and nothing else, persist empty
        # filters and show a friendly confirmation (no search).
        if remove_keys and not non_null_tags:
            if session_data is not None:
                session_data["active_filters"] = {}
            msg_hi = "सारे फ़िल्टर हटा दिए। अब आप नई शर्त बताइए।"
            msg_en = "All filters cleared. What would you like to see?"
            return {"role": "assistant",
                    "text": msg_hi if lang == "hi" else msg_en,
                    "components": []}
        if non_null_tags:
            # Persist for future turns
            if session_data is not None:
                session_data["active_filters"] = tags

            # LLM-powered pick (preferred)
            llm_result = _llm_pick(session_data, tags, face_data, chat_hint=last_text, limit=6)
            if llm_result and llm_result.get("products"):
                products   = [_enrich_for_frontend(p) for p in llm_result["products"]]
                order = ("budget","product_type","specific_color","color_type","color","trend","lifestyle","vision_need","age_group","gender_pref","frame_shape_pref","min_rating","bestseller_only","trending_only","new_arrival_only","sort_by")
                display_tags = dict(tags)
                if display_tags.get("specific_color"):
                    display_tags.pop("color", None); display_tags.pop("color_type", None)
                desc = " + ".join(_fmt_tag(k, display_tags[k]) for k in order if display_tags.get(k) is not None) or "your criteria"
                # Only surface reasoning when it's actual LLM output, not a fallback note
                reasoning = llm_result.get("reasoning", "") if llm_result.get("source") in ("llm", "rag+llm") else ""
                if lang == "hi":
                    prefix = f"ये रहे आपके लिए बेहतरीन {len(products)} फ्रेम — {desc} के हिसाब से।"
                else:
                    prefix = f"Here are {len(products)} great picks that match {desc}."
                title = "आपके लिए बेस्ट मैच" if lang == "hi" else "Best matches for you"
                text = f"{prefix}\n\n{reasoning}" if reasoning else prefix
                return {
                    "role": "assistant",
                    "text": text,
                    "components": [{"type": "carousel", "title": title, "frames": products}]
                }

            # Fallback: rule-based search
            face_shape = (face_data or {}).get("shape", "").lower() or None
            result = search_products(quiz_tags=tags, face_shape=face_shape)
            if result.get("no_match"):
                msg_hi = "माफ कीजिए, इन फ़िल्टर्स से कोई फ्रेम नहीं मिला। कोई एक शर्त बदलकर कोशिश करें।"
                msg_en = "Sorry, no frames match all those filters. Try loosening one — maybe a different budget or color?"
                return {
                    "role": "assistant",
                    "text": msg_hi if lang == "hi" else msg_en,
                    "components": []
                }
            products = result.get("frames", result.get("products", []))
            if products:
                order = ("budget","product_type","specific_color","color_type","color","trend","lifestyle","vision_need","age_group","gender_pref","frame_shape_pref","min_rating","bestseller_only","trending_only","new_arrival_only","sort_by")
                display_tags = dict(tags)
                if display_tags.get("specific_color"):
                    display_tags.pop("color", None); display_tags.pop("color_type", None)
                filter_desc = [_fmt_tag(k, display_tags[k]) for k in order if display_tags.get(k) is not None]
                desc = " + ".join(filter_desc) if filter_desc else "your criteria"
                if lang == "hi":
                    text = f"{result.get('total_found', len(products))} फ्रेम मिले — {desc} के हिसाब से। ये रहे कुछ बढ़िया विकल्प:"
                    title = "आपके लिए चुने गए"
                else:
                    text = f"Found {result.get('total_found', len(products))} frames matching {desc}. Here are some great options:"
                    title = "Top picks for you"
                return {
                    "role": "assistant",
                    "text": text,
                    "components": [{"type": "carousel", "title": title, "frames": products}]
                }

    # ── GREETING (no prior messages or explicit greeting) ──
    _is_quiz_answer = last_user.get("type") == "quiz_answer"
    if len(user_messages) <= 1 and quiz_step == 0 and not _is_quiz_answer:
        return _demo_greeting(lang)

    # ── PHOTO UPLOADED THIS TURN ──
    # Only fire when the current user message is actually a photo upload.
    # Having face_data in session is NOT a reason to re-show the analysis card —
    # that was the root of the duplicate-response bug when the user's next
    # chat message (e.g. "kisi aur ke liye dikhao") wrongly triggered another
    # face_analysis bubble.
    if last_user.get("type") == "image" or "uploaded" in last_text:
        if face_data and last_shown != "face_analysis":
            # Persist last_shown so subsequent turns don't re-trigger.
            if session_data is not None:
                session_data["last_shown"] = "face_analysis"
            return _demo_face_analysis(face_data)

    # ── LENS INQUIRY ──
    # Skip this shortcut when:
    #   - the user is answering a quiz question (their answer may contain the
    #     literal word "lens" — e.g. the product-type question with answer
    #     "contact lenses"). We don't want to hijack a quiz answer.
    #   - the post-face-analysis mini-quiz is in progress.
    _in_face_miniquiz = last_shown == "face_analysis"
    _is_quiz_ans      = last_user.get("type") == "quiz_answer"
    _in_main_quiz     = 0 < quiz_step < _QUIZ_TOTAL
    if (not _in_face_miniquiz) and (not _is_quiz_ans) and (not _in_main_quiz) \
       and any(w in last_text for w in ["lens", "prescription", "coating", "anti-glare", "blue light"]):
        presc = quiz_answers.get("prescription", "single_vision")
        return _demo_lens_recommendation(presc)

    # ── QUIZ FLOW ──
    if quiz_step < _QUIZ_TOTAL:
        # User answered a quiz step — advance
        return _demo_quiz_step(quiz_step + 1, quiz_answers, lang)

    # ── POST QUIZ: show frames ──
    if quiz_step >= _QUIZ_TOTAL and last_shown != "carousel":
        return _demo_frame_carousel(quiz_answers, face_data, lang)

    # ── FIT SCORE request ──
    if any(w in last_text for w in ["fit", "confidence", "how well", "score", "match"]):
        return _demo_fit_score(quiz_answers, face_data)

    # ── ADD TO CART ──
    if any(w in last_text for w in ["add to cart", "buy", "purchase", "order"]):
        return {
            "role": "assistant",
            "text": "🛒 **Added to cart!** Great choice — you're going to love these!\n\nYour Lenskart cart is ready. Would you like to:\n• Explore more frame options\n• Choose your lens package\n• Proceed to checkout on lenskart.com",
            "components": []
        }

    # ── FALLBACK: general helpful response ──
    return _demo_fallback(last_text, quiz_answers, face_data, lang)


from _quiz_tables import QUIZ_TOTAL as _QUIZ_TOTAL, QUIZ_EN as _QUIZ_EN, QUIZ_HI as _QUIZ_HI

# Greeting is intentionally empty — the frontend speaks a separate intro
# right after the user picks their language, so we don't want the greeting
# bubble to repeat the same welcome. The first quiz question carries all
# the content needed here.
_GREET_TEXT_EN = ""
_GREET_TEXT_HI = ""


def _demo_greeting(lang: str = "en") -> dict:
    q = (_QUIZ_HI if lang == "hi" else _QUIZ_EN)[1]
    return {
        "role": "assistant",
        "text": _GREET_TEXT_HI if lang == "hi" else _GREET_TEXT_EN,
        "components": [{
            "type": "quiz", "question": q["q"], "options": q["opts"],
            "step": 1, "total": _QUIZ_TOTAL, "key": q["key"],
        }]
    }


def _demo_quiz_step(step: int, answers: dict, lang: str = "en") -> dict:
    table = _QUIZ_HI if lang == "hi" else _QUIZ_EN
    # After the final question, trigger the frame search
    if step > _QUIZ_TOTAL:
        return _demo_frame_carousel(answers, None, lang)
    step  = min(max(step, 2), _QUIZ_TOTAL)
    s     = table[step]
    step_data = {
        "text": s["intro"],
        "component": {
            "type": "quiz", "question": s["q"], "options": s["opts"],
            "step": step, "total": _QUIZ_TOTAL, "key": s["key"],
        }
    }
    steps = {step: step_data}

    return {
        "role": "assistant",
        "text": step_data["text"],
        "components": [step_data["component"]]
    }


def _demo_frame_carousel(answers: dict, face_data: dict, lang: str = "en") -> dict:
    """
    Translate free-text or button quiz answers into structured tags via the
    same heuristic used for filter queries, then search the product DB.
    """
    shape = (face_data or {}).get("shape", "").lower() if face_data else None

    # Merge every answer into one text blob; rely on _extract_tags to parse semantically
    blob_parts = [str(v) for v in answers.values() if v]
    blob = " ".join(blob_parts).lower()
    tags = _extract_tags(blob) if blob else {}

    # Helper: only set when current value is falsy (None / "")
    def _set(key, val):
        if val and not tags.get(key):
            tags[key] = val

    # Normalise button-option strings so mapping tables still work
    style_map = {
        "classic & timeless": ("classic", "wayfarer"),
        "bold & trendy":      ("trendy",  "cat-eye"),
        "minimal & clean":    ("minimal", "rimless"),
        "adventurous":        ("bold",    "geometric"),
    }
    sp = (answers.get("style_pref", "") or "").lower()
    for key, (trend, fshape) in style_map.items():
        if key in sp:
            _set("trend", trend)
            _set("frame_shape_pref", fshape)
            break

    color_map = {
        "neutral (black/grey)": "neutral",
        "warm (gold/tortoise)": "warm",
        "cool (blue/silver)":   "cool",
        "statement color":      "statement",
    }
    cp = (answers.get("color_pref", "") or "").lower()
    for key, val in color_map.items():
        if key in cp:
            _set("color", val)
            break

    # Parse budget from the answer text (handles English digits, Hindi words, premium cues)
    bp = (answers.get("budget", "") or "").lower()
    if bp:
        import re as _re
        # Hindi word → digit normalisation (for digits only — "half" modifiers
        # like साढ़े / डेढ़ / ढाई are handled separately below because they
        # modify the following number rather than being digits themselves).
        HINDI_NUMS = {
            "हज़ार":"1000", "हजार":"1000", "thousand":"1000",
            "सौ":"100", "hundred":"100",
            "एक":"1", "दो":"2", "तीन":"3", "चार":"4", "पाँच":"5", "पांच":"5",
            "छह":"6", "सात":"7", "आठ":"8", "नौ":"9",
        }
        bp_norm = bp
        for hi, digit in HINDI_NUMS.items():
            bp_norm = bp_norm.replace(hi, f" {digit} ")

        # "half" modifiers — applied to the *following* number.
        #   "साढ़े 3 हज़ार" → 3500    (3 × 1000 + 500)
        #   "साढ़े 3000"   → 3500    (3000 + 500)
        #   "डेढ़"          → 1500    (standalone, meaning 1.5 thousand)
        #   "ढाई"          → 2500
        nums: list[int] = []
        for m in _re.finditer(r"(?:साढ़े|साढे|सादे|sadhe|saadhe)\s*(\d+(?:\.\d+)?)", bp_norm):
            n = float(m.group(1))
            nums.append(int(n + 500 if n >= 1000 else n * 1000 + 500))
        # Remove matched saadhe-N sequences so the following tokeniser doesn't
        # count "3000" a second time.
        bp_nosaadhe = _re.sub(
            r"(?:साढ़े|साढे|सादे|sadhe|saadhe)\s*\d+(?:\.\d+)?", " ", bp_norm)
        if any(w in bp for w in ("डेढ़", "dedh", "daidh")):
            nums.append(1500)
        if any(w in bp for w in ("ढाई", "dhai", "dhaai")):
            nums.append(2500)

        # Multiply tokens of form "<n> <1000>" → actual thousands, and pick up all numbers
        tokens = _re.findall(r"\d+(?:\.\d+)?", bp_nosaadhe)
        i = 0
        while i < len(tokens):
            n = float(tokens[i])
            if i + 1 < len(tokens) and float(tokens[i+1]) in (100.0, 1000.0):
                n = n * float(tokens[i+1])
                i += 2
            else:
                i += 1
            if 50 <= n <= 10_000_000:
                nums.append(int(n))
        # Fallback: also check for raw multi-digit numbers in the original text
        for m in _re.findall(r"\d[\d,]{2,}", bp):
            try: nums.append(int(m.replace(",", "")))
            except ValueError: pass

        if nums:
            ceiling = max(nums)
            # Detect "above / se jyada / से ज्यादा" as floor indicators
            _floor_cues = ("above", "more than", "over", "at least", "upwards",
                            "se jyada", "se zyada", "se upar",
                            "से ज्यादा", "से ज़्यादा", "से ऊपर", "से अधिक", "से उपर")
            # Detect "around / aaspaas / करीब" as range indicators (±10%).
            _around_cues = ("around", "approximately", "approx", "roughly", "about",
                             "close to", "somewhere around",
                             "aaspaas", "aaspass", "aas paas", "aas-pass", "paas",
                             "karib", "kareeb", "lagbhag", "tqreeban", "takreeban",
                             "के आसपास", "के आस-पास", "के पास", "करीब", "लगभग", "तकरीबन")
            is_around = any(w in bp for w in _around_cues) and ceiling >= 300
            is_floor  = (any(w in bp for w in _floor_cues) or "+" in bp or "3500+" in bp)

            if is_around:
                _set("budget", f"around_{ceiling}")
            elif is_floor:
                _set("budget", f"above_{ceiling}" if ceiling >= 1000 else "above_1000")
            elif "premium" in bp or "प्रीमियम" in bp:
                _set("budget", "above_3000")
            elif ceiling <= 500:   _set("budget", "under_500")
            elif ceiling <= 1000:  _set("budget", "under_1000")
            elif ceiling <= 1500:  _set("budget", "under_1500")
            elif ceiling <= 2000:  _set("budget", "under_2000")
            elif ceiling <= 2500:  _set("budget", "under_2500")
            elif ceiling <= 3500:  _set("budget", "under_3000")
            else:                  _set("budget", "above_3000")

    # Vision need — parse natural descriptions ("trouble seeing close" etc.)
    pr = (answers.get("prescription", "") or "").lower()
    if pr:
        if any(w in pr for w in (
                "zero power", "bina power", "sirf style", "fashion only",
                "ज़ीरो पावर", "जीरो पावर", "बिना पावर", "फैशन के लिए", "सिर्फ फैशन")):
            _set("vision_need", "zero_power")
        elif any(w in pr for w in (
                "progressive", "bifocal", "बाइफोकल", "प्रोग्रेसिव",
                "both near and far", "दूर और पास", "all round", "all-round")):
            _set("vision_need", "progressive")
        elif any(w in pr for w in (
                "computer", "reading", "near", "close",
                "पास", "पढ़न", "कंप्यूटर", "पास का नहीं दिखता")):
            # Computer/reading glasses typically map to single_vision
            _set("vision_need", "single_vision")
        elif any(w in pr for w in (
                "single vision", "single", "minus", "plus", "far",
                "सिंगल विज़न", "पावर वाला", "नंबर वाला", "दूर का नहीं दिखता")):
            _set("vision_need", "single_vision")

    # ── Explicit product-type answer → product_type ──────────────────────────
    pt_raw = (answers.get("product_type", "") or "").lower()
    if pt_raw:
        if any(w in pt_raw for w in ("sunglass", "goggle", "dhoop",
                                      "सनग्लास", "गॉगल", "धूप")):
            _set("product_type", "sunglasses")
        elif any(w in pt_raw for w in ("contact", "lens ", "lenses",
                                        "कॉन्टैक्ट", "कांटेक्ट")):
            _set("product_type", "contact_lens")
        elif any(w in pt_raw for w in ("eyeglass", "specs", "frame", "chashma",
                                        "चश्म", "आईग्लास", "फ्रेम")):
            _set("product_type", "eyeglasses")

    # ── Gender answer → gender_pref ──────────────────────────────────────────
    # Handles natural phrasings: "for me", "उनके लिए (पुरुष)", "this is for her".
    # If the user says "for me" via voice, the frontend attaches a voice_gender
    # hint (from pitch analysis) in answers["gender_voice"]. Use that if present.
    gen_raw = (answers.get("gender", "") or "").lower()
    voice_gender = (answers.get("gender_voice", "") or "").lower()
    if gen_raw:
        is_self = any(w in gen_raw for w in (
            "for me", "myself", "my own", "it's mine", "its mine",
            "मेरे लिए", "अपने लिए", "मेरा", "खुद"))
        said_him = any(w in gen_raw for w in (
            "for him", "for my husband", "for my brother", "for my son", "for my dad",
            "उनके लिए (पुरुष)", "पति", "भाई", "बेटा", "पापा", "पिता"))
        said_her = any(w in gen_raw for w in (
            "for her", "for my wife", "for my sister", "for my daughter", "for my mom",
            "उनके लिए (महिला)", "पत्नी", "बहन", "बेटी", "माँ", "मम्मी"))
        said_everyone = any(w in gen_raw for w in (
            "for everyone", "everyone", "unisex", "both",
            "सबके लिए", "सब के लिए", "कोई भी"))

        if is_self and voice_gender in ("male", "female"):
            _set("gender_pref", voice_gender)
        elif said_him or any(w in gen_raw for w in ("male", "men", "पुरुष", "आदमी", "लड़का", "boys")):
            _set("gender_pref", "male")
        elif said_her or any(w in gen_raw for w in ("female", "women", "ladies", "महिला", "लेडीज", "औरत", "लड़की", "girls")):
            _set("gender_pref", "female")
        elif said_everyone:
            _set("gender_pref", "unisex")
        elif is_self:
            # Typed "for me" without voice-pitch info → default to male per prod spec.
            _set("gender_pref", "male")

    # ── Age answer → age_group ──────────────────────────────────────────────
    # Now expects a NUMERIC age. Extract the number and bucket internally.
    age_raw = (answers.get("age", "") or "").lower()
    if age_raw:
        import re as _ra
        # Hindi number words for ages
        HI_NUMS = {"तेरह":"13","बीस":"20","पच्चीस":"25","तीस":"30","पैंतीस":"35",
                   "चालीस":"40","पैंतालिस":"45","पचास":"50","साठ":"60","सत्तर":"70"}
        blob = age_raw
        for hi, n in HI_NUMS.items():
            blob = blob.replace(hi, f" {n} ")
        nums = [int(x) for x in _ra.findall(r"\d{1,3}", blob) if 1 <= int(x) <= 110]
        if nums:
            age_years = max(nums)
            if   age_years < 13: _set("age_group", "kids")
            elif age_years < 50: _set("age_group", "adult")
            else:                _set("age_group", "aged")
        else:
            # Fallback to keyword matching if no number was given
            if   any(w in age_raw for w in ("kid", "child", "बच्च", "13 से कम", "under 13")):
                _set("age_group", "kids")
            elif any(w in age_raw for w in ("senior", "50+", "50 या", "elderly", "सीनियर", "बुज़ुर्ग", "बड़े")):
                _set("age_group", "aged")
            elif any(w in age_raw for w in ("teen", "young", "टीन", "यंग", "adult", "एडल्ट")):
                _set("age_group", "adult")

    # Ensure all tag keys are present (None for missing)
    for k in ("price","lifestyle","trend","color","color_type","budget",
              "vision_need","product_type","age_group","gender_pref","frame_shape_pref",
              "specific_color"):
        tags.setdefault(k, None)

    # ── LLM-powered recommendation (preferred) ──
    session_ctx = {"quiz_answers": answers}
    llm_result = _llm_pick(session_ctx, tags, face_data, chat_hint="", limit=6)
    if llm_result and llm_result.get("products"):
        products  = [_enrich_for_frontend(p) for p in llm_result["products"]]
        # Only surface reasoning when it's actual LLM output, not a fallback note
        reasoning = llm_result.get("reasoning", "") if llm_result.get("source") in ("llm", "rag+llm") else ""
        if lang == "hi":
            intro = f"बहुत बढ़िया! आपके जवाबों के हिसाब से मैंने {len(products)} फ्रेम चुने हैं"
            if face_data:
                intro += f" (आपके {face_data.get('shape')} चेहरे के लिए)"
            intro += "।"
            tail  = "\n\nकिसी भी फ्रेम पर टैप करें — डिटेल्स और फिट स्कोर देखें।"
            title = "आपके लिए चुने गए"
        else:
            intro = f"Great — I've picked {len(products)} frames based on your answers"
            if face_data:
                intro += f" (for your {face_data.get('shape')} face shape)"
            intro += "."
            tail  = "\n\nTap any frame to see details and your personal fit score."
            title = "Curated just for you"
        text = f"{intro}\n\n{reasoning}" if reasoning else intro
        return {
            "role": "assistant",
            "text": text + tail,
            "components": [{"type": "carousel", "title": title, "frames": products}]
        }

    # ── Fallback: rule-based search ──
    result   = search_products(quiz_tags=tags, face_shape=shape)
    products = result.get("frames") or result.get("products") or []
    if not products:
        msg = ("माफ कीजिए, आपके जवाबों से कोई फ्रेम नहीं मिला। कोई एक शर्त बदलकर देखें।"
               if lang == "hi"
               else "Hmm, no frames matched all your answers. Try loosening one — budget or color maybe?")
        return {"role": "assistant", "text": msg, "components": []}
    if lang == "hi":
        text  = f"ये रहे आपके जवाबों के हिसाब से {len(products)} फ्रेम!\n\nकिसी भी फ्रेम पर टैप करें।"
        title = "आपके लिए चुने गए"
    else:
        text  = f"Here are {len(products)} frames based on your answers.\n\nTap any frame for details."
        title = "Curated just for you"
    return {
        "role": "assistant",
        "text": text,
        "components": [{"type": "carousel", "title": title, "frames": products}]
    }


def _demo_face_analysis(face_data: dict) -> dict:
    shape = face_data.get("shape", "Oval")
    results = search_frames(shape=shape.lower(), limit=6)

    return {
        "role": "assistant",
        "text": f"✨ Face analysis complete! You have a **{shape}** face shape — here's what that means for your frames:",
        "components": [
            {
                "type": "face_analysis",
                "shape": face_data.get("shape", "Oval"),
                "description": face_data.get("description", ""),
                "face_width": face_data.get("face_width", "Medium"),
                "recommended_styles": face_data.get("recommended_styles", []),
                "key_feature": face_data.get("key_feature", ""),
                "celebrity_match": face_data.get("celebrity_match", ""),
                "confidence": face_data.get("confidence", 92)
            },
            {
                "type": "carousel",
                "title": f"Best Frames for {shape} Face Shape",
                "frames": results["frames"]
            }
        ]
    }


def _demo_lens_recommendation(prescription_type: str = "single_vision") -> dict:
    from tools import get_lens_recommendation
    result = get_lens_recommendation(prescription_type, screen_time_hours=7)
    return {
        "role": "assistant",
        "text": "Great question! Here are the lens packages I'd recommend based on your vision needs and lifestyle:\n\nEach package includes professional fitting at any Lenskart store. 👓",
        "components": [
            {
                "type": "lens_rec",
                "packages": result["packages"],
                "reasoning": result["reasoning"]
            }
        ]
    }


def _demo_fit_score(answers: dict, face_data: dict) -> dict:
    from tools import calculate_fit_confidence
    frame_id = "LK-HC-001"  # Default to bestseller
    shape = (face_data or {}).get("shape", "Oval")
    width = (face_data or {}).get("face_width", "Medium")

    result = calculate_fit_confidence(frame_id, face_shape=shape, face_width=width)

    return {
        "role": "assistant",
        "text": f"Here's your personalized fit analysis for the **{result['frame_name']}**:",
        "components": [
            {
                "type": "fit_score",
                "frame_id": result["frame_id"],
                "frame_name": result["frame_name"],
                "score": result["score"],
                "verdict": result["verdict"],
                "reasons": result["reasons"]
            }
        ]
    }


def _demo_fallback(text: str, answers: dict, face_data: dict, lang: str = "en") -> dict:
    if lang == "hi":
        msg = ("मैं आपकी परफेक्ट फ्रेम ढूँढने में मदद करूँगी। "
               "एक selfie भेजिए चेहरे के शेप के लिए, या क्विज़ पूरी करें।")
    else:
        msg = ("Happy to help you find the perfect pair. "
               "You can upload a selfie for a face-shape read, or keep going with the quiz.")
    # Marker so run_agent knows this was a "nothing scripted matched" answer
    # and can hand the turn over to Gemini for free-form Q&A.
    return {"role": "assistant", "text": msg, "components": [], "_fallback": True}
