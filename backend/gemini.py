#!/usr/bin/env python3
"""
Lenskart Claire AI — Google Gemini Integration
Real-time quiz response analysis: language detection, translation, tag extraction.
Uses Gemini 2.0 Flash for ultra-fast inference (< 1s typical).
"""
import json
import http.client
import os

# ─── Config ───────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Do NOT change to gemini-3 / gemini-3.1 — those model IDs return 404.
# `gemini-2.5-flash` is the current stable text model on the public API.
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_HOST    = "generativelanguage.googleapis.com"

# ─── Prompt ───────────────────────────────────────────────────────────────────

QUIZ_ANALYSIS_PROMPT = """\
You are a multilingual AI assistant for Lenskart, an Indian eyewear brand.

A customer answered a quiz step about their eyewear preferences. Analyse the
response and extract structured recommendation tags.

QUIZ STEP CONTEXT: {context}
CUSTOMER RESPONSE: {response}

Return ONLY a valid JSON object — no markdown fences, no extra text:
{{
  "original_response": "<exact user response>",
  "detected_language": "<ISO 639-1 code — en/hi/te/ta/bn/mr/gu/kn/ml/pa/other>",
  "language_name": "<human-readable: English / Hindi / Telugu / Tamil …>",
  "english_translation": "<English; same as original if already English>",
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
    "frame_shape_pref": "<wayfarer | aviator | round | rectangular | square | geometric | rimless | clubmaster | oval | cat-eye | butterfly | hexagonal | shield | wrap-around — or null>"
  }},
  "confidence": <integer 0-100>
}}

Semantic extraction rules — interpret liberally:
- PRICE / BUDGET (extract the tightest applicable bucket):
  "500 se niche" / "under 500" / "5 se niche" (₹) → budget:under_500
  "1000 se kam" / "under 1000" / "cheap" / "sasta" → budget:under_1000
  "1500 budget" → budget:under_1500
  "2000 tak" / "do hazaar" → budget:under_2000
  "2500" → budget:under_2500
  "3000 budget" → budget:under_3000
  "3500+" / "premium" / "mehnga chalega" / "best quality" → budget:above_3000 + price:premium
  "affordable" / "budget-friendly" → budget:under_1000 + price:budget

- VISION / PRODUCT TYPE:
  "power wala" / "power chahiye" / "number wali" / "number hai" / "prescription" → vision_need:single_vision
  "zero power" / "bina power" / "sirf style" / "fashion only" → vision_need:zero_power
  "progressive" / "bifocal" / "reading" / "door paas dono" → vision_need:progressive
  "sunglass" / "dhoop wali" / "goggles" / "UV wali" → product_type:sunglasses
  "eyeglasses" / "chashma" / "specs" → product_type:eyeglasses
  "contact" / "lens" / "contact lens" → product_type:contact_lens

- COLOR TYPE:
  "light color" / "halka rang" / "light wali" / "light frames" → color_type:light
  "dark color" / "dark wali" / "gehra rang" / "dark frames" → color_type:dark
  "black" / "grey" / "gunmetal" → color:neutral + color_type:dark
  "gold" / "tortoise" / "brown" / "warm" → color:warm
  "blue" / "silver" / "cool" → color:cool
  "bright" / "bold rang" / "statement" / "colorful" / "rangeen" → color:statement

- LIFESTYLE:
  "office" / "corporate" / "professional" / "formal" → lifestyle:professional
  "sports" / "gym" / "active" / "outdoor" / "khel" → lifestyle:active
  "creative" / "artist" / "design" / "art" → lifestyle:creative
  "fashion" / "trendy" / "stylish" / "swag" → lifestyle:fashion

- TREND:
  "classic" / "timeless" / "traditional" → trend:classic
  "bold" / "statement" / "adventurous" → trend:bold
  "minimal" / "simple" / "clean" / "sadha" → trend:minimal
  "trendy" / "latest" / "modern" / "new" → trend:trendy

- AGE / GENDER:
  "kids" / "bachon ke liye" / "children" / "child" → age_group:kids
  "uncle" / "aunty" / "bade" / "elderly" / "old age" / "senior" → age_group:aged
  "ladies" / "women" / "female" / "girls" → gender_pref:female
  "gents" / "men" / "male" / "boys" / "ladkon ke liye" → gender_pref:male

- FRAME SHAPE:
  "round" / "gol" → frame_shape_pref:round
  "square" / "chaukona" → frame_shape_pref:square
  "cat eye" / "cat-eye" / "bilauti" → frame_shape_pref:cat-eye
  "aviator" / "pilot" → frame_shape_pref:aviator
  "wayfarer" → frame_shape_pref:wayfarer
  "butterfly" → frame_shape_pref:butterfly
  "rimless" / "bina frame" → frame_shape_pref:rimless

Rules:
- Detect Hindi in Roman script (Hinglish) → "hi"
- Extract EVERY applicable tag; use null only if truly undetermined
- Mixed answers (button + free text): merge all clues
- Output raw JSON only — no markdown, no explanation
"""

# ─── Gemini caller ────────────────────────────────────────────────────────────

_TEXT_FALLBACKS = ("gemini-3.1-flash-tts-preview", "gemini-flash-latest", "gemini-2.0-flash")
_PRO_FALLBACKS  = ("gemini-2.5-pro",   "gemini-pro-latest",   "gemini-2.5-flash")


def _call_gemini(prompt: str, max_tokens: int = 1200, model: str = None,
                  json_mode: bool = False) -> str:
    """Try the configured model; on 404 / NOT_FOUND, try fallbacks automatically.
    When the caller expects JSON, set json_mode=True so Gemini emits strict
    application/json (prevents truncated responses)."""
    gen_cfg = {"maxOutputTokens": max_tokens, "temperature": 0.1, "topP": 0.9}
    if json_mode:
        gen_cfg["responseMimeType"] = "application/json"
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": gen_cfg,
    })
    headers = {"Content-Type": "application/json"}
    from ssl_ctx import SSL_CTX

    primary = model or GEMINI_MODEL
    fallbacks = _PRO_FALLBACKS if "pro" in primary.lower() else _TEXT_FALLBACKS
    models_to_try = [primary] + [m for m in fallbacks if m != primary]

    last_raw = ""; last_status = 0
    for m in models_to_try:
        path = f"/v1beta/models/{m}:generateContent?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=15, context=SSL_CTX)
        try:
            conn.request("POST", path, payload, headers)
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8")
        finally:
            conn.close()
        last_raw = raw; last_status = resp.status
        if resp.status == 200:
            data = json.loads(raw)
            cands = data.get("candidates", [])
            if not cands:
                raise RuntimeError(f"Gemini returned no candidates: {raw[:300]}")
            parts = cands[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts).strip()
        if resp.status == 404 or "NOT_FOUND" in raw:
            print(f"  [gemini] model {m} unavailable, trying next")
            continue
        raise RuntimeError(f"Gemini API error {resp.status}: {raw[:400]}")
    raise RuntimeError(f"Gemini: all models failed (last {last_status}: {last_raw[:200]})")


# ─── Public API ───────────────────────────────────────────────────────────────

def analyze_quiz_response(user_response: str, quiz_context: str = "") -> dict:
    """
    Analyse a quiz answer with Gemini 2.0 Flash.

    Accepts mixed answers: button selection, free text, or both combined.
    Returns full tag bundle including extended keys (color_type, product_type,
    age_group, gender_pref, frame_shape_pref, fine-grained budget thresholds).
    """
    if not user_response or not user_response.strip():
        return _fallback_result(user_response or "", "Empty response")

    prompt = QUIZ_ANALYSIS_PROMPT.format(
        context=quiz_context or "General eyewear preference question",
        response=user_response.strip(),
    )

    try:
        raw_text = _call_gemini(prompt, max_tokens=1200, json_mode=True)
    except Exception as exc:
        print(f"  [Gemini] API call failed: {exc}")
        return _fallback_result(user_response, str(exc))

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"  [Gemini] JSON parse error: {exc}\n  Raw: {cleaned[:300]}")
        return _fallback_result(user_response, f"JSON parse error: {exc}")

    result["success"] = True

    tags = result.setdefault("tags", {})
    for key in _ALL_TAG_KEYS:
        tags.setdefault(key, None)

    return result


# ─── Fallback (no network / API error) ───────────────────────────────────────

_ALL_TAG_KEYS = (
    "price", "lifestyle", "trend", "color", "color_type",
    "budget", "vision_need", "product_type", "age_group",
    "gender_pref", "frame_shape_pref", "specific_color",
    # Chat-driven refinements:
    "min_rating", "bestseller_only", "new_arrival_only", "trending_only",
    "sort_by",
)


# Internal key groups used for REMOVE intent. Each target word maps to a list
# of tag keys that should be cleared from active_filters.
_REMOVE_TARGETS = {
    "color":     ("color", "color_type", "specific_color"),
    "colour":    ("color", "color_type", "specific_color"),
    "रंग":       ("color", "color_type", "specific_color"),
    "price":     ("budget", "price"),
    "budget":    ("budget", "price"),
    "बजट":       ("budget", "price"),
    "rating":    ("min_rating",),
    "stars":     ("min_rating",),
    "rate":      ("min_rating",),
    "bestseller":("bestseller_only",),
    "new":       ("new_arrival_only",),
    "trending":  ("trending_only",),
    "shape":     ("frame_shape_pref",),
    "style":     ("trend", "frame_shape_pref"),
    "type":      ("product_type",),
    "gender":    ("gender_pref",),
    "age":       ("age_group",),
    "vision":    ("vision_need",),
    "power":     ("vision_need",),
    "sort":      ("sort_by",),
    "lifestyle": ("lifestyle",),
}

def parse_restart_intent(text: str) -> bool:
    """
    Detect whether the user wants to restart the whole session / start over.
    Examples that return True:
        "start over"           "let's start again"
        "reset"                "begin again"
        "वापस शुरू करो"        "फिर से शुरू"
        "phir se suru kro"     "नया शुरू"
        "new conversation"     "restart"
    """
    t = (text or "").lower().strip()
    if not t:
        return False
    patterns = [
        "start over", "start again", "begin again", "restart",
        "reset session", "reset everything", "new conversation", "new chat",
        "let's start", "lets start", "start fresh", "from scratch",
        "वापस शुरू", "फिर से शुरू", "नया शुरू", "दोबारा शुरू",
        "phir se", "dobara shuru", "suru se", "shuru se",
        "rset", "riset",
    ]
    return any(p in t for p in patterns)


def parse_change_gender(text: str) -> str | None:
    """
    Detect an in-flight gender switch.
    Returns 'male' | 'female' | 'unisex' when an explicit "for my X" type
    phrase is detected, otherwise None.
    Examples:
        "actually show me for my wife"       → female
        "meri wife ke liye dikhao"           → female
        "show for my husband"                → male
        "मेरे पति के लिए दिखाओ"              → male
        "unisex dikhao"                      → unisex
    """
    t = (text or "").lower().strip()
    if not t:
        return None
    # Must be a change-of-audience phrase, not a generic answer.
    trigger = any(w in t for w in (
        "for my", "show for", "dikhao", "दिखाओ", "दिखाइए",
        "instead", "actually", "change to", "switch to",
        "बदल", "के लिए दिखा", "ke liye", "के लिए", "meri ", "mera ",
    ))
    if not trigger:
        return None
    female = any(w in t for w in (
        "wife", "girlfriend", "sister", "mom", "mother", "daughter",
        "aunt", "for her", "women", "female", "lady", "ladies",
        "behen", "behan", "biwi", "patni", "maa", "mummy", "beti",
        "didi", "bua", "mausi", "chachi", "bhabhi",
        "पत्नी", "बहन", "माँ", "माता", "बेटी", "पत्नि", "मम्मी",
        "wife ke", "girlfriend ke", "sister ke", "behen ke", "beti ke",
    ))
    male = any(w in t for w in (
        "husband", "boyfriend", "brother", "dad", "father", "son",
        "uncle", "for him", "men", "male", "gent", "gents",
        "pati", "bhai", "papa", "pita", "beta", "dada", "chacha",
        "पति", "भाई", "पिता", "पापा", "बेटा", "दादा", "चाचा",
        "husband ke", "brother ke", "pati ke", "bhai ke", "beta ke",
    ))
    unisex = any(w in t for w in (
        "unisex", "everyone", "both", "any gender", "सबके लिए", "सब के लिए",
    ))
    if unisex: return "unisex"
    if female: return "female"
    if male:   return "male"
    return None


def parse_remove_intent(text: str) -> list:
    """
    If the user is asking to drop/remove filter(s), return the list of tag
    keys to clear. Empty list otherwise.
    Examples that trigger this:
        "remove the color filter"
        "drop budget"
        "no color"
        "without rating filter"
        "बजट हटा दो"
        "color nikal do"
        "clear all"
    """
    t = (text or "").lower().strip()
    if not t:
        return []

    remove_triggers = [
        "remove", "drop", "clear", "without", "no more", "no ", "cancel",
        "reset", "हटा", "निकाल", "साफ", "बिना",
    ]
    if not any(tr in t for tr in remove_triggers):
        return []

    # "clear all" / "reset all" / "remove everything" → wipe everything
    if any(w in t for w in ("all", "everything", "सब", "सारे")):
        return ["__ALL__"]

    keys = []
    for target, tag_keys in _REMOVE_TARGETS.items():
        if target in t:
            keys.extend(tag_keys)
    # Dedup while preserving order
    seen = set(); out = []
    for k in keys:
        if k not in seen:
            seen.add(k); out.append(k)
    return out


def _fallback_result(original: str, error: str) -> dict:
    return {
        "success":             False,
        "original_response":   original,
        "detected_language":   "en",
        "language_name":       "English",
        "english_translation": original,
        "tags":                _heuristic_tags(original.lower()),
        "confidence":          35,
        "error":               error,
    }


def _heuristic_tags(text: str) -> dict:
    """
    Pure-Python keyword fallback — no external calls.
    Covers common English + Hindi/Hinglish quiz options including the
    new semantic examples: "500 se niche", "power wala", "light color", etc.
    """
    t = text.lower()
    tags = {k: None for k in _ALL_TAG_KEYS}

    # ── Budget / price ────────────────────────────────────────────────────────
    def _has(*words):
        return any(w in t for w in words)

    if _has("500 se niche", "under 500", "below 500", "500 se kam"):
        tags["budget"] = "under_500";  tags["price"] = "budget"
    elif _has("1000 se kam", "under 1000", "under ₹1", "sasta", "1000 ke andar"):
        tags["budget"] = "under_1000"; tags["price"] = "budget"
    elif "cheap" in t and not _has("cheapest", "cheap first", "sort", "order", "first", "low to high"):
        # "cheap" means budget only if the user isn't asking to sort
        tags["budget"] = "under_1000"; tags["price"] = "budget"
    elif _has("under 1500", "1500 budget", "1500 tak", "1500 ke andar"):
        tags["budget"] = "under_1500"; tags["price"] = "budget"
    elif _has("under 2000", "2000 tak", "do hazaar", "2000 ke andar"):
        tags["budget"] = "under_2000"; tags["price"] = "mid-range"
    elif _has("under 2500", "2500 tak", "2500 ke andar"):
        tags["budget"] = "under_2500"; tags["price"] = "mid-range"
    elif _has("under 3000", "3000 tak", "teen hazaar"):
        tags["budget"] = "under_3000"; tags["price"] = "mid-range"
    elif _has("premium", "3500+", "luxury", "mehnga", "above 3000", "best quality", "no budget"):
        tags["budget"] = "above_3000"; tags["price"] = "premium"
    elif _has("affordable", "budget"):
        tags["budget"] = "under_1000"; tags["price"] = "budget"

    # Numeric fallback — preserve the EXACT ceiling user typed (under 300, under 750, etc.)
    # e.g. "under 300" → under_300, "₹2,000 – ₹3,500" → under_3500, "above 3000" → above_3000
    import re as _re

    # Hindi/Hinglish "half" modifiers applied to thousands:
    #   "साढ़े 3000" / "sadhe 3000"   = 3500
    #   "साढ़े 3 हज़ार" / "sadhe 3 k"  = 3500
    #   "डेढ़" / "dedh"                = 1500  (one-and-a-half thousand)
    #   "ढाई" / "dhai"                 = 2500  (two-and-a-half thousand)
    _HALF_BEFORE = _re.compile(
        r"(?:साढ़े|साढे|सादे|sadhe|saadhe|saade)\s*(\d+)\s*(?:हज़ार|हजार|k|thousand)?",
        _re.IGNORECASE,
    )
    half_nums: list[int] = []
    for m in _HALF_BEFORE.finditer(t):
        n = int(m.group(1))
        # "साढ़े 3"  → 3500   (treats "3" as thousands)
        # "साढ़े 3000" → 3500 (already in rupees)
        half_nums.append(n + 500 if n >= 1000 else n * 1000 + 500)
    # Erase matched substrings so the following numeric regex doesn't re-capture them
    t_nums = _HALF_BEFORE.sub(" ", t)
    # Standalone "डेढ़" / "ढाई" → 1500 / 2500 (only when unambiguously about money;
    # cheap gate: require a money-context word nearby)
    _MONEY_CTX = ("रुपये", "rupees", "rs", "₹", "budget", "price", "कीमत",
                  "मूल्य", "पैसे", "हज़ार", "हजार", "k ", "thousand", "tak",
                  "niche", "se kam", "se jyada", "se zyada", "se upar",
                  "से कम", "से ज्यादा", "से ज़्यादा", "से ऊपर")
    _has_money = any(w in t for w in _MONEY_CTX)
    if _has_money and any(w in t for w in ("डेढ़", "dedh", "daidh")):
        half_nums.append(1500)
    if _has_money and any(w in t for w in ("ढाई", "dhai", "dhaai")):
        half_nums.append(2500)

    reg_nums = [int(n.replace(",", "")) for n in _re.findall(r"(\d[\d,]{1,7})", t_nums)]
    nums = half_nums + reg_nums
    # Accept any positive number — the caller will produce "no match" for
    # unrealistically high floors (e.g. "above 350000"), which is the right
    # answer rather than silently dropping the filter.
    nums = [n for n in nums if 50 <= n <= 10_000_000]
    if nums:
        ceiling = max(nums)
        # Detect price floor indicators (English + Hindi/Hinglish).
        _FLOOR_WORDS = (
            "above", "more than", "over", "at least", "upwards",
            "upar", "se upar", "se jyada", "se zyada",
            "से ऊपर", "से ज्यादा", "से ज़्यादा", "से अधिक", "से उपर",
        )
        # Detect "around / approximately" — sets a ±10 percent band instead of
        # a hard ceiling. Handles English + Hindi + Hinglish.
        _AROUND_WORDS = (
            "around", "approximately", "approx", "roughly", "about",
            "near ", "close to", "somewhere around", "give or take",
            "aas paas", "aaspaas", "aaspass", "aas-pass", "paas",
            "karib", "kareeb", "lagbhag",
            "के आसपास", "के आस-पास", "के पास", "करीब", "लगभग", "तकरीबन",
        )
        is_around = any(w in t for w in _AROUND_WORDS) and ceiling >= 300
        is_floor  = (any(w in t for w in _FLOOR_WORDS) or "+" in t) and ceiling >= 1000

        if is_around:
            new_budget = f"around_{ceiling}"
            tags["price"] = "mid-range"
        elif is_floor:
            new_budget = f"above_{ceiling}"
            tags["price"] = "premium"
        else:
            new_budget = f"under_{ceiling}"
            tags["price"] = "budget" if ceiling <= 1500 else "mid-range"
        tags["budget"] = new_budget

    # ── Vision need / product type (English + Hindi Devanagari) ──────────────
    # NOTE: do NOT include bare "फैशन" / "fashion" here — many lifestyle
    # answers contain it (fashion-forward customers, etc.) and it would
    # incorrectly force zero-power. Only match explicit "for fashion only"-
    # style phrases.
    if _has("zero power", "bina power", "sirf style", "fashion only",
            "no power", "no prescription", "without power", "just fashion",
            "ज़ीरो पावर", "जीरो पावर", "बिना पावर", "सिर्फ स्टाइल",
            "सिर्फ फैशन", "सिर्फ़ फैशन", "फैशन के लिए",
            "फैशन ही", "सिर्फ फैशन के लिए"):
        tags["vision_need"] = "zero_power"
    elif _has("power wala", "power chahiye", "number wali", "number hai",
              "single vision", "prescription", "minus", "plus",
              "सिंगल विज़न", "सिंगल विजन", "पावर वाला", "पावर लेंस", "प्रिस्क्रिप्शन", "नंबर वाला"):
        tags["vision_need"] = "single_vision"
    elif _has("progressive", "bifocal", "reading", "door paas",
              "प्रोग्रेसिव", "बाइफोकल", "बायफोकल", "पढ़ने"):
        tags["vision_need"] = "progressive"
    elif _has("not sure", "pata nahi", "unsure", "पता नहीं"):
        tags["vision_need"] = "not_sure"

    if _has("sunglass", "goggle", "dhoop", "uv wali", "shade",
            "सनग्लास", "गॉगल", "धूप", "यूवी"):
        tags["product_type"] = "sunglasses"
    elif _has("contact lens", "contacts ",
              "कॉन्टैक्ट लेंस", "कांटेक्ट"):
        tags["product_type"] = "contact_lens"
    elif _has("chashma", "specs", "eyeglass", "frame",
              "चश्मा", "आईग्लास", "फ्रेम"):
        tags["product_type"] = "eyeglasses"

    # ── Color type (light / dark) ─────────────────────────────────────────────
    if _has("light color", "halka rang", "light wali", "light frame"):
        tags["color_type"] = "light"
    elif _has("dark color", "dark wali", "gehra rang", "dark frame"):
        tags["color_type"] = "dark"

    # ── Specific color (exact match, highest priority) ────────────────────────
    SPECIFIC_COLORS = [
        ("blue",   ["blue", "neela", "neeli"]),
        ("red",    ["red", "laal"]),
        ("black",  ["black", "kala"]),
        ("white",  ["white", "safed"]),
        ("green",  ["green", "hara"]),
        ("gold",   ["gold", "sona", "golden"]),
        ("silver", ["silver", "chandi"]),
        ("tortoise", ["tortoise"]),
        ("brown",  ["brown", "bhura"]),
        ("rose-gold", ["rose gold", "rose-gold", "rosegold", "pink gold"]),
        ("maroon", ["maroon", "wine"]),
        ("grey",   ["grey", "gray", "gunmetal"]),
        ("pink",   ["pink", "gulabi"]),
        ("clear",  ["clear", "transparent"]),
    ]
    for canonical, keywords in SPECIFIC_COLORS:
        if any(k in t for k in keywords):
            tags["specific_color"] = canonical
            break

    # ── Color family (broad tag, English + Hindi) ────────────────────────────
    if _has("neutral", "black", "grey", "gray", "brown", "gunmetal",
            "न्यूट्रल", "काला", "काले", "ग्रे", "भूरा", "ब्राउन"):
        tags["color"] = "neutral"
    elif _has("warm", "gold", "tortoise", "amber",
              "वार्म", "गोल्ड", "सोना", "सुनहरा", "टॉर्टॉइज़"):
        tags["color"] = "warm"
    elif _has("cool", "blue", "silver",
              "कूल", "नीला", "नीले", "सिल्वर", "चांदी"):
        tags["color"] = "cool"
    elif _has("bright", "colorful", "statement", "bold rang", "rang", "rangeen",
              "बोल्ड रंग", "रंगीन", "चमकीला"):
        tags["color"] = "statement"

    # ── Lifestyle (English + Devanagari) ─────────────────────────────────────
    if _has("office", "professional", "corporate", "work", "business", "formal",
            "ऑफिस", "प्रोफेशनल", "कॉर्पोरेट", "ऑफ़िस", "दफ़्तर", "नौकरी"):
        tags["lifestyle"] = "professional"
    elif _has("creative", "artist", "design", "art",
              "क्रिएटिव", "आर्टिस्ट", "कला", "डिज़ाइन"):
        tags["lifestyle"] = "creative"
    elif _has("active", "sport", "outdoor", "gym", "fitness", "sporty", "khel",
              "एक्टिव", "स्पोर्ट्स", "खेल", "आउटडोर", "जिम"):
        tags["lifestyle"] = "active"
    elif _has("fashion", "trendy", "stylish", "forward", "swag",
              "फैशन", "स्टाइलिश", "फॉरवर्ड"):
        tags["lifestyle"] = "fashion"

    # ── Trend ─────────────────────────────────────────────────────────────────
    if _has("classic", "timeless", "traditional",
            "क्लासिक", "टाइमलेस", "पुराने", "ट्रेडिशनल"):
        tags["trend"] = "classic"
    elif _has("bold", "statement", "adventurous",
              "बोल्ड", "एडवेंचरस", "स्टेटमेंट"):
        tags["trend"] = "bold"
    elif _has("minimal", "clean", "simple", "sadha",
              "मिनिमल", "क्लीन", "सादा", "सिंपल"):
        tags["trend"] = "minimal"
    elif _has("trendy", "trending", "modern", "latest", "new",
              "ट्रेंडी", "ट्रेंडिंग", "मॉडर्न", "लेटेस्ट", "नया"):
        tags["trend"] = "trendy"

    # ── Age group ─────────────────────────────────────────────────────────────
    if _has("kids", "child", "bachon", "children", "bachche",
            "बच्च", "किड्स"):
        tags["age_group"] = "kids"
    elif _has("elderly", "senior", "old age", "uncle", "aunty", "bade", "aged", "50+",
              "सीनियर", "बुज़ुर्ग", "बड़े"):
        tags["age_group"] = "aged"
    elif _has("adult", "एडल्ट", "teen", "young adult", "टीन", "यंग"):
        tags["age_group"] = "adult"

    # ── Gender ────────────────────────────────────────────────────────────────
    if _has("ladies", "women", "female", "girls", "ladki",
            "महिला", "लेडी", "औरत", "लड़की", "स्त्री"):
        tags["gender_pref"] = "female"
    elif _has("gents", "men", "male", "boys", "ladka", "ladkon",
              "पुरुष", "आदमी", "लड़का", "जेंट्स", "मर्द"):
        tags["gender_pref"] = "male"
    elif _has("unisex", "everyone", "both", "सब के लिए", "सबके लिए", "दोनों"):
        tags["gender_pref"] = "unisex"

    # ── Frame shape preference ────────────────────────────────────────────────
    shape_map = {
        "round": ["round", "gol"],
        "square": ["square", "chaukona"],
        "cat-eye": ["cat eye", "cat-eye", "bilauti"],
        "aviator": ["aviator", "pilot"],
        "wayfarer": ["wayfarer"],
        "rimless": ["rimless", "bina frame"],
        "butterfly": ["butterfly"],
        "oval": ["oval"],
        "rectangular": ["rectangular", "rect"],
    }
    for shape, kws in shape_map.items():
        if _has(*kws):
            tags["frame_shape_pref"] = shape
            break

    # ── Minimum rating ("4+ rating", "above 4.5 stars", "अच्छी रेटिंग") ──
    import re as _rr
    rating_ctx = _rr.search(
        r"(rating|stars|star|रेटिंग)",
        t, flags=_rr.IGNORECASE,
    )
    rating_num = None
    if rating_ctx or "+" in t or "above 4" in t or "4 से ऊपर" in t:
        m = _rr.search(r"(\d(?:\.\d)?)\s*(?:\+|or\s*(?:more|above|up|higher)|से\s*ऊपर)?",
                       t.split("rating")[0] + " " + t.split("rating")[-1]
                       if "rating" in t else t)
        if m:
            try:
                n = float(m.group(1))
                if 1.0 <= n <= 5.0:
                    rating_num = n
            except Exception:
                pass
    if rating_num:
        tags["min_rating"] = rating_num
    elif any(w in t for w in ("highly rated", "top rated", "best rated",
                              "अच्छी रेटिंग", "हाई रेटिंग")):
        tags["min_rating"] = 4.0

    # ── Special flags ─────────────────────────────────────────────────────────
    if any(w in t for w in ("bestseller", "best seller", "best-seller",
                             "popular", "most loved", "बेस्टसेलर", "पॉपुलर")):
        tags["bestseller_only"] = True
    if any(w in t for w in ("new arrival", "latest", "brand new",
                             "नई", "नए", "नया आया", "न्यू अराइवल")):
        tags["new_arrival_only"] = True
    if any(w in t for w in ("trending", "in trend", "hot", "ट्रेंडिंग")):
        tags["trending_only"] = True

    # ── Sort intent ───────────────────────────────────────────────────────────
    # "sort by rating", "highest rated first", "low to high price", "cheapest"
    if _has("highest rated", "best rated", "by rating", "rating first",
            "top rated first", "रेटिंग से", "रेटिंग के हिसाब"):
        tags["sort_by"] = "rating"
    elif _has("cheapest", "low to high", "price low", "lowest price",
              "cheap first", "सस्ते पहले", "कम कीमत"):
        tags["sort_by"] = "price_asc"
    elif _has("expensive first", "high to low", "price high", "costliest",
              "महंगे पहले"):
        tags["sort_by"] = "price_desc"
    elif _has("newest first", "latest first", "नए पहले"):
        tags["sort_by"] = "newest"

    return tags
