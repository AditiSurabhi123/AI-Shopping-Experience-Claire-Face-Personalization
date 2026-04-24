#!/usr/bin/env python3
"""
Lenskart Claire AI — RAG-Powered Recommender

Flow:
  1. Build a natural-language query from the user's quiz answers + filters + chat.
  2. Use the RAG index (embedding cosine similarity, built at boot) to retrieve
     top-K semantically similar products, respecting hard constraints.
  3. Optionally ask Gemini to pick the final top-N from those K and write a
     one-line reason per pick. Falls back to the RAG top-N when LLM is down.

Designed for single-user prototype: fast (<500 ms total), cached per context.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from gemini import _call_gemini

# Heavier reasoning — do NOT use gemini-3.x (404 on public API).
# `gemini-2.5-pro` is the working heavier model.
GEMINI_PRO_MODEL = "gemini-3.1-pro-preview"
from rag     import get_index
from product_db import get_db

# ─── In-process cache ─────────────────────────────────────────────────────────
_CACHE: dict = {}
_CACHE_MAX = 32


def _cache_key(ctx: dict, limit: int) -> str:
    payload = {
        "qa":   ctx.get("quiz_answers") or {},
        "af":   ctx.get("active_filters") or {},
        "fd":   (ctx.get("face_data") or {}).get("shape"),
        "chat": ctx.get("chat_hint") or "",
        "lim":  limit,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _cache_put(key: str, value: dict) -> None:
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)), None)
    _CACHE[key] = value


# ─── Build the search query from user context ────────────────────────────────

def _build_query(ctx: dict) -> str:
    """
    Compose a rich natural-language query for semantic retrieval. Translates
    canonical tags into the same phrasing used in _doc_text so the query and
    document embeddings align well.
    """
    af = ctx.get("active_filters") or {}
    parts: list = []

    # Product type
    pt = af.get("product_type")
    if pt:
        parts.append(f"{pt.replace('_',' ')} type frame")

    # Budget → natural-language bucket
    budget = af.get("budget") or ""
    import re as _re
    m = _re.match(r"^(under|above)_(\d+)$", str(budget))
    if m:
        kind, n = m.group(1), int(m.group(2))
        n_int = n
        if kind == "under":
            if   n_int <= 500:  parts.append("budget very affordable under 500 rupees")
            elif n_int <= 1000: parts.append("affordable under 1000 rupees")
            elif n_int <= 2000: parts.append("mid range under 2000 rupees")
            elif n_int <= 3000: parts.append("upper mid range under 3000 rupees")
            else:               parts.append("premium luxury above 3000 rupees")
        else:
            parts.append(f"premium luxury above {n_int} rupees")

    # Color
    sc = af.get("specific_color")
    if sc:
        parts.append(f"{sc} color")
    else:
        if af.get("color_type"): parts.append(f"{af['color_type']} colored")
        if af.get("color"):      parts.append(f"{af['color']} tones")

    # Shape
    if af.get("frame_shape_pref"):
        parts.append(f"{af['frame_shape_pref']} frame shape style")

    # Trend / lifestyle → audience-flavour words
    if af.get("trend"):
        parts.append({
            "classic": "classic timeless professional",
            "trendy":  "trendy modern new arrival",
            "bold":    "bold statement fashion",
            "minimal": "minimal clean lightweight rimless",
        }.get(af["trend"], af["trend"]))
    if af.get("lifestyle"):
        parts.append({
            "professional": "professional office formal work",
            "active":       "active sports outdoor gym",
            "creative":     "creative artistic unique",
            "fashion":      "fashion trendy stylish statement",
        }.get(af["lifestyle"], af["lifestyle"]))

    # Vision / prescription
    vn = af.get("vision_need")
    if vn == "zero_power":
        parts.append("zero power fashion only no prescription")
    elif vn in ("single_vision", "progressive"):
        parts.append("prescription lenses available")

    # Gender / age
    gp = af.get("gender_pref")
    if   gp == "male":   parts.append("for men gents")
    elif gp == "female": parts.append("for women ladies")
    elif gp == "unisex": parts.append("unisex for everyone")

    ag = af.get("age_group")
    if   ag == "kids":  parts.append("for kids children")
    elif ag == "aged":  parts.append("for seniors elderly")
    elif ag == "adult": parts.append("for adults")

    # Face shape from face analysis
    fd = ctx.get("face_data") or {}
    if fd.get("shape"):
        parts.append(f"suits {fd['shape'].lower()} face shape")

    # Chat hint (latest free-text refinement)
    if ctx.get("chat_hint"):
        parts.append(ctx["chat_hint"])

    return ". ".join(parts) or "popular highly rated eyewear frames"


# ─── Hard filter predicate ───────────────────────────────────────────────────

def _make_hard_filter(ctx: dict):
    ptype          = (ctx.get("product_type") or "").lower()
    budget_max     = ctx.get("budget_max")
    budget_min     = ctx.get("budget_min")
    has_power      = ctx.get("has_power")
    age            = (ctx.get("age_group") or "").lower()
    frame_shape    = (ctx.get("frame_shape") or "").lower()
    color_type     = (ctx.get("color_type") or "").lower()
    specific_color = (ctx.get("specific_color") or "").lower()
    gender         = (ctx.get("gender") or "").lower()
    min_rating       = ctx.get("min_rating")
    bestseller_only  = bool(ctx.get("bestseller_only"))
    new_arrival_only = bool(ctx.get("new_arrival_only"))
    trending_only    = bool(ctx.get("trending_only"))
    # Trend is normally a soft signal — but when the customer explicitly asked
    # for "bold" / "adventurous" we don't want the LLM re-ranker to surface
    # classic office frames. So we hard-filter out the OPPOSITE trend when
    # the user's trend is on the extreme end of the scale.
    trend            = (ctx.get("trend") or "").lower()
    # Tag families that each trend should NOT include.
    TREND_EXCLUDE = {
        "bold":    ("classic", "timeless", "minimalist"),
        "trendy":  ("classic", "timeless"),
        "classic": ("bold", "statement"),
        "minimal": ("bold", "statement"),
    }
    # Tag families we REQUIRE (at least one) when the customer asked for it.
    TREND_REQUIRE = {
        "bold":    ("bold", "statement", "modern", "trending"),
        "trendy":  ("trending", "new-arrival", "modern", "bold"),
        "classic": ("classic", "timeless", "heritage"),
        "minimal": ("minimalist", "lightweight", "rimless", "clean"),
    }

    # Map color_type → concrete colors present in catalog
    COLOR_MATCH = {
        "dark":  ("black", "navy", "gunmetal", "matte-black", "glossy-black", "maroon"),
        "light": ("clear", "white", "silver", "rose", "rose-gold", "gold", "transparent"),
        "warm":  ("gold", "tortoise", "brown", "amber", "rose-gold", "warm"),
        "cool":  ("blue", "silver", "grey", "gray", "navy", "teal", "cool"),
    }

    def _fn(p: dict) -> bool:
        if ptype and (p.get("type") or "").lower() != ptype: return False
        price = float(p.get("price") or 0)
        if budget_max is not None and price > budget_max:    return False
        if budget_min is not None and price < budget_min:    return False
        if has_power is True  and not any(pw != 0 for pw in (p.get("powers") or [0])): return False
        if has_power is False and     any(pw != 0 for pw in (p.get("powers") or [0])): return False
        if age in {"adult","kids","aged"} and (p.get("age") or "").lower() != age: return False
        if frame_shape and (p.get("frame_shape") or "").lower() != frame_shape: return False
        if gender and gender != "unisex":
            g = (p.get("gender") or "").lower()
            if g not in (gender, "unisex"): return False
        # Specific color (exact match) takes precedence over broad color_type
        if specific_color:
            col = (p.get("color") or "").lower()
            if specific_color not in col:
                return False
        elif color_type in COLOR_MATCH:
            col = (p.get("color") or "").lower()
            if not any(c in col for c in COLOR_MATCH[color_type]): return False

        # Rating floor
        if min_rating is not None:
            try:
                if float(p.get("avg_rating") or p.get("rating") or 0) < float(min_rating):
                    return False
            except Exception:
                return False

        # Tag-based toggles
        tags_l = [str(t).lower() for t in (p.get("tags") or [])]
        if bestseller_only  and "bestseller"  not in tags_l: return False
        if trending_only    and "trending"    not in tags_l: return False
        if new_arrival_only and "new-arrival" not in tags_l and "new_arrival" not in tags_l: return False

        # Trend alignment — reject products that contradict the stated trend
        # (e.g. "adventurous" / bold user shouldn't get classic office frames).
        if trend in TREND_EXCLUDE:
            bad = TREND_EXCLUDE[trend]
            if any(b in tags_l for b in bad): return False
        if trend in TREND_REQUIRE:
            need = TREND_REQUIRE[trend]
            if not any(n in tags_l for n in need): return False

        return True
    return _fn


# ─── Optional LLM re-rank ────────────────────────────────────────────────────

_RERANK_PROMPT = """You are Claire, Lenskart's expert AI eyewear stylist.

The customer has told you their preferences below. Your job: look at the
candidate products (already hard-filtered to satisfy price/type/gender/age),
and select the TOP {n} that best match the customer's STATED preferences —
lifestyle, style, colour family, frame shape, face shape, trend, vision needs.

Strict rules:
- Respect the stated colour family / specific colour closely.
- Respect the stated frame-shape preference if given.
- Prefer bestsellers and high-rated picks when the fit is equal.
- Do NOT repeat the same product. Do NOT invent IDs.
- Do NOT include items that contradict the preferences.

Customer preferences:
{context}

Candidate products (filtered and ranked by semantic similarity):
{catalog}

Return ONLY valid JSON in this exact shape:
{{
  "picks": [{{"id": "<exact product id from catalog>", "reason": "<one short sentence explaining why it fits this customer>"}}, ... {n} items],
  "overall": "<one warm sentence summarising why these picks suit the customer>"
}}
"""


def _compact(p: dict) -> dict:
    return {
        "id":    p.get("id"),
        "name":  p.get("name"),
        "type":  p.get("type"),
        "price": p.get("price"),
        "color": p.get("color"),
        "shape": p.get("frame_shape"),
        "face":  p.get("face_shape_recommendation") or [],
        "age":   p.get("age"),
        "gender": p.get("gender"),
        "tags":  (p.get("tags") or [])[:8],
        "rating": p.get("avg_rating"),
    }


def _build_context_block(ctx: dict) -> str:
    lines = []
    qa = ctx.get("quiz_answers") or {}
    if qa:
        lines.append("Quiz answers:")
        for k, v in qa.items():
            if v: lines.append(f"  - {k}: {v}")
    af = ctx.get("active_filters") or {}
    if af:
        lines.append("Chat refinements:")
        for k, v in af.items():
            if v: lines.append(f"  - {k}: {v}")
    fd = ctx.get("face_data") or {}
    if fd:
        lines.append(f"Face: shape={fd.get('shape')}, best_styles={fd.get('recommended_styles')}")
    if ctx.get("chat_hint"):
        lines.append(f"Latest message: {ctx['chat_hint']}")
    return "\n".join(lines) or "No explicit preferences."


# ─── Public API ──────────────────────────────────────────────────────────────

def _boost_face_shape(products: list, face_shape: str) -> list:
    """
    Face-scan PLP ordering: push products whose `face_shape_recommendation`
    lists the user's detected face shape to the top. Stable sort so the LLM
    order is preserved among equally-good matches.
    """
    if not face_shape or not products:
        return products
    fs = face_shape.lower()
    def _score(p):
        recs = [str(r).lower() for r in (p.get("face_shape_recommendation")
                                         or p.get("shape_suitability") or [])]
        # Primary: exact match for the user's face shape
        primary = 1 if fs in recs else 0
        # Secondary: any face-shape suggestion at all (small tie-breaker)
        any_rec  = 1 if recs else 0
        return (-primary, -any_rec)
    return sorted(products, key=_score)


def _apply_sort(products: list, sort_by: str) -> list:
    """Sort the final product list per user's sort_by preference."""
    if not sort_by or not products:
        return products
    key = None
    reverse = False
    if sort_by == "rating":
        key = lambda p: float(p.get("avg_rating") or p.get("rating") or 0)
        reverse = True
    elif sort_by == "price_asc":
        key = lambda p: float(p.get("price") or 0)
    elif sort_by == "price_desc":
        key = lambda p: float(p.get("price") or 0); reverse = True
    elif sort_by == "newest":
        # Products tagged 'new-arrival' sort first, else rating desc
        def _k(p):
            tags = [str(t).lower() for t in (p.get("tags") or [])]
            is_new = 1 if ("new-arrival" in tags or "new_arrival" in tags) else 0
            return (-is_new, -float(p.get("avg_rating") or 0))
        return sorted(products, key=_k)
    if key is not None:
        return sorted(products, key=key, reverse=reverse)
    return products


def recommend(user_context: dict, products: list = None, limit: int = 6,
              rerank_with_llm: bool = True) -> dict:
    """
    RAG-powered recommendation.

    Returns {success, source, products, reasoning, total_considered}.
    """
    # ── Cache check ──
    cache_k = _cache_key(user_context, limit)
    if cache_k in _CACHE:
        cached = _CACHE[cache_k]
        by_id  = {p["id"]: p for p in get_db().all_products()}
        restored = [dict(by_id[pid], ai_reason=r) for pid, r in cached["picks"] if pid in by_id]
        if restored:
            return {
                "success": True, "source": "cache",
                "products": restored,
                "reasoning": cached.get("reasoning", ""),
                "total_considered": cached.get("total_considered", len(restored)),
            }

    # ── Step 1: retrieve with RAG ──
    idx    = get_index()
    # Auto-rebuild the vector index if the product catalog has drifted since
    # the last build (new products added, prices changed, tags edited, etc.).
    try: idx.refresh_if_stale()
    except Exception as _exc: print(f"  [rag] refresh_if_stale failed: {_exc}")
    query  = _build_query(user_context)
    # Pull a generous candidate pool so the LLM re-ranker has breathing room.
    hits   = idx.retrieve(query, k=40, hard_filter=_make_hard_filter(user_context))

    if not hits:
        return {
            "success": False, "source": "none",
            "reason":  "No products match your hard constraints (budget/type/age).",
            "products": [], "reasoning": "",
        }

    # If no LLM re-rank, return top-N from RAG directly (<100 ms typical)
    if not rerank_with_llm:
        picks = hits[:limit]
        return {
            "success": True, "source": "rag",
            "products": picks,
            "reasoning": f"Top {len(picks)} by semantic similarity to: {query[:80]}",
            "total_considered": len(hits),
        }

    # ── Step 2: LLM re-ranks the top-20 ──
    catalog = "\n".join(json.dumps(_compact(p)) for p in hits)
    prompt  = _RERANK_PROMPT.format(
        n=limit, context=_build_context_block(user_context), catalog=catalog,
    )

    sort_by   = user_context.get("sort_by")
    face_shape = (user_context.get("face_data") or {}).get("shape", "")

    def _finalise(lst):
        """Always boost face-shape matches first; if user asked for a sort
        order, apply it on top of the face-shape tiers."""
        lst = _boost_face_shape(lst, face_shape) if face_shape else lst
        if sort_by:
            lst = _apply_sort(lst, sort_by)
        return lst

    try:
        raw = _call_gemini(prompt, max_tokens=600, model=GEMINI_PRO_MODEL)
    except Exception as exc:
        print(f"  [recommend] LLM rerank failed: {exc} — returning RAG top-N")
        return {
            "success": True, "source": "rag",
            "products": _finalise(hits[:limit]),
            "reasoning": "Retrieved by semantic similarity (LLM re-rank unavailable).",
            "total_considered": len(hits),
        }

    cleaned = raw.strip()
    if cleaned.startswith("```"): cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):   cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return {
            "success": True, "source": "rag",
            "products": _finalise(hits[:limit]),
            "reasoning": "Retrieved by semantic similarity (LLM JSON parse failed).",
            "total_considered": len(hits),
        }

    by_id = {p["id"]: p for p in hits}
    out   = []
    for item in parsed.get("picks", []):
        pid = item.get("id")
        if pid in by_id:
            prod = dict(by_id[pid])
            prod["ai_reason"] = item.get("reason", "")
            out.append(prod)
        if len(out) >= limit:
            break

    if not out:
        return {
            "success": True, "source": "rag",
            "products": _finalise(hits[:limit]),
            "reasoning": "Retrieved by semantic similarity.",
            "total_considered": len(hits),
        }

    # Apply face-shape boost + user sort on top of the LLM ordering.
    out = _finalise(out)

    _cache_put(cache_k, {
        "picks":     [(p["id"], p.get("ai_reason", "")) for p in out],
        "reasoning": parsed.get("overall", ""),
        "total_considered": len(hits),
    })

    return {
        "success": True,
        "source": "rag+llm",
        "products": out,
        "reasoning": parsed.get("overall", ""),
        "total_considered": len(hits),
    }
