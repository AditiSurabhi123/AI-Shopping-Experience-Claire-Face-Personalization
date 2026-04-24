#!/usr/bin/env python3
"""
One-shot catalog retagger — fixes tag quality by actually LOOKING at each
product image via Gemini Vision, then rewrites `data/products.db.json` +
`data/products.csv` + invalidates `data/rag.index.json`.

What it corrects:
  1. `color` / `color_hex` / `tags` color-family — the keyword matcher in
     build_catalog.py often picked the wrong colour (e.g. sunglasses titled
     "JJ Tints Sunglasses Green …" were tagged grey because "Green" came
     after "Grey" in the description).
  2. `frame_shape` — sometimes mis-detected from ambiguous titles.
  3. `face_shape_recommendation` — derived from the real frame silhouette
     rather than a static shape→faces lookup table.
  4. Contact-lens power — removes `zero_power` from contact lenses whose
     photo shows a clear / colourless lens (those are prescription lenses,
     not cosmetic colour contacts), and replaces it with `single_vision`.

The script is **resumable**: each product's vision response is appended to
`data/retag_results.jsonl`, so re-running picks up where it left off. Pass
`--fresh` to start from scratch.

Usage:
    python3 backend/retag_catalog.py                  # retag all 500
    python3 backend/retag_catalog.py --limit 50       # try it on a slice
    python3 backend/retag_catalog.py --workers 6      # tune concurrency
    python3 backend/retag_catalog.py --dry-run        # don't overwrite files
    python3 backend/retag_catalog.py --embed          # rebuild RAG in-process
"""
import argparse
import base64
import csv
import http.client
import json
import os
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ssl_ctx import SSL_CTX

DATA_DIR       = Path(__file__).parent / "data"
JSON_PATH      = DATA_DIR / "products.db.json"
CSV_PATH       = DATA_DIR / "products.csv"
INDEX_PATH     = DATA_DIR / "rag.index.json"
CACHE_PATH     = DATA_DIR / "retag_results.jsonl"

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "",
)
GEMINI_HOST    = "generativelanguage.googleapis.com"
VISION_MODEL   = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
_FALLBACKS     = ("gemini-2.5-flash", "gemini-flash-latest", "gemini-2.0-flash")

# ─── Prompt templates ────────────────────────────────────────────────────────

_PROMPT_FRAME = """You are a professional eyewear catalog tagger. Look at the product image and return ONLY a compact JSON object describing what you actually SEE:

{
  "color":       "<the dominant VISIBLE frame colour: black|white|grey|brown|blue|green|red|yellow|pink|purple|orange|gold|silver|tortoise|transparent|rose-gold|gunmetal|navy-blue|beige|multi — pick one that best matches>",
  "color_hex":   "<approximate hex colour of the frame, e.g. #1a1a1a>",
  "frame_shape": "<wayfarer|aviator|round|rectangular|square|cat-eye|oval|geometric|rimless|clubmaster|hexagonal|butterfly|sports — pick the shape that best matches>",
  "face_shape_recommendation": ["<pick 2-4 from: oval, round, square, heart, diamond, oblong — the face shapes this frame suits best>"],
  "is_sunglass": <true if lenses are darkly tinted (not clear), else false>
}

Tagging rules:
- `color` must reflect the FRAME itself, not the lens tint.
- If the frame is multi-colour or a gradient, pick the most prominent.
- `face_shape_recommendation` should be genuine matches (not all six).
- Output raw JSON only — no markdown fences, no commentary.
"""

_PROMPT_CONTACT = """You are a contact-lens catalog tagger. Look at this contact-lens product image and return ONLY JSON:

{
  "is_colored":  <true if the lens has a visible colour tint (blue/green/hazel/grey/brown etc.) used for cosmetic/beauty; false if it's clear/transparent>,
  "color":       "<the visible lens tint: clear|blue|green|grey|brown|hazel|violet|turquoise|black — or 'clear' for transparent>",
  "color_hex":   "<approximate hex; use #cfd4dc for clear>"
}

Rules:
- Clear/transparent daily-wear contacts → is_colored=false, color="clear".
- Coloured cosmetic contacts → is_colored=true, color=<tint>.
- Output raw JSON only — no markdown fences, no commentary.
"""


# ─── Image fetch ─────────────────────────────────────────────────────────────

def _fetch_image(url: str, timeout: int = 8) -> tuple[str, str] | None:
    """Download an image, return (base64_payload, mime). None on failure."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return None
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(parsed.hostname, timeout=timeout,
                    context=SSL_CTX if parsed.scheme == "https" else None)
    try:
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        conn.request("GET", path, headers={"User-Agent": "lenskart-claire-retagger/1.0"})
        resp = conn.getresponse()
        if resp.status != 200:
            resp.read()  # drain
            return None
        raw = resp.read()
    except Exception:
        return None
    finally:
        conn.close()

    mime = "image/jpeg"
    ct = resp.getheader("Content-Type", "").lower()
    if "png" in ct:  mime = "image/png"
    elif "webp" in ct: mime = "image/webp"
    elif "gif" in ct:  mime = "image/gif"
    return base64.b64encode(raw).decode("ascii"), mime


# ─── Gemini vision call ──────────────────────────────────────────────────────

def _call_gemini_vision(prompt: str, image_b64: str, mime: str) -> dict | None:
    payload = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": mime, "data": image_b64}},
            ]
        }],
        "generationConfig": {
            # Gemini 2.5 consumes reasoning tokens before output — 400 was
            # hitting MAX_TOKENS after the first `{`. 2048 gives plenty of
            # headroom for both reasoning + the 5-field JSON response.
            "maxOutputTokens": 2048,
            "temperature":     0.1,
            "responseMimeType": "application/json",
            # Zero-reasoning mode: we want pure extraction, not chain-of-thought.
            "thinkingConfig":  {"thinkingBudget": 0},
        },
    })

    tried: list = []
    last_status, last_raw = 0, ""
    for model in [VISION_MODEL, *_FALLBACKS]:
        if model in tried:
            continue
        tried.append(model)
        path = f"/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(GEMINI_HOST, timeout=30, context=SSL_CTX)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8")
            status = resp.status
        except Exception as exc:
            last_status, last_raw = 0, str(exc)
            continue
        finally:
            conn.close()
        if status == 200:
            try:
                data  = json.loads(raw)
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text  = (parts[0].get("text") if parts else "") or ""
                text  = text.strip().lstrip("`").rstrip("`").strip()
                if text.startswith("json"): text = text[4:].strip()
                return json.loads(text)
            except Exception as exc:
                if os.environ.get("RETAG_DEBUG"):
                    print(f"    [parse-fail] model={model} exc={exc} raw={raw[:300]}")
                return None
        last_status, last_raw = status, raw
        if os.environ.get("RETAG_DEBUG"):
            print(f"    [vision-{status}] model={model} raw={raw[:200]}")
        if status == 404 or "NOT_FOUND" in raw: continue
        # transient 429/5xx → back off briefly
        if status in (429, 500, 502, 503, 504):
            time.sleep(1.2)
            continue
        return None

    return None


# ─── Tag helpers ─────────────────────────────────────────────────────────────

_COLOR_TO_FAMILY = {
    # color → list of tag families it should contribute
    "black":        ["dark", "neutral"],
    "matte-black":  ["dark", "neutral"],
    "glossy-black": ["dark", "neutral"],
    "white":        ["light"],
    "grey":         ["cool"],
    "silver":       ["light", "cool"],
    "gold":         ["warm"],
    "rose-gold":    ["light", "warm"],
    "brown":        ["warm"],
    "tortoise":     ["warm"],
    "beige":        ["warm", "light"],
    "blue":         ["cool"],
    "navy-blue":    ["cool", "dark"],
    "gunmetal":     ["dark", "cool"],
    "green":        ["statement"],
    "red":          ["statement", "bold"],
    "pink":         ["statement"],
    "purple":       ["statement"],
    "yellow":       ["statement", "bold"],
    "orange":       ["statement", "bold"],
    "transparent":  ["light"],
    "clear":        ["light"],
    "multi":        ["statement"],
}

# Tags we remove before recomputing color families so old wrong tags don't
# linger. Budget / age / power / shape tags stay.
_COLOR_FAMILY_TAGS_TO_DROP = {
    "dark", "light", "warm", "cool", "neutral", "statement", "bold",
}

_SHAPE_FACE_FIT_FALLBACK = {
    "wayfarer":    ["oval", "round", "heart", "diamond"],
    "aviator":     ["oval", "square", "heart"],
    "round":       ["square", "oblong", "heart"],
    "rectangular": ["round", "oval", "heart"],
    "square":      ["round", "oval"],
    "oval":        ["square", "oblong", "diamond"],
    "cat-eye":     ["round", "oval", "heart", "square"],
    "geometric":   ["round", "oval", "heart"],
    "rimless":     ["oval", "square", "heart", "round", "diamond", "oblong"],
    "clubmaster":  ["oval", "square", "heart"],
    "hexagonal":   ["round", "oval", "oblong"],
    "butterfly":   ["round", "heart", "diamond"],
    "sports":      ["oval", "oblong", "square"],
}


def _apply_frame_vision(p: dict, v: dict) -> dict:
    """Update a product dict with frame-vision results."""
    # Color + hex
    c = (v.get("color") or "").strip().lower()
    if c:
        p["color"] = c
    if v.get("color_hex"):
        p["color_hex"] = v["color_hex"]

    # Frame shape
    fs = (v.get("frame_shape") or "").strip().lower()
    if fs and fs != "none":
        p["frame_shape"] = fs
        p["style"]       = fs  # legacy alias kept in sync

    # Face-shape recommendation
    recs = v.get("face_shape_recommendation") or []
    recs = [s.strip().lower() for s in recs if isinstance(s, str)]
    recs = [s for s in recs if s in {"oval", "round", "square", "heart", "diamond", "oblong"}]
    if recs:
        p["face_shape_recommendation"] = recs
    elif p.get("frame_shape") in _SHAPE_FACE_FIT_FALLBACK:
        p["face_shape_recommendation"] = _SHAPE_FACE_FIT_FALLBACK[p["frame_shape"]]

    # If the vision says it's a sunglass but we have it as eyeglasses, leave the
    # type alone — type is product-feed authoritative. But record a shade flag.
    if v.get("is_sunglass") is True and p.get("type") == "sunglasses":
        p["shade"] = "tinted"

    # Re-compute colour-family tags
    tags = [t for t in (p.get("tags") or []) if t not in _COLOR_FAMILY_TAGS_TO_DROP]
    for fam in _COLOR_TO_FAMILY.get(p["color"], []):
        if fam not in tags:
            tags.append(fam)
    p["tags"] = tags
    return p


def _apply_contact_vision(p: dict, v: dict) -> dict:
    """Update a contact-lens product with contact-vision results."""
    is_colored = bool(v.get("is_colored"))
    colour = (v.get("color") or "").strip().lower() or ("clear" if not is_colored else "blue")
    p["color"] = "transparent" if colour == "clear" else colour
    if v.get("color_hex"):
        p["color_hex"] = v["color_hex"]

    # Power rule: clear (non-coloured) contacts are prescription single-vision.
    # Coloured/cosmetic contacts stay as whatever the feed said, minus any
    # wrongly-applied zero_power if we can't confirm it.
    tags = [t for t in (p.get("tags") or []) if t not in _COLOR_FAMILY_TAGS_TO_DROP]
    if not is_colored:
        tags = [t for t in tags if t != "zero_power"]
        if "single_vision" not in tags:
            tags.append("single_vision")
    # Add colour-family tags for the tint
    for fam in _COLOR_TO_FAMILY.get(p["color"], []):
        if fam not in tags:
            tags.append(fam)
    p["tags"] = tags
    return p


# ─── Cache (resumability) ────────────────────────────────────────────────────

def _read_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    out = {}
    with open(CACHE_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                out[row["id"]] = row
            except Exception:
                pass
    return out


def _append_cache(row: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ─── Per-product worker ──────────────────────────────────────────────────────

def _retag_one(p: dict) -> dict:
    pid  = p["id"]
    urls = p.get("image_urls") or []
    if not urls:
        return {"id": pid, "ok": False, "reason": "no_image"}

    url = urls[0]
    img = _fetch_image(url)
    if not img:
        return {"id": pid, "ok": False, "reason": "fetch_failed", "url": url}
    b64, mime = img

    prompt = _PROMPT_CONTACT if p.get("type") == "contact_lens" else _PROMPT_FRAME
    vision = _call_gemini_vision(prompt, b64, mime)
    if not vision:
        return {"id": pid, "ok": False, "reason": "vision_failed"}

    return {"id": pid, "ok": True, "vision": vision, "type": p.get("type")}


# ─── CSV writer (same schema as build_catalog.py) ────────────────────────────

_CSV_COLUMNS = [
    "id", "sku", "name", "type", "gender", "age", "color", "color_hex",
    "powers", "quantity", "price", "strikeout_price", "tags",
    "frame_shape", "face_shape_recommendation", "shade", "image_urls",
    "avg_rating", "brand", "material", "style", "description", "product_url",
]


def _write_csv(products: list, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLUMNS)
        for p in products:
            row = []
            for col in _CSV_COLUMNS:
                v = p.get(col, "")
                if isinstance(v, list):
                    v = "|".join(str(x) for x in v)
                row.append(v)
            w.writerow(row)


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",   type=int, default=0,
                    help="Only retag the first N products (0 = all).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel vision calls (default 4).")
    ap.add_argument("--fresh",   action="store_true",
                    help="Delete the checkpoint cache before running.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't overwrite CSV/JSON/index.")
    ap.add_argument("--embed",   action="store_true",
                    help="Also rebuild the RAG embedding index at the end.")
    args = ap.parse_args()

    if not JSON_PATH.exists():
        print(f"  [retag] missing {JSON_PATH} — run build_catalog.py first", file=sys.stderr)
        return 1
    products = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    if args.fresh and CACHE_PATH.exists():
        CACHE_PATH.unlink()
        print(f"  [retag] cleared checkpoint cache")

    done = _read_cache()
    print(f"  [retag] cached results: {len(done)} / {len(products)}")

    todo = [p for p in products if p["id"] not in done]
    if args.limit > 0:
        todo = todo[: args.limit]
    print(f"  [retag] scheduling vision for {len(todo)} products "
          f"(workers={args.workers})")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_retag_one, p): p["id"] for p in todo}
        completed = 0
        for fut in as_completed(futures):
            res = fut.result()
            _append_cache(res)
            done[res["id"]] = res
            completed += 1
            if completed % 10 == 0 or completed == len(todo):
                ok = sum(1 for r in done.values() if r.get("ok"))
                eta = (time.time() - t0) / completed * (len(todo) - completed)
                print(f"  [retag] {completed}/{len(todo)} "
                      f"(ok so far: {ok}) — eta {eta:0.0f}s")

    # Merge vision results back into products
    updated = 0
    cleared_zero = 0
    colour_fixed = 0
    for p in products:
        r = done.get(p["id"])
        if not r or not r.get("ok"):
            continue
        v = r["vision"]
        prev_color = p.get("color")
        prev_tags  = list(p.get("tags") or [])
        if r.get("type") == "contact_lens":
            _apply_contact_vision(p, v)
        else:
            _apply_frame_vision(p, v)
        if p.get("color") != prev_color:
            colour_fixed += 1
        if "zero_power" in prev_tags and "zero_power" not in p.get("tags", []):
            cleared_zero += 1
        updated += 1

    print(f"  [retag] merged {updated} products — "
          f"colour corrected: {colour_fixed}, zero_power cleared: {cleared_zero}")

    if args.dry_run:
        print(f"  [retag] --dry-run: not writing products.json/csv/rag.index")
        return 0

    # Persist
    JSON_PATH.write_text(json.dumps(products, ensure_ascii=False), encoding="utf-8")
    _write_csv(products, CSV_PATH)
    print(f"  [retag] wrote {JSON_PATH.name} + {CSV_PATH.name}")

    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
        print(f"  [retag] invalidated {INDEX_PATH.name} — RAG rebuilds on next boot")

    if args.embed:
        try:
            from rag import get_index
            n = get_index().rebuild()
            print(f"  [retag] rebuilt RAG embeddings for {n} products")
        except Exception as exc:
            print(f"  [retag] embedding rebuild failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
