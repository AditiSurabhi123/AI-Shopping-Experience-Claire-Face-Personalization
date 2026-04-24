#!/usr/bin/env python3
"""
Catalog builder — converts `data/raw.tsv` (Lenskart product feed) into:
  - `data/products.csv`        (flat CSV consumed by product_db.py)
  - `data/products.db.json`    (parsed list consumed by product_db.py)

Tagging rules (keyword-based, deterministic — no LLM call per row):
  - age     ← derived from size (kids buckets) / "Kids"/"Junior" in title
  - gender  ← Lenskart feed gender (Unisex/Male/Female)
  - style   ← classic | trendy | bold | minimalist (from shape + colour cues)
  - power   ← zero_power | single_vision | progressive (from text cues)

After writing the catalog, deletes `data/rag.index.json` so the RAG layer
auto-rebuilds embeddings on the next server boot (see rag.refresh_if_stale()).

Run:
    python3 backend/build_catalog.py            # default: ~500 products
    python3 backend/build_catalog.py --all      # all rows in raw.tsv
    python3 backend/build_catalog.py --max 800  # explicit cap

Env:
    GEMINI_API_KEY  — if set and --embed flag passed, embeddings are built
                      synchronously here instead of on next boot.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

DATA_DIR   = Path(__file__).parent / "data"
TSV_PATH   = DATA_DIR / "raw.tsv"
CSV_PATH   = DATA_DIR / "products.csv"
JSON_PATH  = DATA_DIR / "products.db.json"
INDEX_PATH = DATA_DIR / "rag.index.json"

# ─── Keyword → tag tables ────────────────────────────────────────────────────

_SHAPE_KEYWORDS = [
    # order matters — check more-specific first
    ("cat-eye",      r"\bcat[\s-]?eye\b"),
    ("clubmaster",   r"\bclubmaster\b|\bbrowline\b"),
    ("butterfly",    r"\bbutterfly\b"),
    ("hexagonal",    r"\bhexagonal\b|\bhexagon\b"),
    ("geometric",    r"\bgeometric\b|\bpolygon\b"),
    ("aviator",      r"\baviator\b|\bpilot\b"),
    ("wayfarer",     r"\bwayfarer\b"),
    ("rimless",      r"\brimless\b"),
    ("rectangular",  r"\brectangle\b|\brectangular\b"),
    ("round",        r"\bround\b"),
    ("oval",         r"\boval\b"),
    ("square",       r"\bsquare\b"),
    ("sports",       r"\bsports?\b|\bwrap[\s-]?around\b|\bshield\b"),
]

# color → normalized name + hex (first match wins)
_COLOR_TABLE = [
    ("tortoise",       "#8b5a2b", r"\btortoise\b|\bdemi\b"),
    ("rose-gold",      "#b76e79", r"\brose\s*gold\b"),
    ("gunmetal",       "#2a3439", r"\bgun[\s-]?metal\b"),
    ("navy-blue",      "#1b2a4e", r"\bnavy\b"),
    ("transparent",    "#cfd4dc", r"\btransparent\b|\bclear\b|\bcrystal\b"),
    ("matte-black",    "#111111", r"\bmatte\s+black\b"),
    ("glossy-black",   "#0d0d0d", r"\bglossy\s+black\b"),
    ("black",          "#111111", r"\bblack\b"),
    ("white",          "#f5f5f5", r"\bwhite\b"),
    ("grey",           "#808080", r"\bgrey\b|\bgray\b"),
    ("silver",         "#c0c0c0", r"\bsilver\b"),
    ("gold",           "#d4af37", r"\bgold\b"),
    ("brown",          "#7a4f2a", r"\bbrown\b|\bbeige\b|\bcoffee\b"),
    ("blue",           "#3563a8", r"\bblue\b"),
    ("green",          "#2f7a3a", r"\bgreen\b|\bolive\b"),
    ("red",            "#b5352b", r"\bred\b|\bmaroon\b|\bburgundy\b"),
    ("pink",           "#e98ca8", r"\bpink\b|\brose\b"),
    ("purple",         "#6a4b86", r"\bpurple\b|\bviolet\b"),
    ("yellow",         "#e2c344", r"\byellow\b|\bmustard\b"),
    ("orange",         "#e07a3c", r"\borange\b"),
]

_MATERIALS = [
    ("TR90",            r"\btr[\s-]?90\b"),
    ("Acetate",         r"\bacetate\b"),
    ("Italian Acetate", r"\bitalian\s+acetate\b"),
    ("Stainless Steel", r"\bstainless\s+steel\b"),
    ("Titanium",        r"\btitanium\b"),
    ("Metal",           r"\bmetal\b"),
    ("Polycarbonate",   r"\bpolycarbonate\b|\bnylon\b"),
]

# product_type (raw feed) → canonical type
_TYPE_MAP = {
    "eyeglasses":                "eyeglasses",
    "sunglasses":                "sunglasses",
    "reading glasses":           "eyeglasses",
    "computer screen glasses":   "eyeglasses",
    "clip-on eyeglasses":        "eyeglasses",
    "smart glasses":             "eyeglasses",
    "contact lens":              "contact_lens",
    "contact lens-spherical":    "contact_lens",
    "contact lens-cylindrical":  "contact_lens",
    "contact lens-fast moving":  "contact_lens",
    "contact lens-toric":        "contact_lens",
}

# Frame shape → face shapes that suit it
_SHAPE_FACE_FIT = {
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

_KID_SIZES = {"2-5 yrs", "5-8 yrs", "8-12 yrs", "teens"}

_PRICE_BAND_TAG = {
    "Below 499": "under_500",
    "500-999":   "under_1000",
    "1000-1999": "under_2000",
    "2000-2999": "under_3000",
    "3000+":     "above_3000",
}

# trend/style cue words lifted from product copy
_TRENDY_CUES   = ("sleek", "surrealist", "aw22", "neo", "craft", "new arrival",
                  "studio", "flex", "hustlr")
_MINIMAL_CUES  = ("air", "rimless", "half rim", "lightweight", "thin",
                  "slim", "air flex")
_BOLD_CUES     = ("bold", "statement", "oversized", "polarized")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _strip(s: str) -> str:
    return (s or "").strip().strip('"').strip()


def _match_first(text: str, table):
    """table: list of (tag, pattern) — return first matching tag or None."""
    for tag, pat in table:
        if re.search(pat, text, re.I):
            return tag
    return None


def _price_from_str(s: str) -> float:
    """'1500 INR' → 1500.0, '' → 0.0"""
    if not s:
        return 0.0
    m = re.search(r"(\d+)", s)
    return float(m.group(1)) if m else 0.0


def _detect_color(text: str):
    for name, hexv, pat in _COLOR_TABLE:
        if re.search(pat, text, re.I):
            return name, hexv
    return "black", "#111111"


def _detect_material(text: str) -> str:
    for mat, pat in _MATERIALS:
        if re.search(pat, text, re.I):
            return mat
    return "Acetate"


def _detect_shape(text: str) -> str:
    for shape, pat in _SHAPE_KEYWORDS:
        if re.search(pat, text, re.I):
            return shape
    return "rectangular"


def _detect_power_tag(text: str, prod_type: str, label0: str) -> str:
    t = text.lower()
    l0 = (label0 or "").lower()
    if "zero power" in t or "zero power" in l0 or "screen glasses" in l0:
        return "zero_power"
    if "bifocal" in t or "progressive" in t:
        return "progressive"
    if prod_type == "contact_lens":
        return "single_vision"
    # Default eyewear — has-power (single vision) is the safe prescription default
    return "single_vision"


def _detect_age(size_val: str, text: str) -> str:
    s = (size_val or "").lower().strip()
    if s in _KID_SIZES:
        return "kids"
    if re.search(r"\bkids?\b|\bjuniors?\b|\blkj\b", text, re.I):
        return "kids"
    if re.search(r"\baged?\b|\bseniors?\b|\breaders?\b", text, re.I):
        return "aged"
    return "adult"


def _detect_gender(raw_gender: str, text: str) -> str:
    g = (raw_gender or "").lower().strip()
    if g in ("male", "men"):    return "male"
    if g in ("female", "women"): return "female"
    # Check description — "Women" only vs "Men" only
    has_women = bool(re.search(r"\bwomen\b|\bladies\b|\bfemale\b", text, re.I))
    has_men   = bool(re.search(r"\bmen\b|\bgents\b|\bmale\b", text, re.I))
    if has_women and not has_men: return "female"
    if has_men and not has_women: return "male"
    return "unisex"


def _derive_style_tag(shape: str, color: str, text: str) -> str:
    t = text.lower()
    if any(c in t for c in _BOLD_CUES) or shape in ("cat-eye", "butterfly", "hexagonal"):
        return "bold"
    if any(c in t for c in _MINIMAL_CUES) or shape == "rimless":
        return "minimalist"
    if any(c in t for c in _TRENDY_CUES):
        return "trendy"
    return "classic"


def _build_tags(*, style: str, power: str, age: str, gender: str,
                color: str, price: float, budget_band_tag: str | None,
                shape: str, text: str) -> list:
    tags = [style, power, age, gender]
    # Color-family tags
    if color in ("black", "matte-black", "glossy-black", "navy-blue", "gunmetal"):
        tags += ["dark", "neutral"]
    elif color in ("white", "transparent", "silver", "rose-gold", "clear"):
        tags += ["light"]
    elif color in ("gold", "tortoise", "brown"):
        tags += ["warm"]
    elif color in ("blue", "grey"):
        tags += ["cool"]
    elif color in ("red", "pink", "purple", "yellow", "orange", "green"):
        tags += ["statement", "bold"]
    # Budget bucket tag (from feed band first; fallback by price)
    if budget_band_tag:
        tags.append(budget_band_tag)
    else:
        for thresh in (500, 1000, 1500, 2000, 2500, 3000):
            if price <= thresh:
                tags.append(f"under_{thresh}")
                break
        else:
            tags.append("above_3000")
    # Style-taste tags
    tl = text.lower()
    if "polarized" in tl:                          tags.append("polarized")
    if "screen" in tl or "blu" in tl or "blue light" in tl:
        tags += ["blue-cut", "computer"]
    if "rimless" in tl or "half rim" in tl:        tags.append("lightweight")
    if shape in ("aviator", "wayfarer"):           tags.append("timeless")
    if shape in ("sports",):                       tags += ["sports", "active"]
    if age == "kids":                              tags.append("kids")
    # Dedupe, preserve order
    seen, out = set(), []
    for t in tags:
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


# ─── Row → product dict ──────────────────────────────────────────────────────

def row_to_product(row: dict) -> dict | None:
    """Return a parsed product dict, or None if the row isn't a stockable frame."""
    raw_type = _strip(row.get("product_type") or row.get("google_product_category") or "").lower()
    if "solution" in raw_type or "gold membership" in raw_type:
        return None
    ptype = _TYPE_MAP.get(raw_type)
    if ptype is None:
        # Fall back: infer from label0
        l0 = _strip(row.get("custom_label_0") or "").lower()
        if "sunglass" in l0:   ptype = "sunglasses"
        elif "contact" in l0:  ptype = "contact_lens"
        elif l0:               ptype = "eyeglasses"
        else:                  return None

    title = _strip(row.get("title") or row.get("display_ads_title") or "")
    desc  = _strip(row.get("description") or "")
    text  = f"{title} {desc}"

    # Skip rows missing an image — can't render
    image = _strip(row.get("image_link") or "")
    if not image:
        return None

    raw_id   = _strip(row.get("id") or "")
    if not raw_id:
        return None
    pid = f"LK-{raw_id}"

    shape   = _detect_shape(text)
    color, color_hex = _detect_color(text)
    material = _detect_material(text)

    price        = _price_from_str(row.get("price"))
    sale_price   = _price_from_str(row.get("sale_price"))
    effective    = sale_price if sale_price > 0 else price
    strike       = price if sale_price > 0 and sale_price < price else max(effective * 1.4, effective + 500)

    age    = _detect_age(_strip(row.get("size")), text)
    gender = _detect_gender(_strip(row.get("gender")), text)
    power  = _detect_power_tag(text, ptype, _strip(row.get("custom_label_0")))
    style  = _derive_style_tag(shape, color, text)

    band_tag = _PRICE_BAND_TAG.get(_strip(row.get("custom_label_1")))
    tags = _build_tags(
        style=style, power=power, age=age, gender=gender,
        color=color, price=effective, budget_band_tag=band_tag,
        shape=shape, text=text,
    )

    face_fit = _SHAPE_FACE_FIT.get(shape, ["oval"])

    brand = _strip(row.get("brand") or "Lenskart")

    return {
        "id":                         pid,
        "sku":                        _strip(row.get("item_group_id")) or pid,
        "name":                       title or f"{brand} {shape.title()}",
        "type":                       ptype,
        "gender":                     gender,
        "age":                        age,
        "color":                      color,
        "color_hex":                  color_hex,
        "powers":                     [0.0] if power == "zero_power" else [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        "quantity":                   25,
        "price":                      round(effective or 0.0, 2),
        "strikeout_price":            round(strike, 2),
        "tags":                       tags,
        "frame_shape":                shape,
        "face_shape_recommendation":  face_fit,
        "shade":                      "clear" if ptype != "sunglasses" else "tinted",
        "image_urls":                 [image],      # feed URLs are already absolute
        "avg_rating":                 4.2,
        "brand":                      brand,
        "material":                   material,
        "style":                      shape,        # legacy alias used by some callers
        "description":                desc or title,
        "product_url":                _strip(row.get("link") or ""),
    }


# ─── TSV reader ──────────────────────────────────────────────────────────────

def _read_tsv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _balanced_sample(products: list, max_total: int) -> list:
    """
    Ensure catalog variety when capping. Distributes the cap across product
    types proportional to their presence in the source, guaranteeing at least
    some of each.
    """
    if max_total >= len(products):
        return products
    by_type: dict = {}
    for p in products:
        by_type.setdefault(p["type"], []).append(p)
    total = len(products)
    out = []
    for t, bucket in by_type.items():
        share = max(1, round(max_total * len(bucket) / total))
        out.extend(bucket[:share])
    return out[:max_total]


# ─── CSV writer ──────────────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "id", "sku", "name", "type", "gender", "age", "color", "color_hex",
    "powers", "quantity", "price", "strikeout_price", "tags",
    "frame_shape", "face_shape_recommendation", "shade", "image_urls",
    "avg_rating", "brand", "material", "style", "description", "product_url",
]


def _write_csv(products: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLUMNS)
        for p in products:
            row = []
            for col in _CSV_COLUMNS:
                v = p.get(col, "")
                if isinstance(v, list):
                    v = "|".join(str(x) for x in v)
                row.append(v)
            writer.writerow(row)
    print(f"  [build_catalog] wrote {len(products)} rows → {path.relative_to(path.parent.parent)}")


def _write_json(products: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False)
    print(f"  [build_catalog] wrote JSON snapshot    → {path.relative_to(path.parent.parent)}")


def _invalidate_index() -> None:
    """Delete the RAG index so rag.refresh_if_stale() rebuilds embeddings on
    next boot. (It would notice the catalog hash change anyway, but wiping
    the file makes the 'first boot after rebuild' state unambiguous.)"""
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
        print(f"  [build_catalog] invalidated {INDEX_PATH.name} — RAG will rebuild on next boot")


# ─── Embedding (optional, synchronous) ───────────────────────────────────────

def _build_embeddings_now() -> None:
    """Force the RAG index to rebuild in this process (blocks on Gemini calls)."""
    try:
        # Import here so the build script runs even without google deps configured.
        from rag import get_index
        idx = get_index()
        n = idx.rebuild()
        print(f"  [build_catalog] embeddings built for {n} products")
    except Exception as exc:
        print(f"  [build_catalog] embedding build skipped: {exc}")
        print(f"  [build_catalog] the RAG layer will rebuild on next server boot anyway")


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max",   type=int, default=500,
                    help="Cap number of products (default: 500; use --all for no cap).")
    ap.add_argument("--all",   action="store_true", help="Keep every row in raw.tsv.")
    ap.add_argument("--embed", action="store_true",
                    help="Build embeddings now instead of deferring to next boot.")
    args = ap.parse_args()

    if not TSV_PATH.exists():
        print(f"  [build_catalog] missing {TSV_PATH}", file=sys.stderr)
        return 1

    parsed, skipped = [], 0
    for row in _read_tsv(TSV_PATH):
        p = row_to_product(row)
        if p is None:
            skipped += 1
            continue
        parsed.append(p)

    # Dedupe on id — raw feed has colour variants sharing ids sometimes.
    by_id: dict = {}
    for p in parsed:
        by_id.setdefault(p["id"], p)
    unique = list(by_id.values())

    if not args.all:
        unique = _balanced_sample(unique, args.max)

    print(f"  [build_catalog] parsed {len(parsed)} rows "
          f"({skipped} skipped, {len(by_id) - len(unique)} trimmed by cap)")
    print(f"  [build_catalog] final catalog size:   {len(unique)}")

    _write_csv(unique, CSV_PATH)
    _write_json(unique, JSON_PATH)
    _invalidate_index()

    if args.embed:
        _build_embeddings_now()

    # Breakdown summary
    by_type: dict = {}
    for p in unique:
        by_type[p["type"]] = by_type.get(p["type"], 0) + 1
    print(f"  [build_catalog] by type: {by_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
