#!/usr/bin/env python3
"""
Lenskart Claire — Product Catalog Generator
Deterministically generates 500 products (200 eyeglasses, 200 sunglasses, 100 contact lenses).
Run standalone:  python3 generate_products.py
Called automatically by product_db.py on first startup when CSV is absent.
"""
import csv
import json
import random
from pathlib import Path

SEED = 42
DATA_DIR = Path(__file__).parent / "data"
CSV_PATH = DATA_DIR / "products.csv"

# ─── Brand pools ──────────────────────────────────────────────────────────────
EG_BRANDS = ["Vincent Chase", "John Jacobs", "Lenskart Blu", "Lenskart Gold", "Fastrack", "Oakley LK"]
SG_BRANDS = ["Vincent Chase", "Lenskart Blu", "Fastrack", "Oakley LK", "Scott LK", "Carrera LK"]
CL_BRANDS = ["Air Optix", "Acuvue Moist", "FreshLook", "Bausch + Lomb", "Lenskart Contacts"]

# ─── Frame shapes ─────────────────────────────────────────────────────────────
EG_SHAPES = [
    "wayfarer", "aviator", "round", "rectangular", "square",
    "geometric", "rimless", "clubmaster", "oval", "cat-eye",
    "butterfly", "hexagonal",
]
SG_SHAPES = [
    "wayfarer", "aviator", "round", "shield", "wrap-around",
    "cat-eye", "square", "butterfly", "clubmaster",
]

# ─── Color table: (name, hex) ─────────────────────────────────────────────────
EG_COLORS = [
    ("matte-black", "#2c2c2c"), ("glossy-black", "#0d0d0d"),
    ("tortoise",    "#8B4513"), ("clear",         "#e8f4fd"),
    ("navy-blue",   "#1a237e"), ("rose-gold",     "#B76E79"),
    ("gold",        "#c9a84c"), ("silver",        "#C0C0C0"),
    ("gunmetal",    "#2c3e50"), ("brown",         "#795548"),
    ("red",         "#c62828"), ("white",         "#f5f5f5"),
    ("maroon",      "#880e4f"), ("blue",          "#1565c0"),
    ("green",       "#2e7d32"),
]
SG_COLORS = [
    ("matte-black",  "#2c2c2c"), ("glossy-black",  "#0d0d0d"),
    ("tortoise",     "#8B4513"), ("white",         "#f5f5f5"),
    ("gold",         "#c9a84c"), ("silver",        "#C0C0C0"),
    ("red",          "#c62828"), ("blue",          "#1565c0"),
    ("transparent",  "#e8f4fd"), ("brown",         "#795548"),
]
SG_SHADES = [
    "grey", "amber", "green", "blue", "yellow", "pink",
    "mirror-silver", "mirror-gold", "gradient-brown", "gradient-grey", "rose",
]

# ─── Face-shape recommendations per frame shape ────────────────────────────────
FACE_RECS = {
    "wayfarer":    ["oval", "heart", "diamond", "oblong"],
    "aviator":     ["oval", "round", "heart"],
    "round":       ["square", "diamond", "oblong"],
    "rectangular": ["oval", "round", "heart"],
    "square":      ["oval", "round", "heart"],
    "geometric":   ["oval", "round", "heart"],
    "rimless":     ["oval", "square", "oblong", "heart"],
    "clubmaster":  ["oval", "square", "heart", "diamond"],
    "oval":        ["square", "diamond", "heart"],
    "cat-eye":     ["round", "oval", "square"],
    "butterfly":   ["square", "diamond", "oblong"],
    "hexagonal":   ["round", "oval", "square"],
    "shield":      ["oval", "square", "heart"],
    "wrap-around": ["oval", "square", "oblong"],
}

# ─── Style tags per frame shape ───────────────────────────────────────────────
SHAPE_TAGS = {
    "wayfarer":    ["classic", "timeless"],
    "aviator":     ["classic", "cool"],
    "round":       ["vintage", "retro"],
    "rectangular": ["professional", "office"],
    "square":      ["bold", "modern"],
    "geometric":   ["trendy", "fashion"],
    "rimless":     ["minimalist", "lightweight"],
    "clubmaster":  ["heritage", "retro"],
    "oval":        ["versatile", "everyday"],
    "cat-eye":     ["feminine", "bold"],
    "butterfly":   ["statement", "bold"],
    "hexagonal":   ["trendy", "geometric"],
    "shield":      ["sporty", "active"],
    "wrap-around": ["sports", "outdoor"],
}

MATERIALS = ["Acetate", "Metal", "TR90 Nylon", "Titanium", "Stainless Steel", "Carbon Fibre"]

# Powers pools
ADULT_POWERS = [-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0,
                -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
KIDS_POWERS  = [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
AGED_POWERS  = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Color character classifiers
_LIGHT  = {"clear", "white", "transparent", "rose-gold", "silver", "gold"}
_DARK   = {"matte-black", "glossy-black", "navy-blue", "gunmetal", "maroon"}
_WARM   = {"tortoise", "gold", "brown", "rose-gold", "red", "amber"}
_COOL   = {"blue", "silver", "gunmetal", "navy-blue", "grey", "green"}
_NEUT   = {"matte-black", "glossy-black", "silver", "gunmetal", "clear", "white"}

# Brand → (min_price, max_price)
EG_BRAND_PRICE = {
    "Vincent Chase": (499, 2499), "John Jacobs": (1499, 4999),
    "Lenskart Blu":  (699, 2999), "Lenskart Gold": (2499, 6999),
    "Fastrack":      (399, 1499), "Oakley LK":    (2999, 7999),
}
SG_BRAND_PRICE = {
    "Vincent Chase": (699, 3499), "Lenskart Blu":  (799, 2999),
    "Fastrack":      (499, 1799), "Oakley LK":    (3499, 9999),
    "Scott LK":      (1499, 4999), "Carrera LK":  (2499, 7999),
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _price_tags(price: float) -> list:
    for t in [500, 1000, 1500, 2000, 2500, 3000]:
        if price <= t:
            return [f"under_{t}"]
    return ["premium"] + (["luxury"] if price > 5000 else [])


def _color_tags(color: str) -> list:
    tags = []
    if color in _LIGHT:
        tags.append("light")
    if color in _DARK:
        tags.append("dark")
    if color in _WARM:
        tags.append("warm")
    if color in _COOL:
        tags.append("cool")
    if color in _NEUT:
        tags.append("neutral")
    return tags


def _round_price(value: float) -> int:
    return int(round(value / 50) * 50)


def _gender(i: int) -> str:
    g = (i * 7) % 30
    return "male" if g < 9 else ("female" if g < 19 else "unisex")


def _age(i: int) -> str:
    a = (i * 3) % 20
    return "kids" if a < 2 else ("aged" if a >= 17 else "adult")


def _rating(i: int) -> float:
    return round(3.2 + (i * 17 % 18) / 10, 1)


def _qty(i: int) -> int:
    return 30 + (i * 13) % 200


def _pick_powers(rng, pool, n_min=4, n_max=8) -> list:
    n = rng.randint(n_min, n_max)
    return sorted(set(rng.choices(pool, k=n)))


# ─── Eyeglasses generator ─────────────────────────────────────────────────────

def _eyeglasses(rng: random.Random) -> list:
    products = []
    for i in range(200):
        shape   = EG_SHAPES[i % len(EG_SHAPES)]
        brand   = EG_BRANDS[i % len(EG_BRANDS)]
        color, color_hex = EG_COLORS[i % len(EG_COLORS)]
        gender  = _gender(i)
        age     = _age(i)
        pool    = KIDS_POWERS if age == "kids" else (AGED_POWERS if age == "aged" else ADULT_POWERS)
        powers  = _pick_powers(rng, pool)

        pmin, pmax = EG_BRAND_PRICE[brand]
        price    = _round_price(pmin + (i * 31 + i * i * 7) % max(1, pmax - pmin))
        sp       = _round_price(price * rng.uniform(1.3, 2.0))

        tags = list(dict.fromkeys(
            SHAPE_TAGS.get(shape, ["classic"])
            + _price_tags(price)
            + _color_tags(color)
            + (["kids"] if age == "kids" else [])
            + (["reading", "progressive"] if age == "aged" else [])
            + (["bestseller"] if i % 15 == 0 else [])
            + (["trending"] if i % 10 == 0 else [])
            + (["new-arrival"] if i % 7 == 0 else [])
            + (["zero-power", "fashion"] if 0.0 in powers else ["prescription"])
        ))

        sku = f"EG-{i+1:04d}"
        products.append({
            "id":                     sku,
            "sku":                    sku,
            "name":                   f"{brand} {shape.replace('-',' ').title()} {color.replace('-',' ').title()}",
            "type":                   "eyeglasses",
            "gender":                 gender,
            "age":                    age,
            "color":                  color,
            "color_hex":              color_hex,
            "powers":                 powers,
            "quantity":               _qty(i),
            "price":                  price,
            "strikeout_price":        sp,
            "tags":                   tags,
            "frame_shape":            shape,
            "face_shape_recommendation": FACE_RECS.get(shape, ["oval", "round", "square"]),
            "shade":                  "clear",
            "image_urls":             [f"/static/products/eg/{sku}-1.jpg", f"/static/products/eg/{sku}-2.jpg"],
            "avg_rating":             _rating(i),
            "brand":                  brand,
            "material":               MATERIALS[i % len(MATERIALS)],
            "style":                  shape,
            "description":            f"{brand} {shape} frames in {color} for {age} {gender}.",
        })
    return products


# ─── Sunglasses generator ─────────────────────────────────────────────────────

def _sunglasses(rng: random.Random) -> list:
    products = []
    for i in range(200):
        shape   = SG_SHAPES[i % len(SG_SHAPES)]
        brand   = SG_BRANDS[i % len(SG_BRANDS)]
        color, color_hex = SG_COLORS[i % len(SG_COLORS)]
        shade   = SG_SHADES[i % len(SG_SHADES)]
        gender  = _gender(i + 100)
        age     = _age(i + 50)

        # Mostly zero power; ~20% prescription sunglasses
        if i % 5 == 0:
            powers = _pick_powers(rng, [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5], n_min=3, n_max=5)
        else:
            powers = [0.0]

        pmin, pmax = SG_BRAND_PRICE[brand]
        price    = _round_price(pmin + (i * 37 + i * i * 11) % max(1, pmax - pmin))
        sp       = _round_price(price * rng.uniform(1.4, 2.2))

        tags = list(dict.fromkeys(
            SHAPE_TAGS.get(shape, ["cool"])
            + ["sunglasses", "uv-protection"]
            + _price_tags(price)
            + _color_tags(color)
            + (["polarized"] if i % 3 == 0 else [])
            + (["outdoor", "sporty"] if shape in ("shield", "wrap-around") else [])
            + (["kids"] if age == "kids" else [])
            + (["bestseller"] if i % 12 == 0 else [])
            + (["trending"] if i % 8 == 0 else [])
            + (["new-arrival"] if i % 6 == 0 else [])
            + (["zero-power", "fashion"] if powers == [0.0] else ["prescription"])
        ))

        sku = f"SG-{i+1:04d}"
        products.append({
            "id":                     sku,
            "sku":                    sku,
            "name":                   f"{brand} {shape.replace('-',' ').title()} {color.replace('-',' ').title()} Sunglass",
            "type":                   "sunglasses",
            "gender":                 gender,
            "age":                    age,
            "color":                  color,
            "color_hex":              color_hex,
            "powers":                 powers,
            "quantity":               _qty(i + 30),
            "price":                  price,
            "strikeout_price":        sp,
            "tags":                   tags,
            "frame_shape":            shape,
            "face_shape_recommendation": FACE_RECS.get(shape, ["oval", "round"]),
            "shade":                  shade,
            "image_urls":             [f"/static/products/sg/{sku}-1.jpg", f"/static/products/sg/{sku}-2.jpg"],
            "avg_rating":             _rating(i + 50),
            "brand":                  brand,
            "material":               MATERIALS[(i + 2) % len(MATERIALS)],
            "style":                  shape,
            "description":            f"{brand} {shape} sunglasses with {shade} lens in {color}.",
        })
    return products


# ─── Contact lenses generator ─────────────────────────────────────────────────

CL_TYPES     = ["daily", "monthly", "yearly"]
CL_SHADES    = ["clear", "blue", "green", "grey", "hazel", "brown", "violet"]
CL_POWERS    = [-6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0,
                -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
CL_QTY_MAP   = {"daily": 30, "monthly": 6, "yearly": 2}
CL_PRICE_MAP = {"daily": (199, 999), "monthly": (399, 1499), "yearly": (699, 2499)}


def _contacts(rng: random.Random) -> list:
    products = []
    for i in range(100):
        disp  = CL_TYPES[i % len(CL_TYPES)]
        brand = CL_BRANDS[i % len(CL_BRANDS)]
        shade = CL_SHADES[i % len(CL_SHADES)]
        colored = shade != "clear"

        powers = _pick_powers(rng, CL_POWERS, n_min=5, n_max=12)
        qty = CL_QTY_MAP[disp]

        pmin, pmax = CL_PRICE_MAP[disp]
        price = _round_price(pmin + (i * 29 + i * i * 5) % max(1, pmax - pmin))
        sp    = _round_price(price * rng.uniform(1.3, 1.8))

        tags = list(dict.fromkeys(
            ["contact-lens", disp]
            + _price_tags(price)
            + (["colored"] if colored else ["clear"])
            + (["zero-power", "fashion"] if 0.0 in powers else ["prescription"])
            + (["bestseller"] if i % 10 == 0 else [])
            + (["trending"] if i % 7 == 0 else [])
        ))

        sku = f"CL-{i+1:04d}"
        products.append({
            "id":                     sku,
            "sku":                    sku,
            "name":                   f"{brand} {disp.title()} {'Colored ' if colored else ''}{shade.title()} Contact Lens",
            "type":                   "contact_lens",
            "gender":                 "unisex",
            "age":                    "adult",
            "color":                  shade,
            "color_hex":              "#a0c4d8" if not colored else "#6fa8dc",
            "powers":                 powers,
            "quantity":               qty,
            "price":                  price,
            "strikeout_price":        sp,
            "tags":                   tags,
            "frame_shape":            "none",
            "face_shape_recommendation": [],
            "shade":                  shade,
            "image_urls":             [f"/static/products/cl/{sku}-1.jpg"],
            "avg_rating":             _rating(i + 70),
            "brand":                  brand,
            "material":               "Silicone Hydrogel",
            "style":                  disp,
            "description":            f"{brand} {disp} contact lenses in {shade} shade.",
        })
    return products


# ─── CSV writer ───────────────────────────────────────────────────────────────

COLUMNS = [
    "id", "sku", "name", "type", "gender", "age", "color", "color_hex",
    "powers", "quantity", "price", "strikeout_price", "tags",
    "frame_shape", "face_shape_recommendation", "shade", "image_urls",
    "avg_rating", "brand", "material", "style", "description",
]


def _encode(v) -> str:
    """Encode list fields as pipe-separated strings."""
    if isinstance(v, list):
        return "|".join(str(x) for x in v)
    return str(v) if v is not None else ""


def write_csv(products: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for p in products:
            w.writerow([_encode(p.get(col, "")) for col in COLUMNS])
    print(f"  [generate_products] wrote {len(products)} products → {path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def generate(force: bool = False) -> Path:
    """Generate products.csv if absent (or force=True). Returns CSV path."""
    if CSV_PATH.exists() and not force:
        return CSV_PATH

    rng = random.Random(SEED)
    products = _eyeglasses(rng) + _sunglasses(rng) + _contacts(rng)
    write_csv(products, CSV_PATH)
    return CSV_PATH


if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    p = generate(force=force)
    print(f"Done. CSV at: {p}  ({sum(1 for _ in open(p))-1} products)")
