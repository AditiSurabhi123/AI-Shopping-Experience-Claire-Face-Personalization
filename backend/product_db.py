#!/usr/bin/env python3
"""
Lenskart Claire — Product In-Memory Database
Loads from CSV on startup, builds inverted indexes for fast filtered search,
persists parsed data as JSON so subsequent boots skip CSV parsing.

Public surface:
    db = ProductDB()
    db.load()                         # auto-generate CSV if needed, parse, index
    results = db.search(filters)      # raw field filters
    results = db.search_by_quiz_tags(quiz_tags, face_shape, gender)
"""
import csv
import json
import os
from pathlib import Path

DATA_DIR  = Path(__file__).parent / "data"
CSV_PATH  = DATA_DIR / "products.csv"
JSON_PATH = DATA_DIR / "products.db.json"

# ─── Tag → filter mapping tables ──────────────────────────────────────────────

# budget tag → max price (inclusive). None means no upper bound.
BUDGET_PRICE_MAP = {
    "under_500":     {"max": 500},
    "under_1000":    {"max": 1000},
    "under_1500":    {"max": 1500},
    "under_2000":    {"max": 2000},
    "under_2500":    {"max": 2500},
    "under_3000":    {"max": 3000},
    "above_3000":    {"min": 3000},   # premium — floor only
    # legacy keys
    "budget":        {"max": 1000},
    "mid-range":     {"max": 2500},
    "premium":       {"min": 3000},
    "ultra-premium": {"min": 5000},
}

# vision_need → product filter
VISION_TYPE_MAP = {
    "zero_power":    {"has_power": False},
    "single_vision": {"has_power": True,  "prod_types": ["eyeglasses", "contact_lens"]},
    "progressive":   {"has_power": True,  "prod_types": ["eyeglasses"]},
    "not_sure":      {},
}

# lifestyle → desired product tags (any match boosts score)
LIFESTYLE_TAGS = {
    "active":        ["sports", "outdoor", "active", "sporty", "wrap-around"],
    "professional":  ["professional", "office", "minimalist", "rimless", "rectangular"],
    "creative":      ["trendy", "bold", "statement", "geometric", "fashion"],
    "fashion":       ["fashion", "trending", "bold", "statement", "trendy"],
}

# trend → desired product tags
TREND_TAGS = {
    "classic":  ["classic", "timeless", "heritage", "retro"],
    "trendy":   ["trending", "trendy", "new-arrival"],
    "minimal":  ["minimalist", "lightweight", "rimless"],
    "bold":     ["bold", "statement", "fashion"],
}

# color → desired product tags / colors
COLOR_TAGS = {
    "neutral":   ["neutral", "dark", "matte-black", "silver", "gunmetal"],
    "warm":      ["warm", "gold", "tortoise", "brown"],
    "cool":      ["cool", "blue", "silver", "navy-blue", "grey"],
    "statement": ["statement", "bold", "red", "maroon", "green", "blue"],
    "light":     ["light", "clear", "white", "rose-gold", "silver"],
    "dark":      ["dark", "matte-black", "glossy-black", "navy-blue", "gunmetal"],
}


# ─── ProductDB ────────────────────────────────────────────────────────────────

class ProductDB:
    def __init__(self):
        self._products: dict = {}          # id → product dict
        self._idx_type: dict  = {}         # type  → set of ids
        self._idx_gender: dict = {}        # gender → set of ids
        self._idx_age: dict   = {}         # age   → set of ids
        self._idx_tag: dict   = {}         # tag   → set of ids
        self._idx_shape: dict = {}         # frame_shape → set of ids
        self._idx_face: dict  = {}         # face_shape_rec → set of ids
        self._idx_shade: dict = {}         # shade → set of ids
        self._idx_price: dict = {}         # bucket → set of ids
        self._loaded = False

    # ── Public load ───────────────────────────────────────────────────────────

    def load(self) -> int:
        """
        Load order: persisted JSON → CSV (re-parse) → generate CSV then parse.
        Returns number of products loaded.
        """
        if JSON_PATH.exists():
            self._load_json(JSON_PATH)
        elif CSV_PATH.exists():
            self._load_csv(CSV_PATH)
            self._save_json(JSON_PATH)
        else:
            from generate_products import generate
            generate()
            self._load_csv(CSV_PATH)
            self._save_json(JSON_PATH)

        self._loaded = True
        return len(self._products)

    def reload_from_csv(self) -> int:
        """Force re-parse from CSV (e.g. after regeneration)."""
        self._products.clear()
        self._clear_indexes()
        self._load_csv(CSV_PATH)
        self._save_json(JSON_PATH)
        self._loaded = True
        return len(self._products)

    # ── JSON persistence ──────────────────────────────────────────────────────

    def _save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(self._products.values()), f, ensure_ascii=False)
        print(f"  [product_db] saved {len(self._products)} products → {path.name}")

    def _load_json(self, path: Path) -> None:
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            self._index(row)
        print(f"  [product_db] loaded {len(self._products)} products from {path.name}")

    # ── CSV parsing ───────────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                p = self._parse_row(raw)
                self._index(p)
        print(f"  [product_db] parsed {len(self._products)} products from {path.name}")

    @staticmethod
    def _parse_row(raw: dict) -> dict:
        def _split(s):
            return [x.strip() for x in s.split("|") if x.strip()] if s else []

        def _floats(s):
            out = []
            for x in _split(s):
                try:
                    out.append(float(x))
                except ValueError:
                    pass
            return out

        p = dict(raw)
        p["powers"]                   = _floats(raw.get("powers", ""))
        p["tags"]                     = _split(raw.get("tags", ""))
        p["face_shape_recommendation"] = _split(raw.get("face_shape_recommendation", ""))
        p["image_urls"]               = _split(raw.get("image_urls", ""))
        try:
            p["price"]          = float(raw.get("price", 0))
            p["strikeout_price"] = float(raw.get("strikeout_price", 0))
            p["avg_rating"]     = float(raw.get("avg_rating", 3.0))
            p["quantity"]       = int(raw.get("quantity", 0))
        except (ValueError, TypeError):
            pass
        return p

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _clear_indexes(self) -> None:
        for d in (self._idx_type, self._idx_gender, self._idx_age,
                  self._idx_tag, self._idx_shape, self._idx_face,
                  self._idx_shade, self._idx_price):
            d.clear()

    def _index(self, p: dict) -> None:
        pid = p["id"]
        self._products[pid] = p

        def add(idx, key):
            if key:
                idx.setdefault(str(key).lower(), set()).add(pid)

        add(self._idx_type,   p.get("type"))
        add(self._idx_gender, p.get("gender"))
        add(self._idx_age,    p.get("age"))
        add(self._idx_shape,  p.get("frame_shape"))
        add(self._idx_shade,  p.get("shade"))

        for tag in p.get("tags", []):
            add(self._idx_tag, tag)

        for face in p.get("face_shape_recommendation", []):
            add(self._idx_face, face)

        # Price bucket index
        price = float(p.get("price", 0))
        for threshold in [500, 1000, 1500, 2000, 2500, 3000]:
            if price <= threshold:
                self._idx_price.setdefault(f"under_{threshold}", set()).add(pid)
        if price > 3000:
            self._idx_price.setdefault("above_3000", set()).add(pid)

    # ── Low-level search ──────────────────────────────────────────────────────

    def search(self, filters: dict, limit: int = 3) -> dict:
        """
        Direct field-based search.

        Accepted filter keys:
          type, gender, age, price_max (int), price_min (int),
          frame_shape, face_shape, shade, tags (list[str]),
          has_power (bool), powers (list[float]), specific_color (str)

        Returns:
          {success, no_match, total_found, results, applied_filters}
        """
        if not self._loaded:
            self.load()

        candidates = set(self._products.keys())

        def _intersect(idx, key):
            nonlocal candidates
            s = idx.get(str(key).lower(), set())
            candidates &= s

        applied = {}

        # Hard filters — reduce candidate set
        if filters.get("type"):
            _intersect(self._idx_type, filters["type"])
            applied["type"] = filters["type"]

        if filters.get("gender") and filters["gender"] != "unisex":
            gen_ids = (
                self._idx_gender.get(filters["gender"].lower(), set())
                | self._idx_gender.get("unisex", set())
            )
            candidates &= gen_ids
            applied["gender"] = filters["gender"]

        if filters.get("age"):
            _intersect(self._idx_age, filters["age"])
            applied["age"] = filters["age"]

        if filters.get("frame_shape"):
            _intersect(self._idx_shape, filters["frame_shape"])
            applied["frame_shape"] = filters["frame_shape"]

        if filters.get("shade"):
            _intersect(self._idx_shade, filters["shade"])
            applied["shade"] = filters["shade"]

        if filters.get("price_max") is not None:
            price_max = float(filters["price_max"])
            price_ids = {pid for pid, p in self._products.items()
                         if float(p.get("price", 0)) <= price_max}
            candidates &= price_ids
            applied["price_max"] = price_max

        if filters.get("price_min") is not None:
            price_min = float(filters["price_min"])
            candidates = {pid for pid in candidates
                          if float(self._products[pid].get("price", 0)) >= price_min}
            applied["price_min"] = price_min

        if "has_power" in filters:
            want_power = bool(filters["has_power"])
            candidates = {
                pid for pid in candidates
                if _has_nonzero_power(self._products[pid]) == want_power
            }
            applied["has_power"] = want_power

        if filters.get("face_shape"):
            face_ids = self._idx_face.get(filters["face_shape"].lower(), set())
            if face_ids:
                candidates &= face_ids
                applied["face_shape"] = filters["face_shape"]

        if filters.get("specific_color"):
            sc = filters["specific_color"].lower()
            candidates = {pid for pid in candidates
                          if sc in (self._products[pid].get("color") or "").lower()}
            applied["specific_color"] = sc

        # If hard filters yielded nothing — no match
        if not candidates:
            return {
                "success":    True,
                "no_match":   True,
                "reason":     _no_match_reason(applied),
                "total_found": 0,
                "results":    [],
                "applied_filters": applied,
            }

        # Soft tag scoring
        desired_tags = [t.lower() for t in filters.get("tags", [])]
        ranked = _score_and_rank(
            [self._products[pid] for pid in candidates],
            desired_tags,
            limit,
        )

        return {
            "success":    True,
            "no_match":   False,
            "total_found": len(candidates),
            "results":    ranked,
            "applied_filters": applied,
        }

    # ── Quiz-tag smart search ─────────────────────────────────────────────────

    def search_by_quiz_tags(
        self,
        quiz_tags: dict,
        face_shape: str = None,
        gender: str = None,
        age: str = None,
    ) -> dict:
        """
        Translate quiz tag bundle → product filters → top-3 results.

        quiz_tags keys (all optional, null = ignored):
          price, lifestyle, trend, color, budget, vision_need
          [extended] product_type, age_group, gender_pref, color_type, frame_shape_pref
        """
        filters: dict = {}
        desired_tags: list = []

        # ── Budget / price ────────────────────────────────────────────────────
        budget_key = (
            quiz_tags.get("budget")
            or _price_tag_from_price(quiz_tags.get("price"))
        )
        if budget_key:
            price_range = BUDGET_PRICE_MAP.get(budget_key)
            if price_range is None:
                # Dynamic parse for arbitrary buckets like "under_300",
                # "above_4000", "around_2000" (±10 percent band).
                import re as _re
                m = _re.match(r"^(under|above|around)_(\d+)$", str(budget_key))
                if m:
                    kind, n = m.group(1), int(m.group(2))
                    if kind == "around":
                        # ±10 percent, widened for larger values
                        band = max(200, int(n * 0.10))
                        price_range = {"min": max(0, n - band), "max": n + band}
                    elif kind == "under":
                        price_range = {"max": n}
                    else:
                        price_range = {"min": n}
                else:
                    price_range = {}
            if "max" in price_range:
                filters["price_max"] = price_range["max"]
            if "min" in price_range:
                filters["price_min"] = price_range["min"]

        # ── Vision need → type + has_power ───────────────────────────────────
        vision = quiz_tags.get("vision_need")
        explicit_type = quiz_tags.get("product_type")  # "eyeglasses"|"sunglasses"|"contact_lens"

        if explicit_type:
            filters["type"] = explicit_type.replace(" ", "_")
        elif vision and vision in VISION_TYPE_MAP:
            vmap = VISION_TYPE_MAP[vision]
            if "has_power" in vmap:
                filters["has_power"] = vmap["has_power"]
            if "prod_types" in vmap and not explicit_type:
                # prefer eyeglasses as default unless sunglasses clue
                filters["type"] = vmap["prod_types"][0]

        # ── Gender ────────────────────────────────────────────────────────────
        gen = quiz_tags.get("gender_pref") or gender
        if gen:
            filters["gender"] = gen

        # ── Age ───────────────────────────────────────────────────────────────
        ag = quiz_tags.get("age_group") or age
        if ag:
            filters["age"] = ag
        else:
            # Default to adult unless the user explicitly signalled kids/aged,
            # so professional/office/fashion searches don't surface kids frames.
            lifestyle_hint = (quiz_tags.get("lifestyle") or "").lower()
            if lifestyle_hint in {"professional", "fashion", "creative", "active"}:
                filters["age"] = "adult"

        # ── Face shape ────────────────────────────────────────────────────────
        if face_shape:
            filters["face_shape"] = face_shape

        # ── Frame shape preference ────────────────────────────────────────────
        fshape = quiz_tags.get("frame_shape_pref")
        if fshape:
            filters["frame_shape"] = fshape

        # ── Specific color (exact match takes priority over color_type) ───────
        specific_color = quiz_tags.get("specific_color")
        if specific_color:
            filters["specific_color"] = specific_color

        # ── Soft-tag accumulation ─────────────────────────────────────────────
        lifestyle = quiz_tags.get("lifestyle")
        if lifestyle:
            desired_tags.extend(LIFESTYLE_TAGS.get(lifestyle, [lifestyle]))

        trend = quiz_tags.get("trend")
        if trend:
            desired_tags.extend(TREND_TAGS.get(trend, [trend]))

        # color / color_type
        color = quiz_tags.get("color_type") or quiz_tags.get("color")
        if color:
            desired_tags.extend(COLOR_TAGS.get(color, [color]))

        filters["tags"] = desired_tags

        return self.search(filters)

    # ── Convenience accessors ─────────────────────────────────────────────────

    def get(self, product_id: str) -> dict | None:
        return self._products.get(product_id)

    def all_products(self) -> list:
        return list(self._products.values())

    def count(self) -> int:
        return len(self._products)


# ─── Module-level singleton ───────────────────────────────────────────────────

_db: ProductDB | None = None


def get_db() -> ProductDB:
    global _db
    if _db is None:
        _db = ProductDB()
        n = _db.load()
        print(f"  [product_db] database ready — {n} products indexed")
    return _db


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _has_nonzero_power(p: dict) -> bool:
    return any(float(pw) != 0.0 for pw in p.get("powers", [0.0]))


def _score_and_rank(products: list, desired_tags: list, limit: int) -> list:
    dt_set = set(desired_tags)

    def _score(p):
        prod_tags = set(t.lower() for t in p.get("tags", []))
        tag_hits  = len(dt_set & prod_tags)
        # Bonus for bestseller/trending
        bonus = (2 if "bestseller" in prod_tags else 0) + (1 if "trending" in prod_tags else 0)
        return (tag_hits * 3 + bonus, float(p.get("avg_rating", 3.0)))

    ranked = sorted(products, key=_score, reverse=True)
    return ranked[:limit]


def _price_tag_from_price(price_label: str | None) -> str | None:
    """Map qualitative price label to budget bucket key."""
    if not price_label:
        return None
    return price_label  # keys like "budget","mid-range","premium" exist directly in BUDGET_PRICE_MAP


def _no_match_reason(applied: dict) -> str:
    parts = []
    if "type" in applied:
        parts.append(applied["type"])
    if "price_max" in applied:
        parts.append(f"under ₹{int(applied['price_max'])}")
    if "has_power" in applied:
        parts.append("with prescription" if applied["has_power"] else "zero-power")
    if "gender" in applied:
        parts.append(f"for {applied['gender']}")
    if "age" in applied:
        parts.append(f"for {applied['age']}")
    if parts:
        return f"No products found matching: {', '.join(parts)}."
    return "No products matched the given filters."
