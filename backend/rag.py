#!/usr/bin/env python3
"""
Lenskart Claire AI — RAG Layer

At startup: builds semantic embeddings for every product once, persists them
to disk. At query time: embeds the user's context and retrieves top-K matches
by cosine similarity — all in-memory, <50 ms per query.

Embeddings: Google `text-embedding-004` (768 dims), batched for fast boot.
Fallback: if the embedding API is unreachable, uses a lightweight token-overlap
scorer so the app still works offline.

Public surface:
    idx = get_index()
    hits = idx.retrieve(query_text, k=20, hard_filter=...)
    idx.rebuild()     # force re-embed all products
"""
import http.client
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from product_db import get_db

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
# Do NOT change to gemini-embedding-2 / gemini-3-* — those IDs 404.
# `gemini-embedding-001` is the current working embedding model.
EMBED_MODEL    = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-2")
_EMBED_FALLBACKS = ("gemini-embedding-001", "text-embedding-004")
EMBED_HOST     = "generativelanguage.googleapis.com"
INDEX_PATH     = Path(__file__).parent / "data" / "rag.index.json"

# Bump whenever _doc_text format changes so boot rebuilds the index.
INDEX_VERSION  = 2


# ─── Product → searchable text ────────────────────────────────────────────────

def _doc_text(p: dict) -> str:
    """
    Build a rich, embedding-friendly paragraph per product. Writes every
    salient signal in natural language so the query embedding can match on
    semantics (audience, occasion, material, bestseller, price bucket).
    """
    price   = float(p.get("price") or 0)
    if   price <= 500:  bucket = "budget under 500 rupees very affordable"
    elif price <= 1000: bucket = "affordable under 1000 rupees budget friendly"
    elif price <= 2000: bucket = "mid range between 1000 and 2000 rupees"
    elif price <= 3000: bucket = "upper mid range under 3000 rupees"
    else:               bucket = "premium above 3000 rupees luxury"

    has_power = any(float(pw) != 0 for pw in (p.get("powers") or [0]))
    rx_line   = "prescription lenses available single vision" if has_power else "zero power fashion only no prescription"

    gender = (p.get("gender") or "unisex").lower()
    age    = (p.get("age") or "adult").lower()
    aud_en = {"male":"for men boys gents","female":"for women ladies girls","unisex":"unisex for everyone"}.get(gender, f"for {gender}")
    age_en = {"kids":"kids children 13 and under","adult":"adults men and women","aged":"seniors elderly 50 plus"}.get(age, age)

    tags   = p.get("tags") or []
    occ    = []
    if "bestseller" in tags:   occ.append("bestseller top rated")
    if "trending"   in tags:   occ.append("trending popular")
    if "new-arrival" in tags:  occ.append("new arrival latest")
    if "classic"    in tags:   occ.append("classic timeless professional office work")
    if "bold"       in tags or "statement" in tags: occ.append("bold statement stylish trendy")
    if "minimalist" in tags or "rimless" in tags:   occ.append("minimal clean lightweight")
    if "sporty"     in tags or "active" in tags:    occ.append("sports active outdoor gym running")

    faces = p.get("face_shape_recommendation") or p.get("shape_suitability") or []
    face_line = ("suits " + " ".join(faces) + " face shapes") if faces else ""

    parts = [
        p.get("name", ""),
        (p.get("brand") or "") + " brand",
        f"{p.get('type','')} type frame",
        f"{p.get('frame_shape','')} frame shape style",
        f"{p.get('color','')} color",
        aud_en,
        age_en,
        bucket,
        rx_line,
        " ".join(occ),
        face_line,
        "features: " + " ".join(tags),
        p.get("description", ""),
    ]
    return ". ".join(x for x in parts if x).strip()


# ─── Gemini embedding calls (single + batch) ──────────────────────────────────

def _embed_one(text: str) -> list:
    payload = json.dumps({"content": {"parts": [{"text": text}]}})
    from ssl_ctx import SSL_CTX
    models_to_try = [EMBED_MODEL] + [m for m in _EMBED_FALLBACKS if m != EMBED_MODEL]
    for model in models_to_try:
        path = f"/v1beta/models/{model}:embedContent?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(EMBED_HOST, timeout=10, context=SSL_CTX)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8")
        finally:
            conn.close()
        if resp.status == 200:
            return json.loads(raw).get("embedding", {}).get("values") or []
        if resp.status != 404 and "NOT_FOUND" not in raw:
            raise RuntimeError(f"Embed error {resp.status}: {raw[:200]}")
    raise RuntimeError(f"Embed: all models returned 404")


def _embed_batch(texts: list) -> list:
    """Batch embed — up to 100 per call. Returns list of vectors (may be empty on error)."""
    from ssl_ctx import SSL_CTX
    models_to_try = [EMBED_MODEL] + [m for m in _EMBED_FALLBACKS if m != EMBED_MODEL]
    for model in models_to_try:
        requests = [{
            "model": f"models/{model}",
            "content": {"parts": [{"text": t}]},
        } for t in texts]
        payload = json.dumps({"requests": requests})
        path = f"/v1beta/models/{model}:batchEmbedContents?key={GEMINI_API_KEY}"
        conn = http.client.HTTPSConnection(EMBED_HOST, timeout=30, context=SSL_CTX)
        try:
            conn.request("POST", path, payload, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw  = resp.read().decode("utf-8")
        finally:
            conn.close()
        if resp.status == 200:
            items = json.loads(raw).get("embeddings", [])
            return [e.get("values") or [] for e in items]
        if resp.status != 404 and "NOT_FOUND" not in raw:
            raise RuntimeError(f"Batch embed error {resp.status}: {raw[:200]}")
    raise RuntimeError("Batch embed: all models returned 404")


# ─── Vector ops ───────────────────────────────────────────────────────────────

def _cosine(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y; na += x * x; nb += y * y
    return dot / (math.sqrt(na) * math.sqrt(nb) + 1e-9)


# ─── Index ────────────────────────────────────────────────────────────────────

class RAGIndex:
    def __init__(self):
        self._vectors: dict = {}      # pid → list[float]
        self._docs:    dict = {}      # pid → flat text
        self._dim:     int  = 0
        self._embedding_ok = True     # flipped to False on first API failure
        self._current_hash: str = ""  # latest catalog hash the index was built with

    # ── Catalog hash — detects when products change so we can auto-rebuild ──
    @staticmethod
    def _catalog_hash(products: list) -> str:
        """
        Stable hash covering every salient product field. If the catalog is
        edited (price change, new product, colour/shape change, etc.) the hash
        changes and we rebuild automatically on next boot.
        """
        import hashlib as _h
        h = _h.sha256()
        for p in sorted(products, key=lambda x: x.get("id", "")):
            row = "|".join(str(p.get(k, "")) for k in
                           ("id","name","type","gender","age","color","frame_shape",
                            "price","avg_rating","tags","face_shape_recommendation"))
            h.update(row.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()

    # ── Persistence ─────────────────────────────────────────────────────────
    def _save(self, catalog_hash: str) -> None:
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(INDEX_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "version":      INDEX_VERSION,
                "catalog_hash": catalog_hash,
                "dim":          self._dim,
                "vectors":      self._vectors,
                "docs":         self._docs,
            }, f)

    def _load(self, expected_hash: str = None) -> bool:
        if not INDEX_PATH.exists():
            return False
        try:
            with open(INDEX_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") != INDEX_VERSION:
                print(f"  [rag] index version {data.get('version')} != {INDEX_VERSION} — will rebuild")
                return False
            if expected_hash and data.get("catalog_hash") != expected_hash:
                print(f"  [rag] catalog changed — will rebuild index")
                return False
            self._vectors = data.get("vectors", {})
            self._docs    = data.get("docs", {})
            self._dim     = data.get("dim", 0)
            self._current_hash = data.get("catalog_hash") or ""
            return bool(self._vectors)
        except Exception:
            return False

    # ── Build ───────────────────────────────────────────────────────────────
    def build(self, force: bool = False) -> int:
        db       = get_db()
        products = db.all_products()
        cat_hash = self._catalog_hash(products)

        # Load existing index IF it matches the current catalog hash
        if not force and self._load(expected_hash=cat_hash) and len(self._vectors) == len(products):
            print(f"  [rag] loaded existing index — {len(self._vectors)} vectors")
            return len(self._vectors)

        print(f"  [rag] embedding {len(products)} products "
              f"(catalog hash {cat_hash[:10]})…")
        self._vectors = {}
        self._docs = {p["id"]: _doc_text(p) for p in products}
        ids        = list(self._docs.keys())
        texts      = [self._docs[i] for i in ids]

        try:
            # Batch in chunks of 90 to respect Gemini's per-call limit
            vectors = []
            for i in range(0, len(texts), 90):
                vectors.extend(_embed_batch(texts[i : i + 90]))
            for pid, vec in zip(ids, vectors):
                if vec:
                    self._vectors[pid] = vec
                    self._dim = len(vec)
            self._save(cat_hash)
            self._current_hash = cat_hash
            print(f"  [rag] indexed {len(self._vectors)} products · dim={self._dim}")
        except Exception as exc:
            print(f"  [rag] embedding API failed: {exc}")
            print("  [rag] falling back to lexical retrieval (tokens only)")
            self._embedding_ok = False

        return len(self._vectors)

    def rebuild(self) -> int:
        self._vectors.clear(); self._docs.clear(); self._dim = 0
        return self.build(force=True)

    # ── Runtime auto-refresh ────────────────────────────────────────────────
    # Call this from any endpoint that knows the catalog may have just changed
    # (admin import, price update, new product). If the hash has drifted from
    # the one stored in the index, rebuild automatically.
    def refresh_if_stale(self) -> bool:
        db       = get_db()
        products = db.all_products()
        new_hash = self._catalog_hash(products)
        cur      = getattr(self, "_current_hash", None)
        if cur != new_hash:
            print(f"  [rag] catalog drift detected — rebuilding index")
            self.rebuild()
            return True
        return False

    # ── Retrieve ────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: int = 20, hard_filter=None) -> list:
        """
        Return top-k product dicts by similarity. hard_filter is an optional
        predicate `fn(product) → bool` — candidates failing it are dropped
        BEFORE scoring (so k is always satisfied if pool is large enough).
        """
        db = get_db()
        candidates = db.all_products()
        if hard_filter:
            candidates = [p for p in candidates if hard_filter(p)]
        if not candidates:
            return []

        if self._embedding_ok and self._vectors:
            try:
                qvec = _embed_one(query)
            except Exception as exc:
                print(f"  [rag] query embed failed: {exc} — using lexical fallback")
                qvec = None
            if qvec:
                scored = [
                    (_cosine(qvec, self._vectors.get(p["id"], [])), p)
                    for p in candidates if p["id"] in self._vectors
                ]
                scored.sort(key=lambda x: x[0], reverse=True)
                return [p for s, p in scored[:k] if s > 0.0]

        # Lexical fallback (token overlap)
        q_tokens = set(query.lower().split())
        def _score(p):
            doc = (self._docs.get(p["id"]) or _doc_text(p)).lower()
            overlap = sum(1 for t in q_tokens if t in doc)
            rating  = float(p.get("avg_rating") or 3.0)
            return (overlap, rating)
        candidates.sort(key=_score, reverse=True)
        return candidates[:k]


# ─── Module singleton ─────────────────────────────────────────────────────────

_idx: RAGIndex | None = None

def get_index() -> RAGIndex:
    global _idx
    if _idx is None:
        _idx = RAGIndex()
        _idx.build()
    return _idx
