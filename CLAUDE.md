# Lenskart Claire — AI Shopping Assistant (Developer Deep-Dive)

> This is the contributor-facing guide. For the pitch / 1-pager see [CLAIRE.md](./CLAIRE.md); for the user-facing setup instructions see [README.md](./README.md).

## Project Overview

Claire is a **unified AI-powered eyewear shopping assistant** for Lenskart, built as a zero-dependency Python HTTP backend + CDN-based React SPA frontend. It combines five AI-driven shopping features into a bilingual (English + Hindi) conversational chat interface with voice I/O.

**Five core features:**
1. **8-step Frame Quiz** — guided preference discovery (gender, age, product type, lifestyle, style, colour, budget, vision needs) with Gemini-powered multilingual tag extraction
2. **Face-Based Recommendations** — Gemini Vision detects face shape from an uploaded photo and curates frames from a 500-product catalog
3. **Lens Guidance** — personalized lens packages by prescription type + screen-time
4. **Face-Scan PLP Ordering** — RAG-backed reranking of the 500-product catalog from quiz + face signals
5. **Photo-Based Size-Confidence Widget** — 0–100 fit score with 3 customer-facing reasons, scored from the user's photo + frame metadata

**Design philosophy:**
- **Offline-first.** The UI boots and runs end-to-end via `clientDemoReply()` when no backend is reachable; API calls only enhance the experience.
- **Scripted-first routing.** A deterministic conversation manager owns the quiz and documented flows; the LLM only takes over for free-form follow-ups. This guarantees the user always sees the quiz questions in order.
- **RAG over real SKUs.** 500 parsed Lenskart products are embedded with `gemini-embedding-2` and retrieved by cosine similarity at inference time.

---

## Repo Layout

```
lenskart-claire/
├── backend/
│   ├── server.py            # stdlib HTTP server, routes, upload handling, boot
│   ├── agent.py             # scripted-first router + Gemini function-calling ReAct loop
│   ├── tools.py             # 4 tool implementations + legacy MOCK_FRAMES/MOCK_LENSES/FACE_SHAPES
│   ├── _quiz_tables.py      # QUIZ_EN / QUIZ_HI — 8-step bilingual quiz
│   ├── product_db.py        # in-memory 500-SKU ProductDB + inverted indexes
│   ├── rag.py               # 3072-dim embedding index with catalog-hash auto-refresh
│   ├── recommend.py         # RAG-first recommender with optional Gemini LLM rerank
│   ├── face_llm.py          # Gemini Vision face-shape + fit-score analysis
│   ├── gemini.py            # quiz-analyze via Gemini + heuristic fallback
│   ├── bedrock.py           # AWS Bedrock Claude Sonnet 4 fallback (quiz-analyze only)
│   ├── tts.py               # Gemini TTS (preview) + Google Translate fallback, disk-cached
│   ├── stt.py               # Gemini audio transcription (handler wired, route currently disabled)
│   ├── build_catalog.py     # raw.tsv → products.csv / products.db.json
│   ├── retag_catalog.py     # vision-retag colour/shape/face-fit per product; --embed rebuilds RAG
│   ├── generate_products.py # synthetic catalog generator (fallback when raw.tsv missing)
│   ├── ssl_ctx.py           # shared ssl.SSLContext for Gemini/Bedrock HTTPS calls
│   ├── data/
│   │   ├── raw.tsv          # Lenskart product feed (placeholder; 0 bytes in this checkout)
│   │   ├── products.csv     # 500 curated SKUs (~407 KB)
│   │   ├── products.db.json # parsed + indexed ProductDB snapshot (~590 KB)
│   │   ├── rag.index.json   # 500 × 3072-dim embeddings (~21 MB)
│   │   ├── retag_results.jsonl
│   │   └── tts_cache/       # ~87 cached audio blobs, keyed by hash(text+lang)
│   └── uploads/             # user-uploaded selfies, {uuid}.jpg
├── frontend/
│   └── index.html           # React 18 SPA — 3,754 lines, no build step
├── CLAIRE.md                # submission 1-pager (problem, solution, business impact)
├── CLAUDE.md                # this file — developer deep-dive
├── README.md                # user-facing setup + API reference + scripts
├── claire-1-pager.pdf       # PDF version of the 1-pager
├── requirements.txt         # stdlib only; `anthropic` SDK optional and unused by the chat loop
└── start.sh                 # bootstrap: kill port, start `python3 backend/server.py`
```

**No build step.** Frontend uses CDN React 18, CDN Tailwind, CDN Babel (standalone JSX transpiler). Drop `index.html` anywhere static and it renders.

---

## Running Locally

```bash
# Minimum (demo mode works client-side; most backend responses succeed via heuristics)
bash start.sh

# Recommended — Gemini is the primary model for chat, vision, TTS, embeddings
GEMINI_API_KEY=... bash start.sh

# Optional extras
BEDROCK_BEARER_TOKEN=... GEMINI_API_KEY=... bash start.sh    # quiz-analyze fallback
PORT=3000 GEMINI_API_KEY=... bash start.sh                    # custom port
```

`start.sh` kills anything bound to `$PORT` before booting, so restarts stay clean. There is **no `.env.example`** in the checkout today — export env vars directly or wrap `start.sh` in your own shell script.

On first boot without a pre-built catalog:
```bash
python3 backend/build_catalog.py        # parse raw.tsv → products.csv/db.json (500 SKUs)
python3 backend/retag_catalog.py --embed  # vision retag + rebuild rag.index.json
```

**Health check:** `GET /health` → `{"status": "ok", "service": "Claire AI", "version": "1.0.0", "product_db": 500}`

---

## Environment Variables

| Variable | Default (code) | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | *(empty)* | Primary key: chat agent, vision, embeddings, TTS, quiz-analyze. Demo mode kicks in if absent. |
| `GEMINI_AGENT_MODEL` | `gemini-2.5-flash` | Main ReAct/tool-use model. **Do not set to `gemini-3.*`** — those IDs 404 on the public API (see [agent.py:151](backend/agent.py:151)). Fallback chain: `gemini-2.5-flash → gemini-flash-latest → gemini-2.0-flash`. |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-2` | RAG embeddings (3072-dim, [rag.py:31](backend/rag.py:31)). Falls back to `embedding-001` / `text-embedding-004` on 404. |
| `BEDROCK_BEARER_TOKEN` | *(empty)* | Optional AWS Bedrock fallback for `/api/quiz-analyze` only. Not used for the main chat loop. |
| `BEDROCK_REGION` | `ap-south-1` | Bedrock region. |
| `BEDROCK_MODEL` | `apac.anthropic.claude-sonnet-4-20250514-v1:0` | Bedrock model id (Claude Sonnet 4, inference profile for APAC). |
| `ANTHROPIC_API_KEY` | *(empty)* | Legacy. `start.sh` / `server.py` still print status for this, but the main chat loop no longer calls Anthropic directly. |
| `PORT` | `8000` | HTTP listen port. |

> **Provider cascade for quiz analysis** ([server.py:38](backend/server.py:38)): Gemini → Bedrock → pure-Python heuristic. Chat agent routing is independent — scripted manager first, Gemini second (no Bedrock path).

---

## HTTP API

All routes live in `backend/server.py`. CORS headers are set for all origins.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` , `/index.html` | Serves `frontend/index.html`. |
| `GET` | `/health` | Service + product-DB count. |
| `GET` | `/uploads/{filename}` | Serves a previously uploaded selfie. |
| `GET` | `/api/tts?text=...&lang=en\|hi` | Gemini TTS (falls back to Google Translate TTS). Returns base64 audio chunks. |
| `POST` | `/api/chat` | Main chat handler → `run_agent()`. |
| `POST` | `/api/upload` | Base64 image → `/uploads/{uuid}.jpg` → `analyze_face()` (deterministic mock). |
| `POST` | `/api/face-analyze` | Base64 image → `face_llm.analyze_image()` (Gemini Vision). |
| `POST` | `/api/fit-score` | Base64 image + frame → `face_llm.analyze_fit()`. |
| `POST` | `/api/quiz-analyze` | Free-text answer → language + tags (Gemini → Bedrock → heuristic). |
| `POST` | `/api/search` | Quiz-tag / filter-driven catalog search via `tools.search_products()`. |
| *(disabled)* | `POST /api/stt` | Handler `_handle_stt` exists in [server.py:382](backend/server.py:382) but the route dispatch is commented out at [server.py:181](backend/server.py:181) — rolled back for latency. Re-enable by wiring `path == "/api/stt"` in `do_POST`. |

### `/api/chat` payload

```json
{
  "messages": [{"role": "user", "content": "hi"}],
  "session_data": {
    "quiz_step": 0,
    "quiz_answers": {},
    "face_data": null,
    "active_filters": {},
    "lang": "en"
  }
}
```

If a message has `type: "image"` and a `data:...` URI in `content`, the server auto-saves the upload and injects `image_url` before calling the agent.

### `/api/chat` response

```json
{
  "success": true,
  "message": {
    "role": "assistant",
    "text": "Here are 6 frames curated for you — cat-eye + under ₹1,500",
    "components": [
      {"type": "quiz",           "step": 1, "total": 8, "key": "gender",  "question": "...", "options": [...]},
      {"type": "carousel",       "title": "...", "frames": [...]},
      {"type": "face_analysis",  "shape": "Oval", "confidence": 87, ...},
      {"type": "lens_rec",       "lenses": [...]},
      {"type": "fit_score",      "score": 87, "verdict": "Perfect Match", "reasons": [...]}
    ]
  },
  "session_data": { ... }
}
```

Components are emitted by the scripted manager or parsed out of Gemini's final text via regex tags (`[CAROUSEL]{...}`, `[QUIZ]{...}`, etc. — see [agent.py:314](backend/agent.py:314) `parse_response`).

---

## Feature Deep Dives

### 1. Frame Quiz

**Files:** [backend/_quiz_tables.py](backend/_quiz_tables.py) · [backend/agent.py:559](backend/agent.py:559) (`run_demo_agent` quiz block) · [frontend/index.html](frontend/index.html) `QuizComponent` (line ~1047)

**8 steps**, keyed by field name in `QUIZ_EN` / `QUIZ_HI`:

| # | Key | Example EN prompt |
|---|---|---|
| 1 | `gender`       | "Are these frames for you, for him, for her, or for everyone?" |
| 2 | `age`          | "And the age — just a number in years, like 10, 20, 35, or 55?" |
| 3 | `product_type` | "Are we looking at eyeglasses, sunglasses, or contact lenses?" |
| 4 | `lifestyle`    | "Office / creative / active / fashion-forward?" |
| 5 | `style_pref`   | "Classic / bold / minimal / adventurous?" |
| 6 | `color_pref`   | "Neutral / warm / cool / statement?" |
| 7 | `budget`       | "Under ₹1K / ₹1K–₹2K / ₹2K–₹3.5K / premium ₹3.5K+" |
| 8 | `prescription` | "Zero power / single vision / progressive / computer-reading" |

**Multilingual tag extraction** (non-blocking): Whenever the user free-texts an answer, the frontend `POST`s to `/api/quiz-analyze`. Response shape:

```json
{
  "original_response": "main fashion ke liye leke mujhe dedh hazaar ke aaspaas ka chahiye",
  "detected_language": "hi",
  "language_name": "Hindi",
  "english_translation": "I want it for fashion, around ₹1500",
  "tags": {"budget": "around_1500", "lifestyle": "fashion", "trend": "trendy"},
  "confidence": 92,
  "provider": "gemini"
}
```

**Provider cascade:** Gemini → AWS Bedrock (Claude Sonnet 4) → `_heuristic_tags()` regex matcher ([gemini.py](backend/gemini.py)).

---

### 2. Face-Based Recommendations

**Two paths coexist:**

- **LLM vision (primary):** `POST /api/face-analyze` → [face_llm.py](backend/face_llm.py) `analyze_image()` sends base64 image to Gemini Vision. Strict detection: returns `{has_face: false, reason}` for blurry / multi-face / no-face. Successful output includes `shape, gender, age, age_group, hair_color, skin_tone, face_width, recommended_styles, key_feature, confidence`.
- **Deterministic mock (fallback / `/api/upload`):** [tools.py](backend/tools.py) `analyze_face()` hashes the image URL to pick one of 6 shapes. Used on the classic `/api/upload` path and for client-side offline mode (`clientAnalyzeFace()` in the frontend mirrors this hash logic so the UI works with no backend).

`FACE_SHAPES` dict ([tools.py:212](backend/tools.py:212)) defines 6 profiles — Oval, Round, Square, Heart, Diamond, Oblong — each with `description, width, recommended_styles, avoid_styles, key_feature, celebrity_match`.

---

### 3. Lens Guidance

**File:** [tools.py:415](backend/tools.py:415) `get_lens_recommendation()` + frontend `LensRecommendation` component.

**Input:** `prescription_type` (`zero_power` | `single_vision` | `progressive` | `bifocal`) + `screen_time_hours` (number).

**Output:** up to 3 lens packages filtered from `MOCK_LENSES` ([tools.py:149](backend/tools.py:149)):

- `BluBlock Pro` — blue-light, ~6h+ screen time
- `ClearVision Single` — default for single_vision
- `HiIndex Thin` — high prescription
- `Progressive Elite` — default for progressive/bifocal
- `Transitions Smart` — always added for outdoor use
- `StyleZero` — default for zero_power

Each package has `price, features, tagline, badge, best_for, thickness`.

---

### 4. Face-Scan PLP Ordering (RAG-backed)

**Pipeline:**

```
quiz answers + face analysis
        │
        ▼
agent._llm_pick(ctx, limit=6)     ←  agent.py:71
        │
        ▼
recommend.recommend(ctx)          ←  recommend.py
        │
        ├── rag.get_index().retrieve(query, k=50, filters)
        │       │
        │       ▼
        │   cosine similarity over 500 × 3072-dim index
        │
        └── optional Gemini re-rank (rerank_with_llm=True, ~400ms)
                │
                ▼
returns top-N with 1-line per-pick reason
```

**ProductDB** ([product_db.py](backend/product_db.py)) loads 500 SKUs from `products.db.json` (or parses `products.csv` on first boot, or generates synthetic products as a last resort). Inverted indexes: type, gender, age, tag, frame_shape, face_shape, shade, price-bucket.

**Tag → filter maps** live in [product_db.py:22–72](backend/product_db.py:22):
- `BUDGET_PRICE_MAP` — tag → `{min, max}` (e.g. `under_500 → {max: 500}`, `around_1500 → {min: ~1350, max: ~1650}`).
- `VISION_TYPE_MAP` — vision_need → filter dict.
- `LIFESTYLE_TAGS`, `TREND_TAGS`, `COLOR_TAGS` — keyword boosters for embedding query text.

**RAG index** ([rag.py](backend/rag.py)) is persisted to `data/rag.index.json` (~21 MB). `INDEX_VERSION` + a hash of the catalog contents are stored alongside the vectors; if either changes, the index is rebuilt on boot. Embeddings are computed in batches of up to 100 documents.

---

### 5. Photo-Based Size-Confidence Widget

**Two implementations:**

- **Vision-based (primary):** `POST /api/fit-score` → [face_llm.py](backend/face_llm.py) `analyze_fit(img_b64, frame)` asks Gemini Vision to score the frame on the person in the photo. Returns `{score: 0–99, verdict, size_match, reasons: [3], face_shape, face_width}`.
- **Deterministic (fallback):** [tools.py:476](backend/tools.py:476) `calculate_fit_confidence()` — base 60 pts + ±32 for shape-match + ±8 for face-width compat, plus a small deterministic jitter from `hash(frame_id) % 10`.

**UI:** SVG circular progress arc (animated `stroke-dashoffset`) with score, verdict, and up to 3 reason bullets.

---

## Backend Code Map

### `server.py` ([519 lines](backend/server.py))

- `ClaireHandler(BaseHTTPRequestHandler)` — handles all HTTP with CORS.
- Boot sequence (lines 77–104):
  1. `product_db.get_db()` — load 500 SKUs (JSON fast path, CSV parse fallback).
  2. `rag.get_index()` — warm or rebuild the 3072-dim embedding index.
  3. `tts.preload_disk_cache()` (sync) + `tts.precache_common(async_mode=True)` (background).
- `analyze_quiz_response()` at [line 38](backend/server.py:38) orchestrates the Gemini → Bedrock → heuristic cascade.

### `agent.py` ([1,216 lines](backend/agent.py))

- `TOOLS` (line 246) — 4 Anthropic-flavoured tool schemas: `analyze_face`, `search_frames`, `get_lens_recommendation`, `calculate_fit_confidence`. Auto-translated to Gemini `functionDeclarations` via `_tools_to_gemini` (line 380).
- `SYSTEM_PROMPT` (line 162) — Claire persona + tool docs + response-format spec (`[CAROUSEL]`, `[QUIZ]`, `[FACE_ANALYSIS]`, `[LENS_REC]`, `[FIT_SCORE]`).
- `parse_response()` (line 314) — regex-extracts tagged JSON components from LLM output.
- `call_gemini_agent()` (line 401) — HTTPS call with model fallback chain.
- `run_agent()` (line 459) — **scripted-first**: calls `run_demo_agent`; if it returns `_fallback=True`, hands off to Gemini with up to 8 ReAct iterations.
- `run_demo_agent()` (line 559) — the deterministic conversation manager. Key branches: restart intent, gender switch, budget/color/style filter parse, greeting, photo-uploaded, lens inquiry, quiz-step advance, post-quiz carousel, fit-score, add-to-cart, else `_fallback=True`.
- `_llm_pick()` (line 71) — build a context dict and delegate to `recommend.recommend()`.

### `tools.py` ([607 lines](backend/tools.py))

- `MOCK_FRAMES` (line 14) — 12 frames used only as an offline / demo-mode fallback. Real catalog lives in `ProductDB`.
- `MOCK_LENSES` (line 149) — 6 lens packages.
- `FACE_SHAPES` (line 212) — 6 shape profiles.
- Tool functions: `analyze_face`, `search_frames`, `search_products`, `get_lens_recommendation`, `calculate_fit_confidence`.
- `execute_tool(name, input)` (line 571) — dispatcher for Gemini function calls.

### `product_db.py` ([497 lines](backend/product_db.py))

- `ProductDB` class with `load()`, `reload_from_csv()`, `count()`, `search()`, `search_by_quiz_tags()`.
- Loads `products.db.json` (fast path) → `products.csv` → `generate_products.generate(500)`.
- Inverted indexes built in `_index(p)`.
- Constants: `BUDGET_PRICE_MAP`, `VISION_TYPE_MAP`, `LIFESTYLE_TAGS`, `TREND_TAGS`, `COLOR_TAGS`.

### `rag.py` ([317 lines](backend/rag.py))

- `EMBED_MODEL = gemini-embedding-2` (3072-dim), with fallbacks `embedding-001`, `text-embedding-004`.
- `INDEX_PATH = data/rag.index.json`, `INDEX_VERSION = 2`.
- `_doc_text(p)` builds rich embedding text per product.
- `RAGIndex.retrieve(query, k, filters)` — embed query once, cosine-rank docs, apply post-filters (budget, gender, age, product_type).
- Catalog hash is stored alongside the index; re-embeds automatically when catalog or `INDEX_VERSION` changes.

### `recommend.py` ([470 lines](backend/recommend.py))

- `recommend(ctx, limit, rerank_with_llm)` — RAG retrieve → optional Gemini rerank with per-item reason.
- In-memory LRU (32 entries) keyed by context hash.
- Returns `{success, products, reasoning, source: "rag" | "rag+llm" | "llm"}`.

### `gemini.py` ([661 lines](backend/gemini.py))

- `analyze_quiz_response(user_response, quiz_context)` — language detection + tag extraction via Gemini (json_mode).
- `parse_restart_intent`, `parse_change_gender`, `parse_remove_intent` — lightweight intent parsers used by the scripted agent.
- `_heuristic_tags()` — regex-based fallback tag extractor. Always available.
- `_fallback_result()` — safe response shape when API unavailable.

### `bedrock.py` ([199 lines](backend/bedrock.py))

- `BEDROCK_MODEL = apac.anthropic.claude-sonnet-4-20250514-v1:0` (Sonnet 4).
- Bearer-token auth via `BEDROCK_BEARER_TOKEN`.
- Exposes the same `analyze_quiz_response()` shape as Gemini for drop-in fallback.

### `face_llm.py` ([355 lines](backend/face_llm.py))

- Gemini Vision wrapper.
- `analyze_image(img_b64, mime)` — face shape + width + recommended styles + `has_face` strict check.
- `analyze_fit(img_b64, frame, mime)` — fit score 0–99 with 3 reasons.

### `tts.py` ([469 lines](backend/tts.py))

- Primary: Gemini TTS preview model (configurable; known-working IDs hardcoded as last-resort fallback).
- Secondary: Google Translate free TTS (no auth).
- Output: 24 kHz mono PCM wrapped in WAV header, split into ≤180-char sentence chunks, base64-encoded.
- `preload_disk_cache()` at boot; `precache_common(async_mode=True)` fills gaps for UI phrases.
- Disk cache: `data/tts_cache/{hash}.{mp3|wav}` (~87 cached phrases in the current checkout).

### `stt.py` ([122 lines](backend/stt.py))

- `transcribe(audio_b64, mime, lang)` — Gemini audio understanding.
- Currently **not routed** (see note on `/api/stt` above) — frontend uses the browser `SpeechRecognition` API instead.

### `build_catalog.py`, `retag_catalog.py`, `generate_products.py`

Catalog tooling. Typical workflow:
```bash
python3 backend/build_catalog.py          # raw.tsv → products.csv (balanced 500)
python3 backend/build_catalog.py --all    # keep every row in feed
python3 backend/retag_catalog.py          # vision-retag via Gemini (resumable via retag_results.jsonl)
python3 backend/retag_catalog.py --embed  # also rebuild rag.index.json
```

---

## Frontend Code Map (`frontend/index.html`, 3,754 lines)

CDN dependencies: React 18, ReactDOM 18, Babel standalone, Tailwind.

| Approx. lines | Section / component |
|---|---|
| 1–100     | `<head>`, CDN imports, CSS custom properties, keyframe animations |
| 100–245   | `CLIENT_FRAMES`, `CLIENT_FACE_SHAPES`, `I18N` (EN+HI), `PRICE_CHIPS`/`TYPE_CHIPS`/`COLOR_CHIPS` |
| 245–570   | Voice layer: `useSpeechRecognition`, TTS playback pipeline, Web Audio pitch detection for gender hints |
| 620–700   | `LanguageSelector` — first-run: pick language + voice-vs-text mode |
| 700–1045  | `VoiceOnlyView` — dedicated full-voice interface |
| 1029–1200 | `QuizTagBadge`, `QuizComponent` (question + 4 options + free-text + progress bar) |
| 1200–1600 | `FrameCard` (SVG frame viz, badges, add-to-cart), `ProductCarousel` (scroll + filter chips) |
| 1600–2000 | `FaceAnalysisCard`, `FitConfidenceWidget`, `LensRecommendation` |
| 2000–2400 | `MessageBubble` (component dispatch), `PhotoUploadModal` |
| 2400–2800 | `Sidebar`, `ChatInput`, modal chrome |
| 2800–3754 | `App` — state (`lang`, `messages`, `sessionData`, `faceData`, `cart`, `loading`, ...), `sendMessage`, `clientDemoReply`, `handlePhotoUpload`, `handleQuizAnswer`, `clientSearchFrames`, `clientAnalyzeFace` |

> Line ranges are approximate — the single-file frontend keeps shifting. Use Grep for the exact component/function name.

**Key frontend features beyond the core components:**
- **Voice I/O.** Browser `SpeechRecognition` drives mic input; `GET /api/tts` plays back Gemini / Google synthesized audio, chunked and queued.
- **Bilingual UI.** `I18N` pack covers quiz, labels, intro text. Hindi answers are parsed with Devanagari + Hinglish-aware regex both client-side and server-side.
- **Offline fallback.** `clientDemoReply()` handles quiz, face analysis, lens, fit score entirely client-side; `clientAnalyzeFace()` mirrors the backend's MD5-hash mock.
- Hardcoded `const API = 'http://localhost:8000'` — update when deploying elsewhere.

---

## Agent Routing (scripted-first)

```
User message
    │
    ▼
run_demo_agent (scripted)   ← agent.py:559
    │
    ├─ restart intent ──────────────► wipe session + greet + Q1
    ├─ "for my wife" / gender switch ► update gender filter + re-run search
    ├─ explicit filter query ───────► parse budget/color/style + RAG search
    ├─ greeting / empty message ────► greet + Q1
    ├─ photo uploaded ──────────────► face analysis card + carousel
    ├─ lens inquiry ────────────────► lens_rec card
    ├─ quiz_step < 8 ───────────────► next quiz question
    ├─ post-quiz first turn ────────► _llm_pick → carousel
    ├─ fit keywords ────────────────► fit_score
    └─ no match ────────────────────► _fallback = true
                                            │
                                            ▼
                              GEMINI_API_KEY set?  ── no ──► return scripted reply as-is
                                            │ yes
                                            ▼
                              call_gemini_agent (ReAct, max 8 iters)
                                            │
                                            ▼
                              parse_response → extract tagged components
```

---

## AI Model Configuration (as called by code)

| Provider | Model (default) | Used for | Fallback |
|---|---|---|---|
| Google Gemini | `gemini-2.5-flash` | Main chat + function calling | `gemini-flash-latest` → `gemini-2.0-flash` → scripted reply |
| Google Gemini | `gemini-embedding-2` (3072-dim) | RAG embeddings | `embedding-001` → `text-embedding-004` |
| Google Gemini | Gemini Vision (model resolved dynamically) | Face shape + fit-score | Deterministic mock in `tools.py` |
| Google Gemini | Gemini TTS preview | `/api/tts` | Google Translate free TTS |
| Google Gemini | Gemini audio | `stt.py` | Browser `SpeechRecognition` (frontend) |
| AWS Bedrock | `apac.anthropic.claude-sonnet-4-20250514-v1:0` | `/api/quiz-analyze` fallback only | `_heuristic_tags()` regex |

> The agent explicitly refuses `gemini-3.*` — those IDs 404. See the warning comment at [agent.py:151](backend/agent.py:151).

---

## State Management

**Backend:** Stateless. `session_data` is passed in every request body and returned updated. Uploaded images are written to `backend/uploads/{uuid}.jpg` and served from `/uploads/`.

**Frontend (React hooks in `App`):**
- `lang` — "en" | "hi"
- `messages` — chat history `{id, role, type, content, text, time, components, answered}`
- `sessionData` — `{quiz_step, quiz_answers, face_data, active_filters, lang, last_shown}`
- `faceData` — latest face analysis result
- `cart` — selected frames
- `loading`, `showUpload`, `showCart`, `voiceMode` — UI flags

---

## Design System

- **Colours:** Teal `#00C3C3` · Navy `#003D5B` · Dark background `#070E1C`
- **Animations:** `msgIn`, `dotBounce`, `scoreRise`, `shimmer`, `fadeUp`, `glow`
- **Frame SVG viz:** `FrameCard` maps colour names to hex and renders an inline SVG silhouette per frame shape.

---

## Known Limitations & Gotchas

- **`raw.tsv` ships empty** (0 bytes) in this checkout. The catalog builder falls through to `generate_products.generate(500)` for synthetic data unless you drop in a real Lenskart feed.
- **No automated tests.** `tools.py` and `product_db.py` functions are pure and deterministic → good unit-test targets.
- **Frontend API URL hardcoded** — `const API = 'http://localhost:8000'` in `index.html`. Update for non-localhost deploys.
- **`/api/stt` is disabled** — handler exists, routing is commented out at [server.py:181](backend/server.py:181). Frontend uses browser `SpeechRecognition` today.
- **`ANTHROPIC_API_KEY` is legacy.** `start.sh` and `server.py`'s banner still mention it; the chat loop does not call Anthropic.
- **Do not set `GEMINI_AGENT_MODEL=gemini-3.*`** — those IDs 404 on the public Gemini API.
- **Single-file frontend.** `index.html` at 3,754 lines; should be componentized for maintainability before scale-up.
- **PII.** Selfies are stored as raw jpgs by uuid; production must add encryption-at-rest + TTL + consent gate.
- **RAG scale.** 500 SKUs fits comfortably in a flat cosine-scan; for ≥ 50k, switch to FAISS / pgvector.

---

## Adding Tests (Suggested)

- **Backend unit:** `tools.py` (`analyze_face`, `calculate_fit_confidence`, `get_lens_recommendation`) is fully deterministic — golden-value tests work. `product_db.py` search + tag maps similarly.
- **RAG contract:** lock in `_doc_text()` format via snapshot test; the embedding index auto-rebuilds when the doc format changes, so breaking this invalidates all cached vectors.
- **API integration:** drive `server.py` with a mock `http.client`; assert on component tags in responses.
- **Frontend E2E:** Playwright for the 8-step quiz, photo upload → carousel, voice-mode toggle, Hindi path.
