# Claire AI — Unified Eyewear Shopping Assistant

**Problem #15 · Online · AI Shopping Experience — Claire + Face Personalization**
**Score:** 82 ⭐⭐⭐⭐⭐

---

## Problem Statement

**Online eyewear shopping has a 65–80% drop-off rate** because customers can't self-diagnose three things at once: (1) which frame **shape** suits their face, (2) which **lens** their prescription needs, (3) whether the frame will **fit**. The friction compounds for non-English speakers — Lenskart's core India audience mixes Hindi, Hinglish, and regional dialects that generic chat UIs can't parse.

**Baseline today**
- Frame-selection time: **8–12 minutes**, typically 4+ PLP page changes
- Prescription/lens confusion causes **~22% of returns**
- Only **~9%** of mobile users complete a frame-try-on journey (no face-fit confidence)
- Hindi/Hinglish customers abandon at **2×** the English rate

---

## Your Solution — Claire

A conversational AI stylist that unifies **five discovery tools** into one chat:

- **Few-step Frame Quiz** with multilingual free-text tag extraction (English / Hindi Devanagari / Hinglish)
- **Photo-based Face Analysis** → face shape, width and match → auto-curated frames
- **Lens Guidance** — prescription + screen-time → 3 lens packages with blue-cut / progressive / hi-index rationale
- **Face-scan PLP Ordering** — reorders the 500-product catalog to surface the best-suited frames first
- **Size-Confidence Widget** — 0–100 fit score with 3 customer-facing reasons per highlighted frame

**AI techniques used**
- **RAG over 500 real Lenskart SKUs** (embedding-2, 3072-dim) with catalog-hash auto-refresh
- **Tool-use / ReAct loop** via Gemini function calling — 4 tools: `analyze_face`, `search_frames`, `get_lens_recommendation`, `calculate_fit_confidence`
- **Vision model** retags the entire catalog from product images (fixes colour / shape / face-fit labels the text feed gets wrong)
- **Scripted-first routing** — a deterministic conversation manager owns the quiz and scripted flows; the LLM only takes over for free-form follow-ups, guaranteeing the customer always sees the listed questions in order

---

## Tech Stack

| Layer | Technology                                                                                                    |
|---|---------------------------------------------------------------------------------------------------------------|
| Agent LLM (tool-use loop) | **Gemini 3.1 Flash** (function calling, 8-iter ReAct, AUTO tool-config)                                       |
| Enterprise fallback | **AWS Bedrock** — (auto-recovers from inference-profile errors)                                               |
| Embeddings / RAG | **gemini-embedding-2** (3072-dim), cosine similarity, catalog-hash invalidation                               |
| Vision | **Gemini 3.1 Flash Vision** — face shape detection + fit scoring + one-shot catalog retag from product images |
| TTS | Gemini 3.1 Flash TTS preview — disk-cached, preloaded on boot, common-phrase precache                         |
| Data | 500 real Lenskart products (parsed from the public product feed → tagged via vision)                          |
| Serving | Python stdlib HTTP (no framework), React 18 + Tailwind via CDN (no build step)                                |
| Deploy | Add AI Key & Single `bash start.sh`, port 8000; zero external services required to demo offline               |

---

## Business Impact

| Metric | Baseline | With Claire | Lift |
|---|---|---|---|
| Time-to-recommendation | 8–12 min | **~90 seconds** (quiz + face analysis) | **≈ 6× faster** |
| Return rate (wrong fit/lens) | ~22% | Projected **~8%** (fit-score + lens guidance) | **–14 pts** |
| Hindi/Hinglish completion rate | ~45% | **≥ 85%** (native Devanagari + Hinglish + voice) | **+40 pts** |
| Mobile try-on completion | ~9% | Projected **~35%** (photo → shape → curated PLP) | **≈ 4× lift** |
| PLP page-changes per session | 4+ | **0–1** (carousel returns pre-ranked top 6) | **↓ 75%** |
| Avg order value | — | **+12–18%** projected via lens attach-rate (BluBlock Pro / progressive upsell) | |

**Direct revenue lever:** every 1% lift in lens attach rate on the existing eyewear GMV is estimated at **₹18–22 Cr/year** for Lenskart India online.

---

## Assumptions & Limitations

**Assumed**
- Customer is willing to upload a selfie or answer quiz questions (instrumented opt-out at every step)
- Product feed (`raw.tsv`) refreshes nightly; our retag script is **idempotent** and resumable
- AWS Bedrock + Gemini API keys available in deployment env — demo mode runs fully client-side if neither is present

**Limitations → Production-readiness gaps**
- **CV accuracy** — face-shape detection uses Gemini Vision; needs eval against a ground-truth set of 2,000 Indian faces before full rollout
- **Catalog scale** — tested at 500 SKUs; the RAG index + product DB are designed for ≥ 50k with ANN (FAISS / pgvector) when plugged into Lenskart's live catalog API
- **Latency** — current P95 chat turn ~1.8s; needs <1s for voice-mode parity (prefetch + partial-stream rendering planned)
- **PII / safety** — selfies are stored to disk by `uuid.jpg`; production must add encryption-at-rest + 30-day TTL + consent gate
- **A/B gate** — recommend rolling out behind a GrowthBook flag at 10% traffic before full ramp

---