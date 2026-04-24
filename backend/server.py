#!/usr/bin/env python3
"""
Lenskart Claire AI — Backend HTTP Server
Serves the React frontend and handles all /api/* routes.
"""
import base64
import http.server
import json
import mimetypes
import os
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))
from agent import run_agent
from tools import analyze_face

# Gemini is primary (fast, multilingual); Bedrock is fallback
try:
    from gemini import analyze_quiz_response as _gemini_analyze
    _USE_GEMINI = True
except Exception as _e:
    print(f"  [server] Gemini import failed ({_e}), falling back to Bedrock")
    _USE_GEMINI = False

try:
    from bedrock import analyze_quiz_response as _bedrock_analyze
    _USE_BEDROCK = True
except Exception as _e:
    print(f"  [server] Bedrock import failed ({_e})")
    _USE_BEDROCK = False


def analyze_quiz_response(user_response: str, quiz_context: str = "") -> dict:
    """Try Gemini first, then Bedrock, then pure heuristic fallback."""
    if _USE_GEMINI:
        result = _gemini_analyze(user_response, quiz_context)
        if result.get("success"):
            result["provider"] = "gemini"
            return result
        # Gemini failed — try Bedrock
        print(f"  [server] Gemini analysis unsuccessful, trying Bedrock…")

    if _USE_BEDROCK:
        result = _bedrock_analyze(user_response, quiz_context)
        if result.get("success"):
            result["provider"] = "bedrock"
            return result
        print(f"  [server] Bedrock analysis also failed — using heuristic fallback")

    # Both unavailable — return Gemini's heuristic fallback (already computed)
    if _USE_GEMINI:
        from gemini import _fallback_result
        r = _fallback_result(user_response, "All providers unavailable")
        r["provider"] = "heuristic"
        return r

    return {
        "success": False, "provider": "none",
        "original_response": user_response,
        "detected_language": "en", "language_name": "English",
        "english_translation": user_response,
        "tags": {k: None for k in ("price","lifestyle","trend","color","budget","vision_need")},
        "confidence": 0, "error": "No analysis provider available"
    }

ROOT_DIR    = Path(__file__).parent.parent
FRONTEND    = ROOT_DIR / "frontend"
BACKEND     = Path(__file__).parent
UPLOADS     = BACKEND / "uploads"
UPLOADS.mkdir(exist_ok=True)

# ── Boot: load (or generate) the product database ─────────────────────────────
try:
    from product_db import get_db as _get_db
    _product_db = _get_db()
    _DB_READY = True
except Exception as _e:
    print(f"  [server] ⚠  Product DB failed to load: {_e}")
    _product_db = None
    _DB_READY = False

# ── Warm RAG embedding index (builds once, persists to disk) ──────────────────
try:
    from rag import get_index as _get_rag
    _rag_idx = _get_rag()
    _RAG_READY = True
except Exception as _e:
    print(f"  [server] ⚠  RAG index failed: {_e}")
    _rag_idx   = None
    _RAG_READY = False

# ── TTS cache: preload on-disk clips, then precache any new common phrases ───
try:
    from tts import preload_disk_cache as _preload_tts, precache_common as _precache_tts
    _preload_tts()                    # synchronous, fast — loads existing disk cache
    _precache_tts(async_mode=True)    # background: fill gaps for any new phrases
except Exception as _e:
    print(f"  [server] ⚠  TTS cache setup failed: {_e}")


class ClaireHandler(http.server.BaseHTTPRequestHandler):

    # ── Suppress verbose default logging ──────────────────
    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else "?"
        method = self.command
        path   = self.path.split("?")[0][:60]
        print(f"  {method:6} {status}  {path}")

    # ── CORS helper ────────────────────────────────────────
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Session-ID")
        self.send_header("Access-Control-Max-Age",       "86400")

    # ── OPTIONS pre-flight ─────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── GET ────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path == "/health":
            self._json({
                "status": "ok",
                "service": "Claire AI",
                "version": "1.0.0",
                "product_db": _product_db.count() if _DB_READY else 0,
            })

        elif path == "/api/tts":
            self._handle_tts(parsed.query)
            return

        elif path.startswith("/uploads/"):
            fname   = Path(path[9:]).name      # strip directory traversal
            fpath   = UPLOADS / fname
            ct, _   = mimetypes.guess_type(fname)
            self._file(fpath, ct or "image/jpeg")

        elif path in ("/", "/index.html"):
            self._file(FRONTEND / "index.html", "text/html; charset=utf-8")

        else:
            # Try serving from frontend directory
            rel = path.lstrip("/")
            candidate = FRONTEND / rel
            if candidate.exists() and candidate.is_file():
                ct, _ = mimetypes.guess_type(str(candidate))
                self._file(candidate, ct or "application/octet-stream")
            else:
                self.send_response(404)
                self._cors()
                self.end_headers()

    # ── POST ───────────────────────────────────────────────
    def do_POST(self):
        path           = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", 0))
        body           = self.rfile.read(content_length) if content_length else b""

        try:
            if path == "/api/chat":
                self._handle_chat(body)
            elif path == "/api/upload":
                self._handle_upload(body)
            elif path == "/api/face-analyze":
                self._handle_face_analyze(body)
            elif path == "/api/fit-score":
                self._handle_fit_score(body)
            # /api/stt was rolled back for latency — see stt.py for the handler
            # if you want to re-enable server-side transcription later.
            elif path == "/api/quiz-analyze":
                self._handle_quiz_analyze(body)
            elif path == "/api/search":
                self._handle_search(body)
            else:
                self._json({"error": "Not found"}, 404)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._json({
                "success": False,
                "error": str(exc),
                "message": {
                    "role": "assistant",
                    "text": "I hit a small snag — please try again in a moment! 💙",
                    "components": []
                }
            }, 500)

    # ── /api/chat ──────────────────────────────────────────
    def _handle_chat(self, body: bytes):
        data         = json.loads(body.decode("utf-8"))
        messages     = data.get("messages", [])
        session_data = data.get("session_data", {})

        # If the last user message contains an image (base64 data URI), upload it
        # and inject the URL into the message before sending to the agent
        for msg in messages:
            if msg.get("role") == "user" and msg.get("type") == "image":
                img_data = msg.get("content", "")
                if img_data.startswith("data:"):
                    header, b64 = img_data.split(",", 1)
                    img_bytes   = base64.b64decode(b64)
                    fname       = f"{uuid.uuid4()}.jpg"
                    (UPLOADS / fname).write_bytes(img_bytes)
                    img_url     = f"/uploads/{fname}"
                    msg["image_url"] = img_url
                    # Run face analysis and store in session_data
                    if not session_data.get("face_data"):
                        face_result = analyze_face(img_url)
                        if face_result.get("success"):
                            session_data["face_data"] = face_result

        response = run_agent(messages, session_data)
        self._json({"success": True, "message": response, "session_data": session_data})

    # ── /api/search ────────────────────────────────────────────────────────────
    def _handle_search(self, body: bytes):
        """
        Smart product search driven by quiz tags or raw filters.

        Accepts:
          { "quiz_tags": {...}, "face_shape": str, "gender": str, "age": str }
          { "filters": {"type":..., "price_max":..., "tags":[...], ...} }

        Returns:
          { success, no_match, reason, total_found, products:[top-3], applied_filters }
        """
        if not _DB_READY:
            self._json({"success": False, "error": "Product database unavailable"}, 503)
            return

        data = json.loads(body.decode("utf-8"))
        from tools import search_products
        result = search_products(
            quiz_tags=data.get("quiz_tags"),
            face_shape=data.get("face_shape"),
            gender=data.get("gender"),
            age=data.get("age"),
            filters=data.get("filters"),
        )
        self._json(result)

    # ── /api/tts ───────────────────────────────────────────
    def _handle_tts(self, qs: str):
        from urllib.parse import parse_qs
        params = parse_qs(qs or "")
        text   = (params.get("text", [""])[0] or "").strip()
        lang   = (params.get("lang", ["en"])[0] or "en").strip().lower()
        if not text:
            self._json({"success": False, "error": "Missing text"}, 400)
            return
        try:
            from tts import synthesise
            payload = synthesise(text, lang)
            if not payload.get("chunks"):
                self._json({"success": False, "error": "TTS unavailable — check server logs"}, 503)
                return
            self._json({
                "success": True,
                "chunks":  payload["chunks"],
                "mime":    payload["mime"],
                "source":  payload["source"],
            })
        except Exception as exc:
            print(f"  [tts] failed: {exc}")
            self._json({"success": False, "error": str(exc)}, 503)

    # ── /api/face-analyze (LLM vision) ─────────────────────
    def _handle_face_analyze(self, body: bytes):
        """
        Accept a base64 image (optionally as a data URI) and run it through
        Gemini vision. Returns the structured analysis or { has_face: False }.
        """
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._json({"success": False, "error": "Invalid JSON"}, 400)
            return

        img = (data.get("image") or "").strip()
        if not img:
            self._json({"success": False, "error": "image required"}, 400)
            return

        mime = "image/jpeg"
        if img.startswith("data:"):
            header, img = img.split(",", 1)
            m = re.match(r"data:(image/[a-zA-Z0-9.+-]+);", header)
            if m: mime = m.group(1)

        try:
            from face_llm import analyze_image
            result = analyze_image(img, mime=mime)
            if not result.get("has_face"):
                self._json({"success": True, "has_face": False,
                             "error": "no_face",
                             "reason": result.get("reason", "no_face")}, 200)
                return
            # Also save the image to disk for display
            import uuid
            img_bytes = base64.b64decode(img)
            fname = f"{uuid.uuid4()}.jpg"
            (UPLOADS / fname).write_bytes(img_bytes)
            result["url"]     = f"/uploads/{fname}"
            result["success"] = True
            self._json(result)
        except Exception as exc:
            print(f"  [face-analyze] failed: {exc}")
            self._json({"success": False, "error": str(exc)}, 503)

    # ── /api/fit-score ─────────────────────────────────────
    def _handle_fit_score(self, body: bytes):
        """
        Accepts: {
          "image":  "<data URI or base64>",   # optional if image_url given
          "image_url": "/uploads/<file>.jpg",  # optional
          "frame":  { id, name, type, frame_shape, color, frame_width,
                      face_shape_recommendation: [...] }
        }
        Returns: { success, has_face, score, verdict, size_match, reasons,
                   face_shape, face_width }
        """
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._json({"success": False, "error": "Invalid JSON"}, 400)
            return

        frame = data.get("frame") or {}
        if not frame:
            self._json({"success": False, "error": "frame required"}, 400)
            return

        img_b64 = (data.get("image") or "").strip()
        mime    = "image/jpeg"
        if img_b64.startswith("data:"):
            header, img_b64 = img_b64.split(",", 1)
            m = re.match(r"data:(image/[a-zA-Z0-9.+-]+);", header)
            if m: mime = m.group(1)

        # Accept an image_url pointing to a previously uploaded photo
        if not img_b64:
            img_url = (data.get("image_url") or "").strip()
            if img_url.startswith("/uploads/"):
                fp = UPLOADS / Path(img_url[9:]).name
                if fp.exists():
                    img_b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
                    mime = mimetypes.guess_type(str(fp))[0] or "image/jpeg"

        if not img_b64:
            self._json({"success": False, "error": "image or image_url required",
                         "no_image": True}, 400)
            return

        try:
            from face_llm import analyze_fit
            result = analyze_fit(img_b64, frame, mime=mime)
            if not result.get("has_face"):
                self._json({"success": True, "has_face": False,
                             "reasons": result.get("reasons") or ["No face detected"]})
                return
            result["success"] = True
            self._json(result)
        except Exception as exc:
            print(f"  [fit-score] failed: {exc}")
            self._json({"success": False, "error": str(exc)}, 503)

    # ── /api/stt (speech-to-text via Gemini audio) ─────────
    def _handle_stt(self, body: bytes):
        """
        Accepts JSON: { "audio": "<data URI or base64>", "mime": "audio/webm",
                        "lang": "en" | "hi" }
        Returns: { success, text } or { success: false, error }.
        """
        try:
            data = json.loads(body.decode("utf-8"))
        except Exception:
            self._json({"success": False, "error": "Invalid JSON"}, 400)
            return
        audio = (data.get("audio") or "").strip()
        if not audio:
            self._json({"success": False, "error": "audio required"}, 400)
            return
        mime = (data.get("mime") or "audio/webm").strip()
        if audio.startswith("data:"):
            header, audio = audio.split(",", 1)
            m = re.match(r"data:([a-zA-Z0-9/+.-]+);", header)
            if m: mime = m.group(1)
        lang = (data.get("lang") or "en").strip().lower()

        try:
            from stt import transcribe
            result = transcribe(audio, mime=mime, lang=lang)
            self._json(result if result.get("success") else {**result, "success": False},
                        200 if result.get("success") else 503)
        except Exception as exc:
            print(f"  [stt] failed: {exc}")
            self._json({"success": False, "error": str(exc)}, 503)

    # ── /api/quiz-analyze ──────────────────────────────────
    def _handle_quiz_analyze(self, body: bytes):
        """
        Accepts: { "response": str, "context": str (optional) }
        Returns structured language detection + tag extraction via AWS Bedrock.
        Falls back to heuristic analysis if Bedrock is unreachable.
        """
        data    = json.loads(body.decode("utf-8"))
        user_resp    = data.get("response", "").strip()
        quiz_context = data.get("context", "")

        if not user_resp:
            self._json({"success": False, "error": "No response provided"}, 400)
            return

        result = analyze_quiz_response(user_resp, quiz_context)
        self._json(result)

    # ── /api/upload ────────────────────────────────────────
    def _handle_upload(self, body: bytes):
        content_type = self.headers.get("Content-Type", "")

        if "application/json" in content_type:
            data      = json.loads(body.decode("utf-8"))
            img_data  = data.get("image", "")

            if img_data.startswith("data:"):
                _, b64 = img_data.split(",", 1)
                img_bytes = base64.b64decode(b64)
            else:
                img_bytes = base64.b64decode(img_data)

            fname  = f"{uuid.uuid4()}.jpg"
            fpath  = UPLOADS / fname
            fpath.write_bytes(img_bytes)

            img_url     = f"/uploads/{fname}"
            face_result = analyze_face(img_url)

            self._json({
                "success":    True,
                "url":        img_url,
                "filename":   fname,
                "face_data":  face_result if face_result.get("success") else None
            })
        else:
            self._json({"success": False, "error": "Send JSON with base64 image"}, 400)

    # ── Helpers ────────────────────────────────────────────
    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _file(self, fpath: Path, content_type: str):
        if not fpath.exists():
            self.send_response(404)
            self.end_headers()
            return
        data = fpath.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",   content_type)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run(port: int = 8000):
    import socketserver

    has_key  = bool(os.environ.get("ANTHROPIC_API_KEY"))
    key_line = "✅ ANTHROPIC_API_KEY found — live Claude AI enabled" if has_key \
               else "⚠️  No ANTHROPIC_API_KEY — running in demo mode"
    db_line  = f"🛍️  Product DB: {_product_db.count()} products loaded" if _DB_READY \
               else "⚠️  Product DB not available"

    print("\n" + "─" * 56)
    print("  🔵  Lenskart Claire AI Backend")
    print("─" * 56)
    print(f"  🚀  http://localhost:{port}")
    print(f"  {key_line}")
    print(f"  {db_line}")
    print("  Press Ctrl+C to stop")
    print("─" * 56 + "\n")

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", port), ClaireHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  👋  Server stopped. Goodbye!\n")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    run(port)
