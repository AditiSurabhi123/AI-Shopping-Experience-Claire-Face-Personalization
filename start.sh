#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
#  Lenskart Claire AI — Quick Start Script
# ─────────────────────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source .env if present, so GEMINI_API_KEY / BEDROCK_BEARER_TOKEN / etc.
# don't need to be exported manually on every run.
if [ -f "${SCRIPT_DIR}/.env" ]; then
  set -o allexport
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/.env"
  set +o allexport
fi

PORT="${PORT:-8000}"

echo ""
echo "  ██████╗██╗      █████╗ ██╗██████╗ ██████╗ ███████╗"
echo "  ██╔════╝██║     ██╔══██╗██║██╔══██╗██╔══██╗██╔════╝"
echo "  ██║     ██║     ███████║██║██████╔╝██████╔╝█████╗"
echo "  ██║     ██║     ██╔══██║██║██╔══██╗██╔══██╗██╔══╝"
echo "  ╚██████╗███████╗██║  ██║██║██║  ██║██║  ██║███████╗"
echo "   ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝"
echo ""
echo "  AI-powered eyewear assistant — Claire"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "  ❌  Python 3 is required. Install from https://python.org"
  exit 1
fi

# Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo "  ℹ️   No ANTHROPIC_API_KEY set — running in demo mode"
  echo "  💡  To enable live Claude AI:"
  echo "      export ANTHROPIC_API_KEY='sk-ant-...'"
  echo ""
else
  echo "  ✅  ANTHROPIC_API_KEY found — live Claude AI enabled"
  echo ""
fi

# Auto-kill any process already bound to $PORT so restarts are clean
EXISTING_PID="$(lsof -ti tcp:${PORT} 2>/dev/null || true)"
if [ -n "$EXISTING_PID" ]; then
  echo "  ⚠️   Port ${PORT} in use by PID ${EXISTING_PID} — killing it"
  kill -9 $EXISTING_PID 2>/dev/null || true
  sleep 0.5
fi

# Start server
cd "$SCRIPT_DIR"
echo "  🚀  Starting server on http://localhost:${PORT}"
echo "  📱  Open your browser to http://localhost:${PORT}"
echo "  ⌨️   Press Ctrl+C to stop"
echo ""

exec python3 backend/server.py
