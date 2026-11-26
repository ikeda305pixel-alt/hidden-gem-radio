#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WAKE_LOCK_ACQUIRED=0

cleanup() {
  if [ "${WAKE_LOCK_ACQUIRED}" -eq 1 ] && command -v termux-wake-unlock >/dev/null 2>&1; then
    termux-wake-unlock || true
    echo "[Hidden Gem Radio] termux-wake-unlock completed."
  fi
}
trap cleanup EXIT INT TERM

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
  WAKE_LOCK_ACQUIRED=1
  echo "[Hidden Gem Radio] termux-wake-lock enabled (background audio protection)."
fi

# Refresh stations.yaml automatically if watch-history.json exists and is newer than stations.yaml
if [ -f "data/watch-history.json" ]; then
  if [ ! -f "config/stations.yaml" ] || [ "data/watch-history.json" -nt "config/stations.yaml" ]; then
    echo "[Hidden Gem Radio] Updating config/stations.yaml from data/watch-history.json..."
    python3 -m src.digger.history_importer --stations 5 --offline || true
  fi
fi

echo "[Hidden Gem Radio] Starting server on http://${HOST}:${PORT}/ (Brave/Android ready)"
exec python3 -m uvicorn src.radio_server:app --host "${HOST}" --port "${PORT}"
