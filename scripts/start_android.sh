#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WAKE_LOCK_ACQUIRED=0
SERVER_PID=""

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [ "${WAKE_LOCK_ACQUIRED}" -eq 1 ]; then
    WAKE_LOCK_ACQUIRED=0
    if command -v termux-wake-unlock >/dev/null 2>&1; then
      termux-wake-unlock || true
      echo "[Hidden Gem Radio] termux-wake-unlock completed."
    fi
  fi
}

on_signal() {
  cleanup
  exit 0
}

trap cleanup EXIT
trap on_signal INT TERM

if command -v termux-wake-lock >/dev/null 2>&1; then
  termux-wake-lock || true
  WAKE_LOCK_ACQUIRED=1
  echo "[Hidden Gem Radio] termux-wake-lock enabled (background audio protection)."
fi

# Refresh stations.yaml automatically if watch-history.json exists and is newer than stations.yaml
if [ -f "data/watch-history.json" ]; then
  if [ ! -f "config/stations.yaml" ] || [ "data/watch-history.json" -nt "config/stations.yaml" ] || ! grep -q "ukg_bass_bounce" "config/stations.yaml"; then
    echo "[Hidden Gem Radio] Updating config/stations.yaml from data/watch-history.json..."
    python3 -m src.digger.history_importer --stations 8 --offline || true
  fi
fi

echo "[Hidden Gem Radio] Starting server on http://${HOST}:${PORT}/ (Brave/Android ready)"
python3 -m uvicorn src.radio_server:app --host "${HOST}" --port "${PORT}" &
SERVER_PID=$!

# Poll /health up to 15 seconds (30 x 0.5s) for readiness & early crash detection
HEALTH_URL="http://${HOST}:${PORT}/health"
READY=0
for _ in $(seq 1 30); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[Hidden Gem Radio] Server exited early during startup." >&2
    break
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 1 "${HEALTH_URL}" >/dev/null 2>&1; then
      READY=1
      break
    fi
  else
    if python3 -c "import urllib.request; urllib.request.urlopen('${HEALTH_URL}', timeout=1).read()" >/dev/null 2>&1; then
      READY=1
      break
    fi
  fi
  sleep 0.5
done

if [ "${READY}" -eq 1 ]; then
  echo "[Hidden Gem Radio] Server healthy at ${HEALTH_URL}"
fi

wait "${SERVER_PID}"
