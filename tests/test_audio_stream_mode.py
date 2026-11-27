"""Regression tests for backend /api/audio/{video_id} Range proxy, batch prefetch, and HTML5 <audio> mode."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from src.adapters.audio_stream import AudioStreamResolver, CachedAudioStream
from src.models import TrackEntity
from src.radio_server import create_app

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _extract_inline_js() -> str:
    html = WEB_INDEX.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    assert scripts, "No inline <script> block found in web/index.html"
    return "\n".join(scripts)


def test_audio_stream_proxy_range_206_and_batch_background_prefetch(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 4  # 1024-byte synthetic audio payload
    extract_calls: list[str] = []

    def fake_extractor(video_id: str) -> CachedAudioStream | None:
        extract_calls.append(video_id)
        return CachedAudioStream(
            video_id=video_id,
            url=f"https://upstream.example.test/audio/{video_id}.m4a",
            mime_type="audio/mp4",
            content_length=len(payload),
            http_headers={"User-Agent": "TestAgent/1.0"},
        )

    def mock_upstream_handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start_s, end_s = rng.replace("bytes=", "").split("-", 1)
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(payload) - 1
            end = min(end, len(payload) - 1)
            chunk = payload[start : end + 1]
            return httpx.Response(
                status_code=206,
                content=chunk,
                headers={
                    "Content-Type": "audio/mp4",
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                    "Content-Length": str(len(chunk)),
                },
            )
        return httpx.Response(
            status_code=200,
            content=payload,
            headers={
                "Content-Type": "audio/mp4",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(payload)),
            },
        )

    resolver = AudioStreamResolver(extractor=fake_extractor, ttl_seconds=1800.0)
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
        audio_resolver=resolver,
        audio_http_factory=lambda: httpx.Client(transport=httpx.MockTransport(mock_upstream_handler)),
    )
    db = app.state.runtime.db
    for idx in range(4):
        db.upsert_track(
            "pes_passion_groove",
            TrackEntity.build(
                artist=f"Audio Artist {idx}",
                title=f"Audio Song {idx}",
                view_count=40_000,
                video_id=f"AUD_VID_{idx:03d}",
            ),
        )

    client = TestClient(app)

    # 1. /health reports audio_stream_available=True
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["audio_stream_available"] is True

    # 2. /batch triggers background prefetch for top 3 tracks
    batch_res = client.get("/api/stations/pes_passion_groove/batch", params={"count": 4})
    assert batch_res.status_code == 200
    items = batch_res.json()["items"]
    assert len(items) == 4
    top3_vids = [item["video_id"] for item in items[:3]]

    # Wait briefly for background prefetch threads to populate cache
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if all(resolver.get_cached(v) is not None for v in top3_vids):
            break
        time.sleep(0.02)
    assert all(resolver.get_cached(v) is not None for v in top3_vids), "Top 3 batch tracks must be prefetched in cache"

    # 3. GET /api/audio/{vid} with Range: bytes=0-255 serves 206 Partial Content from cache (zero re-extraction!)
    target_vid = top3_vids[0]
    calls_before = len(extract_calls)
    r_206 = client.get(f"/api/audio/{target_vid}", headers={"Range": "bytes=0-255"})
    assert r_206.status_code == 206
    assert r_206.headers.get("accept-ranges") == "bytes"
    assert r_206.headers.get("content-range") == f"bytes 0-255/{len(payload)}"
    assert r_206.headers.get("content-type", "").startswith("audio/mp4")
    assert r_206.content == payload[:256]
    assert len(extract_calls) == calls_before, "Prefetched video_id must serve directly from memory cache"

    # 4. Full GET /api/audio/{vid} returns 200 with complete payload
    r_200 = client.get(f"/api/audio/{target_vid}")
    assert r_200.status_code == 200
    assert r_200.headers.get("accept-ranges") == "bytes"
    assert r_200.content == payload

    # 5. Invalid video_id rejected with 400
    assert client.get("/api/audio/bad%20id").status_code == 400


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS vm tests")
def test_web_ui_audio_mode_zero_await_ended_background_and_fallback_node() -> None:
    inline_js = _extract_inline_js()
    harness = r"""
const vm = require("vm");

function makeElement(id) {
  const listeners = {};
  return {
    id,
    textContent: "",
    innerHTML: "",
    value: "",
    src: "",
    style: {},
    className: "",
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
      toggle(c, force) {
        if (force === undefined) {
          if (this._set.has(c)) this._set.delete(c);
          else this._set.add(c);
        } else if (force) {
          this._set.add(c);
        } else {
          this._set.delete(c);
        }
      },
    },
    children: [],
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(evt, fn) {
      listeners[evt] = listeners[evt] || [];
      listeners[evt].push(fn);
    },
    dispatchSync(evt) {
      for (const fn of (listeners[evt] || [])) {
        fn({ target: this, type: evt });
      }
    },
    async dispatchAsync(evt) {
      for (const fn of (listeners[evt] || [])) {
        await fn({ target: this, type: evt });
      }
    },
  };
}

function createAudioEnv(fetchImpl) {
  const elements = {};
  const audioPlayCalls = [];
  const ytLoadCalls = [];

  const getEl = (id) => {
    if (!elements[id]) {
      const el = makeElement(id);
      if (id === "nativeAudioPlayer") {
        el.paused = true;
        el.muted = false;
        el.currentTime = 0;
        el.duration = 200;
        el.play = function () {
          el.paused = false;
          audioPlayCalls.push(el.src);
          return Promise.resolve();
        };
        el.pause = function () {
          el.paused = true;
        };
      }
      elements[id] = el;
    }
    return elements[id];
  };

  const mediaSessionObj = {
    metadata: null,
    playbackState: "none",
    actions: {},
    setActionHandler(name, fn) { this.actions[name] = fn; },
  };
  const docObj = {
    hidden: false,
    getElementById: getEl,
    createElement: (tag) => makeElement(tag),
    addEventListener: () => {},
  };

  const sandbox = {
    console,
    Date,
    Math,
    Number,
    String,
    Boolean,
    Array,
    Object,
    Set,
    Map,
    Promise,
    Error,
    encodeURIComponent,
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    AbortController: typeof AbortController !== "undefined" ? AbortController : undefined,
    document: docObj,
    window: {
      location: { origin: "http://127.0.0.1:8765" },
      addEventListener: () => {},
    },
    navigator: { mediaSession: mediaSessionObj },
    MediaMetadata: function (obj) { return obj; },
    YT: {
      PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
      Player: function (_elId, opts) {
        setTimeout(() => opts.events.onReady(), 0);
        return {
          loadVideoById(vid) { ytLoadCalls.push(vid); },
          playVideo() {},
          stopVideo() {},
          pauseVideo() {},
          isMuted() { return false; },
          mute() {},
          unMute() {},
          getPlayerState() { return 1; },
          getCurrentTime() { return 0; },
          getDuration() { return 200; },
        };
      },
    },
    fetch: fetchImpl,
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(
    INLINE_JS +
      "\n;globalThis.__exports = { state, el, isAudioModeActive, playTrack, playNextFromQueue, setPlaybackMode };",
    sandbox
  );
  sandbox.window.onYouTubeIframeAPIReady();
  return {
    exports: sandbox.__exports,
    docObj,
    audioPlayCalls,
    ytLoadCalls,
    mediaSessionObj,
  };
}

async function runAll() {
  // Scenario A: Audio mode synchronous zero-await ended handler in background (document.hidden === true)
  {
    const fetchCalls = [];
    const env = createAudioEnv((url, opts = {}) => {
      fetchCalls.push({ url, opts });
      if (url === "/feedback") {
        // Deliberately NEVER resolves! If ended awaited /feedback, next track would NEVER start.
        return new Promise(() => {});
      }
      if (url === "/health") {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ status: "ok", audio_stream_available: true }) });
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ items: [], stations: [], playlists: [] }) });
    });

    const { state, el, isAudioModeActive, playNextFromQueue } = env.exports;
    state.activeStationId = "pes_passion_groove";
    state.audioStreamAvailable = true;
    state.playbackMode = "audio";
    if (!isAudioModeActive()) {
      throw new Error("Scenario A failed: audio mode should be active by default when nativeAudioPlayer has .play()");
    }

    state.queue = [
      { track_id: "trk_a1", video_id: "VID_AUD_1", artist: "Parcels", title: "Audio Song 1", duration_seconds: 195 },
      { track_id: "trk_a2", video_id: "VID_AUD_2", artist: "Phoenix", title: "Audio Song 2", duration_seconds: 210 },
      { track_id: "trk_a3", video_id: "VID_AUD_3", artist: "Oberhofer", title: "Audio Song 3", duration_seconds: 180 },
    ];

    await playNextFromQueue();
    if (el.nativeAudioPlayer.src !== "/api/audio/VID_AUD_1") {
      throw new Error(`Scenario A failed: expected /api/audio/VID_AUD_1, got ${el.nativeAudioPlayer.src}`);
    }

    // Switch to background (LINE / Gemini open) and fire ended SYNCHRONOUSLY
    env.docObj.hidden = true;
    fetchCalls.length = 0;
    env.audioPlayCalls.length = 0;

    el.nativeAudioPlayer.dispatchSync("ended");

    // MUST already have updated audio.src & called audio.play() SYNCHRONOUSLY before any await!
    if (el.nativeAudioPlayer.src !== "/api/audio/VID_AUD_2") {
      throw new Error(`Scenario A failed: audio.src not updated synchronously on ended (got ${el.nativeAudioPlayer.src})`);
    }
    if (env.audioPlayCalls[0] !== "/api/audio/VID_AUD_2") {
      throw new Error(`Scenario A failed: audio.play() not called synchronously for VID_AUD_2`);
    }
    if (state.currentTrack.video_id !== "VID_AUD_2") {
      throw new Error(`Scenario A failed: currentTrack not updated to VID_AUD_2`);
    }
    if (!env.mediaSessionObj.metadata || env.mediaSessionObj.metadata.title !== "Audio Song 2") {
      throw new Error("Scenario A failed: MediaSession metadata not updated synchronously");
    }
    const fb = fetchCalls.find((c) => c.url === "/feedback");
    if (!fb || fb.opts.keepalive !== true || JSON.parse(fb.opts.body).event_type !== "COMPLETE") {
      throw new Error("Scenario A failed: fire-and-forget keepalive COMPLETE not sent for Track 1");
    }
  }

  // Scenario B: Audio mode errors NEVER call /unplayable (no 30d blacklist) and auto-fallback to YouTube after 3 errors
  {
    const unplayableCalls = [];
    const env = createAudioEnv((url) => {
      if (url.includes("/unplayable")) unplayableCalls.push(url);
      if (url === "/health") {
        return Promise.resolve({ ok: true, status: 200, json: async () => ({ status: "ok", audio_stream_available: true }) });
      }
      return Promise.resolve({ ok: true, status: 200, json: async () => ({ items: [], stations: [], playlists: [] }) });
    });
    const { state, el, playNextFromQueue } = env.exports;
    state.playerReady = true;
    state.activeStationId = "pes_passion_groove";
    state.audioStreamAvailable = true;
    state.playbackMode = "audio";
    state.queue = [
      { track_id: "trk_e1", video_id: "VID_ERR_1", artist: "Art 1", title: "Err 1" },
      { track_id: "trk_e2", video_id: "VID_ERR_2", artist: "Art 2", title: "Err 2" },
      { track_id: "trk_e3", video_id: "VID_ERR_3", artist: "Art 3", title: "Err 3" },
    ];
    await playNextFromQueue();

    // 1st error on track 1 -> retries once (?retry=1)
    await el.nativeAudioPlayer.dispatchAsync("error");
    if (!el.nativeAudioPlayer.src.includes("retry=1")) {
      throw new Error("Scenario B failed: expected 1x 4G retry on first audio error");
    }
    // Trigger remaining errors to reach MAX_CONSECUTIVE_AUDIO_ERRORS (3) -> auto-fallback to YouTube mode
    await el.nativeAudioPlayer.dispatchAsync("error"); // err #1 -> moves to VID_ERR_2
    await el.nativeAudioPlayer.dispatchAsync("error"); // retry VID_ERR_2
    await el.nativeAudioPlayer.dispatchAsync("error"); // err #2 -> moves to VID_ERR_3
    await el.nativeAudioPlayer.dispatchAsync("error"); // retry VID_ERR_3
    await el.nativeAudioPlayer.dispatchAsync("error"); // err #3 -> auto-fallback to youtube

    if (unplayableCalls.length !== 0) {
      throw new Error(`Scenario B failed: audio errors must NOT call /unplayable (got ${unplayableCalls.length})`);
    }
    if (state.playbackMode !== "youtube") {
      throw new Error(`Scenario B failed: expected auto-fallback to youtube mode, got ${state.playbackMode}`);
    }
  }

  console.log(JSON.stringify({ ok: true }));
  process.exit(0);
}

runAll().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    script = f"const INLINE_JS = {json.dumps(inline_js)};\n{harness}"
    proc = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"Node audio mode vm test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert '"ok":true' in proc.stdout.replace(" ", "")
