"""Real-execution regression tests for all 13 blind-audit findings (Critical 2 / High 5 / Medium 4 / Low 2)."""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.db import RadioDatabase, build_youtube_watch_videos_urls, sanitize_video_id
from src.digger.history_importer import (
    _iter_json_array_objects,
    build_weighted_theory_stations,
    classify_entry_to_station,
    parse_takeout_watch_history,
)
from src.models import StationConfig, TrackEntity, save_stations_yaml
from src.radio_server import create_app
from src.recommender.theory_features import TheoryFeatureExtractor

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_station_app(tmp_path: Path, history_index: dict[str, list[str]] | None = None):
    stations_yaml = tmp_path / "stations.yaml"
    save_stations_yaml(
        stations_yaml,
        [
            StationConfig(
                station_id="audit_fm",
                name="Audit FM",
                seed_tags=["uk garage", "hip hop"],
                seed_artists=["Audit Artist"],
                max_popularity_views=500_000,
                exploration_rate=0.9,
            )
        ],
        history_index=history_index,
    )
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=stations_yaml,
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    return app, app.state.runtime


# ---------------------------------------------------------------------------
# [Critical #1, High #3, High #4, Medium #10, Medium #11]
# Real Node.js execution of web/index.html inline <script>
# ---------------------------------------------------------------------------
def test_web_ui_inline_script_real_node_execution() -> None:
    """Execute the real web/index.html <script> in Node.js vm and verify all 5 UI fixes."""
    html = (PROJECT_ROOT / "web/index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, flags=re.DOTALL)
    inline_js = "\n".join(s for s in scripts if s.strip())
    assert inline_js, "Inline script not found in web/index.html"

    harness = r"""
const assert = require("assert");
const vm = require("vm");
const fs = require("fs");

const inlineCode = fs.readFileSync(process.argv[1], "utf8");

function makeClassList() {
  const set = new Set();
  return {
    add: (...cls) => cls.forEach((c) => set.add(c)),
    remove: (...cls) => cls.forEach((c) => set.delete(c)),
    toggle: (c, force) => (force === undefined ? (set.has(c) ? set.delete(c) : set.add(c)) : (force ? set.add(c) : set.delete(c))),
    contains: (c) => set.has(c),
  };
}

function makeElement(id) {
  const listeners = {};
  return {
    id,
    textContent: "",
    innerHTML: "",
    value: "pl_favorites",
    className: "",
    children: [],
    classList: makeClassList(),
    appendChild(child) { this.children.push(child); return child; },
    addEventListener(type, fn) { (listeners[type] = listeners[type] || []).push(fn); },
    async dispatch(type) {
      for (const fn of listeners[type] || []) {
        await fn({ target: this });
      }
    },
  };
}

// Deterministic virtual clock so 7000ms watchdog & 2500ms cooldown run instantaneously
let nowMs = 1_000_000;
let nextTimerId = 1;
const timers = new Map();

function vSetTimeout(fn, delay) {
  const id = nextTimerId++;
  timers.set(id, { id, fn, runAt: nowMs + Number(delay || 0), interval: null });
  return id;
}
function vClearTimeout(id) { timers.delete(id); }
function vSetInterval(fn, delay) {
  const id = nextTimerId++;
  timers.set(id, { id, fn, runAt: nowMs + Number(delay || 0), interval: Math.max(1, Number(delay || 1)) });
  return id;
}
function vClearInterval(id) { timers.delete(id); }

async function advanceTime(ms) {
  const target = nowMs + ms;
  while (true) {
    let next = null;
    for (const t of timers.values()) {
      if (t.runAt <= target && (!next || t.runAt < next.runAt)) next = t;
    }
    if (!next) break;
    nowMs = next.runAt;
    if (next.interval !== null) {
      next.runAt = nowMs + next.interval;
    } else {
      timers.delete(next.id);
    }
    await next.fn();
    await new Promise((r) => setImmediate(r));
  }
  nowMs = target;
}

const elements = {};
const documentStub = {
  hidden: false,
  getElementById(id) {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  },
  createElement(tag) { return makeElement(tag); },
};

const mediaHandlers = {};
const navigatorStub = {
  mediaSession: {
    metadata: null,
    playbackState: "none",
    setActionHandler(name, fn) { mediaHandlers[name] = fn; },
  },
};

let fetchHandler = async (url, opts) => ({ ok: true, status: 200, json: async () => ({}) });
const fetchCalls = [];
async function fetchStub(url, opts) {
  fetchCalls.push({ url, opts });
  return fetchHandler(url, opts);
}

const YTStub = {
  PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
  Player: function (_id, config) {
    this._config = config;
    this._state = -1;
    this._time = 0;
    this._muted = false;
    this.loadedVideos = [];
    this.playCalls = 0;
    this.pauseCalls = 0;
    this.loadVideoById = (vid) => { this.loadedVideos.push(vid); };
    this.playVideo = () => { this.playCalls += 1; this._state = 1; };
    this.pauseVideo = () => { this.pauseCalls += 1; this._state = 2; };
    this.getPlayerState = () => this._state;
    this.getCurrentTime = () => this._time;
    this.getDuration = () => 180;
    this.isMuted = () => this._muted;
    this.mute = () => { this._muted = true; };
    this.unMute = () => { this._muted = false; };
  },
};

const DateProxy = class extends Date {
  static now() { return nowMs; }
};

const sandbox = {
  console,
  document: documentStub,
  navigator: navigatorStub,
  MediaMetadata: function (m) { Object.assign(this, m); },
  YT: YTStub,
  fetch: fetchStub,
  Date: DateProxy,
  setTimeout: vSetTimeout,
  clearTimeout: vClearTimeout,
  setInterval: vSetInterval,
  clearInterval: vClearInterval,
  encodeURIComponent,
  Promise,
  Set,
  Array,
  Object,
  Number,
  String,
  Boolean,
  Math,
  JSON,
  Error,
};
sandbox.window = sandbox;

// Initial bootstrap returns empty stations so tests control state deterministically
fetchHandler = async (url) => {
  if (url === "/stations") return { ok: true, status: 200, json: async () => ({ stations: [] }) };
  if (url === "/api/playlists") return { ok: true, status: 200, json: async () => ({ playlists: [{ playlist_id: "pl_favorites", name: "Good お気に入り", track_count: 0 }] }) };
  if (url.startsWith("/api/playlists/")) return { ok: true, status: 200, json: async () => ({ tracks: [], track_count: 0, youtube_watch_urls: [] }) };
  if (url === "/api/youtube/playlists") return { ok: true, status: 200, json: async () => ({ authenticated: false, playlists: [] }) };
  if (url.startsWith("/api/profile/theory")) return { ok: true, status: 200, json: async () => ({ updates: 0, axes: [] }) };
  return { ok: true, status: 200, json: async () => ({ items: [] }) };
};

const ctx = vm.createContext(sandbox);
vm.runInContext(
  inlineCode + "\nwindow.__testExports = { state, el, MAX_CONSECUTIVE_ERRORS, PLAYBACK_WATCHDOG_MS, EMPTY_BATCH_COOLDOWN_MS, armPlaybackWatchdog, handleUnplayableSkip, ensurePrefetchQueue, playTrack, playNextFromQueue, switchStation };",
  ctx
);

(async () => {
  await new Promise((r) => setImmediate(r));
  const api = sandbox.__testExports;
  sandbox.onYouTubeIframeAPIReady();
  api.state.player._config.events.onReady();
  api.state.stations = [
    { station_id: "st_a", name: "Station A", candidate_count: 10 },
    { station_id: "st_b", name: "Station B", candidate_count: 10 },
  ];
  api.state.activeStationId = "st_a";

  // -------------------------------------------------------------------------
  // Case 1 [High #3]: Watchdog MUST NOT fire after confirmed playback + PAUSE,
  // and MUST NOT blacklist on initial autoplay-blocked UNSTARTED state.
  // -------------------------------------------------------------------------
  fetchCalls.length = 0;
  const trackAutoplay = { track_id: "t_auto", video_id: "VID_AUTO_01", artist: "Art A", title: "Song A" };
  api.state.hasUserPlaybackStarted = false;
  api.state.player._state = YTStub.PlayerState.UNSTARTED;
  api.state.player._time = 0;
  api.playTrack(trackAutoplay);
  await advanceTime(8000);
  assert.strictEqual(
    fetchCalls.filter((c) => c.url.includes("/unplayable")).length,
    0,
    "Autoplay-blocked initial UNSTARTED track must NOT be blacklisted by watchdog"
  );

  // Confirmed playback (>0.3s) followed by user pause -> watchdog must never trigger
  fetchCalls.length = 0;
  const trackConfirmed = { track_id: "t_conf", video_id: "VID_CONF_01", artist: "Art B", title: "Song B" };
  api.state.player._state = YTStub.PlayerState.PLAYING;
  api.state.player._time = 0.0;
  api.playTrack(trackConfirmed);
  api.state.player._time = 1.2;
  await advanceTime(600); // poll confirms playback
  assert.strictEqual(api.state.trackConfirmedPlayingVideoId, "VID_CONF_01");
  assert.strictEqual(api.state.hasUserPlaybackStarted, true);
  api.state.player._state = YTStub.PlayerState.PAUSED;
  await api.state.player._config.events.onStateChange({ data: YTStub.PlayerState.PAUSED });
  await advanceTime(9000);
  assert.strictEqual(
    fetchCalls.filter((c) => c.url.includes("/unplayable")).length,
    0,
    "Confirmed + paused track must NOT be blacklisted by watchdog"
  );

  // Genuine stall after user playback started -> watchdog MUST call /unplayable
  fetchCalls.length = 0;
  const trackStalled = { track_id: "t_stall", video_id: "VID_STALL_01", artist: "Art C", title: "Stall" };
  api.state.player._state = YTStub.PlayerState.BUFFERING;
  api.state.player._time = 0;
  api.playTrack(trackStalled);
  await advanceTime(7500);
  assert.strictEqual(
    fetchCalls.filter((c) => c.url.includes("/api/tracks/VID_STALL_01/unplayable")).length,
    1,
    "Genuine stalled track after playback started must trigger /unplayable once"
  );

  // -------------------------------------------------------------------------
  // Case 2 [High #4]: handleUnplayableSkip re-entrancy lock & stale station/epoch guard
  // -------------------------------------------------------------------------
  fetchCalls.length = 0;
  api.state.activeStationId = "st_a";
  api.state.consecutiveErrors = 0;
  api.state.circuitBroken = false;
  api.state.queue = [];
  let resolveUnplayable = null;
  fetchHandler = (url) => {
    if (url.includes("/unplayable")) {
      return new Promise((resolve) => {
        resolveUnplayable = () => resolve({
          ok: true,
          status: 200,
          json: async () => ({
            replacement: { track_id: "stale_rep", video_id: "VID_STALE_REP", artist: "Stale", title: "From A" },
          }),
        });
      });
    }
    if (url.includes("/api/stations/st_b/batch")) {
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          items: [{ track_id: "b_track_1", video_id: "VID_B_00001", artist: "Artist B", title: "Song B1" }],
        }),
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => ({ items: [] }) });
  };

  const trackDup = { track_id: "t_dup", video_id: "VID_DUP_001", artist: "Art D", title: "Dup" };
  api.state.currentTrack = trackDup;
  const p1 = api.handleUnplayableSkip(trackDup, "yt_error_150", 150);
  const p2 = api.handleUnplayableSkip(trackDup, "stall_watchdog_7s", null); // re-entrant duplicate!
  await p2;
  assert.strictEqual(
    fetchCalls.filter((c) => c.url.includes("/api/tracks/VID_DUP_001/unplayable")).length,
    1,
    "Re-entrant handleUnplayableSkip for same video_id must be ignored"
  );

  // Switch to station B before station A's /unplayable resolves
  await api.switchStation("st_b");
  assert.strictEqual(api.state.activeStationId, "st_b");
  resolveUnplayable();
  await p1;
  assert.strictEqual(api.state.currentTrack.video_id, "VID_B_00001", "Stale unplayable response must not override station B track");
  assert.strictEqual(
    api.state.queue.some((q) => q.track_id === "stale_rep"),
    false,
    "Stale replacement from station A must not pollute station B queue"
  );

  // -------------------------------------------------------------------------
  // Case 3 [Critical #1]: True Circuit Breaker halts completely at MAX_CONSECUTIVE_ERRORS
  // -------------------------------------------------------------------------
  fetchCalls.length = 0;
  api.state.activeStationId = "st_a";
  api.state.consecutiveErrors = 0;
  api.state.circuitBroken = false;
  api.state.queue = [];
  let seq = 0;
  fetchHandler = async (url) => {
    if (url.includes("/unplayable")) {
      seq += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          replacement: { track_id: `t_bad_${seq}`, video_id: `VID_BAD_00${seq}`, artist: `Bad ${seq}`, title: `Fail ${seq}` },
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({ items: [] }) };
  };

  api.state.currentTrack = { track_id: "t_bad_0", video_id: "VID_BAD_000", artist: "Bad 0", title: "Fail 0" };
  for (let i = 0; i < api.MAX_CONSECUTIVE_ERRORS; i += 1) {
    await api.handleUnplayableSkip(api.state.currentTrack, "yt_error_150", 150);
  }
  assert.strictEqual(api.state.circuitBroken, true, "Circuit breaker must trip at MAX_CONSECUTIVE_ERRORS");
  assert.strictEqual(api.state.consecutiveErrors, api.MAX_CONSECUTIVE_ERRORS, "consecutiveErrors must NOT reset on trip");
  assert.strictEqual(api.state.retryTimerId, null, "scheduleQueueRetry must NOT be scheduled when circuit broken");
  assert.strictEqual(api.state.currentTrack, null);
  assert.ok(api.el.statusPill.textContent.includes("連続再生失敗のため自動スキップを停止しました"));

  // Advancing time must NOT trigger any automatic retries
  const callsAfterTrip = fetchCalls.length;
  await advanceTime(20_000);
  assert.strictEqual(fetchCalls.length, callsAfterTrip, "Zero network requests allowed while circuit breaker is tripped");

  // Manual play click resets circuit breaker and resumes immediately
  api.state.queue = [];
  fetchHandler = async (url) => {
    if (url.includes("/batch")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({ items: [{ track_id: "t_rec", video_id: "VID_REC_001", artist: "Recovered", title: "Back" }] }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  await api.el.btnPlayMute.dispatch("click");
  assert.strictEqual(api.state.circuitBroken, false, "Manual play button must reset circuitBroken");
  assert.strictEqual(api.state.consecutiveErrors, 0, "Manual play button must reset consecutiveErrors");
  assert.strictEqual(api.state.currentTrack.video_id, "VID_REC_001");

  // -------------------------------------------------------------------------
  // Case 4 [Medium #10]: Good button UI rollback on fetch error
  // -------------------------------------------------------------------------
  api.state.currentRatedGood = false;
  fetchHandler = async (url) => {
    if (url === "/feedback") return { ok: false, status: 500, json: async () => ({}) };
    return { ok: true, status: 200, json: async () => ({}) };
  };
  await api.el.btnGood.dispatch("click");
  assert.strictEqual(api.state.currentRatedGood, false, "currentRatedGood must roll back on failure");
  assert.strictEqual(api.el.btnGood.classList.contains("rated"), false, ".rated must be removed on failure");
  assert.strictEqual(api.el.postGoodBar.classList.contains("visible"), false, ".visible must be removed on failure");
  assert.ok(api.el.statusPill.textContent.includes("送信に失敗しました"));

  // Retrying Good when server recovers must now work
  fetchHandler = async (url) => {
    if (url === "/feedback") return { ok: true, status: 200, json: async () => ({ model_updates: 7 }) };
    if (url === "/api/playlists") return { ok: true, status: 200, json: async () => ({ playlists: [] }) };
    return { ok: true, status: 200, json: async () => ({ updates: 7, axes: [] }) };
  };
  await api.el.btnGood.dispatch("click");
  assert.strictEqual(api.state.currentRatedGood, true, "Retry after rollback must succeed");
  assert.strictEqual(api.el.btnGood.classList.contains("rated"), true);

  // -------------------------------------------------------------------------
  // Case 5 [Medium #11]: ensurePrefetchQueue callerEpoch guard + empty-batch cooldown
  // -------------------------------------------------------------------------
  fetchCalls.length = 0;
  api.state.activeStationId = "st_a";
  api.state.queue = [];
  api.state.prefetchEpoch += 1;
  api.state.prefetchPromise = null;
  let resolveEmptyBatch = null;
  fetchHandler = (url) => {
    if (url.includes("/batch")) {
      return new Promise((resolve) => {
        resolveEmptyBatch = () => resolve({ ok: true, status: 200, json: async () => ({ items: [] }) });
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
  };

  const w1 = api.ensurePrefetchQueue(3);
  const w2 = api.ensurePrefetchQueue(3);
  resolveEmptyBatch();
  await Promise.all([w1, w2]);
  assert.strictEqual(
    fetchCalls.filter((c) => c.url.includes("/batch")).length,
    1,
    "Empty batch cooldown must prevent second waiter from hammering /batch"
  );

  // Epoch change while waiting must also abort waiter cleanly
  fetchCalls.length = 0;
  api.state.prefetchEpoch += 1; // clears cooldown for new epoch
  let resolveInFlight = null;
  fetchHandler = (url) => {
    if (url.includes("/batch")) {
      return new Promise((resolve) => {
        resolveInFlight = () => resolve({ ok: true, status: 200, json: async () => ({ items: [{ track_id: "x", video_id: "VID_X0001", artist: "X", title: "X" }] }) });
      });
    }
    return Promise.resolve({ ok: true, status: 200, json: async () => ({}) });
  };
  const e1 = api.ensurePrefetchQueue(3);
  const e2 = api.ensurePrefetchQueue(3);
  api.state.prefetchEpoch += 1; // simulate BAD/station switch mid-flight
  resolveInFlight();
  await Promise.all([e1, e2]);
  assert.strictEqual(
    fetchCalls.filter((c) => c.url.includes("/batch")).length,
    1,
    "Epoch mismatch waiter must abort without firing a stale /batch request"
  );
  assert.strictEqual(api.state.queue.length, 0, "Stale epoch batch items must be discarded");

  console.log("ALL_WEB_UI_VM_ASSERTIONS_PASSED");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    js_file = PROJECT_ROOT / ".tmp_web_inline_test.js"
    try:
        js_file.write_text(inline_js, encoding="utf-8")
        res = subprocess.run(
            ["node", "-e", harness, str(js_file)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res.returncode == 0, f"Node UI test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "ALL_WEB_UI_VM_ASSERTIONS_PASSED" in res.stdout
    finally:
        js_file.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# [Critical #2] Streaming Takeout parser + tightened non-music filter
# ---------------------------------------------------------------------------
def test_streaming_takeout_parser_and_non_music_filter(tmp_path: Path) -> None:
    entries = [
        # Must be REJECTED by tightened non-music channel/episode filter
        {
            "header": "YouTube",
            "title": "Watched Linus Tech Tips - Tech Video #1",
            "titleUrl": "https://www.youtube.com/watch?v=TECHVID0001",
            "subtitles": [{"name": "Linus Tech Tips"}],
            "time": "2026-06-01T10:00:00Z",
        },
        {
            "header": "YouTube",
            "title": "Watched Coding Academy - Python Tips Ep. 12",
            "titleUrl": "https://www.youtube.com/watch?v=CODEVID0002",
            "subtitles": [{"name": "Coding Academy"}],
            "time": "2026-06-01T10:05:00Z",
        },
    ]
    # Add 1200+ entries (>250KB) to force multi-chunk streaming decode
    for i in range(1200):
        entries.append(
            {
                "header": "YouTube Music",
                "title": f"Watched Track Number {i}",
                "titleUrl": f"https://music.youtube.com/watch?v=MUSVID_{i:04d}",
                "subtitles": [{"name": "Avicii - Topic"}],
                "time": "2026-06-01T11:00:00Z",
            }
        )

    takeout_path = tmp_path / "watch-history.json"
    takeout_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    assert takeout_path.stat().st_size > 200_000

    # Verify small chunk_size streaming decodes every single element
    streamed = list(_iter_json_array_objects(takeout_path, chunk_size=1024))
    assert len(streamed) == len(entries)

    records = parse_takeout_watch_history(takeout_path)
    video_ids = {r["video_id"] for r in records}
    assert "TECHVID0001" not in video_ids
    assert "CODEVID0002" not in video_ids
    assert len(records) == 1200
    assert "MUSVID_0000" in video_ids and "MUSVID_1199" in video_ids


# ---------------------------------------------------------------------------
# [High #5] 30-day blacklist TTL automatic recovery + transient YT errors 2 & 5
# ---------------------------------------------------------------------------
def test_blacklist_30d_ttl_recovers_and_transient_errors_not_blacklisted(tmp_path: Path) -> None:
    app, runtime = _make_station_app(tmp_path)
    db: RadioDatabase = runtime.db

    t_ttl = TrackEntity.build("TTL Artist", "TTL Song", video_id="VID_TTL_001", view_count=10_000)
    t_err2 = TrackEntity.build("Err2 Artist", "Err2 Song", video_id="VID_ERR_002", view_count=10_000)
    t_err5 = TrackEntity.build("Err5 Artist", "Err5 Song", video_id="VID_ERR_005", view_count=10_000)
    t_perm = TrackEntity.build("Perm Artist", "Perm Song", video_id="VID_PRM_150", view_count=10_000)
    t_ok = TrackEntity.build("Safe Artist", "Safe Song", video_id="VID_SAFE_99", view_count=10_000)
    for t in (t_ttl, t_err2, t_err5, t_perm, t_ok):
        db.upsert_track("audit_fm", t)

    # 1. 30-day TTL check: blocked at +15d, automatically recovered at +31d
    t0 = 1_700_000_000.0
    db.blacklist_unplayable("VID_TTL_001", reason="stall_watchdog_7s", ttl_days=30, now=t0)
    mid = db.get_eligible_candidates("audit_fm", 500_000, now=t0 + 15 * 86400)
    assert "VID_TTL_001" not in {t.video_id for t in mid}
    after = db.get_eligible_candidates("audit_fm", 500_000, now=t0 + 31 * 86400)
    assert "VID_TTL_001" in {t.video_id for t in after}, "Track must auto-recover after 30-day TTL expires"

    # 2. Transient YT errors 2 and 5 MUST NOT blacklist or mark tracks.unplayable=1
    client = TestClient(app)
    for vid, code in (("VID_ERR_002", 2), ("VID_ERR_005", 5)):
        resp = client.post(
            f"/api/tracks/{vid}/unplayable",
            json={"station_id": "audit_fm", "reason": f"yt_error_{code}", "yt_error_code": code},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["blacklisted"] is False
        assert body["blacklisted_days"] == 0
        assert body["expires_at"] is None
        assert body["bad_learning_polluted"] is False
        assert body["replacement"] is not None
        assert vid not in db.get_blacklisted_video_ids()
        assert db.get_track(video_id=vid).unplayable is False

    # 3. Permanent YT error 150 MUST blacklist AND mark tracks.unplayable=1
    resp_perm = client.post(
        "/api/tracks/VID_PRM_150/unplayable",
        json={"station_id": "audit_fm", "reason": "yt_error_150", "yt_error_code": 150},
    )
    assert resp_perm.status_code == 200
    assert resp_perm.json()["blacklisted"] is True
    assert "VID_PRM_150" in db.get_blacklisted_video_ids()
    assert db.get_track(video_id="VID_PRM_150").unplayable is True


# ---------------------------------------------------------------------------
# [High #6] Per-station dig token set prevents waiting background token loss
# ---------------------------------------------------------------------------
def test_dig_token_set_preserves_waiting_background_token_after_sync_dig(tmp_path: Path) -> None:
    _, runtime = _make_station_app(tmp_path)
    station = runtime.get_station("audit_fm")

    sync_entered = threading.Event()
    release_sync = threading.Event()
    bg_entered = threading.Event()
    release_bg = threading.Event()
    call_order: list[str] = []

    class ControlledDigger:
        def dig_station(self, st, target_pool_size=60, **_kwargs):
            if not sync_entered.is_set():
                call_order.append("sync")
                sync_entered.set()
                release_sync.wait(timeout=5.0)
            else:
                call_order.append("bg")
                bg_entered.set()
                release_bg.wait(timeout=5.0)
            return {"station_id": st.station_id, "added": 0, "total_playable": 0}

    runtime._digger = ControlledDigger()

    # T2 acquires station dig lock synchronously
    t2 = threading.Thread(target=runtime._run_station_dig, args=(station,))
    t2.start()
    assert sync_entered.wait(timeout=3.0)

    # Simulate T1 waiting background dig registered while T2 holds lock
    t1_token = object()
    runtime._add_dig_token(station.station_id, t1_token)
    t1 = threading.Thread(
        target=runtime._run_station_dig,
        kwargs={"station": station, "owner_token": t1_token},
    )
    t1.start()
    time.sleep(0.05)  # allow T1 to block on station lock

    # Release T2: T2 finishes and releases ONLY T2's token; T1's token MUST remain active!
    release_sync.set()
    t2.join(timeout=3.0)
    assert bg_entered.wait(timeout=3.0)

    assert station.station_id in runtime._digging_stations
    assert runtime._trigger_background_dig(station) is False, "Duplicate T3 background dig must be blocked"

    release_bg.set()
    t1.join(timeout=3.0)
    assert station.station_id not in runtime._digging_stations
    assert call_order == ["sync", "bg"]


# ---------------------------------------------------------------------------
# [High #7] scripts/start_android.sh real subprocess SIGTERM & single wake-unlock
# ---------------------------------------------------------------------------
def test_start_android_sh_runs_wake_unlock_exactly_once_on_sigterm(tmp_path: Path) -> None:
    # Allocate a free ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    lock_log = tmp_path / "lock.log"
    unlock_log = tmp_path / "unlock.log"

    def _write_exec(name: str, body: str) -> None:
        p = fake_bin / name
        p.write_text(body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)

    _write_exec("termux-wake-lock", f"#!/usr/bin/env bash\necho locked >> '{lock_log}'\n")
    _write_exec("termux-wake-unlock", f"#!/usr/bin/env bash\necho unlocked >> '{unlock_log}'\n")

    # Fake python3 that handles `python3 -m uvicorn ...` by serving /health until SIGTERM
    real_python = sys.executable
    _write_exec(
        "python3",
        f"""#!/usr/bin/env bash
if [ "$1" = "-m" ] && [ "$2" = "uvicorn" ]; then
  exec "{real_python}" -c '
import http.server, signal, sys
port = int(sys.argv[1])
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{{\\"status\\":\\"ok\\"}}")
    def log_message(self, *args): pass
srv = http.server.HTTPServer(("127.0.0.1", port), H)
def _stop(_sig, _frame):
    sys.exit(0)
signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)
srv.serve_forever()
' "${{PORT}}"
else
  exec "{real_python}" "$@"
fi
""",
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["HOST"] = "127.0.0.1"
    env["PORT"] = str(free_port)

    proc = subprocess.Popen(
        ["bash", str(PROJECT_ROOT / "scripts/start_android.sh")],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        healthy_seen = False
        deadline = time.time() + 10.0
        output_lines: list[str] = []
        while time.time() < deadline:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                output_lines.append(line)
                if "Server healthy" in line:
                    healthy_seen = True
                    break
            elif proc.poll() is not None:
                break
        assert healthy_seen, f"start_android.sh did not reach /health ready:\n{''.join(output_lines)}"

        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=3.0)

    assert lock_log.exists() and len(lock_log.read_text(encoding="utf-8").splitlines()) == 1
    assert unlock_log.exists(), "termux-wake-unlock was NEVER called on SIGTERM!"
    unlock_lines = unlock_log.read_text(encoding="utf-8").splitlines()
    assert len(unlock_lines) == 1, f"Expected termux-wake-unlock exactly once, got {len(unlock_lines)}"


# ---------------------------------------------------------------------------
# [Medium #8] Fallback known-history exclusion & recency x frequency cap ordering
# ---------------------------------------------------------------------------
def test_fallback_excludes_known_history_and_known_index_is_weight_sorted(tmp_path: Path) -> None:
    known_track = TrackEntity.build("Known Artist", "Known Hit", video_id="VID_KNOWN_01", view_count=50_000)
    known_key_track = TrackEntity.build("Key Artist", "Key Hit", video_id="VID_OTHER_02", view_count=50_000)
    fresh_completed = TrackEntity.build("Gem Artist", "Completed Gem", video_id="VID_GEM_0003", view_count=50_000)

    _, runtime = _make_station_app(
        tmp_path,
        history_index={
            "known_video_ids": ["VID_KNOWN_01"],
            "known_track_keys": [known_key_track.canonical_key],
        },
    )
    for t in (known_track, known_key_track, fresh_completed):
        runtime.db.upsert_track("audit_fm", t)
        # Mark all 3 as previously COMPLETE so next_queue enters the heard-fallback branch
        runtime.db.record_feedback("audit_fm", t, "COMPLETE", 0.5)

    queue = runtime.next_queue("audit_fm", count=3)
    returned_vids = [item["video_id"] for item in queue]
    assert returned_vids == ["VID_GEM_0003"], f"Known history leaked into fallback: {returned_vids}"

    # Verify build_weighted_theory_stations sorts known_video_ids by recency x frequency weight
    ref = datetime(2026, 6, 1, 12, 0, 0)  # Also exercises [Low #13] naive reference_time!
    records = [
        {"artist": "Old One", "title": "Old", "video_id": "VID_OLD_LOW1", "played_at": "2024-01-01T00:00:00Z"},
        {"artist": "Old Two", "title": "Old2", "video_id": "VID_OLD_LOW2", "played_at": "2024-01-02T00:00:00Z"},
        *[
            {"artist": "Avicii", "title": "Levels", "video_id": "VID_HOT_TOP1", "played_at": "2026-05-31T00:00:00Z"}
            for _ in range(8)
        ],
        *[
            {"artist": "Drake", "title": "Hot", "video_id": "VID_HOT_TOP2", "played_at": "2026-05-30T00:00:00Z"}
            for _ in range(4)
        ],
    ]
    _, history_idx = build_weighted_theory_stations(records, reference_time=ref, max_known_items=2)
    assert history_idx["known_video_ids"] == ["VID_HOT_TOP1", "VID_HOT_TOP2"]
    assert len(history_idx["known_track_keys"]) == 2


# ---------------------------------------------------------------------------
# [Medium #9] Artist transliteration, collab split, and artist-before-remix priority
# ---------------------------------------------------------------------------
def test_station_classification_transliteration_collabs_and_artist_priority() -> None:
    assert classify_entry_to_station("¥ellow Bucks", "My Resort") == "japan_hiphop_street"
    assert classify_entry_to_station("STUTS & PUNPEE", "夜を使いはたして") == "japan_hiphop_street"
    assert classify_entry_to_station("JJJ feat. KID FRESINO", "2024") == "japan_hiphop_street"
    assert classify_entry_to_station("KEIJU feat. IO", "Play Fast") == "japan_hiphop_street"
    assert classify_entry_to_station("Drake & 21 Savage", "Rich Flex") == "us_hiphop_bouncy"
    assert (
        classify_entry_to_station("Kendrick Lamar", "HUMBLE. (SKRILLEX Remix)") == "us_hiphop_bouncy"
    ), "Explicit US Hip-Hop artist must not be hijacked to EDM by 'Remix' in title"
    assert classify_entry_to_station("Unknown Producer", "Midnight (Club Remix)") == "edm_dance_party"


# ---------------------------------------------------------------------------
# [Medium #10] Strict video_id sanitization & HTTP 400/422 validation
# ---------------------------------------------------------------------------
def test_video_id_sanitization_and_api_rejection(tmp_path: Path) -> None:
    assert sanitize_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert sanitize_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123") == "dQw4w9WgXcQ"
    assert sanitize_video_id("https://youtu.be/dQw4w9WgXcQ?t=42") == "dQw4w9WgXcQ"
    assert sanitize_video_id("abc&list=evil") is None
    assert sanitize_video_id("../etc/passwd") is None
    assert sanitize_video_id("short") is None

    urls = build_youtube_watch_videos_urls(["dQw4w9WgXcQ", "bad&inject=1", "VID_OK_1234"])
    assert urls == ["https://www.youtube.com/watch_videos?video_ids=dQw4w9WgXcQ,VID_OK_1234"]

    app, _ = _make_station_app(tmp_path)
    client = TestClient(app)

    assert (
        client.post(
            "/api/playlists/pl_favorites/tracks",
            json={"video_id": "bad&list=evil", "artist": "A", "title": "B"},
        ).status_code
        == 422
    )
    assert client.post("/api/youtube/sync", json={"video_id": "bad id!"}).status_code == 422
    assert client.post("/api/tracks/bad%20id/unplayable", json={}).status_code == 400
    assert client.delete("/api/playlists/pl_favorites/tracks/bad%20id").status_code == 400


# ---------------------------------------------------------------------------
# [Low #12] Smooth continuous differentiation for legacy tagless DB tracks
# ---------------------------------------------------------------------------
def test_theory_extractor_differentiates_legacy_tagless_tracks_continuously() -> None:
    extractor = TheoryFeatureExtractor()
    legacy_a = TrackEntity.build(
        "Legacy One", "Track One", video_id="VID_LEG_001", tags=[], duration_seconds=185, release_year=2023, view_count=12_000
    )
    legacy_b = TrackEntity.build(
        "Legacy Two", "Track Two", video_id="VID_LEG_002", tags=[], duration_seconds=310, release_year=2006, view_count=280_000
    )
    legacy_c = TrackEntity.build(
        "Legacy Three", "Track Three", video_id="VID_LEG_003", tags=[], duration_seconds=135, release_year=2016, view_count=45_000
    )

    va = extractor.extract_vector(legacy_a, hour=12)[:5]
    vb = extractor.extract_vector(legacy_b, hour=12)[:5]
    vc = extractor.extract_vector(legacy_c, hour=12)[:5]

    assert not np.allclose(va, vb)
    assert not np.allclose(va, vc)
    assert not np.allclose(vb, vc)
    # Check smooth continuity around old 120s step boundary (no abrupt jump > 0.02 over 2 seconds)
    v_119 = extractor.extract_vector(
        TrackEntity.build("S", "A", video_id="VID_C119", tags=[], duration_seconds=119), hour=12
    )[4]
    v_121 = extractor.extract_vector(
        TrackEntity.build("S", "B", video_id="VID_C121", tags=[], duration_seconds=121), hour=12
    )[4]
    assert abs(float(v_121 - v_119)) < 0.02
