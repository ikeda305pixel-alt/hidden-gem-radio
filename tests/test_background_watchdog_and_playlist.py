"""Regression tests for background stall_watchdog fix, DB cleanup, visibilitychange recovery, and loadPlaylist."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.db import RadioDatabase
from src.models import TrackEntity
from src.radio_server import create_app

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _extract_inline_js() -> str:
    html = WEB_INDEX.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    assert scripts, "No inline <script> block found in web/index.html"
    return "\n".join(scripts)


def test_stall_watchdog_never_blacklists_30d_and_db_init_cleans_legacy_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "radio.db"
    db_pre = RadioDatabase(db_path)

    # Seed legacy mis-blacklisted stall_watchdog_7s rows + 1 legitimate permanent error row
    now = time.time()
    expires = now + 30 * 86400
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO unplayable_blacklist (video_id, reason, yt_error_code, created_at, expires_at)
            VALUES ('LEGACY_STALL_01', 'stall_watchdog_7s', NULL, ?, ?),
                   ('LEGACY_STALL_02', 'stall_watchdog_15s', NULL, ?, ?),
                   ('PERM_ERR_150', 'yt_error_150', 150, ?, ?)
            """,
            (now, expires, now, expires, now, expires),
        )
        conn.commit()

    # Re-initializing RadioDatabase (e.g. app startup) must auto-clean stall_watchdog* rows but keep yt_error_150
    app = create_app(
        db_path=db_path,
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    db: RadioDatabase = app.state.runtime.db
    bl = db.get_blacklisted_video_ids()
    assert "LEGACY_STALL_01" not in bl
    assert "LEGACY_STALL_02" not in bl
    assert "PERM_ERR_150" in bl

    # Seed two playable tracks
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(artist="Artist One", title="Track One", view_count=50_000, video_id="VID_STALL_A"),
    )
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(artist="Artist Two", title="Track Two", view_count=50_000, video_id="VID_STALL_B"),
    )

    client = TestClient(app)

    # POST /unplayable with stall_watchdog_7s & yt_error_code=None must NOT blacklist for 30 days
    r1 = client.post(
        "/api/tracks/VID_STALL_A/unplayable",
        json={"station_id": "pes_passion_groove", "reason": "stall_watchdog_7s", "yt_error_code": None, "ttl_days": 30},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["blacklisted"] is False
    assert b1["blacklisted_days"] == 0
    assert b1["expires_at"] is None
    assert b1["replacement"]["video_id"] == "VID_STALL_B"
    assert "VID_STALL_A" not in db.get_blacklisted_video_ids()

    # Same for stall_watchdog_15s
    r2 = client.post(
        "/api/tracks/VID_STALL_B/unplayable",
        json={"station_id": "pes_passion_groove", "reason": "stall_watchdog_15s", "yt_error_code": None, "ttl_days": 30},
    )
    assert r2.status_code == 200
    assert r2.json()["blacklisted"] is False
    assert "VID_STALL_B" not in db.get_blacklisted_video_ids()
    assert db_pre is not None


@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS vm tests")
def test_web_ui_background_watchdog_visibilitychange_buffering_and_loadplaylist_node() -> None:
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
    async trigger(evt) {
      for (const fn of (listeners[evt] || [])) {
        await fn({ target: this });
      }
    },
  };
}

function createEnv(fetchImpl) {
  const elements = {};
  const docListeners = {};
  const getEl = (id) => {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  };

  let playerEvents = null;
  let playerStateVal = -1;
  let currentTimeVal = 0;
  let durationVal = 200;
  let currentVideoId = null;
  let currentPlaylistIndex = 0;
  let loadedPlaylist = [];

  const playerCalls = {
    loadVideoById: [],
    loadPlaylist: [],
    nextVideo: 0,
    playVideo: 0,
    stopVideo: 0,
    pauseVideo: 0,
  };

  // Deterministic manual timer queue so we can fast-forward 15s watchdog instantaneously
  let nextTimerId = 1;
  const timers = new Map();
  function fakeSetTimeout(fn, ms) {
    const id = nextTimerId++;
    timers.set(id, { fn, ms, interval: false });
    return id;
  }
  function fakeClearTimeout(id) { timers.delete(id); }
  function fakeSetInterval(fn, ms) {
    const id = nextTimerId++;
    timers.set(id, { fn, ms, interval: true });
    return id;
  }
  function fakeClearInterval(id) { timers.delete(id); }
  async function fireTimeoutsByMs(minMs) {
    const due = [];
    for (const [id, t] of timers.entries()) {
      if (!t.interval && t.ms >= minMs) due.push([id, t]);
    }
    for (const [id, t] of due) {
      timers.delete(id);
      await t.fn();
    }
  }

  const mediaActions = {};
  const mediaSessionObj = {
    metadata: null,
    playbackState: "none",
    setActionHandler(name, fn) { mediaActions[name] = fn; },
  };

  const docObj = {
    hidden: false,
    getElementById: getEl,
    createElement: (tag) => makeElement(tag),
    addEventListener(evt, fn) {
      docListeners[evt] = docListeners[evt] || [];
      docListeners[evt].push(fn);
    },
    async dispatch(evt) {
      for (const fn of (docListeners[evt] || [])) {
        await fn({ type: evt });
      }
    },
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
    setTimeout: fakeSetTimeout,
    clearTimeout: fakeClearTimeout,
    setInterval: fakeSetInterval,
    clearInterval: fakeClearInterval,
    AbortController: typeof AbortController !== "undefined" ? AbortController : undefined,
    document: docObj,
    window: {
      location: { origin: "http://127.0.0.1:8765" },
      addEventListener: () => {},
    },
    navigator: { mediaSession: mediaSessionObj },
    MediaMetadata: function(obj) { return obj; },
    YT: {
      PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
      Player: function(_elId, opts) {
        playerEvents = opts.events;
        return {
          loadVideoById(vid) {
            playerCalls.loadVideoById.push(vid);
            currentVideoId = vid;
          },
          loadPlaylist(vids, startIndex) {
            playerCalls.loadPlaylist.push({ vids, startIndex });
            loadedPlaylist = [...vids];
            currentPlaylistIndex = startIndex || 0;
            currentVideoId = loadedPlaylist[currentPlaylistIndex] || null;
          },
          nextVideo() {
            playerCalls.nextVideo += 1;
            if (currentPlaylistIndex + 1 < loadedPlaylist.length) {
              currentPlaylistIndex += 1;
              currentVideoId = loadedPlaylist[currentPlaylistIndex];
            }
          },
          getVideoData() { return { video_id: currentVideoId }; },
          getPlaylistIndex() { return currentPlaylistIndex; },
          playVideo() { playerCalls.playVideo += 1; },
          stopVideo() { playerCalls.stopVideo += 1; },
          pauseVideo() { playerCalls.pauseVideo += 1; },
          isMuted() { return false; },
          mute() {},
          unMute() {},
          getPlayerState() { return playerStateVal; },
          getCurrentTime() { return currentTimeVal; },
          getDuration() { return durationVal; },
        };
      },
    },
    fetch: fetchImpl,
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(
    INLINE_JS +
      "\n;globalThis.__exports = { state, PLAYBACK_WATCHDOG_MS, armPlaybackWatchdog, clearPlaybackWatchdog, playTrack, playNextFromQueue, handleManualSkip };",
    sandbox
  );
  sandbox.window.onYouTubeIframeAPIReady();
  playerEvents.onReady();

  return {
    exports: sandbox.__exports,
    docObj,
    playerCalls,
    mediaSessionObj,
    mediaActions,
    fireTimeoutsByMs,
    getPlayerEvents: () => playerEvents,
    setPlayerState: (st) => { playerStateVal = st; },
    setTimes: (cur, dur) => { currentTimeVal = cur; durationVal = dur; },
    setPlayerVideo: (vid, idx) => { currentVideoId = vid; currentPlaylistIndex = idx; },
  };
}

async function runAll() {
  // Scenario 1: document.hidden === true pauses watchdog -> NO skip, NO error increment, NO circuit breaker
  {
    const unplayableCalls = [];
    const env = createEnv(async (url) => {
      if (url.includes("/unplayable")) unplayableCalls.push(url);
      return { ok: true, status: 200, json: async () => ({ items: [] }) };
    });
    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.hasUserPlaybackStarted = true;
    st.currentTrack = { track_id: "trk_bg_1", video_id: "VID_BG_1", artist: "Parcels", title: "Hidden Song" };
    env.setPlayerState(5); // CUED
    env.setTimes(0, 200);

    // Arm watchdog in foreground, then switch to background (LINE) before 15s fires
    env.exports.armPlaybackWatchdog(st.currentTrack);
    env.docObj.hidden = true;
    await env.fireTimeoutsByMs(15000);

    if (unplayableCalls.length !== 0) {
      throw new Error(`Scenario 1 failed: /unplayable called ${unplayableCalls.length} times while hidden`);
    }
    if (st.consecutiveErrors !== 0 || st.circuitBroken !== false) {
      throw new Error(`Scenario 1 failed: errors=${st.consecutiveErrors}, broken=${st.circuitBroken}`);
    }
  }

  // Scenario 2: visibilitychange returning to visible while CUED auto-kicks playVideo() & re-arms watchdog
  {
    const env = createEnv(async () => ({ ok: true, status: 200, json: async () => ({ items: [] }) }));
    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.hasUserPlaybackStarted = true;
    st.circuitBroken = false;
    st.currentTrack = { track_id: "trk_vis_1", video_id: "VID_VIS_1", artist: "Phoenix", title: "Armistice" };
    st.trackConfirmedPlayingVideoId = null;
    env.setPlayerState(5); // CUED
    env.docObj.hidden = true;
    await env.docObj.dispatch("visibilitychange");

    // Return to foreground
    env.docObj.hidden = false;
    await env.docObj.dispatch("visibilitychange");

    if (env.playerCalls.playVideo < 1) {
      throw new Error("Scenario 2 failed: visibilitychange did not call playVideo() when CUED");
    }
    if (st.watchdogTimerId === null) {
      throw new Error("Scenario 2 failed: visibilitychange did not re-arm watchdog");
    }
  }

  // Scenario 3: 4G BUFFERING grants 1x watchdog grace extension before skipping
  {
    const unplayableCalls = [];
    const env = createEnv(async (url) => {
      if (url.includes("/unplayable")) unplayableCalls.push(url);
      return { ok: true, status: 200, json: async () => ({ items: [] }) };
    });
    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.hasUserPlaybackStarted = true;
    st.currentTrack = { track_id: "trk_buf_1", video_id: "VID_BUF_1", artist: "Royel Otis", title: "Oysters" };
    env.setPlayerState(3); // BUFFERING
    env.setTimes(0, 200);
    env.docObj.hidden = false;

    env.exports.armPlaybackWatchdog(st.currentTrack);
    await env.fireTimeoutsByMs(15000);
    if (unplayableCalls.length !== 0 || st.watchdogBufferingExtended !== true) {
      throw new Error("Scenario 3 failed: BUFFERING did not grant 1x grace extension");
    }
  }

  // Scenario 4: loadPlaylist multi-track batch + internal playlist transition -> keepalive COMPLETE + UI/MediaSession sync + nextVideo()
  {
    const feedbackCalls = [];
    const env = createEnv(async (url, opts = {}) => {
      if (url === "/feedback") {
        feedbackCalls.push({ url, opts, body: JSON.parse(opts.body) });
        return { ok: true, status: 200, json: async () => ({ status: "ok", model_updates: 1 }) };
      }
      return { ok: true, status: 200, json: async () => ({ items: [], updates: 1, layers: [], axes: [] }) };
    });
    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.queue = [
      { track_id: "trk_p1", video_id: "VID_PL_1", artist: "Parcels", title: "Track 1", duration_seconds: 210 },
      { track_id: "trk_p2", video_id: "VID_PL_2", artist: "Phoenix", title: "Track 2", duration_seconds: 190 },
      { track_id: "trk_p3", video_id: "VID_PL_3", artist: "Oberhofer", title: "Track 3", duration_seconds: 200 },
      { track_id: "trk_p4", video_id: "VID_PL_4", artist: "Vampire Weekend", title: "Track 4", duration_seconds: 205 },
    ];

    await env.exports.playNextFromQueue();
    if (env.playerCalls.loadPlaylist.length !== 1) {
      throw new Error(`Scenario 4 failed: expected 1 loadPlaylist call, got ${env.playerCalls.loadPlaylist.length}`);
    }
    if (env.playerCalls.loadPlaylist[0].vids.join(",") !== "VID_PL_1,VID_PL_2,VID_PL_3,VID_PL_4") {
      throw new Error(`Scenario 4 failed: unexpected playlist vids ${env.playerCalls.loadPlaylist[0].vids}`);
    }
    if (!env.mediaSessionObj.metadata || env.mediaSessionObj.metadata.title !== "Track 1") {
      throw new Error("Scenario 4 failed: MediaSession metadata not set for Track 1");
    }
    if (typeof env.mediaActions.play !== "function" || typeof env.mediaActions.nexttrack !== "function") {
      throw new Error("Scenario 4 failed: MediaSession action handlers not registered");
    }

    // Confirm Track 1 playing
    env.setPlayerState(1); // PLAYING
    env.setTimes(5.0, 210);
    await env.getPlayerEvents().onStateChange({ data: 1 });
    if (st.trackConfirmedPlayingVideoId !== "VID_PL_1") {
      throw new Error("Scenario 4 failed: Track 1 not confirmed playing");
    }

    // Simulate YouTube internal auto-advance to Track 2 (even in background!)
    env.setPlayerVideo("VID_PL_2", 1);
    env.setTimes(0.5, 190);
    await env.getPlayerEvents().onStateChange({ data: 1 });

    if (st.currentTrack.video_id !== "VID_PL_2") {
      throw new Error(`Scenario 4 failed: currentTrack did not sync to VID_PL_2 (got ${st.currentTrack.video_id})`);
    }
    if (env.mediaSessionObj.metadata.title !== "Track 2") {
      throw new Error("Scenario 4 failed: MediaSession metadata not updated to Track 2");
    }
    const completeCall = feedbackCalls.find((c) => c.body.video_id === "VID_PL_1" && c.body.event_type === "COMPLETE");
    if (!completeCall || completeCall.opts.keepalive !== true) {
      throw new Error("Scenario 4 failed: non-blocking keepalive COMPLETE not sent for Track 1");
    }

    // Manual skip should use player.nextVideo() to jump immediately to Track 3 inside loaded playlist
    await env.exports.handleManualSkip();
    if (env.playerCalls.nextVideo !== 1 || st.currentTrack.video_id !== "VID_PL_3") {
      throw new Error(`Scenario 4 failed: expected nextVideo() -> VID_PL_3, got calls=${env.playerCalls.nextVideo}, cur=${st.currentTrack?.video_id}`);
    }
  }

  console.log(JSON.stringify({ ok: true }));
}

runAll().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    script = f"const INLINE_JS = {json.dumps(inline_js)};\n{harness}"
    proc = subprocess.run(["node", "-e", script], check=False, capture_output=True, text=True, timeout=15)
    assert proc.returncode == 0, f"Node vm test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert '"ok":true' in proc.stdout.replace(" ", "")
