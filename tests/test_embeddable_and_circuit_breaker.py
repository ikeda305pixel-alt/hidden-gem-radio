"""Regression tests for Stage 1 oEmbed embeddability pre-filter, inventory floors, and separated circuit breakers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase
from src.digger.pool import CandidatePoolDigger
from src.models import StationConfig, TrackEntity
from src.radio_server import LOW_POOL_BG_DIG_THRESHOLD, create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_youtube_oembed_embeddability_check_cache_and_fallback() -> None:
    """oEmbed 200->True (cached), 401/403->False (cached), 5xx/timeout->True (uncached + circuit breaker)."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_param = request.url.params.get("url", "")
        vid = url_param.split("v=")[-1]
        calls.append(vid)
        if vid == "VID_OK_0001":
            return httpx.Response(200, json={"title": "Playable Official Song"})
        if vid == "VID_401_001":
            return httpx.Response(401, text="Unauthorized")
        if vid == "VID_403_001":
            return httpx.Response(403, text="Forbidden")
        if vid == "VID_500_001":
            return httpx.Response(500, text="Internal Server Error")
        if vid.startswith("VID_NET_"):
            raise httpx.ConnectTimeout("timed out")
        return httpx.Response(200, json={"title": "Default OK"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = YouTubeMusicAdapter(client=object(), http_client=client, check_embeddable=True)

    # 200 -> True and cached
    assert adapter.is_video_embeddable("VID_OK_0001") is True
    assert adapter.is_video_embeddable("VID_OK_0001") is True
    assert calls.count("VID_OK_0001") == 1

    # 401 & 403 -> False and cached
    assert adapter.is_video_embeddable("VID_401_001") is False
    assert adapter.is_video_embeddable("VID_401_001") is False
    assert calls.count("VID_401_001") == 1

    assert adapter.is_video_embeddable("VID_403_001") is False
    assert adapter.is_video_embeddable("VID_403_001") is False
    assert calls.count("VID_403_001") == 1

    # 500 -> safe fallback True and NOT cached as False
    assert adapter.is_video_embeddable("VID_500_001") is True
    assert adapter.is_video_embeddable("VID_500_001") is True
    assert calls.count("VID_500_001") == 2

    # 4 consecutive network errors open the network circuit breaker (returns True without extra HTTP calls)
    # (2 calls to VID_500_001 already incremented error count to 2, so 2 timeouts reach 4)
    assert adapter.is_video_embeddable("VID_NET_001") is True
    assert adapter.is_video_embeddable("VID_NET_002") is True
    calls_before_open = len(calls)
    assert adapter.is_video_embeddable("VID_NET_003") is True
    assert len(calls) == calls_before_open


def test_stage1_dig_excludes_unembeddable_and_blacklists_before_db_insert(tmp_path: Path) -> None:
    """Stage 1 dig must filter out oEmbed 403 tracks before DB upsert and blacklist them for 30 days."""
    db = RadioDatabase(tmp_path / "radio.db")

    class FakeYTMusicClient:
        def search(self, query: str, filter: str | None = None, limit: int = 5) -> list[dict[str, Any]]:
            _ = (query, filter, limit)
            return [
                {
                    "videoId": "SEED_VID_01",
                    "title": "Seed Track",
                    "resultType": "song",
                    "videoType": "MUSIC_VIDEO_TYPE_ATV",
                    "artists": [{"name": "Seed Artist", "id": "UC_SEED_001"}],
                    "album": {"name": "Seed Album"},
                    "duration_seconds": 200,
                }
            ]

        def get_watch_playlist(self, videoId: str, limit: int = 25) -> dict[str, Any]:
            _ = (videoId, limit)
            return {
                "tracks": [
                    {
                        "videoId": "EMB_OK_0001",
                        "title": "Playable Gem 1",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Indie One", "id": "UC_IND_01"}],
                        "album": {"name": "Album 1"},
                        "length": "3:15",
                        "views": "25K",
                    },
                    {
                        "videoId": "EMB_NO_0001",
                        "title": "Major Label Blocked 1",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Major Blocked A", "id": "UC_MAJ_01"}],
                        "album": {"name": "Album M1"},
                        "length": "3:20",
                        "views": "40K",
                    },
                    {
                        "videoId": "EMB_OK_0002",
                        "title": "Playable Gem 2",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Indie Two", "id": "UC_IND_02"}],
                        "album": {"name": "Album 2"},
                        "length": "3:40",
                        "views": "18K",
                    },
                    {
                        "videoId": "EMB_NO_0002",
                        "title": "Major Label Blocked 2",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Major Blocked B", "id": "UC_MAJ_02"}],
                        "album": {"name": "Album M2"},
                        "length": "3:05",
                        "views": "32K",
                    },
                ]
            }

        def get_artist(self, channel_id: str) -> dict[str, Any]:
            _ = channel_id
            return {}

    def oembed_handler(request: httpx.Request) -> httpx.Response:
        url_param = request.url.params.get("url", "")
        if "EMB_NO_" in url_param:
            return httpx.Response(403, text="Forbidden")
        return httpx.Response(200, json={"title": "Embeddable"})

    http_client = httpx.Client(transport=httpx.MockTransport(oembed_handler))
    yt_adapter = YouTubeMusicAdapter(client=FakeYTMusicClient(), http_client=http_client, check_embeddable=True)
    digger = CandidatePoolDigger(db=db, yt_adapter=yt_adapter, enable_mir_enrichment=False)

    station = StationConfig(
        station_id="test_hiphop",
        name="Test Hip-Hop",
        seed_tags=["hip-hop"],
        seed_artists=["Seed Artist"],
        spotify_anchor_seeds=["Seed Artist - Seed Track"],
        max_popularity_views=500_000,
    )
    summary = digger.dig_station(station, target_pool_size=10)

    assert summary["added"] == 2
    assert summary["skipped_unembeddable"] == 2

    playable = db.get_eligible_candidates("test_hiphop", max_popularity_views=500_000)
    playable_vids = {t.video_id for t in playable}
    assert playable_vids == {"EMB_OK_0001", "EMB_OK_0002"}

    blacklisted = db.get_blacklisted_video_ids()
    assert "EMB_NO_0001" in blacklisted
    assert "EMB_NO_0002" in blacklisted


def test_unplayable_permanent_error_triggers_bg_dig_when_pool_below_10_and_startup_stations_dig(
    tmp_path: Path,
) -> None:
    """Permanent error (150) depleting pool < 10 and GET /stations with < 5 tracks trigger background dig."""
    stations_yaml = tmp_path / "stations.yaml"
    stations_yaml.write_text(
        json.dumps(
            {
                "stations": [
                    {
                        "station_id": "st_low",
                        "name": "Low Pool Station",
                        "description": "Test",
                        "genres": ["hip-hop"],
                        "seed_artists": ["Seed A"],
                        "max_popularity_views": 500000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=stations_yaml,
        model_dir=tmp_path / "models",
        auto_dig=True,
    )
    runtime = app.state.runtime
    bg_dig_calls: list[tuple[str, int]] = []

    def fake_trigger_bg_dig(
        station: StationConfig,
        target_pool_size: int = 60,
        min_unheard: int = 25,
    ) -> bool:
        _ = target_pool_size
        bg_dig_calls.append((station.station_id, min_unheard))
        return True

    runtime._trigger_background_dig = fake_trigger_bg_dig  # type: ignore[method-assign]

    client = TestClient(app)

    # 1. Startup GET /stations with 0 candidates (< 5) triggers non-blocking background dig
    res_stations = client.get("/stations")
    assert res_stations.status_code == 200
    assert len(bg_dig_calls) == 1
    assert bg_dig_calls[0][0] == "st_low"

    # 2. Seed exactly LOW_POOL_BG_DIG_THRESHOLD (10) tracks; marking one 150 drops playable count to 9 (< 10)
    bg_dig_calls.clear()
    for i in range(LOW_POOL_BG_DIG_THRESHOLD):
        runtime.db.upsert_track(
            "st_low",
            TrackEntity(
                track_id=f"tr_{i:02d}",
                canonical_key=f"artist {i}::track {i}",
                title=f"Track {i}",
                artist=f"Artist {i}",
                video_id=f"VID_POOL_{i:02d}",
                view_count=12_000,
                duration_seconds=200,
            ),
        )

    res_unp = client.post(
        "/api/stations/st_low/tracks/VID_POOL_00/unplayable",
        json={"track_id": "tr_00", "reason": "yt_error_150", "yt_error_code": 150},
    )
    assert res_unp.status_code == 200
    body = res_unp.json()
    assert body["blacklisted"] is True
    assert body["background_dig_triggered"] is True
    assert any(sid == "st_low" for sid, _ in bg_dig_calls)


def test_web_ui_separate_circuit_breakers_and_resume_usability() -> None:
    """Real Node.js vm test verifying separated embed vs system circuit breakers, ▶ 再開 label, and ⏭ 次曲 recovery."""
    html_path = PROJECT_ROOT / "web" / "index.html"
    node_script = r"""
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[1], "utf8");
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(scriptMatch, "Inline script not found in web/index.html");

function makeElement(id) {
  const listeners = {};
  const classes = new Set();
  return {
    id,
    textContent: "",
    innerHTML: "",
    value: "",
    children: [],
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c, force) => (force === undefined ? (classes.has(c) ? classes.delete(c) : classes.add(c)) : (force ? classes.add(c) : classes.delete(c))),
      contains: (c) => classes.has(c),
    },
    appendChild(child) { this.children.push(child); },
    addEventListener(type, fn) { (listeners[type] ||= []).push(fn); },
    async dispatch(type) { for (const fn of listeners[type] || []) await fn({}); },
  };
}

const elements = {};
const documentStub = {
  hidden: false,
  getElementById: (id) => (elements[id] ||= makeElement(id)),
  createElement: (tag) => makeElement(tag),
};

const fetchCalls = [];
let fetchHandler = async () => ({ ok: true, status: 200, json: async () => ({ items: [] }) });
async function fetchStub(url, options) {
  fetchCalls.push({ url, options });
  return fetchHandler(url, options);
}

const YTStub = {
  PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
  Player: function (_id, config) {
    this._config = config;
    this._state = 1;
    this._time = 0;
    this._muted = false;
    this.loadedVideos = [];
    this.loadVideoById = (vid) => { this.loadedVideos.push(vid); };
    this.playVideo = () => { this._state = 1; };
    this.pauseVideo = () => { this._state = 2; };
    this.getPlayerState = () => this._state;
    this.getCurrentTime = () => this._time;
    this.getDuration = () => 180;
    this.isMuted = () => this._muted;
    this.mute = () => { this._muted = true; };
    this.unMute = () => { this._muted = false; };
  },
};

const sandbox = {
  console,
  document: documentStub,
  navigator: { mediaSession: { setActionHandler: () => {} } },
  MediaMetadata: function (m) { Object.assign(this, m); },
  YT: YTStub,
  fetch: fetchStub,
  Date,
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  encodeURIComponent,
  Promise, Set, Array, Object, Number, String, Boolean, Math, JSON, Error,
};
sandbox.window = sandbox;

fetchHandler = async (url) => {
  if (url === "/stations") return { ok: true, status: 200, json: async () => ({ stations: [] }) };
  if (url === "/api/playlists") return { ok: true, status: 200, json: async () => ({ playlists: [{ playlist_id: "pl_favorites", name: "Good", track_count: 0 }] }) };
  if (url.startsWith("/api/playlists/")) return { ok: true, status: 200, json: async () => ({ tracks: [], track_count: 0, youtube_watch_urls: [] }) };
  if (url === "/api/youtube/playlists") return { ok: true, status: 200, json: async () => ({ authenticated: false, playlists: [] }) };
  if (url.startsWith("/api/profile/theory")) return { ok: true, status: 200, json: async () => ({ updates: 0, axes: [] }) };
  return { ok: true, status: 200, json: async () => ({ items: [] }) };
};

const ctx = vm.createContext(sandbox);
vm.runInContext(
  scriptMatch[1] + "\nwindow.__exports = { state, el, MAX_CONSECUTIVE_ERRORS, MAX_CONSECUTIVE_EMBED_SKIPS, handleUnplayableSkip, handleManualSkip, updatePlayMuteButton };",
  ctx
);

(async () => {
  await new Promise((r) => setImmediate(r));
  const api = sandbox.__exports;
  sandbox.onYouTubeIframeAPIReady();
  api.state.player._config.events.onReady();
  api.state.stations = [{ station_id: "st_hip", name: "Hip-Hop", candidate_count: 20 }];
  api.state.activeStationId = "st_hip";

  assert.strictEqual(api.MAX_CONSECUTIVE_ERRORS, 4);
  assert.strictEqual(api.MAX_CONSECUTIVE_EMBED_SKIPS, 10);

  // 1. 4 consecutive Error 150/101 skips MUST NOT trip circuit breaker
  let seq = 0;
  fetchHandler = async (url) => {
    if (url.includes("/unplayable")) {
      seq += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          replacement: { track_id: `t_rep_${seq}`, video_id: `VID_REP_${seq}`, artist: `Art ${seq}`, title: `Song ${seq}` },
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({ items: [] }) };
  };

  api.state.currentTrack = { track_id: "t_0", video_id: "VID_EMB_0", artist: "Major 0", title: "Blocked 0" };
  for (let i = 0; i < 4; i += 1) {
    const code = i % 2 === 0 ? 150 : 101;
    await api.handleUnplayableSkip(api.state.currentTrack, `yt_error_${code}`, code);
  }
  assert.strictEqual(api.state.circuitBroken, false, "4 consecutive embed errors (150/101) must NOT trip circuit breaker");
  assert.strictEqual(api.state.consecutiveEmbedSkips, 4);
  assert.strictEqual(api.state.consecutiveErrors, 0);
  assert.ok(api.state.currentTrack !== null, "Playback must seamlessly continue after 4 embed skips");

  // 2. Reaching 10 consecutive embed skips trips circuit breaker and updates btnPlayMute to ▶ 再開
  for (let i = 4; i < 10; i += 1) {
    await api.handleUnplayableSkip(api.state.currentTrack, "yt_error_150", 150);
  }
  assert.strictEqual(api.state.circuitBroken, true, "10 consecutive embed skips must trip circuit breaker");
  assert.strictEqual(api.state.consecutiveEmbedSkips, 10);
  assert.strictEqual(api.el.btnPlayMute.textContent, "▶ 再開", "Bottom play/mute button must show ▶ 再開 when circuitBroken");

  // 3. Clicking ⏭ 次曲 (handleManualSkip) resets circuitBroken & resumes immediately
  api.state.queue = [];
  fetchCalls.length = 0;
  fetchHandler = async (url) => {
    if (url.includes("/batch")) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          items: [{ track_id: "t_resumed", video_id: "VID_RESUMED", artist: "Resumed Art", title: "Resumed Song" }],
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({}) };
  };
  await api.el.btnSkip.dispatch("click");
  assert.strictEqual(api.state.circuitBroken, false, "⏭ 次曲 must reset circuitBroken");
  assert.strictEqual(api.state.consecutiveEmbedSkips, 0, "⏭ 次曲 must reset consecutiveEmbedSkips");
  assert.strictEqual(api.state.consecutiveErrors, 0, "⏭ 次曲 must reset consecutiveErrors");
  assert.strictEqual(api.state.currentTrack.video_id, "VID_RESUMED", "⏭ 次曲 must immediately play next track");
  assert.strictEqual(
    fetchCalls.filter((c) => c.url === "/feedback").length,
    0,
    "Resuming from circuitBroken via ⏭ 次曲 must NOT send bogus implicit feedback"
  );

  // 4. Transient/system errors (e.g. yt_error_5) still trip strictly at MAX_CONSECUTIVE_ERRORS (4)
  fetchHandler = async (url) => {
    if (url.includes("/unplayable")) {
      seq += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          replacement: { track_id: `t_sys_${seq}`, video_id: `VID_SYS_${seq}`, artist: `Sys ${seq}`, title: `Err ${seq}` },
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({ items: [] }) };
  };
  for (let i = 0; i < 4; i += 1) {
    await api.handleUnplayableSkip(api.state.currentTrack, "yt_error_5", 5);
  }
  assert.strictEqual(api.state.circuitBroken, true, "System/transient errors must trip strictly at 4");
  assert.strictEqual(api.state.consecutiveErrors, 4);
  assert.strictEqual(api.el.btnPlayMute.textContent, "▶ 再開");

  console.log("EMBED_AND_CIRCUIT_BREAKER_UI_OK");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    proc = subprocess.run(
        ["node", "-e", node_script, str(html_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert proc.returncode == 0, f"Node UI test failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "EMBED_AND_CIRCUIT_BREAKER_UI_OK" in proc.stdout
