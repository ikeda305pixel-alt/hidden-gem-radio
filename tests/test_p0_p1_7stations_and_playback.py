"""Regression tests for P0/P1 real-device fixes, 7-station thematic split, and 5-layer anti-overfitting stoppers."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase
from src.digger.history_importer import (
    classify_entry_to_station,
    default_theory_stations,
    generate_stations_yaml,
)
from src.digger.pool import CandidatePoolDigger
from src.models import (
    StationConfig,
    TrackEntity,
    is_japanese_music_entity,
    normalize_artist,
    normalize_track_key,
)
from src.radio_server import create_app
from src.recommender.audio_mir import MIR_DIM
from src.recommender.bandit import ContextualBanditRecommender
from src.recommender.calibrate_profile import _collect_youtube_station_top_tracks

WEB_INDEX = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _extract_inline_js() -> str:
    html = WEB_INDEX.read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    assert scripts, "No inline <script> block found in web/index.html"
    return "\n".join(scripts)


# ---------------------------------------------------------------------------
# 1. Stage 1 ytmusicapi UGC rescue & radioId fallback
# ---------------------------------------------------------------------------
def test_watch_playlist_ugc_rescue_and_strict_slop_rejection() -> None:
    class FakeYTMusic:
        def search(self, query: str, filter: str = "songs", limit: int = 8) -> list[dict[str, Any]]:
            if filter == "songs":
                return [{"videoId": "SEED_VID_01", "title": "Seed Song", "artists": [{"name": "Seed Artist"}]}]
            return []

        def get_watch_playlist(self, videoId: str | None = None, playlistId: str | None = None, limit: int = 25) -> dict[str, Any]:
            return {
                "tracks": [
                    # Valid official-artist UGC (has artists[0].id and 60-660s duration) -> RESCUED
                    {
                        "videoId": "UGC_VALID_01",
                        "title": "Midnight Parisian Groove",
                        "artists": [{"name": "Parcels", "id": "UC_PARCELS_OFFICIAL"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "3:42",
                        "views": "450K views",
                    },
                    # UGC without artists[0].id -> REJECTED
                    {
                        "videoId": "UGC_NOID_01",
                        "title": "Midnight Groove",
                        "artists": [{"name": "Random Uploader"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "3:20",
                    },
                    # UGC with cover/nightcore/lyric/ai slop -> REJECTED
                    {
                        "videoId": "UGC_COVER_01",
                        "title": "Midnight Groove (Acoustic Cover)",
                        "artists": [{"name": "Some Singer", "id": "UC_COVER"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "3:20",
                    },
                    {
                        "videoId": "UGC_NIGHT_01",
                        "title": "Midnight Groove (Nightcore)",
                        "artists": [{"name": "NC Channel", "id": "UC_NC"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "2:50",
                    },
                    {
                        "videoId": "UGC_AISLOP_01",
                        "title": "AI Music Chill Lo-Fi",
                        "artists": [{"name": "Suno AI Music", "id": "UC_AI"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "3:10",
                    },
                    # UGC too short (<60s) -> REJECTED
                    {
                        "videoId": "UGC_SHORT_01",
                        "title": "Quick Teaser",
                        "artists": [{"name": "Parcels", "id": "UC_PARCELS_OFFICIAL"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "0:45",
                    },
                    # UGC too long (>660s) -> REJECTED
                    {
                        "videoId": "UGC_LONG_01",
                        "title": "Full Live Set",
                        "artists": [{"name": "Parcels", "id": "UC_PARCELS_OFFICIAL"}],
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "length": "15:00",
                    },
                ]
            }

    adapter = YouTubeMusicAdapter(client=FakeYTMusic(), check_embeddable=False)
    tracks = adapter.get_related_tracks("SEED_VID_01", limit=10)
    vids = [t.video_id for t in tracks]
    assert vids == ["UGC_VALID_01"]


def test_artist_deep_cuts_falls_back_to_radio_id_when_related_absent() -> None:
    class FakeYTMusicNoRelated:
        def get_artist(self, channelId: str) -> dict[str, Any]:
            # No "related" key, only radioId
            return {"name": "Parcels", "radioId": "RDAMVM_PARCELS_RADIO"}

        def get_watch_playlist(self, playlistId: str | None = None, limit: int = 15, **kwargs: Any) -> dict[str, Any]:
            if playlistId == "RDAMVM_PARCELS_RADIO":
                return {
                    "tracks": [
                        {
                            "videoId": "RADIO_FALLBACK_01",
                            "title": "L'Impératrice - Peur des filles",
                            "artists": [{"name": "L'Impératrice", "id": "UC_IMPERATRICE"}],
                            "videoType": "MUSIC_VIDEO_TYPE_UGC",
                            "length": "3:35",
                        }
                    ]
                }
            return {"tracks": []}

    adapter = YouTubeMusicAdapter(client=FakeYTMusicNoRelated(), check_embeddable=False)
    tracks = adapter.get_artist_related_deep_cuts("UC_ARTIST_01")
    assert [t.video_id for t in tracks] == ["RADIO_FALLBACK_01"]
    assert tracks[0].source == "artist_radio"


# ---------------------------------------------------------------------------
# 2. Non-blocking /batch & background dig semaphore cap (max 2)
# ---------------------------------------------------------------------------
def test_batch_endpoint_never_blocks_on_cold_pool_and_caps_bg_dig_concurrency(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    runtime = app.state.runtime

    active_digs = 0
    max_concurrent_digs = 0
    lock = threading.Lock()
    dig_started = threading.Event()
    release_dig = threading.Event()

    def slow_dig_station(station: StationConfig, **kwargs: Any) -> dict[str, Any]:
        nonlocal active_digs, max_concurrent_digs
        with lock:
            active_digs += 1
            max_concurrent_digs = max(max_concurrent_digs, active_digs)
            dig_started.set()
        release_dig.wait(timeout=2.0)
        with lock:
            active_digs -= 1
        return {"added": 0, "skipped_region_mismatch": 0}

    runtime.digger.dig_station = slow_dig_station  # type: ignore[method-assign]

    client = TestClient(app)

    # Cold pool /batch must return immediately without waiting for slow_dig_station
    t0 = time.monotonic()
    res = client.get("/api/stations/pes_passion_groove/batch", params={"count": 3})
    elapsed = time.monotonic() - t0
    assert res.status_code == 200
    assert res.json()["items"] == []
    assert elapsed < 0.6, f"/batch blocked synchronously for {elapsed:.2f}s"

    # Trigger background digs across 5 stations simultaneously -> semaphore must cap at 2
    for sid in list(runtime.stations.keys())[:5]:
        runtime._trigger_background_dig(runtime.stations[sid])

    assert dig_started.wait(timeout=1.5)
    time.sleep(0.15)
    assert max_concurrent_digs <= 2, f"Background dig concurrency exceeded cap: {max_concurrent_digs}"
    release_dig.set()


# ---------------------------------------------------------------------------
# 3. Node.js vm real execution tests for web/index.html (ENDED / CUED / BAD / timeout)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS vm tests")
def test_web_ui_ended_1track_cued_autoplay_rapid_bad_and_fetch_timeout() -> None:
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
  const getEl = (id) => {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  };
  let playerEvents = null;
  const playerCalls = { loadVideoById: [], playVideo: 0, stopVideo: 0, pauseVideo: 0 };
  let playerStateVal = 1;
  let currentTimeVal = 180;
  let durationVal = 200;

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
    document: {
      getElementById: getEl,
      createElement: (tag) => makeElement(tag),
    },
    window: {
      location: { origin: "http://127.0.0.1:8765" },
      addEventListener: () => {},
    },
    navigator: {},
    MediaMetadata: function(obj) { return obj; },
    YT: {
      PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
      Player: function(_elId, opts) {
        playerEvents = opts.events;
        return {
          loadVideoById(vid) { playerCalls.loadVideoById.push(vid); },
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
      "\n;globalThis.__exports = { state, armPlaybackWatchdog, clearPlaybackWatchdog, fetchJson };",
    sandbox
  );
  sandbox.window.onYouTubeIframeAPIReady();
  playerEvents.onReady();
  return {
    sandbox,
    exports: sandbox.__exports,
    elements,
    getEl,
    playerCalls,
    getPlayerEvents: () => playerEvents,
    setTimes: (cur, dur) => { currentTimeVal = cur; durationVal = dur; },
    setPlayerState: (st) => { playerStateVal = st; },
  };
}

async function runAll() {
  // Scenario A: 1-track inventory natural ENDED -> COMPLETE -> /batch retry without exclude -> loadVideoById
  {
    const requests = [];
    const env = createEnv(async (url, opts = {}) => {
      requests.push({ url, method: opts.method || "GET", body: opts.body || null });
      if (url.startsWith("/feedback")) {
        return { ok: true, status: 200, json: async () => ({ status: "ok", model_updates: 1 }) };
      }
      if (url.includes("/batch")) {
        // When exclude contains trk_only_1, simulate 1-track inventory returning empty
        if (url.includes("exclude=trk_only_1")) {
          return { ok: true, status: 200, json: async () => ({ items: [] }) };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            items: [{ track_id: "trk_only_1", video_id: "VID_ONLY_1", artist: "Parcels", title: "Tieduprightnow", duration_seconds: 200 }],
          }),
        };
      }
      return { ok: true, status: 200, json: async () => ({ updates: 1, layers: [], axes: [], playlists: [], stations: [] }) };
    });

    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.hasUserPlaybackStarted = true;
    st.currentTrack = { track_id: "trk_only_1", video_id: "VID_ONLY_1", artist: "Parcels", title: "Tieduprightnow", duration_seconds: 200 };
    st.trackConfirmedPlayingVideoId = "VID_ONLY_1";
    st.queue = [];
    env.setTimes(198, 200);

    await env.getPlayerEvents().onStateChange({ data: 0 }); // ENDED
    env.exports.clearPlaybackWatchdog();
    const fbReq = requests.find((r) => r.url === "/feedback");
    if (!fbReq || JSON.parse(fbReq.body).event_type !== "COMPLETE") {
      throw new Error("Scenario A failed: COMPLETE feedback not sent");
    }
    if (!env.playerCalls.loadVideoById.includes("VID_ONLY_1")) {
      throw new Error("Scenario A failed: 1-track ENDED did not reload fallback track");
    }
  }

  // Scenario B: hasUserPlaybackStarted=true + CUED(5) -> calls playVideo() & keeps watchdog armed
  {
    const env = createEnv(async () => ({ ok: true, status: 200, json: async () => ({ items: [] }) }));
    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.hasUserPlaybackStarted = true;
    st.circuitBroken = false;
    st.currentTrack = { track_id: "trk_cued_1", video_id: "VID_CUED_1", artist: "L'Imperatrice", title: "Sonate" };
    st.trackConfirmedPlayingVideoId = null;
    env.exports.armPlaybackWatchdog(st.currentTrack);

    await env.getPlayerEvents().onStateChange({ data: 5 }); // CUED
    const watchdogAfter = st.watchdogTimerId;
    env.exports.clearPlaybackWatchdog();
    if (env.playerCalls.playVideo < 1) {
      throw new Error("Scenario B failed: CUED did not call playVideo()");
    }
    if (watchdogAfter === null) {
      throw new Error("Scenario B failed: CUED cleared watchdog while hasUserPlaybackStarted=true");
    }
  }

  // Scenario C: 3x rapid BAD taps -> immediate stopVideo(), BAD sent only ONCE, skippingInProgress released
  {
    let badFeedbackCount = 0;
    let resolveFeedback = null;
    const env = createEnv(async (url, opts = {}) => {
      if (url === "/feedback") {
        badFeedbackCount += 1;
        return new Promise((resolve) => {
          resolveFeedback = () => resolve({ ok: true, status: 200, json: async () => ({ status: "ok" }) });
        });
      }
      if (url.includes("/batch")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({
            items: [{ track_id: "trk_next_1", video_id: "VID_NEXT_1", artist: "New Artist", title: "Next Song" }],
          }),
        };
      }
      return { ok: true, status: 200, json: async () => ({ updates: 1, layers: [], axes: [] }) };
    });

    const st = env.exports.state;
    st.activeStationId = "pes_passion_groove";
    st.currentTrack = { track_id: "trk_bad_1", video_id: "VID_BAD_1", artist: "Bad Artist", title: "Skip Me" };
    st.queue = [{ track_id: "trk_next_1", video_id: "VID_NEXT_1", artist: "New Artist", title: "Next Song" }];

    const p1 = env.getEl("btnBad").trigger("click");
    const p2 = env.getEl("btnBad").trigger("click");
    const p3 = env.getEl("btnBad").trigger("click");
    await new Promise((r) => setTimeout(r, 10));
    if (env.playerCalls.stopVideo < 1) {
      throw new Error("Scenario C failed: BAD did not call stopVideo() immediately");
    }
    resolveFeedback();
    await Promise.all([p1, p2, p3]);
    env.exports.clearPlaybackWatchdog();
    if (badFeedbackCount !== 1) {
      throw new Error(`Scenario C failed: expected 1 BAD request, got ${badFeedbackCount}`);
    }
    if (st.skippingInProgress !== false) {
      throw new Error("Scenario C failed: skippingInProgress not released");
    }
    if (!env.playerCalls.loadVideoById.includes("VID_NEXT_1")) {
      throw new Error("Scenario C failed: next track not loaded after BAD");
    }
  }

  // Scenario D: fetchJson timeout rejects cleanly and releases skippingInProgress
  {
    const env = createEnv(async () => new Promise(() => {})); // never resolves
    let timedOut = false;
    try {
      await env.exports.fetchJson("/feedback", { method: "POST" }, 40);
    } catch (err) {
      timedOut = String(err.message || err).includes("timeout");
    }
    if (!timedOut) {
      throw new Error("Scenario D failed: fetchJson did not reject on timeout");
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


# ---------------------------------------------------------------------------
# 4. 7-Station thematic split + 5-Layer anti-overfitting & JP-contamination stoppers
# ---------------------------------------------------------------------------
def test_stopper5_jp_en_artist_alias_and_furigana_subtitle_dedup() -> None:
    t_jp = TrackEntity.build(artist="サザンオールスターズ", title="栞のテーマ", video_id="VID_SAS_JP")
    t_en = TrackEntity.build(
        artist="SOUTHERN ALL STARS",
        title="栞(しおり)のテーマ - Shiori no Theme",
        video_id="VID_SAS_EN",
    )
    assert normalize_artist("サザンオールスターズ") == "southern all stars"
    assert normalize_artist("SOUTHERN ALL STARS") == "southern all stars"
    assert normalize_track_key(t_jp.artist, t_jp.title) == normalize_track_key(t_en.artist, t_en.title)
    assert t_jp.track_id == t_en.track_id
    assert is_japanese_music_entity("サザンオールスターズ", "栞のテーマ") is True
    assert is_japanese_music_entity("SOUTHERN ALL STARS", "Shiori no Theme") is True
    assert is_japanese_music_entity("Parcels", "Tieduprightnow") is False


def test_stopper1_western_vs_japanese_hard_gate_and_station_routing(tmp_path: Path) -> None:
    # Routing check
    assert classify_entry_to_station("サザンオールスターズ", "栞のテーマ") == "jpop_citypop_groove"
    assert classify_entry_to_station("Southern All Stars", "TSUNAMI") == "jpop_citypop_groove"
    assert classify_entry_to_station("imase", "NIGHT DANCER") == "jpop_citypop_groove"
    assert classify_entry_to_station("Royel Otis", "Oysters In My Pocket") == "pes_passion_groove"
    assert classify_entry_to_station("Parcels", "Tieduprightnow") == "parisian_chic_disco"
    assert classify_entry_to_station("L'Impératrice", "Agitations tropicales") == "parisian_chic_disco"

    # Stage 1 pool digger hard gate on western_only station
    class FakeYTWithJapaneseLeak:
        embed_check_enabled = False

        def resolve_track(self, artist: str, title: str = "") -> Any:
            return SimpleNamespace(platform_track_id=f"SEED_{normalize_artist(artist)}", artist_browse_id=None)

        def get_related_tracks(self, platform_track_id: str, **kwargs: Any) -> list[TrackEntity]:
            return [
                TrackEntity.build(
                    artist="サザンオールスターズ",
                    title="栞のテーマ",
                    video_id="VID_JP_LEAK_1",
                    view_count=500_000,
                    source="watch_playlist",
                ),
                TrackEntity.build(
                    artist="SOUTHERN ALL STARS",
                    title="Shiori no Theme",
                    video_id="VID_JP_LEAK_2",
                    view_count=400_000,
                    source="watch_playlist",
                ),
                TrackEntity.build(
                    artist="Parcels",
                    title="Lightenup",
                    video_id="VID_WEST_OK_1",
                    view_count=150_000,
                    source="watch_playlist",
                ),
            ]

        def get_artist_related_deep_cuts(self, artist_browse_id: str, **kwargs: Any) -> list[TrackEntity]:
            return []

    db = RadioDatabase(tmp_path / "radio.db")
    stations = {s.station_id: s for s in default_theory_stations()}
    pes_station = stations["pes_passion_groove"]
    assert pes_station.region_policy == "western_only"

    digger = CandidatePoolDigger(
        db=db,
        yt_adapter=FakeYTWithJapaneseLeak(),  # type: ignore[arg-type]
        enable_mir_enrichment=False,
    )
    res = digger.dig_station(pes_station, target_pool_size=5)
    assert res["skipped_region_mismatch"] >= 2
    cands = db.get_eligible_candidates("pes_passion_groove", 500_000_000, include_heard_fallback=True)
    vids = [c.video_id for c in cands]
    assert "VID_WEST_OK_1" in vids
    assert "VID_JP_LEAK_1" not in vids
    assert "VID_JP_LEAK_2" not in vids


def test_stopper2_known_familiar_artists_compass_only_and_stopper4_pool_artist_cap(tmp_path: Path) -> None:
    ref_now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
    recent_ts = (ref_now - timedelta(days=2)).isoformat()
    # 5 plays of Southern All Stars -> must be extracted into known_familiar_artists
    entries = [
        {
            "header": "YouTube Music",
            "title": f"Watched Track {i}",
            "titleUrl": f"https://music.youtube.com/watch?v=VID_SAS_{i}",
            "subtitles": [{"name": "サザンオールスターズ - Topic"}],
            "time": recent_ts,
        }
        for i in range(5)
    ]
    takeout_path = tmp_path / "watch-history.json"
    takeout_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    stations_yaml = tmp_path / "stations.yaml"

    summary = generate_stations_yaml(
        output_path=stations_yaml,
        browser_auth_path=tmp_path / "missing.json",
        takeout_path=takeout_path,
        num_stations=7,
        offline=True,
        reference_time=ref_now,
    )
    assert summary["station_count"] == 7
    raw = yaml.safe_load(stations_yaml.read_text(encoding="utf-8"))
    familiar = set(raw["history_index"]["known_familiar_artists"])
    assert "southern all stars" in familiar

    # Dig on jpop_citypop_groove: familiar artist must be skipped, and new artist capped at 2 unheard tracks
    class FakeYTArtistSpam:
        embed_check_enabled = False

        def resolve_track(self, artist: str, title: str = "") -> Any:
            return SimpleNamespace(platform_track_id=f"SEED_{normalize_artist(artist)}", artist_browse_id=None)

        def get_related_tracks(self, platform_track_id: str, **kwargs: Any) -> list[TrackEntity]:
            return [
                TrackEntity.build(artist="サザンオールスターズ", title="新しい曲", video_id="VID_FAM_01", view_count=100_000),
                TrackEntity.build(artist="Fresh City Band", title="Song 1", video_id="VID_CAP_01", view_count=50_000),
                TrackEntity.build(artist="Fresh City Band", title="Song 2", video_id="VID_CAP_02", view_count=50_000),
                TrackEntity.build(artist="Fresh City Band", title="Song 3", video_id="VID_CAP_03", view_count=50_000),
                TrackEntity.build(artist="Fresh City Band", title="Song 4", video_id="VID_CAP_04", view_count=50_000),
            ]

        def get_artist_related_deep_cuts(self, artist_browse_id: str, **kwargs: Any) -> list[TrackEntity]:
            return []

    db = RadioDatabase(tmp_path / "radio.db")
    stations = {s.station_id: s for s in default_theory_stations()}
    jpop_station = stations["jpop_citypop_groove"]
    digger = CandidatePoolDigger(
        db=db,
        yt_adapter=FakeYTArtistSpam(),  # type: ignore[arg-type]
        enable_mir_enrichment=False,
    )
    res = digger.dig_station(
        jpop_station,
        target_pool_size=10,
        known_familiar_artists=familiar,
        max_tracks_per_artist_in_pool=2,
    )
    assert res["skipped_familiar_artist"] >= 1
    assert res["skipped_artist_cap"] >= 2
    cands = db.get_eligible_candidates("jpop_citypop_groove", 500_000_000, include_heard_fallback=True)
    vids = [c.video_id for c in cands]
    assert "VID_FAM_01" not in vids
    assert len([c for c in cands if c.artist == "Fresh City Band"]) == 2


def test_stopper3_calibration_artist_cap_and_stopper4_mmr_diversity_penalty() -> None:
    ref_now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
    recent_ts = (ref_now - timedelta(days=1)).isoformat()
    # 6 distinct songs by same artist + 1 song by another artist
    records = [
        {"artist": "サザンオールスターズ", "title": f"Hit {i}", "played_at": recent_ts}
        for i in range(6)
    ] + [
        {"artist": "imase", "title": "NIGHT DANCER", "played_at": recent_ts},
    ]
    grouped = _collect_youtube_station_top_tracks(records, reference_time=ref_now, max_tracks_per_artist=2)
    jpop_items = grouped["jpop_citypop_groove"]
    sas_count = sum(1 for artist, _title, _w in jpop_items if normalize_artist(artist) == "southern all stars")
    assert sas_count == 2

    # Stage 3 MMR diversity penalty: acoustically identical candidate to recent_mir_vectors loses to symmetric diverse candidate
    bandit = ContextualBanditRecommender(
        station_id="pes_passion_groove",
        seed_tags=["indie pop", "funk"],
        exploration_rate=0.0,
    )
    uniform = np.full(MIR_DIM, 0.5, dtype=np.float64)
    bandit._w_anchor = uniform.copy()
    bandit._w_slow = uniform.copy()
    bandit._w_session = uniform.copy()
    bandit._axis_weights = np.ones(MIR_DIM, dtype=np.float64)
    vec_recent = np.array(
        [0.85] * 8 + [0.15] * 8,
        dtype=np.float64,
    )
    vec_diverse = np.array(
        [0.15] * 8 + [0.85] * 8,
        dtype=np.float64,
    )

    t_same = TrackEntity.build(
        artist="Clone Artist",
        title="Clone Track",
        video_id="VID_MMR_SAME",
        view_count=80_000,
        mir_vector=vec_recent.tolist(),
    )
    t_diff = TrackEntity.build(
        artist="Fresh Artist",
        title="Fresh Track",
        video_id="VID_MMR_DIFF",
        view_count=80_000,
        mir_vector=vec_diverse.tolist(),
    )

    ranked_no_mmr = bandit.rank_candidates([t_same, t_diff], seen_artists=set(), limit=2)
    ranked_with_mmr = bandit.rank_candidates(
        [t_same, t_diff],
        seen_artists=set(),
        limit=2,
        recent_mir_vectors=[vec_recent],
    )
    score_same_no_mmr = next(r.score for r in ranked_no_mmr if r.track.video_id == "VID_MMR_SAME")
    score_same_mmr = next(r.score for r in ranked_with_mmr if r.track.video_id == "VID_MMR_SAME")
    assert score_same_mmr < score_same_no_mmr
    assert ranked_with_mmr[0].track.video_id == "VID_MMR_DIFF"
