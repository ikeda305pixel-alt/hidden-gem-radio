"""Regression tests for all 12 Cross-Model Blind Audit findings (commit 80aedc0)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from src.adapters.youtube import (
    YouTubeMusicAdapter,
    _artist_matches,
    _is_official_lyric_channel,
)
from src.db import RadioDatabase
from src.digger.pool import CandidatePoolDigger
from src.models import StationConfig, TrackEntity
from src.radio_server import create_app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_critical1_upsert_track_twice_same_video_id_no_index_error(tmp_path: Path) -> None:
    """[Critical #1] Calling upsert_track twice on the same existing track must not raise IndexError on 'source'."""
    db = RadioDatabase(tmp_path / "radio.db")
    track = TrackEntity.build(
        artist="Kendrick Lamar",
        title="Money Trees",
        video_id="VID_SAME_01",
        view_count=150_000,
        duration_seconds=386,
        source="youtube_music",
    )
    id1 = db.upsert_track("us_hiphop", track)
    id2 = db.upsert_track("us_hiphop", track)
    assert id1 == id2

    fetched = db.get_track(track_id=id1)
    assert fetched is not None
    assert fetched.video_id == "VID_SAME_01"
    assert fetched.source == "youtube_music"

    # Also verify video_id replacement resets unplayable=0 and updates source
    db.mark_unplayable(video_id="VID_SAME_01", track_id=id1)
    replacement = TrackEntity(
        track_id=id1,
        canonical_key=track.canonical_key,
        artist=track.artist,
        title=track.title,
        video_id="VID_OMV_002",
        source="youtube_music_omv",
        unplayable=False,
    )
    db.upsert_track("us_hiphop", replacement)
    updated = db.get_track(track_id=id1)
    assert updated is not None
    assert updated.video_id == "VID_OMV_002"
    assert updated.unplayable is False
    assert updated.source == "youtube_music_omv"


def test_critical2_embed_headers_preview_playability_status_and_oembed_400() -> None:
    """[Critical #2] /embed/{id} receives Referer/Origin headers, parses previewPlayabilityStatus JSON, and rejects oEmbed 400."""
    seen_embed_headers: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oembed":
            url_param = request.url.params.get("url", "")
            if "VID_400_BAD" in url_param:
                return httpx.Response(400, text="Bad Request")
            return httpx.Response(200, json={"title": "Track"})
        if request.url.path.startswith("/embed/"):
            vid = request.url.path.split("/embed/")[-1]
            seen_embed_headers.append(
                {
                    "referer": request.headers.get("referer", ""),
                    "origin": request.headers.get("origin", ""),
                }
            )
            if vid == "VID_PREV_UNPLAY":
                inner = json.dumps({"previewPlayabilityStatus": {"status": "UNPLAYABLE", "reason": "Blocked"}})
                outer_json = json.dumps({"embedded_player_response": inner})
                return httpx.Response(200, text=f"<html><script>ytcfg.set({outer_json});</script></html>")
            if vid == "VID_MISSING_REF":
                inner = json.dumps({"playabilityStatus": {"status": "ERROR", "reason": "MISSING_REFERRER"}})
                outer_json = json.dumps({"embedded_player_response": inner})
                return httpx.Response(200, text=f"<html><script>ytcfg.set({outer_json});</script></html>")
            if vid == "VID_REAL_ERR":
                inner = json.dumps({"playabilityStatus": {"status": "ERROR", "reason": "VIDEO_REMOVED"}})
                outer_json = json.dumps({"embedded_player_response": inner})
                return httpx.Response(200, text=f"<html><script>ytcfg.set({outer_json});</script></html>")
            inner = json.dumps({"previewPlayabilityStatus": {"status": "OK", "playableInEmbed": True}})
            outer_json = json.dumps({"embedded_player_response": inner})
            return httpx.Response(200, text=f"<html><script>ytcfg.set({outer_json});</script></html>")
        return httpx.Response(404)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = YouTubeMusicAdapter(client=object(), http_client=http_client, check_embeddable=True)

    assert adapter.is_video_embeddable("VID_400_BAD") is False
    assert adapter.is_video_embeddable("VID_PREV_UNPLAY") is False
    assert adapter.is_video_embeddable("VID_REAL_ERR") is False
    assert adapter.is_video_embeddable("VID_MISSING_REF") is True
    assert adapter.is_video_embeddable("VID_OK_PREV") is True

    assert seen_embed_headers, "Expected /embed/{id} requests"
    for hdrs in seen_embed_headers:
        assert hdrs["referer"] == "http://127.0.0.1:8000/"
        assert hdrs["origin"] == "http://127.0.0.1:8000"

    html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
    assert '<meta name="referrer" content="strict-origin-when-cross-origin" />' in html


def test_critical3_search_playable_falls_back_to_videos_official_omv_when_songs_empty() -> None:
    """[Critical #3] When filter='songs' returns empty, search_playable falls back to filter='videos' allowing official OMV only."""
    search_filters: list[str | None] = []

    class FakeUnauthYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
            _ = (query, limit)
            search_filters.append(filter)
            if filter == "songs":
                return []
            if filter == "videos":
                return [
                    {
                        "videoId": "VID_UGC_BAD",
                        "title": "Random Fan Upload",
                        "resultType": "video",
                        "videoType": "MUSIC_VIDEO_TYPE_UGC",
                        "artists": [{"name": "Random User"}],
                        "duration": "3:20",
                    },
                    {
                        "videoId": "VID_OMV_OK",
                        "title": "Kendrick Lamar - Alright",
                        "resultType": "video",
                        "videoType": "MUSIC_VIDEO_TYPE_OMV",
                        "artists": [{"name": "Kendrick Lamar", "id": "UC_KL"}],
                        "duration": "3:40",
                    },
                ]
            return []

    adapter = YouTubeMusicAdapter(client=FakeUnauthYT(), check_embeddable=False)
    results = adapter.search_playable("Kendrick Lamar Alright", limit=6)
    assert search_filters == ["songs", "videos"]
    assert [r["videoId"] for r in results] == ["VID_OMV_OK"]


def test_high4_and_low11_embed_breaker_half_open_ttl_recovery_blocks_alt_and_cache_cap() -> None:
    """[High #4 & Low #11] Breaker skips alt confirmation while open, recovers after TTL, caps cache at 4096, and close() respects ownership."""
    fail_mode = True
    http_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        if fail_mode:
            return httpx.Response(503, text="Outage")
        if request.url.path == "/oembed":
            return httpx.Response(200, json={"title": "OK"})
        return httpx.Response(200, text='<html>{"playabilityStatus":{"status":"OK","playableInEmbed":true}}</html>')

    class FakeYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
            _ = (query, filter, limit)
            return [
                {
                    "videoId": "OMV_RECOVER",
                    "title": "Artist X - Song Y (Official Video)",
                    "videoType": "MUSIC_VIDEO_TYPE_OMV",
                    "artists": [{"name": "Artist X"}],
                    "duration": "3:30",
                }
            ]

    injected_client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = YouTubeMusicAdapter(client=FakeYT(), http_client=injected_client, check_embeddable=True)
    adapter._embed_breaker_ttl_sec = 0.05

    # 4 consecutive 503s open the breaker
    for i in range(4):
        assert adapter.is_video_embeddable(f"VID_ERR_{i}") is True
    assert adapter._is_embed_breaker_open() is True

    # While breaker is open, find_embeddable_alternative MUST return None without confirming unverified video
    calls_before = http_calls
    assert adapter.find_embeddable_alternative("Artist X", "Song Y") is None
    assert http_calls == calls_before

    # After TTL expires and network recovers, Half-Open probe succeeds and resets breaker
    fail_mode = False
    time.sleep(0.07)
    assert adapter._is_embed_breaker_open() is False
    alt = adapter.find_embeddable_alternative("Artist X", "Song Y")
    assert alt is not None
    alt_track, alt_kind = alt
    assert alt_kind == "omv"
    assert alt_track.video_id == "OMV_RECOVER"
    assert alt_track.track_id.startswith("trk_")
    assert adapter._embed_consecutive_net_errors == 0

    # [Low #11] close() must NOT close externally injected http_client
    adapter.close()
    assert injected_client.is_closed is False

    # [Low #11] Cache cap enforces max size (4096 default; test with small cap)
    adapter._embed_cache_max_size = 5
    with adapter._embed_lock:
        for i in range(10):
            adapter._cache_embeddable_locked(f"CAP_{i}", True)
    assert len(adapter._embeddable_cache) == 5
    assert "CAP_9" in adapter._embeddable_cache
    assert "CAP_0" not in adapter._embeddable_cache


def test_high5_strict_official_lyric_filter_rejects_generic_channels_and_short_name_false_matches() -> None:
    """[High #5] Generic channels ('7clouds Music', 'AI Music Official') and short-name false matches ('IO' vs 'Studio') are rejected."""
    # Short artist name false-match checks
    assert _artist_matches("IO", "Studio") is False
    assert _artist_matches("Studio", "IO") is False
    assert _artist_matches("IO", "IO - Tokyo (Official Lyric Video)") is True
    assert _artist_matches("IO", "IOVEVO") is True
    assert _artist_matches("IO", "IO - Topic") is True

    # Generic lyric channels must be rejected
    assert _is_official_lyric_channel(
        {"title": "Kendrick Lamar - DNA (Lyrics)", "artists": [{"name": "7clouds Music"}]},
        "Kendrick Lamar",
    ) is False
    assert _is_official_lyric_channel(
        {"title": "Kendrick Lamar - DNA (Official Lyric Video)", "artists": [{"name": "AI Music Official"}]},
        "Kendrick Lamar",
    ) is False
    # Concrete major label with target artist in title is accepted
    assert _is_official_lyric_channel(
        {"title": "Kendrick Lamar - DNA (Official Lyric Video)", "artists": [{"name": "Interscope Records"}]},
        "Kendrick Lamar",
    ) is True
    # Artist VEVO channel is accepted
    assert _is_official_lyric_channel(
        {"title": "DNA (Official Lyric Video)", "artists": [{"name": "KendrickLamarVEVO"}]},
        "Kendrick Lamar",
    ) is True

    # End-to-end find_embeddable_alternative rejects instrumental / reaction / bass boosted / shorts / generic channels
    class SlopLyricYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
            _ = (filter, limit)
            if "lyric" not in query.lower():
                return []
            return [
                {
                    "videoId": "BAD_7CLOUDS",
                    "title": "IO - Tokyo (Lyrics)",
                    "artists": [{"name": "7clouds Music"}],
                    "duration": "3:10",
                },
                {
                    "videoId": "BAD_AIOFF",
                    "title": "IO - Tokyo (Official Lyric Video)",
                    "artists": [{"name": "AI Music Official"}],
                    "duration": "3:10",
                },
                {
                    "videoId": "BAD_STUDIO",
                    "title": "Studio - Tokyo (Official Lyric Video)",
                    "artists": [{"name": "Studio"}],
                    "duration": "3:10",
                },
                {
                    "videoId": "BAD_INST",
                    "title": "IO - Tokyo (Instrumental Lyrics)",
                    "artists": [{"name": "IO"}],
                    "duration": "3:10",
                },
                {
                    "videoId": "BAD_BASS",
                    "title": "IO - Tokyo (Bass Boosted Lyrics)",
                    "artists": [{"name": "IO"}],
                    "duration": "3:10",
                },
                {
                    "videoId": "BAD_SHORTS",
                    "title": "IO - Tokyo (Lyrics Shorts)",
                    "artists": [{"name": "IO"}],
                    "duration": "3:10",
                },
            ]

    def ok_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oembed":
            return httpx.Response(200, json={"title": "OK"})
        return httpx.Response(200, text='<html>{"playabilityStatus":{"status":"OK","playableInEmbed":true}}</html>')

    adapter = YouTubeMusicAdapter(
        client=SlopLyricYT(),
        http_client=httpx.Client(transport=httpx.MockTransport(ok_handler)),
        check_embeddable=True,
    )
    assert adapter.find_embeddable_alternative("IO", "Tokyo") is None


def test_high6_and_medium10_dig_station_caps_alt_searches_and_preserves_unified_track_id(
    tmp_path: Path,
) -> None:
    """[High #6 & Medium #10] dig_station caps alt searches at max_alt_searches_per_dig and preserves TrackEntity.build ID."""
    db = RadioDatabase(tmp_path / "radio.db")
    alt_search_calls = 0

    class ManyBlockedYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
            _ = (query, limit)
            if filter == "songs":
                return [
                    {
                        "videoId": "SEED_CAP_01",
                        "title": "Seed Song",
                        "resultType": "song",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Seed Artist", "id": "UC_SEED"}],
                        "album": {"name": "Seed Album"},
                        "duration_seconds": 200,
                    }
                ]
            return []

        def get_watch_playlist(self, videoId: str, limit: int = 25) -> dict[str, Any]:
            _ = (videoId, limit)
            return {
                "tracks": [
                    {
                        "videoId": f"BLK_VID_{i:02d}",
                        "title": f"Blocked Song {i}",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": f"Artist {i}", "id": f"UC_A_{i}"}],
                        "album": {"name": f"Album {i}"},
                        "length": "3:20",
                        "views": f"{50 - i}K",
                    }
                    for i in range(6)
                ]
            }

        def get_artist(self, channel_id: str) -> dict[str, Any]:
            _ = channel_id
            return {}

    def all_blocked_http(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/oembed":
            return httpx.Response(403, text="Forbidden")
        return httpx.Response(403, text="Forbidden")

    yt_adapter = YouTubeMusicAdapter(
        client=ManyBlockedYT(),
        http_client=httpx.Client(transport=httpx.MockTransport(all_blocked_http)),
        check_embeddable=True,
    )

    def counting_alt_finder(
        artist: str,
        title: str,
        *,
        exclude_video_ids: set[str] | None = None,
    ) -> tuple[TrackEntity, str] | None:
        nonlocal alt_search_calls
        _ = exclude_video_ids
        alt_search_calls += 1
        alt = TrackEntity.build(
            artist=artist,
            title=title,
            video_id=f"ALT_OMV_{alt_search_calls:02d}",
            source="youtube_music_omv",
        )
        return alt, "omv"

    yt_adapter.find_embeddable_alternative = counting_alt_finder  # type: ignore[method-assign]

    digger = CandidatePoolDigger(db=db, yt_adapter=yt_adapter, enable_mir_enrichment=False)
    station = StationConfig(
        station_id="cap_test",
        name="Cap Test",
        seed_tags=["hip-hop"],
        seed_artists=["Seed Artist"],
        spotify_anchor_seeds=["Seed Artist - Seed Song"],
        max_popularity_views=500_000,
    )
    summary = digger.dig_station(station, target_pool_size=20, max_alt_searches_per_dig=2)

    assert alt_search_calls == 2
    assert summary["alt_searches_used"] == 2
    assert summary["replaced_omv"] == 2
    assert summary["added"] == 2

    saved = db.get_eligible_candidates("cap_test", max_popularity_views=500_000)
    assert len(saved) == 2
    for tr in saved:
        assert tr.track_id.startswith("trk_"), f"Expected unified trk_* ID, got {tr.track_id}"
        assert tr.video_id.startswith("ALT_OMV_")


def test_medium8_and_low12_web_ui_delayed_onerror_guard_stopvideo_origin_and_error153(
    tmp_path: Path,
) -> None:
    """[Medium #8 & Low #12] Delayed onError cannot break circuitBroken, player stops on trip, origin passed, Error 153 non-blacklisted."""
    # Backend check: Error 153 is classified as non-permanent config error (blacklisted=False)
    stations_yaml = tmp_path / "stations.yaml"
    stations_yaml.write_text(
        json.dumps(
            {
                "stations": [
                    {
                        "station_id": "st_153",
                        "name": "Err 153 Test",
                        "seed_tags": ["hip-hop"],
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
        auto_dig=False,
    )
    client = TestClient(app)
    res_153 = client.post(
        "/api/tracks/VID_ERR_153/unplayable",
        json={"station_id": "st_153", "reason": "yt_error_153", "yt_error_code": 153},
    )
    assert res_153.status_code == 200
    assert res_153.json()["blacklisted"] is False
    assert "VID_ERR_153" not in app.state.runtime.db.get_blacklisted_video_ids()

    # Frontend real Node.js vm execution
    html_path = PROJECT_ROOT / "web" / "index.html"
    node_script = r"""
const fs = require("fs");
const vm = require("vm");
const assert = require("assert");

const html = fs.readFileSync(process.argv[1], "utf8");
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(scriptMatch, "Inline script not found");

function makeElement(id) {
  const listeners = {};
  const classes = new Set();
  return {
    id, textContent: "", innerHTML: "", value: "", children: [],
    classList: {
      add: (c) => classes.add(c),
      remove: (c) => classes.delete(c),
      toggle: (c, f) => (f ? classes.add(c) : classes.delete(c)),
      contains: (c) => classes.has(c),
    },
    appendChild(c) { this.children.push(c); },
    addEventListener(t, fn) { (listeners[t] ||= []).push(fn); },
    async dispatch(t) { for (const fn of listeners[t] || []) await fn({}); },
  };
}

const elements = {};
const fetchCalls = [];
let fetchHandler = async () => ({ ok: true, status: 200, json: async () => ({ items: [] }) });

let capturedPlayerVars = null;
const YTStub = {
  PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
  Player: function (_id, config) {
    capturedPlayerVars = config.playerVars;
    this._config = config;
    this._state = 1;
    this.stopCalls = 0;
    this.loadedVideos = [];
    this.loadVideoById = (vid) => { this.loadedVideos.push(vid); };
    this.stopVideo = () => { this.stopCalls += 1; this._state = 5; };
    this.pauseVideo = () => { this._state = 2; };
    this.playVideo = () => { this._state = 1; };
    this.getPlayerState = () => this._state;
    this.getCurrentTime = () => 0;
    this.getDuration = () => 180;
    this.isMuted = () => false;
    this.mute = () => {};
    this.unMute = () => {};
  },
};

const sandbox = {
  console,
  location: { origin: "http://127.0.0.1:8000" },
  document: {
    hidden: false,
    getElementById: (id) => (elements[id] ||= makeElement(id)),
    createElement: (tag) => makeElement(tag),
  },
  navigator: { mediaSession: { setActionHandler: () => {} } },
  MediaMetadata: function (m) { Object.assign(this, m); },
  YT: YTStub,
  fetch: async (url, options) => { fetchCalls.push({ url, options }); return fetchHandler(url, options); },
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
  scriptMatch[1] + "\nwindow.__exports = { state, el, handleUnplayableSkip };",
  ctx
);

(async () => {
  await new Promise((r) => setImmediate(r));
  const api = sandbox.__exports;
  sandbox.onYouTubeIframeAPIReady();
  api.state.player._config.events.onReady();

  // [Critical #2] playerVars.origin is set from window.location.origin
  assert.strictEqual(capturedPlayerVars.origin, "http://127.0.0.1:8000");

  api.state.stations = [{ station_id: "st_a", name: "St A", candidate_count: 10 }];
  api.state.activeStationId = "st_a";

  // [Low #12] Error 153 uses strict 4-error circuit breaker (not 10 embed skips) and calls stopVideo()
  let seq = 0;
  fetchHandler = async (url) => {
    if (url.includes("/unplayable")) {
      seq += 1;
      return {
        ok: true,
        status: 200,
        json: async () => ({
          replacement: { track_id: `t_${seq}`, video_id: `VID_153_${seq}`, artist: "A", title: "T" },
        }),
      };
    }
    return { ok: true, status: 200, json: async () => ({ items: [] }) };
  };
  api.state.currentTrack = { track_id: "t_0", video_id: "VID_153_0", artist: "A", title: "T" };
  for (let i = 0; i < 4; i += 1) {
    await api.handleUnplayableSkip(api.state.currentTrack, "yt_error_153", 153);
  }
  assert.strictEqual(api.state.circuitBroken, true, "4x Error 153 must trip strict circuit breaker");
  assert.strictEqual(api.state.consecutiveEmbedSkips, 0, "Error 153 must NOT increment consecutiveEmbedSkips");
  assert.strictEqual(api.state.player.stopCalls, 1, "stopVideo() must be called when circuit breaker trips");
  assert.strictEqual(api.state.currentTrack, null);

  // [Medium #8] Delayed onError after circuitBroken (with currentTrack === null or stale track) MUST be a complete no-op
  const callsBeforeLateError = fetchCalls.length;
  const loadedBeforeLateError = api.state.player.loadedVideos.length;
  await api.state.player._config.events.onError({ data: 150 });
  await api.handleUnplayableSkip({ track_id: "late", video_id: "VID_LATE" }, "yt_error_150", 150);
  assert.strictEqual(api.state.circuitBroken, true, "circuitBroken must remain true after delayed onError");
  assert.strictEqual(api.state.currentTrack, null, "currentTrack must remain null after delayed onError");
  assert.strictEqual(fetchCalls.length, callsBeforeLateError, "Delayed onError must trigger zero network calls");
  assert.strictEqual(api.state.player.loadedVideos.length, loadedBeforeLateError, "Delayed onError must not load next video");

  console.log("AUDIT_12_UI_OK");
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
    assert "AUDIT_12_UI_OK" in proc.stdout


def test_medium9_count_playable_candidates_excludes_known_history(tmp_path: Path) -> None:
    """[Medium #9] count_playable_candidates and GET /stations exclude known_video_ids and known_track_keys."""
    stations_yaml = tmp_path / "stations.yaml"
    stations_yaml.write_text(
        json.dumps(
            {
                "stations": [
                    {
                        "station_id": "st_hist",
                        "name": "History Filter Station",
                        "seed_tags": ["hip-hop"],
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
        auto_dig=False,
    )
    runtime = app.state.runtime
    for i in range(5):
        runtime.db.upsert_track(
            "st_hist",
            TrackEntity.build(
                artist=f"Artist {i}",
                title=f"Song {i}",
                video_id=f"VID_HIST_{i:02d}",
                view_count=10_000,
                duration_seconds=200,
            ),
        )

    # Mark 1 by known_video_ids and 1 by known_track_keys
    runtime.known_video_ids = {"VID_HIST_00"}
    runtime.known_track_keys = {"artist 1::song 1"}

    assert runtime.db.count_playable_candidates("st_hist", 500_000) == 5
    assert (
        runtime.db.count_playable_candidates(
            "st_hist",
            500_000,
            exclude_track_ids=runtime.known_video_ids,
            exclude_canonical_keys=runtime.known_track_keys,
        )
        == 3
    )

    client = TestClient(app)
    res = client.get("/stations")
    assert res.status_code == 200
    assert res.json()["stations"][0]["candidate_count"] == 3
