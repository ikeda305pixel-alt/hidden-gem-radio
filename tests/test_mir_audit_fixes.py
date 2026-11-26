"""Real-execution regression tests for all 11 MIR/Graph blind-audit findings."""

from __future__ import annotations

import io
import math
import re
import sqlite3
import subprocess
import time
import wave
from pathlib import Path

import httpx
import numpy as np
from fastapi.testclient import TestClient

from src.adapters.deezer import DeezerClient
from src.adapters.youtube import YouTubeMusicAdapter, is_slop_or_unofficial
from src.db import RadioDatabase
from src.digger.pool import CandidatePoolDigger
from src.models import StationConfig, TrackEntity, save_stations_yaml
from src.radio_server import create_app
from src.recommender.audio_mir import (
    MIR_DIM,
    TARGET_SR,
    AudioMIRExtractor,
    analyze_pcm_a1_c4,
    compute_yt_views_sweetspot,
)
from src.recommender.bandit import ContextualBanditRecommender
from src.recommender.calibrate_profile import calibrate_all_stations

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _synth_30s_click_wav(bpm: float = 126.0, duration_sec: float = 30.0, sr: int = TARGET_SR) -> bytes:
    n = int(sr * duration_sec)
    y = np.zeros(n, dtype=np.float32)
    interval = round(sr * 60.0 / bpm)
    click_len = int(sr * 0.035)
    t = np.arange(click_len, dtype=np.float32) / np.float32(sr)
    pulse = np.exp(-t * 75.0) * (
        0.65 * np.sin(2.0 * math.pi * 75.0 * t) + 0.35 * np.sin(2.0 * math.pi * 1100.0 * t)
    )
    for pos in range(0, n - click_len, interval):
        y[pos : pos + click_len] += pulse
    pcm16 = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. [Critical #1] Production create_app() dig enables MIR enrichment & no fake 118 BPM
# ---------------------------------------------------------------------------
def test_production_create_app_dig_enables_mir_and_omits_fake_118_bpm(tmp_path: Path) -> None:
    stations_yaml = tmp_path / "stations.yaml"
    save_stations_yaml(
        stations_yaml,
        [
            StationConfig(
                station_id="us_hiphop_bouncy",
                name="US Hip-Hop & Bouncy Rap",
                seed_tags=["hip hop"],
                seed_artists=["bbno$"],
                spotify_anchor_seeds=["bbno$ - edamame"],
                max_popularity_views=500_000_000,
            )
        ],
    )
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=stations_yaml,
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    runtime = app.state.runtime

    class StubYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8):
            return [
                {
                    "videoId": "SEED_BBNO_01",
                    "title": "edamame",
                    "artists": [{"name": "bbno$", "id": "ART_BBNO"}],
                    "resultType": "song",
                    "videoType": "MUSIC_VIDEO_TYPE_ATV",
                    "duration_seconds": 135,
                    "views": "80M",
                }
            ]

        def get_watch_playlist(self, videoId: str, radio: bool = False, limit: int = 25):
            return {
                "tracks": [
                    {
                        "videoId": "VID_MEASURED_1",
                        "title": "Measured Track",
                        "artists": [{"name": "Known Artist", "id": "ART_K1"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "length": "3:05",
                    },
                    {
                        "videoId": "VID_UNMEASURED",
                        "title": "Obscure Unindexed Track",
                        "artists": [{"name": "Unknown Indie", "id": "ART_U1"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "length": "3:10",
                    },
                ]
            }

        def get_artist(self, browseId: str):
            return {}

    def _deezer_transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            q = str(request.url.params.get("q") or "").lower()
            if "measured track" in q:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": 55501,
                                "title": "Measured Track",
                                "artist": {"name": "Known Artist"},
                                "bpm": 132.0,
                                "gain": -9.5,
                                "rank": 410_000,
                                "duration": 185,
                                "preview": "",
                            }
                        ]
                    },
                )
            return httpx.Response(200, json={"data": []})
        return httpx.Response(404)

    runtime._yt_adapter = YouTubeMusicAdapter(client=StubYT())
    runtime._deezer_client = DeezerClient(
        db=runtime.db,
        min_interval_sec=0.0,
        http_client=httpx.Client(transport=httpx.MockTransport(_deezer_transport)),
    )
    runtime.mir_extractor.deezer = runtime._deezer_client
    runtime.mir_extractor.ffmpeg_bin = ""

    # Accessing runtime.digger MUST construct CandidatePoolDigger with enable_mir_enrichment=True
    digger = runtime.digger
    assert digger.enable_mir_enrichment is True
    assert digger.max_mir_tracks_per_dig == 15

    client = TestClient(app)
    dig_res = client.post("/stations/us_hiphop_bouncy/dig").json()
    assert dig_res["added"] == 2
    assert dig_res["mir_enriched"] == 2

    batch = client.get("/api/stations/us_hiphop_bouncy/batch", params={"count": 5}).json()["items"]
    by_vid = {item["video_id"]: item for item in batch}
    assert by_vid["VID_MEASURED_1"]["measured_bpm"] == 132.0
    # Unmeasured track MUST return measured_bpm=None (never fake 118.0!)
    assert by_vid["VID_UNMEASURED"]["measured_bpm"] is None

    # Also verify view_count=None official ATV track gets Deezer rank sweetspot instead of 0.45 penalty
    assert compute_yt_views_sweetspot(None, deezer_rank=410_000) > 0.90
    assert compute_yt_views_sweetspot(None, deezer_rank=None) == 0.60


# ---------------------------------------------------------------------------
# 2. [Critical #2] 30-second WAV analyze_pcm_a1_c4 performance < 250ms
# ---------------------------------------------------------------------------
def test_analyze_pcm_a1_c4_30s_wav_runs_under_250ms() -> None:
    wav_30s = _synth_30s_click_wav(bpm=126.0, duration_sec=30.0)
    with wave.open(io.BytesIO(wav_30s), "rb") as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    # Warm up chroma matrix cache once (production server caches it per process)
    analyze_pcm_a1_c4(pcm[: TARGET_SR * 2])

    t0 = time.perf_counter()
    vec12, est_bpm = analyze_pcm_a1_c4(pcm)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert vec12.shape == (12,)
    assert est_bpm is not None and abs(est_bpm - 126.0) <= 6.0
    assert elapsed_ms < 250.0, f"30s MIR analysis took {elapsed_ms:.1f}ms (target <250ms)"


# ---------------------------------------------------------------------------
# 3. [High #3] Legacy indie_groove_hooks SQLite + YAML + .pkl auto-migration
# ---------------------------------------------------------------------------
def test_legacy_indie_groove_hooks_auto_migrates_db_yaml_and_pkl(tmp_path: Path) -> None:
    db_path = tmp_path / "radio.db"
    model_dir = tmp_path / "models"
    stations_yaml = tmp_path / "stations.yaml"

    # Seed an old-style database with indie_groove_hooks rows
    db_init = RadioDatabase(db_path)
    t_old = TrackEntity.build("Oberhofer", "Gold", video_id="VID_OBER_01", view_count=200_000)
    db_init.upsert_track("indie_groove_cult", t_old)
    # Manually rewrite station_id back to legacy indie_groove_hooks to simulate pre-upgrade DB
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE station_candidates SET station_id = 'indie_groove_hooks'")
        conn.execute(
            """
            INSERT INTO feedback_events (station_id, track_id, video_id, artist_norm, event_type, reward, created_at)
            VALUES ('indie_groove_hooks', ?, 'VID_OBER_01', 'oberhofer', 'GOOD', 1.0, ?)
            """,
            (t_old.track_id, time.time()),
        )
        conn.execute(
            """
            INSERT INTO playlist_tracks (playlist_id, video_id, track_id, artist, title, station_id, added_at)
            VALUES ('pl_favorites', 'VID_OBER_01', ?, 'Oberhofer', 'Gold', 'indie_groove_hooks', ?)
            """,
            (t_old.track_id, time.time()),
        )
        conn.commit()

    # Seed legacy bandit_indie_groove_hooks.pkl with 1 learned update
    old_bandit = ContextualBanditRecommender(
        station_id="indie_groove_hooks",
        seed_tags=["indie pop", "folktronica"],
        model_path=model_dir / "bandit_indie_groove_hooks.pkl",
    )
    old_bandit.update(t_old, event_type="GOOD")
    assert (model_dir / "bandit_indie_groove_hooks.pkl").exists()

    # Seed legacy stations.yaml referencing indie_groove_hooks & old 500k cap
    stations_yaml.write_text(
        """stations:
  - station_id: indie_groove_hooks
    name: Indie Groove & Feel-Good
    seed_tags: [indie pop, folktronica]
    seed_artists: [Oberhofer]
    max_popularity_views: 500000
    exploration_rate: 1.0
""",
        encoding="utf-8",
    )

    app = create_app(
        db_path=db_path,
        stations_path=stations_yaml,
        model_dir=model_dir,
        auto_dig=False,
    )
    runtime = app.state.runtime

    assert "indie_groove_cult" in runtime.stations
    assert "indie_groove_hooks" not in runtime.stations
    assert runtime.stations["indie_groove_cult"].max_popularity_views == 500_000_000
    assert (model_dir / "bandit_indie_groove_cult.pkl").exists()
    assert runtime.bandits["indie_groove_cult"].n_updates == 1
    assert runtime.db.count_feedback("indie_groove_cult") == 1
    assert runtime.db.count_feedback("indie_groove_hooks") == 0
    cands = runtime.db.get_eligible_candidates("indie_groove_cult", 500_000_000, include_heard_fallback=True)
    assert any(c.video_id == "VID_OBER_01" for c in cands)


# ---------------------------------------------------------------------------
# 4. [High #4] Uncalibrated POST /feedback persists station_profiles & external calibration is never overwritten
# ---------------------------------------------------------------------------
def test_uncalibrated_feedback_persists_and_does_not_overwrite_external_calibration(tmp_path: Path) -> None:
    db_path = tmp_path / "radio.db"
    stations_yaml = tmp_path / "stations.yaml"
    model_dir = tmp_path / "models"

    app = create_app(
        db_path=db_path,
        stations_path=stations_yaml,
        model_dir=model_dir,
        auto_dig=False,
    )
    runtime = app.state.runtime
    t1 = TrackEntity.build("Artist A", "Song A", video_id="VID_PERS_01", mir_vector=[0.9] * MIR_DIM)
    t2 = TrackEntity.build("Artist B", "Song B", video_id="VID_PERS_02", mir_vector=[0.1] * MIR_DIM)
    runtime.db.upsert_track("us_hiphop_bouncy", t1)
    runtime.db.upsert_track("us_hiphop_bouncy", t2)

    # Before calibration, station_profiles row does not exist
    assert runtime.db.get_station_profile("us_hiphop_bouncy") is None

    client = TestClient(app)
    r1 = client.post("/feedback", json={"station_id": "us_hiphop_bouncy", "video_id": "VID_PERS_01", "event_type": "GOOD"})
    assert r1.status_code == 200

    # Even uncalibrated, slow_weights MUST now be persisted in SQLite station_profiles
    prof_after_fb1 = runtime.db.get_station_profile("us_hiphop_bouncy")
    assert prof_after_fb1 is not None
    assert len(prof_after_fb1["slow_weights"]) == MIR_DIM

    # Simulate external CLI calibrate_profile running from a SEPARATE RadioDatabase connection
    ext_db = RadioDatabase(db_path)
    time.sleep(0.02)  # guarantee strictly newer updated_at timestamp
    ext_centroid = [0.77] * MIR_DIM
    ext_axis_weights = [1.35] * MIR_DIM
    ext_db.upsert_station_profile(
        station_id="us_hiphop_bouncy",
        centroid=ext_centroid,
        axis_weights=ext_axis_weights,
        slow_weights=prof_after_fb1["slow_weights"],
        sample_counts={"youtube": 10, "spotify": 20, "cult_anchors": 0},
    )

    # Next POST /feedback on running server MUST NOT overwrite external centroid / axis_weights!
    r2 = client.post("/feedback", json={"station_id": "us_hiphop_bouncy", "video_id": "VID_PERS_02", "event_type": "BAD"})
    assert r2.status_code == 200

    prof_after_fb2 = ext_db.get_station_profile("us_hiphop_bouncy")
    assert prof_after_fb2 is not None
    assert np.allclose(prof_after_fb2["centroid"], ext_centroid), "External calibrated centroid was overwritten!"
    assert np.allclose(prof_after_fb2["axis_weights"], ext_axis_weights), "External axis_weights were overwritten!"
    assert not np.allclose(prof_after_fb2["slow_weights"], prof_after_fb1["slow_weights"])
    assert np.allclose(runtime.bandits["us_hiphop_bouncy"].get_calibrated_anchor(), ext_centroid)

    # Also verify calibrate_all_stations switches cleanly to offline on consecutive errors
    err_client = DeezerClient(
        db=ext_db,
        min_interval_sec=0.0,
        max_consecutive_errors=2,
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _req: httpx.Response(503))),
    )
    cal = calibrate_all_stations(
        ext_db,
        stations=[runtime.stations["us_hiphop_bouncy"]],
        mir_extractor=AudioMIRExtractor(db=ext_db, deezer_client=err_client, ffmpeg_bin=""),
        allow_network=True,
        max_spotify_per_station=5,
    )
    assert cal["status"] == "ok"
    assert err_client.is_circuit_open is True


# ---------------------------------------------------------------------------
# 5. [High #5] Expired preview_url re-resolves once (use_cache=False) & succeeds with PCM MIR
# ---------------------------------------------------------------------------
def test_expired_preview_url_refreshes_once_and_analyzes_pcm(tmp_path: Path) -> None:
    db = RadioDatabase(tmp_path / "radio.db")
    wav_bytes = _synth_30s_click_wav(bpm=134.0, duration_sec=6.0)
    calls = {"search": 0, "expired_preview": 0, "fresh_preview": 0}

    # Pre-seed SQLite cache with an EXPIRED preview_url and metadata-only analysis_source
    db.upsert_audio_feature_cache(
        canonical_key="fresh artist::fresh song",
        video_id="VID_EXP_001",
        deezer_id=77701,
        artist="Fresh Artist",
        title="Fresh Song",
        bpm=0.0,
        gain=-8.5,
        rank=300_000,
        duration=180,
        preview_url="https://cdns-preview.dzcdn.net/expired.mp3",
        mir_a1_c4=None,
        analysis_source="deezer_metadata",
    )

    def _transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/search":
            calls["search"] += 1
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": 77701,
                            "title": "Fresh Song",
                            "artist": {"name": "Fresh Artist"},
                            "bpm": 134.0,
                            "gain": -8.5,
                            "rank": 300_000,
                            "duration": 180,
                            "preview": "https://cdns-preview.dzcdn.net/fresh.wav",
                        }
                    ]
                },
            )
        if request.url.path == "/expired.mp3":
            calls["expired_preview"] += 1
            return httpx.Response(403)
        if request.url.path == "/fresh.wav":
            calls["fresh_preview"] += 1
            return httpx.Response(200, content=wav_bytes)
        return httpx.Response(404)

    deezer = DeezerClient(
        db=db,
        min_interval_sec=0.0,
        http_client=httpx.Client(transport=httpx.MockTransport(_transport)),
    )
    # Pass non-empty ffmpeg_bin sentinel so AudioMIRExtractor enters preview branch;
    # RIFF WAV bytes are decoded natively by decode_audio_to_mono_pcm!
    extractor = AudioMIRExtractor(db=db, deezer_client=deezer, ffmpeg_bin="ffmpeg_available")
    vec12, est_bpm, _meta, src = extractor.extract_a1_c4_for_track(
        "Fresh Artist",
        "Fresh Song",
        video_id="VID_EXP_001",
        allow_network=True,
    )
    assert src == "mir_pcm"
    assert vec12.shape == (12,)
    assert est_bpm == 134.0
    assert calls["expired_preview"] == 1
    assert calls["search"] == 1
    assert calls["fresh_preview"] == 1

    # Negative cache: unindexed song returns None and caches analysis_source='not_found' (0 repeat searches)
    def _empty_transport(request: httpx.Request) -> httpx.Response:
        calls["empty"] = calls.get("empty", 0) + 1
        return httpx.Response(200, json={"data": []})

    neg_client = DeezerClient(
        db=db,
        min_interval_sec=0.0,
        http_client=httpx.Client(transport=httpx.MockTransport(_empty_transport)),
    )
    assert neg_client.resolve_track("NoSuchArtistXYZ", "NoSuchSongXYZ") is None
    first_calls = calls["empty"]
    assert neg_client.resolve_track("NoSuchArtistXYZ", "NoSuchSongXYZ") is None
    assert calls["empty"] == first_calls, "Negative cache must block repeat Deezer /search calls"


# ---------------------------------------------------------------------------
# 6. [High #6 & Medium #9] No same-artist self-co-occurrence, OFFICIAL_SOURCE_MUSIC,
#    isAvailable=False, videoType=None verification, & reset_unrated_tracks video_id protection
# ---------------------------------------------------------------------------
def test_no_self_cooccurrence_official_source_music_and_reset_video_id_protection(tmp_path: Path) -> None:
    assert (
        is_slop_or_unofficial(
            {"title": "Official Audio", "videoType": "MUSIC_VIDEO_TYPE_OFFICIAL_SOURCE_MUSIC", "resultType": "song"}
        )
        is False
    )
    assert is_slop_or_unofficial({"title": "Unavailable", "videoType": "MUSIC_VIDEO_TYPE_ATV", "isAvailable": False}) is True
    assert is_slop_or_unofficial({"title": "Unverified Bare Track", "artists": [{"name": "Random"}]}) is True
    assert (
        is_slop_or_unofficial(
            {"title": "Verified Album Track", "artists": [{"name": "Real", "id": "UC123"}], "album": {"name": "LP"}}
        )
        is False
    )

    artist_calls = {"count": 0}

    class SameArtistYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8):
            if "edamame" in query.lower():
                return [
                    {
                        "videoId": "SEED_BBNO_A",
                        "title": "edamame",
                        "artists": [{"name": "bbno$", "id": "ART_BBNO"}],
                        "resultType": "song",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "duration_seconds": 135,
                    }
                ]
            return [
                {
                    "videoId": "SEED_BBNO_B",
                    "title": "bag or die",
                    "artists": [{"name": "bbno$", "id": "ART_BBNO"}],
                    "resultType": "song",
                    "videoType": "MUSIC_VIDEO_TYPE_OFFICIAL_SOURCE_MUSIC",
                    "duration_seconds": 140,
                }
            ]

        def get_watch_playlist(self, videoId: str, radio: bool = False, limit: int = 25):
            return {
                "tracks": [
                    {
                        "videoId": "VID_SHARED_SAME_SEED",
                        "title": "Same Seed Neighbor",
                        "artists": [{"name": "Yung Gravy", "id": "ART_YG"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "length": "2:50",
                    }
                ]
            }

        def get_artist(self, browseId: str):
            artist_calls["count"] += 1
            return {"related": {"results": []}}

    db = RadioDatabase(tmp_path / "radio.db")
    digger = CandidatePoolDigger(db=db, yt_adapter=YouTubeMusicAdapter(client=SameArtistYT()))
    station = StationConfig(
        station_id="us_hiphop_bouncy",
        name="US Hip-Hop",
        seed_tags=["hip hop"],
        seed_artists=["bbno$"],
        spotify_anchor_seeds=["bbno$ - edamame", "bbno$ - bag or die"],
        max_popularity_views=500_000_000,
    )
    digger.dig_station(station, target_pool_size=10)

    # Same artist_browse_id queried only ONCE, and candidate from 2 bbno$ songs has cooccurrence_count == 1!
    assert artist_calls["count"] == 1
    tr = db.get_track(video_id="VID_SHARED_SAME_SEED")
    assert tr is not None
    assert tr.cooccurrence_count == 1
    assert "×" not in (tr.discovery_route or "")

    # [Medium #9] reset_unrated_tracks must protect station_candidates link when playlist_tracks matches via video_id only (track_id=NULL)
    db.add_track_to_playlist(
        "pl_favorites",
        video_id="VID_SHARED_SAME_SEED",
        track_id=None,
        artist="Yung Gravy",
        title="Same Seed Neighbor",
    )
    db.reset_unrated_tracks()
    eligible = db.get_eligible_candidates("us_hiphop_bouncy", 500_000_000, include_heard_fallback=True)
    assert any(c.video_id == "VID_SHARED_SAME_SEED" for c in eligible)


# ---------------------------------------------------------------------------
# 7. [Medium #7 & #8] Node.js real execution: handleManualSkip double-click / station-switch & implicit guards
# ---------------------------------------------------------------------------
def test_web_ui_manual_skip_race_and_implicit_feedback_guards_node() -> None:
    html = (PROJECT_ROOT / "web/index.html").read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, flags=re.DOTALL)
    inline_js = "\n".join(s for s in scripts if s.strip())

    harness = r"""
const assert = require("assert");
const vm = require("vm");
const fs = require("fs");
const inlineCode = fs.readFileSync(process.argv[1], "utf8");

function makeClassList() {
  const s = new Set();
  return {
    add: (...c) => c.forEach((x) => s.add(x)),
    remove: (...c) => c.forEach((x) => s.delete(x)),
    toggle: (c, f) => (f === undefined ? (s.has(c) ? s.delete(c) : s.add(c)) : (f ? s.add(c) : s.delete(c))),
    contains: (c) => s.has(c),
  };
}
function makeEl(id) {
  const listeners = {};
  return {
    id, textContent: "", innerHTML: "", value: "pl_favorites", className: "", children: [],
    classList: makeClassList(),
    appendChild(c) { this.children.push(c); return c; },
    addEventListener(t, fn) { (listeners[t] = listeners[t] || []).push(fn); },
  };
}
const els = {};
const feedbackPosts = [];
let feedbackResolver = null;
let fetchHandler = async (url, opts) => {
  if (url === "/feedback") {
    const body = JSON.parse(opts.body);
    feedbackPosts.push(body);
    if (feedbackResolver) {
      return new Promise((resolve) => {
        const fn = feedbackResolver;
        feedbackResolver = null;
        fn(() => resolve({ ok: true, status: 200, json: async () => ({ status: "ok" }) }));
      });
    }
    return { ok: true, status: 200, json: async () => ({ status: "ok" }) };
  }
  if (url.includes("/api/stations/st_b/batch")) {
    return { ok: true, status: 200, json: async () => ({ items: [{ track_id: "b1", video_id: "VID_B1", artist: "B", title: "B1", duration_seconds: 180 }] }) };
  }
  return { ok: true, status: 200, json: async () => ({ stations: [], playlists: [], items: [], axes: [] }) };
};

const YTStub = {
  PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 },
  Player: function (_id, config) {
    this._config = config;
    this._state = 1;
    this._time = 0;
    this._dur = 180;
    this.loadVideoById = () => {};
    this.playVideo = () => {};
    this.pauseVideo = () => {};
    this.getPlayerState = () => this._state;
    this.getCurrentTime = () => this._time;
    this.getDuration = () => this._dur;
    this.isMuted = () => false;
  },
};

const sandbox = {
  console,
  document: { hidden: false, getElementById: (id) => (els[id] = els[id] || makeEl(id)), createElement: makeEl },
  navigator: { mediaSession: { metadata: null, playbackState: "none", setActionHandler: () => {} } },
  MediaMetadata: function (m) { Object.assign(this, m); },
  YT: YTStub,
  fetch: (u, o) => fetchHandler(u, o),
  setTimeout: () => 1,
  clearTimeout: () => {},
  setInterval: () => 1,
  clearInterval: () => {},
  encodeURIComponent, Promise, Set, Array, Object, Number, String, Boolean, Math, JSON, Error, Date,
};
sandbox.window = sandbox;
const ctx = vm.createContext(sandbox);
vm.runInContext(
  inlineCode + "\nwindow.__exp = { state, handleManualSkip, sendImplicitPlaybackFeedback, switchStation, playTrack };",
  ctx
);

(async () => {
  await new Promise((r) => setImmediate(r));
  const api = sandbox.__exp;
  sandbox.onYouTubeIframeAPIReady();
  api.state.player._config.events.onReady();
  api.state.stations = [
    { station_id: "st_a", name: "A", candidate_count: 5 },
    { station_id: "st_b", name: "B", candidate_count: 5 },
  ];
  api.state.activeStationId = "st_a";

  // 1. Unconfirmed playback -> neither ENDED nor manual skip may emit COMPLETE / EARLY_SKIP
  feedbackPosts.length = 0;
  api.state.currentTrack = { track_id: "t1", video_id: "VID_UNCONF", artist: "A", title: "T1", duration_seconds: 180 };
  api.state.trackConfirmedPlayingVideoId = null;
  api.state.player._time = 5.0;
  await api.sendImplicitPlaybackFeedback(true);
  await api.sendImplicitPlaybackFeedback(false);
  assert.strictEqual(feedbackPosts.length, 0, "Unconfirmed playback must emit zero implicit feedback");

  // 2. Confirmed playback at 0.05s (buffering) -> manual skip must NOT emit EARLY_SKIP (< 1.5s)
  api.state.trackConfirmedPlayingVideoId = "VID_UNCONF";
  api.state.player._time = 0.05;
  await api.sendImplicitPlaybackFeedback(true);
  assert.strictEqual(feedbackPosts.length, 0, "Buffering skip at 0.05s must not emit EARLY_SKIP");

  // 3. Confirmed playback at 5.0s -> rapid double-click on handleManualSkip only advances ONCE
  api.state.currentTrack = { track_id: "t_s1", video_id: "VID_SKIP_1", artist: "A", title: "S1", duration_seconds: 180 };
  api.state.trackConfirmedPlayingVideoId = "VID_SKIP_1";
  api.state.implicitFeedbackSentVideoId = null;
  api.state.queue = [
    { track_id: "t_s2", video_id: "VID_SKIP_2", artist: "A", title: "S2", duration_seconds: 180 },
    { track_id: "t_s3", video_id: "VID_SKIP_3", artist: "A", title: "S3", duration_seconds: 180 },
  ];
  api.state.player._time = 5.0;
  let releaseFb = null;
  feedbackResolver = (done) => { releaseFb = done; };
  const p1 = api.handleManualSkip();
  const p2 = api.handleManualSkip(); // rapid double-click while p1 awaits /feedback!
  await p2;
  releaseFb();
  await p1;
  assert.strictEqual(feedbackPosts.length, 1);
  assert.strictEqual(feedbackPosts[0].event_type, "EARLY_SKIP");
  assert.strictEqual(api.state.currentTrack.video_id, "VID_SKIP_2", "Double-click must NOT skip 2 tracks");

  // 4. Station switch mid-skip must NOT hijack the newly switched station
  api.state.currentTrack = { track_id: "t_s2", video_id: "VID_SKIP_2", artist: "A", title: "S2", duration_seconds: 180 };
  api.state.trackConfirmedPlayingVideoId = "VID_SKIP_2";
  api.state.implicitFeedbackSentVideoId = null;
  api.state.player._time = 6.0;
  let releaseFb2 = null;
  feedbackResolver = (done) => { releaseFb2 = done; };
  const skipPromise = api.handleManualSkip();
  await api.switchStation("st_b");
  assert.strictEqual(api.state.activeStationId, "st_b");
  assert.strictEqual(api.state.currentTrack.video_id, "VID_B1");
  releaseFb2();
  await skipPromise;
  assert.strictEqual(api.state.currentTrack.video_id, "VID_B1", "Stale manual skip must not override station B");

  console.log("ALL_MANUAL_SKIP_AND_IMPLICIT_GUARDS_PASSED");
})().catch((err) => { console.error(err); process.exit(1); });
"""
    js_file = PROJECT_ROOT / ".tmp_web_skip_test.js"
    try:
        js_file.write_text(inline_js, encoding="utf-8")
        res = subprocess.run(
            ["node", "-e", harness, str(js_file)],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert res.returncode == 0, f"Node test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        assert "ALL_MANUAL_SKIP_AND_IMPLICIT_GUARDS_PASSED" in res.stdout
    finally:
        js_file.unlink(missing_ok=True)
