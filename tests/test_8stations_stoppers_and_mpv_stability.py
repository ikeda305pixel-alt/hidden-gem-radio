"""Regression tests for 8-station EDM/UKG split, seed-bias stoppers, amateur cover filter, unified stop button, and mpv 4G/freeze stability."""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.adapters.audio_stream import AudioStreamResolver, CachedAudioStream
from src.adapters.mpv_player import TERMUX_AO_PRIORITY, MpvLocalPlayer
from src.adapters.youtube import YouTubeMusicAdapter, is_slop_or_unofficial
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
    load_known_familiar_artists,
    normalize_artist,
    split_artist_credits,
)
from src.recommender.bandit import ContextualBanditRecommender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"


# ---------------------------------------------------------------------------
# 1. 8-station EDM vs UKG split & routing
# ---------------------------------------------------------------------------
def test_8_stations_split_edm_and_ukg_cleanly(tmp_path: Path) -> None:
    stations = default_theory_stations()
    assert len(stations) == 8
    by_id = {s.station_id: s for s in stations}
    assert "edm_dance_party" in by_id
    assert "ukg_bass_bounce" in by_id

    edm = by_id["edm_dance_party"]
    ukg = by_id["ukg_bass_bounce"]

    # EDM station must be pure 4-on-the-floor festival EDM with zero UKG contamination
    assert edm.name == "ブチ上げ 4つ打ち"
    assert edm.region_policy == "western_only"
    edm_artists_low = {a.lower() for a in edm.seed_artists}
    assert not (edm_artists_low & {"notion", "chrystal", "sammy virji", "interplanetary criminal", "oppidan", "conducta", "mph"})
    assert not any("uk garage" in t or "2-step" in t or "ukg" in t for t in edm.seed_tags)

    # UKG station must have diverse seeds and high syncopation/bounce T2
    assert ukg.name == "跳ねUKG ＆ ベース"
    assert ukg.region_policy == "western_only"
    assert len(ukg.seed_artists) >= 8
    assert ukg.target_theory_vector["T2"] > edm.target_theory_vector["T2"]

    # Routing checks
    assert classify_entry_to_station("Chrystal & NOTION", "The Days (NOTION Remix)") == "ukg_bass_bounce"
    assert classify_entry_to_station("Sammy Virji", "If U Need It") == "ukg_bass_bounce"
    assert classify_entry_to_station("Interplanetary Criminal", "B.O.T.A.") == "ukg_bass_bounce"
    assert classify_entry_to_station("Oppidan", "Mr. Sandman") == "ukg_bass_bounce"
    assert classify_entry_to_station("Conducta", "Whippet") == "ukg_bass_bounce"
    assert classify_entry_to_station("MPH", "Run It") == "ukg_bass_bounce"
    assert classify_entry_to_station("Unknown Producer", "Midnight Speed Garage 2-Step") == "ukg_bass_bounce"

    assert classify_entry_to_station("Avicii", "Wake Me Up") == "edm_dance_party"
    assert classify_entry_to_station("Calvin Harris", "Feel So Close") == "edm_dance_party"
    assert classify_entry_to_station("Tiesto", "Red Lights") == "edm_dance_party"

    # Default generate_stations_yaml produces 8 stations
    out_yaml = tmp_path / "stations.yaml"
    summary = generate_stations_yaml(output_path=out_yaml, offline=True)
    assert summary["station_count"] == 8
    assert "ukg_bass_bounce" in summary["station_ids"]
    assert "edm_dance_party" in summary["station_ids"]


# ---------------------------------------------------------------------------
# 2. Combined-credit splitting & self-exclusion
# ---------------------------------------------------------------------------
def test_combined_credit_splitting_and_seed_self_exclusion(tmp_path: Path) -> None:
    assert split_artist_credits("Chrystal & NOTION") == ["chrystal", "notion"]
    assert split_artist_credits("Artist A feat. Artist B") == ["artist a", "artist b"]
    assert split_artist_credits("Artist A x Artist B") == ["artist a", "artist b"]
    assert split_artist_credits("Artist A × Artist B") == ["artist a", "artist b"]
    assert split_artist_credits("Artist A, Artist B ft. Artist C") == ["artist a", "artist b", "artist c"]

    # load_known_familiar_artists expands combined credits
    st_yaml = tmp_path / "stations.yaml"
    st_yaml.write_text(
        "history_index:\n  known_familiar_artists:\n    - 'Chrystal & NOTION'\nstations: []\n",
        encoding="utf-8",
    )
    familiar = load_known_familiar_artists(st_yaml)
    assert "chrystal" in familiar
    assert "notion" in familiar

    # dig_station excludes standalone NOTION when "Chrystal & NOTION" is a seed or familiar artist
    class FakeYT:
        def search(self, query: str, filter: str = "songs", limit: int = 8):
            _ = (filter, limit)
            if "chrystal" in query.lower():
                return [
                    {
                        "videoId": "SEED_UKG_01",
                        "title": "The Days (NOTION Remix)",
                        "artists": [{"name": "Chrystal & NOTION", "id": "UC_CN"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "resultType": "song",
                        "duration": "3:10",
                        "views": "5M",
                    }
                ]
            return []

        def get_watch_playlist(self, videoId: str, radio: bool = True, limit: int = 25):
            _ = (videoId, radio, limit)
            return {
                "tracks": [
                    {
                        "videoId": "VID_NOTION_SOLO",
                        "title": "Solo Track",
                        "artists": [{"name": "NOTION", "id": "UC_NOTION"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "length": "3:05",
                        "views": "200K",
                    },
                    {
                        "videoId": "VID_FRESH_UKG",
                        "title": "Fresh Garage Cut",
                        "artists": [{"name": "Higgo", "id": "UC_HIGGO"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "length": "3:15",
                        "views": "150K",
                    },
                ]
            }

        def get_artist(self, browseId: str):
            _ = browseId
            return {}

    db = RadioDatabase(tmp_path / "radio.db")
    digger = CandidatePoolDigger(db=db, yt_adapter=YouTubeMusicAdapter(client=FakeYT()))
    st = StationConfig(
        station_id="ukg_bass_bounce",
        name="跳ねUKG ＆ ベース",
        seed_tags=["uk garage"],
        seed_artists=["Chrystal & NOTION"],
        spotify_anchor_seeds=["Chrystal & NOTION - The Days (NOTION Remix)"],
        region_policy="western_only",
    )
    res = digger.dig_station(st, target_pool_size=10)
    eligible = db.get_eligible_candidates("ukg_bass_bounce", 500_000_000)
    vids = {t.video_id for t in eligible}
    assert "VID_NOTION_SOLO" not in vids
    assert "VID_FRESH_UKG" in vids
    assert res["skipped_familiar_artist"] >= 1


# ---------------------------------------------------------------------------
# 3. Single-seed 25% pool cap & round-robin distribution
# ---------------------------------------------------------------------------
def test_per_seed_25pct_pool_cap_and_round_robin_collection(tmp_path: Path) -> None:
    seed_names = ["SeedAlpha", "SeedBeta", "SeedGamma", "SeedDelta"]

    class FakeYTMultiSeed:
        def search(self, query: str, filter: str = "songs", limit: int = 8):
            _ = (filter, limit)
            for idx, name in enumerate(seed_names):
                if name.lower() in query.lower():
                    return [
                        {
                            "videoId": f"SEED_V{idx}",
                            "title": f"Anchor {idx}",
                            "artists": [{"name": name, "id": f"UC_S{idx}"}],
                            "videoType": "MUSIC_VIDEO_TYPE_ATV",
                            "resultType": "song",
                            "duration": "3:00",
                            "views": "1M",
                        }
                    ]
            return []

        def get_watch_playlist(self, videoId: str, radio: bool = True, limit: int = 25):
            _ = (radio, limit)
            if videoId == "SEED_V0":
                # SeedAlpha floods with 16 high-view tracks
                return {
                    "tracks": [
                        {
                            "videoId": f"V_ALPHA_{i:02d}",
                            "title": f"Alpha Gem {i}",
                            "artists": [{"name": f"AlphaArtist {i}", "id": f"UC_A{i}"}],
                            "videoType": "MUSIC_VIDEO_TYPE_ATV",
                            "length": "3:00",
                            "views": f"{20 - i}M",
                        }
                        for i in range(16)
                    ]
                }
            idx = int(videoId[-1])
            return {
                "tracks": [
                    {
                        "videoId": f"V_S{idx}_{i:02d}",
                        "title": f"Seed{idx} Gem {i}",
                        "artists": [{"name": f"S{idx}Artist {i}", "id": f"UC_S{idx}_{i}"}],
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "length": "3:00",
                        "views": "500K",
                    }
                    for i in range(2)
                ]
            }

        def get_artist(self, browseId: str):
            _ = browseId
            return {}

    db = RadioDatabase(tmp_path / "radio.db")
    digger = CandidatePoolDigger(db=db, yt_adapter=YouTubeMusicAdapter(client=FakeYTMultiSeed()))
    st = StationConfig(
        station_id="ukg_bass_bounce",
        name="跳ねUKG ＆ ベース",
        seed_tags=["uk garage"],
        seed_artists=seed_names,
        spotify_anchor_seeds=[f"{name} - Anchor" for name in seed_names],
        region_policy="western_only",
    )
    res = digger.dig_station(st, target_pool_size=20)
    # SeedAlpha capped at 25% of 20 = 5 tracks (11 excess skipped); other 3 seeds contribute 2 each -> 11 total
    assert res["added"] == 11
    assert res["skipped_seed_cap"] == 11

    eligible = db.get_eligible_candidates("ukg_bass_bounce", 500_000_000)
    per_seed: dict[str, int] = {}
    for tr in eligible:
        s = normalize_artist(tr.seed_artist or "")
        per_seed[s] = per_seed.get(s, 0) + 1

    # Max 25% of target_pool_size=20 -> 5 tracks per seed, and all 4 seeds represented
    assert len(per_seed) == 4
    assert per_seed["seedalpha"] == 5
    assert all(cnt <= 5 for cnt in per_seed.values())


# ---------------------------------------------------------------------------
# 4. Stage 3 consecutive-seed penalty & 30-40% forced Explore slots
# ---------------------------------------------------------------------------
def test_stage3_consecutive_seed_penalty_and_forced_explore_ratio(tmp_path: Path) -> None:
    st = StationConfig(
        station_id="ukg_bass_bounce",
        name="跳ねUKG ＆ ベース",
        seed_tags=["uk garage", "2-step"],
        seed_artists=["NOTION", "Sammy Virji", "Conducta", "Oppidan"],
    )
    bandit = ContextualBanditRecommender(
        station_id="ukg_bass_bounce",
        seed_tags=st.seed_tags,
        model_path=tmp_path / "bandit_ukg.pkl",
        station_config=st,
    )
    # Train once so exploit vs explore mode distinction is active
    train_tr = TrackEntity.build(
        artist="Warmup Artist",
        title="Warmup",
        tags=["uk garage", "2-step"],
        view_count=500_000,
        video_id="VID_WARMUP",
        seed_artist="Conducta",
    )
    bandit.update(train_tr, event_type="GOOD")

    # Build 12 candidates across 4 seeds (3 from NOTION, 3 from Sammy Virji, 3 from Conducta, 3 from Oppidan)
    candidates: list[TrackEntity] = []
    for seed_idx, seed_name in enumerate(["NOTION", "Sammy Virji", "Conducta", "Oppidan"]):
        for j in range(3):
            candidates.append(
                TrackEntity.build(
                    artist=f"UKG Artist {seed_idx}_{j}",
                    title=f"Track {seed_idx}_{j}",
                    tags=["uk garage", "2-step"],
                    view_count=1_000_000,
                    video_id=f"VID_CAND_{seed_idx}_{j}",
                    seed_artist=seed_name,
                    cooccurrence_count=2 if seed_name != "NOTION" and j == 0 else 1,
                )
            )

    # Recent 3 played tracks were all from NOTION -> NOTION-seeded candidates must be penalized
    ranked = bandit.rank_candidates(
        candidates,
        seen_artists=set(),
        limit=10,
        recent_seed_artists=["NOTION", "NOTION", "NOTION"],
    )
    assert len(ranked) == 10
    assert normalize_artist(ranked[0].track.seed_artist or "") != "notion"
    assert normalize_artist(ranked[1].track.seed_artist or "") != "notion"

    # At least 30% of batch slots are forced "探索"
    explore_slots = sum(1 for item in ranked if item.mode == "探索")
    assert explore_slots >= 3  # 30-40% of 10

    # Batch of 3 has at least 1 forced Explore slot (33.3%)
    ranked_3 = bandit.rank_candidates(
        candidates,
        seen_artists=set(),
        limit=3,
        recent_seed_artists=["NOTION", "NOTION"],
    )
    assert sum(1 for item in ranked_3 if item.mode == "探索") >= 1


# ---------------------------------------------------------------------------
# 5. Strengthened amateur cover / 歌ってみた filter & [公式MV代替] verification
# ---------------------------------------------------------------------------
def test_amateur_cover_nfkc_filter_and_strict_omv_fallback() -> None:
    base_song = {
        "videoType": "MUSIC_VIDEO_TYPE_ATV",
        "artists": [{"name": "Sammy Virji", "id": "UC_SV"}],
        "album": {"name": "Album"},
    }
    # NFKC full-width and JP/EN amateur cover/piano/acoustic/karaoke/live/instrumental variants rejected
    rejected_titles = [
        "The Days (歌ってみた)",
        "The Days 弾いてみた",
        "The Days (ＣＯＶＥＲ)",
        "The Days (ＰＩＡＮＯ Ver.)",
        "The Days (Acoustic Session)",
        "The Days (Karaoke / カラオケ)",
        "The Days (Live at Club)",
        "The Days (Instrumental)",
        "The Days (ピアノアレンジ)",
        "The Days (インスト)",
    ]
    for title in rejected_titles:
        assert is_slop_or_unofficial({**base_song, "title": title}) is True, f"Should reject: {title}"

    # UGC amateur remix rejected
    ugc_remix = {
        "videoType": "MUSIC_VIDEO_TYPE_UGC",
        "title": "The Days (Bedroom Remix)",
        "artists": [{"name": "RandomDJ", "id": "UC_RND"}],
    }
    assert is_slop_or_unofficial(ugc_remix, allow_verified_watch_ugc=True) is True

    # Legitimate official ATV remix passes
    official_remix = {
        **base_song,
        "title": "The Days (NOTION Remix)",
    }
    assert is_slop_or_unofficial(official_remix) is False

    # [公式MV代替] find_embeddable_alternative rejects amateur personal channel & low-view uploads,
    # and accepts verified official artist/VEVO/Topic channel
    class FakeYTAlt:
        def search(self, query: str, filter: str = "videos", limit: int = 8):
            _ = (query, filter, limit)
            return [
                # Amateur personal uploader with artist name in title -> MUST REJECT
                {
                    "videoId": "VID_AMATEUR_1",
                    "title": "Avicii - Levels (Official Video)",
                    "artists": [{"name": "Taro Guitar Room"}],
                    "videoType": "MUSIC_VIDEO_TYPE_OMV",
                    "duration": "3:18",
                    "views": "850",
                },
                # Amateur piano cover on OMV -> MUST REJECT
                {
                    "videoId": "VID_AMATEUR_2",
                    "title": "Avicii - Levels (ＰＩＡＮＯ Cover)",
                    "artists": [{"name": "Avicii"}],
                    "videoType": "MUSIC_VIDEO_TYPE_OMV",
                    "duration": "3:18",
                    "views": "50K",
                },
                # Genuine official VEVO channel -> MUST ACCEPT
                {
                    "videoId": "VID_OFFICIAL_OK",
                    "title": "Avicii - Levels",
                    "artists": [{"name": "AviciiOfficialVEVO"}],
                    "videoType": "MUSIC_VIDEO_TYPE_OMV",
                    "duration": "3:18",
                    "views": "500M",
                },
            ]

    def ok_handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(200, json={"title": "OK"})

    http_client = httpx.Client(transport=httpx.MockTransport(ok_handler))
    adapter = YouTubeMusicAdapter(client=FakeYTAlt(), http_client=http_client, check_embeddable=True)
    try:
        alt = adapter.find_embeddable_alternative("Avicii", "Levels")
        assert alt is not None
        alt_track, alt_kind = alt
        assert alt_kind == "omv"
        assert alt_track.video_id == "VID_OFFICIAL_OK"
    finally:
        adapter.close()


# ---------------------------------------------------------------------------
# 6. mpv 4G auto-reconnect flags & 12s silence-freeze recovery
# ---------------------------------------------------------------------------
class _MiniMockMpvIpc:
    def __init__(self) -> None:
        self.client_sock, self.server_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client_sock.settimeout(0.2)
        self.server_sock.settimeout(0.2)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.commands: list[list[Any]] = []
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.server_sock.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                msg = json.loads(line.decode("utf-8"))
                cmd = msg.get("command") or []
                with self._lock:
                    self.commands.append(cmd)
                req_id = msg.get("request_id")
                if req_id is not None:
                    reply = (json.dumps({"request_id": req_id, "error": "success", "data": None}) + "\n").encode("utf-8")
                    try:
                        self.server_sock.sendall(reply)
                    except OSError:
                        return

    def wait_for_loadfiles(self, count: int, timeout: float = 3.0) -> list[list[Any]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                loads = [c for c in self.commands if c and c[0] == "loadfile"]
            if len(loads) >= count:
                return loads
            time.sleep(0.02)
        with self._lock:
            return [c for c in self.commands if c and c[0] == "loadfile"]

    def close(self) -> None:
        self._stop.set()
        for s in (self.client_sock, self.server_sock):
            try:
                s.close()
            except OSError:
                pass


def test_mpv_4g_reconnect_flags_and_silence_freeze_recovery(tmp_path: Path) -> None:
    assert TERMUX_AO_PRIORITY == "audiotrack,opensles,aaudio,pulse"

    mock_ipc = _MiniMockMpvIpc()
    resolve_counts: dict[str, int] = {}

    def fake_extract(vid: str) -> CachedAudioStream:
        resolve_counts[vid] = resolve_counts.get(vid, 0) + 1
        return CachedAudioStream(
            video_id=vid,
            url=f"https://audio.example.test/{vid}_try{resolve_counts[vid]}.m4a",
            mime_type="audio/mp4",
        )

    tracks = [
        {"track_id": "t1", "video_id": "FREEZE_VID_1", "artist": "Artist 1", "title": "Song 1", "duration_seconds": 180},
        {"track_id": "t2", "video_id": "FREEZE_VID_2", "artist": "Artist 2", "title": "Song 2", "duration_seconds": 180},
    ]

    def next_provider(station_id: str, count: int, exclude: set[str]) -> list[dict[str, Any]]:
        _ = (station_id, count)
        return [t for t in tracks if t["video_id"] not in exclude and t["track_id"] not in exclude]

    resolver = AudioStreamResolver(extractor=fake_extract)
    mpv = MpvLocalPlayer(
        audio_resolver=resolver,
        next_tracks_provider=next_provider,
        socket_path=tmp_path / "mpv_freeze.sock",
        ipc_connector=lambda _p: mock_ipc.client_sock,
        spawn_process=False,
        stall_timeout_sec=0.12,
        stall_poll_interval_sec=0.03,
    )

    try:
        cmd = mpv._build_mpv_spawn_cmd()
        assert "--ao=audiotrack,opensles,aaudio,pulse" in cmd
        assert "--cache=yes" in cmd
        assert "--demuxer-max-bytes=16MiB" in cmd
        assert "--demuxer-readahead-secs=20" in cmd
        assert "--stream-lavf-o=reconnect=1,reconnect_streamed=1,reconnect_delay_max=5" in cmd
        assert "--audio-stream-silence=no" in cmd

        # Start playing FREEZE_VID_1 with NO time-pos updates -> simulates silent freeze
        mpv.play("ukg_bass_bounce", track=tracks[0])

        # 1st freeze -> 1x invalidate + force_refresh reload of FREEZE_VID_1 (_try2)
        loads2 = mock_ipc.wait_for_loadfiles(2, timeout=2.0)
        assert len(loads2) >= 2
        assert "FREEZE_VID_1_try1.m4a" in loads2[0][1]
        assert "FREEZE_VID_1_try2.m4a" in loads2[1][1]

        # 2nd freeze on same track -> automatically skips to FREEZE_VID_2
        loads3 = mock_ipc.wait_for_loadfiles(3, timeout=2.0)
        assert len(loads3) >= 3
        assert "FREEZE_VID_2_try1.m4a" in loads3[2][1]
    finally:
        mpv.shutdown()
        mock_ipc.close()


# ---------------------------------------------------------------------------
# 7. Single unified Play/Stop toggle button in web/index.html (HTML/CSS + Node vm)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(shutil.which("node") is None, reason="node is required for JS vm tests")
def test_web_ui_single_unified_play_stop_button_no_wrap_node() -> None:
    html = WEB_INDEX_PATH.read_text(encoding="utf-8")
    assert 'id="btnPlayMute"' in html
    assert 'id="btnStop"' not in html
    assert "white-space: nowrap" in html

    scripts = re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, flags=re.DOTALL | re.IGNORECASE)
    inline_js = "\n".join(scripts)

    harness = r"""
const vm = require("vm");

function makeElement(id) {
  const listeners = {};
  return {
    id,
    textContent: "",
    innerHTML: "",
    value: "0",
    src: "",
    paused: false,
    currentTime: 0,
    duration: 180,
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
        } else if (force) this._set.add(c);
        else this._set.delete(c);
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
        await fn({ target: this, type: evt });
      }
    },
    play() { this.paused = false; return Promise.resolve(); },
    pause() { this.paused = true; },
  };
}

async function runAll() {
  const elements = {};
  const getEl = (id) => {
    if (!elements[id]) elements[id] = makeElement(id);
    return elements[id];
  };

  let stopCalls = 0;
  let playCalls = 0;
  let serverState = {
    mpv_available: true,
    active: true,
    stopped: false,
    paused: false,
    station_id: "ukg_bass_bounce",
    position_sec: 12.0,
    duration_sec: 180.0,
    track: { track_id: "t1", video_id: "VID_U1", artist: "Sammy Virji", title: "If U Need It", duration_seconds: 180 },
    queue_preview: [],
  };

  const sandbox = {
    console, Date, Math, Number, String, Boolean, Array, Object, Set, Map, Promise, Error,
    encodeURIComponent, setTimeout, clearTimeout,
    setInterval: () => 1,
    clearInterval: () => {},
    AbortController: typeof AbortController !== "undefined" ? AbortController : undefined,
    document: { hidden: false, getElementById: getEl, createElement: (t) => makeElement(t), addEventListener: () => {} },
    window: { location: { origin: "http://127.0.0.1:8000" }, addEventListener: () => {} },
    navigator: {},
    MediaMetadata: function (o) { return o; },
    YT: { PlayerState: { UNSTARTED: -1, ENDED: 0, PLAYING: 1, PAUSED: 2, BUFFERING: 3, CUED: 5 } },
    fetch: async (url) => {
      if (url === "/health") {
        return { ok: true, status: 200, json: async () => ({ status: "ok", mpv_available: true, audio_stream_available: true }) };
      }
      if (url === "/stations") {
        return { ok: true, status: 200, json: async () => ({ stations: [{ station_id: "ukg_bass_bounce", name: "跳ねUKG ＆ ベース", vibe: "2ステップ", candidate_count: 10 }] }) };
      }
      if (url === "/api/local-player/state") {
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      if (url === "/api/local-player/stop") {
        stopCalls += 1;
        serverState = { ...serverState, active: false, stopped: true, paused: false };
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      if (url === "/api/local-player/play") {
        playCalls += 1;
        serverState = { ...serverState, active: true, stopped: false, paused: false };
        return { ok: true, status: 200, json: async () => ({ ...serverState }) };
      }
      return { ok: true, status: 200, json: async () => ({ playlists: [], updates: 0, layers: [], axes: [] }) };
    },
  };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(INLINE_JS + "\n;globalThis.__exports = { state, el };", sandbox);
  await new Promise((r) => setTimeout(r, 30));

  const { state, el } = sandbox.__exports;
  if (el.btnStop !== undefined) {
    throw new Error("Expected no separate el.btnStop element");
  }
  if (el.btnPlayMute.textContent !== "⏹ 停止") {
    throw new Error(`Expected '⏹ 停止' while playing, got '${el.btnPlayMute.textContent}'`);
  }

  // Click unified button while playing -> calls /api/local-player/stop -> '▶ 再生'
  await el.btnPlayMute.trigger("click");
  if (stopCalls !== 1 || state.userStopped !== true || el.btnPlayMute.textContent !== "▶ 再生") {
    throw new Error(`Stop toggle failed: stopCalls=${stopCalls}, userStopped=${state.userStopped}, label='${el.btnPlayMute.textContent}'`);
  }

  // Click unified button while stopped -> calls /api/local-player/play -> '⏹ 停止'
  await el.btnPlayMute.trigger("click");
  if (playCalls !== 1 || state.userStopped !== false || el.btnPlayMute.textContent !== "⏹ 停止") {
    throw new Error(`Play toggle failed: playCalls=${playCalls}, userStopped=${state.userStopped}, label='${el.btnPlayMute.textContent}'`);
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
    assert proc.returncode == 0, f"Unified play/stop UI test failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    assert '"ok":true' in proc.stdout.replace(" ", "")
