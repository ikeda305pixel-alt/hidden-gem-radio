"""Real-execution tests for 16-dim MIR, ffmpeg fallback, Stage 1 co-occurrence graph, and 3-layer calibrated learning."""

from __future__ import annotations

import io
import math
import time
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import numpy as np

from src.adapters.deezer import DeezerClient, DeezerTrackMeta
from src.adapters.youtube import YouTubeMusicAdapter, is_slop_or_unofficial
from src.db import RadioDatabase
from src.digger.history_importer import default_theory_stations
from src.digger.pool import CandidatePoolDigger
from src.digger.pool import main as pool_cli_main
from src.models import StationConfig, TrackEntity
from src.recommender.audio_mir import (
    MIR_DIM,
    TARGET_SR,
    AudioMIRExtractor,
    compute_cooccurrence_score,
    compute_deezer_rank_sweetspot,
    compute_yt_views_sweetspot,
)
from src.recommender.bandit import ContextualBanditRecommender
from src.recommender.calibrate_profile import calibrate_all_stations


def _synth_wav_bytes(samples: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Encode float32 [-1, 1] mono audio to 16-bit PCM WAV bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def _make_click_track_wav(bpm: float, duration_sec: float = 8.0, sr: int = TARGET_SR) -> bytes:
    """Synthesize a tight periodic percussive click + sub-bass pulse at exact BPM."""
    n = int(sr * duration_sec)
    y = np.zeros(n, dtype=np.float64)
    interval = round(sr * 60.0 / bpm)
    click_len = int(sr * 0.04)
    t_click = np.arange(click_len) / sr
    # Crisp transient + 70Hz kick body
    pulse = np.exp(-t_click * 80.0) * (0.65 * np.sin(2.0 * math.pi * 70.0 * t_click) + 0.35 * np.sin(2.0 * math.pi * 1200.0 * t_click))
    for pos in range(0, n - click_len, interval):
        y[pos : pos + click_len] += pulse
    return _synth_wav_bytes(y, sr=sr)


def _make_chord_wav(freqs_hz: list[float], duration_sec: float = 4.0, sr: int = TARGET_SR) -> bytes:
    """Synthesize a clean harmonic chord WAV (e.g. C major triad vs A minor triad vs rich maj9)."""
    t = np.arange(int(sr * duration_sec), dtype=np.float64) / sr
    y = np.zeros_like(t)
    for f in freqs_hz:
        # Include fundamental + 2nd harmonic for realistic pitch-class profile
        y += 0.75 * np.sin(2.0 * math.pi * f * t) + 0.25 * np.sin(2.0 * math.pi * (2.0 * f) * t)
    y /= max(1e-9, float(np.max(np.abs(y))))
    return _synth_wav_bytes(y, sr=sr)


# ---------------------------------------------------------------------------
# Test 1: Synthetic WAV real MIR analysis (BPM, beat clarity, major/minor, harmonic entropy, speed < 0.5s)
# ---------------------------------------------------------------------------
def test_audio_mir_real_wav_analysis_differentiates_bpm_rhythm_and_harmony() -> None:
    extractor = AudioMIRExtractor()

    # 1a. High BPM (150 BPM) vs Low BPM (84 BPM) vs flat unpulsed drone
    wav_fast = _make_click_track_wav(150.0, duration_sec=8.0)
    wav_slow = _make_click_track_wav(84.0, duration_sec=8.0)
    wav_flat = _make_chord_wav([261.63], duration_sec=4.0)  # Pure C4 drone (no beat)

    t0 = time.perf_counter()
    vec_fast, bpm_fast, _, src_fast = extractor.extract_a1_c4_for_track("Fast", "Click", audio_bytes=wav_fast)
    elapsed = time.perf_counter() - t0
    vec_slow, bpm_slow, _, src_slow = extractor.extract_a1_c4_for_track("Slow", "Click", audio_bytes=wav_slow)
    vec_flat, _, _, _ = extractor.extract_a1_c4_for_track("Flat", "Drone", audio_bytes=wav_flat)

    assert src_fast == "mir_pcm" and src_slow == "mir_pcm"
    assert elapsed < 0.5, f"MIR analysis exceeded 0.5s target: {elapsed:.3f}s"
    assert vec_fast.shape == (12,)
    # High BPM click has significantly higher a1_bpm_norm than low BPM click
    assert vec_fast[0] > vec_slow[0] + 0.25, f"Expected fast a1 ({vec_fast[0]}) >> slow a1 ({vec_slow[0]})"
    assert abs(bpm_fast - 150.0) <= 6.0, f"Estimated fast BPM {bpm_fast} far from 150"
    assert abs(bpm_slow - 84.0) <= 6.0, f"Estimated slow BPM {bpm_slow} far from 84"
    # Periodic click tracks have much higher a2_beat_clarity than flat sine drone
    assert vec_fast[1] > vec_flat[1] + 0.30
    assert vec_slow[1] > vec_flat[1] + 0.30

    # 1b. Harmony: C Major (C-E-G) vs A Minor (A-C-E) for b1_major_minor_score,
    #     and pure single tone vs dense jazz maj9#11 chord for b2_harmonic_entropy
    wav_c_maj = _make_chord_wav([261.63, 329.63, 392.00, 523.25])  # C4 E4 G4 C5
    wav_c_min = _make_chord_wav([261.63, 311.13, 392.00, 523.25])  # C4 Eb4 G4 C5
    wav_rich_jazz = _make_chord_wav([261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88])  # Cmaj13#11

    v_maj, _, _, _ = extractor.extract_a1_c4_for_track("Harm", "CMaj", audio_bytes=wav_c_maj)
    v_min, _, _, _ = extractor.extract_a1_c4_for_track("Harm", "CMin", audio_bytes=wav_c_min)
    v_jazz, _, _, _ = extractor.extract_a1_c4_for_track("Harm", "Jazz", audio_bytes=wav_rich_jazz)

    assert v_maj[4] > v_min[4] + 0.05, f"Major chord b1 ({v_maj[4]}) should exceed minor chord b1 ({v_min[4]})"
    assert v_jazz[5] > vec_flat[5] + 0.25, f"Rich chord entropy ({v_jazz[5]}) should exceed single tone ({vec_flat[5]})"

    # 1c. Verify YouTube 300k-50M sweetspot & Deezer rank sweetspot bell curves
    assert compute_yt_views_sweetspot(2_500_000) == 1.0
    assert compute_yt_views_sweetspot(5_000) < 0.25  # heavy penalty <100k
    assert 0.5 < compute_yt_views_sweetspot(800_000_000) < 0.90  # gentle decay >500M
    assert compute_deezer_rank_sweetspot(350_000) > compute_deezer_rank_sweetspot(500)
    assert compute_cooccurrence_score(3) > compute_cooccurrence_score(2) > compute_cooccurrence_score(1)


# ---------------------------------------------------------------------------
# Test 2: ffmpeg-absent graceful Deezer metadata fallback + SQLite caching (zero duplicate HTTP)
# ---------------------------------------------------------------------------
def test_ffmpeg_absent_falls_back_gracefully_to_deezer_metadata_and_caches(tmp_path: Path) -> None:
    db = RadioDatabase(tmp_path / "radio.db")
    calls = {"search": 0, "track": 0, "preview": 0}

    def _mock_transport(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search":
            calls["search"] += 1
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": 99001,
                            "title": "Oysters In My Pocket",
                            "artist": {"name": "Royel Otis"},
                            "rank": 420_000,
                            "duration": 162,
                            "preview": "https://cdns-preview.dzcdn.net/stream/mock.mp3",
                        }
                    ]
                },
            )
        if path == "/track/99001":
            calls["track"] += 1
            return httpx.Response(
                200,
                json={
                    "id": 99001,
                    "title": "Oysters In My Pocket",
                    "artist": {"name": "Royel Otis"},
                    "bpm": 124.0,
                    "gain": -9.8,
                    "rank": 420_000,
                    "duration": 162,
                    "preview": "https://cdns-preview.dzcdn.net/stream/mock.mp3",
                },
            )
        calls["preview"] += 1
        return httpx.Response(200, content=b"ID3_FAKE_MP3_BYTES")

    http_client = httpx.Client(transport=httpx.MockTransport(_mock_transport))
    deezer = DeezerClient(db=db, min_interval_sec=0.0, http_client=http_client)
    # Explicitly force ffmpeg_bin="" to simulate Termux without ffmpeg installed
    extractor = AudioMIRExtractor(db=db, deezer_client=deezer, ffmpeg_bin="")

    track = TrackEntity.build(
        artist="Royel Otis",
        title="Oysters In My Pocket",
        video_id="VID_ROYEL_01",
        view_count=4_200_000,
        duration_seconds=162,
        cooccurrence_count=2,
    )

    vec1, bpm1, src1 = extractor.extract_16d_vector(track, allow_network=True)
    assert vec1.shape == (MIR_DIM,)
    assert np.all((vec1 >= 0.0) & (vec1 <= 1.0))
    assert bpm1 == 124.0
    assert src1 == "deezer_metadata"
    assert calls["search"] == 1 and calls["track"] == 1
    assert calls["preview"] == 0  # Skips useless preview download when ffmpeg is absent

    # Second call MUST hit SQLite audio_features_cache with ZERO additional Deezer HTTP calls!
    vec2, bpm2, src2 = extractor.extract_16d_vector(track, allow_network=True)
    assert np.allclose(vec1, vec2)
    assert bpm2 == 124.0
    assert src2 == "deezer_metadata"
    assert calls["search"] == 1 and calls["track"] == 1

    # Also verify non-WAV MP3 bytes passed directly with missing ffmpeg falls back without crashing
    meta = DeezerTrackMeta(99001, "Royel Otis", "Oysters In My Pocket", 124.0, -9.8, 420_000, 162)
    v_fb, bpm_fb, _, fb_src = extractor.extract_a1_c4_for_track(
        "Royel Otis",
        "Oysters In My Pocket",
        audio_bytes=b"ID3_UNDECODABLE_WITHOUT_FFMPEG",
        deezer_meta=meta,
    )
    assert v_fb.shape == (12,)
    assert bpm_fb == 124.0
    assert fb_src == "deezer_metadata"


# ---------------------------------------------------------------------------
# Test 3: Stage 1 filters out AI slop & UGC videos, and elevates multi-seed co-occurring gems
# ---------------------------------------------------------------------------
def test_stage1_graph_digger_blocks_ai_slop_ugc_and_boosts_cooccurrence(tmp_path: Path) -> None:
    # Direct unit checks on is_slop_or_unofficial
    assert is_slop_or_unofficial({"title": "Chill Lofi (AI Generated)", "videoType": "MUSIC_VIDEO_TYPE_ATV"}) is True
    assert is_slop_or_unofficial({"title": "bbno$ Type Beat 2026", "videoType": "MUSIC_VIDEO_TYPE_ATV"}) is True
    assert is_slop_or_unofficial({"title": "edamame (Slowed + Reverb)", "videoType": "MUSIC_VIDEO_TYPE_ATV"}) is True
    assert is_slop_or_unofficial({"title": "Suno AI Hip Hop Track", "videoType": "MUSIC_VIDEO_TYPE_OMV"}) is True
    assert is_slop_or_unofficial({"title": "Real Song", "videoType": "MUSIC_VIDEO_TYPE_UGC"}) is True
    assert is_slop_or_unofficial({"title": "Real Song", "resultType": "video"}) is True
    assert is_slop_or_unofficial({"title": "Official Gem", "resultType": "song", "videoType": "MUSIC_VIDEO_TYPE_ATV"}) is False

    class FakeYTMusic:
        def search(self, query: str, filter: str | None = None, limit: int = 8):
            assert filter == "songs", "Stage 1 MUST search strictly with filter='songs'"
            if "bbno" in query.lower():
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
            if "kyle" in query.lower():
                return [
                    {
                        "videoId": "SEED_KYLE_02",
                        "title": "iSpy",
                        "artists": [{"name": "KYLE", "id": "ART_KYLE"}],
                        "resultType": "song",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "duration_seconds": 252,
                        "views": "400M",
                    }
                ]
            return []

        def get_watch_playlist(self, videoId: str, radio: bool = False, limit: int = 25):
            assert radio is True, "Stage 1 MUST request radio=True"
            shared_gem = {
                "videoId": "VID_COOC_GEM",
                "title": "Bouncy Underground Anthem",
                "artists": [{"name": "Amine"}],
                "videoType": "MUSIC_VIDEO_TYPE_ATV",
                "length": "3:05",
                "views": "4.5M",
            }
            slop_ai = {
                "videoId": "VID_SLOP_AI1",
                "title": "Free Type Beat (Suno AI Generated)",
                "artists": [{"name": "BeatBot"}],
                "videoType": "MUSIC_VIDEO_TYPE_ATV",
                "length": "2:40",
                "views": "20K",
            }
            ugc_vid = {
                "videoId": "VID_UGC_0001",
                "title": "Fan Cover Live Clip",
                "artists": [{"name": "RandomUser"}],
                "videoType": "MUSIC_VIDEO_TYPE_UGC",
                "length": "3:10",
                "views": "500K",
            }
            nightcore_vid = {
                "videoId": "VID_NIGHT_01",
                "title": "Song (Sped Up Nightcore)",
                "artists": [{"name": "SpeedChannel"}],
                "videoType": "MUSIC_VIDEO_TYPE_OMV",
                "length": "2:20",
                "views": "1.2M",
            }
            if videoId == "SEED_BBNO_01":
                only_a = {
                    "videoId": "VID_SINGLE_A",
                    "title": "Single Seed Track A",
                    "artists": [{"name": "Yung Gravy"}],
                    "videoType": "MUSIC_VIDEO_TYPE_ATV",
                    "length": "2:55",
                    "views": "3.0M",
                }
                return {"tracks": [shared_gem, only_a, slop_ai, ugc_vid, nightcore_vid]}
            return {"tracks": [shared_gem, slop_ai, ugc_vid]}

        def get_artist(self, browseId: str):
            if browseId == "ART_BBNO":
                return {"related": {"results": [{"browseId": "ART_REL_1", "title": "Duckwrth"}]}}
            if browseId == "ART_REL_1":
                # 5 songs: deep cuts (#3..#5) should be prioritized ahead of #1..#2
                return {
                    "songs": {
                        "results": [
                            {"videoId": f"VID_REL_S{i}", "title": f"Deep Cut {i}", "artists": [{"name": "Duckwrth"}], "videoType": "MUSIC_VIDEO_TYPE_ATV", "duration": "3:12", "views": "1.5M"}
                            for i in range(1, 6)
                        ]
                    }
                }
            return {}

    db = RadioDatabase(tmp_path / "radio.db")
    yt_adapter = YouTubeMusicAdapter(client=FakeYTMusic())
    digger = CandidatePoolDigger(db=db, yt_adapter=yt_adapter)

    station = StationConfig(
        station_id="us_hiphop_bouncy",
        name="US Hip-Hop & Bouncy Rap",
        seed_tags=["hip hop"],
        seed_artists=["bbno$", "KYLE"],
        spotify_anchor_seeds=["bbno$ - edamame", "KYLE - iSpy"],
        max_popularity_views=500_000_000,
    )
    res = digger.dig_station(station, target_pool_size=20)
    assert res["cooccurring_added"] >= 1

    eligible = db.get_eligible_candidates("us_hiphop_bouncy", 500_000_000)
    by_vid = {t.video_id: t for t in eligible}
    # AI slop, UGC, nightcore/sped up completely blocked
    assert "VID_SLOP_AI1" not in by_vid
    assert "VID_UGC_0001" not in by_vid
    assert "VID_NIGHT_01" not in by_vid
    # Co-occurring track + single-seed track + related artist deep cuts present
    assert "VID_COOC_GEM" in by_vid
    assert "VID_SINGLE_A" in by_vid
    assert "VID_REL_S3" in by_vid
    assert by_vid["VID_COOC_GEM"].cooccurrence_count == 2
    assert "共起2シード" in (by_vid["VID_COOC_GEM"].discovery_route or "")

    # Stage 2 scoring MUST rank the multi-seed co-occurring gem strictly above single-seed track
    bandit = ContextualBanditRecommender(
        station_id="us_hiphop_bouncy",
        seed_tags=station.seed_tags,
        model_path=tmp_path / "bandit_us.pkl",
        station_config=station,
        db=db,
    )
    ranked = bandit.rank_candidates(
        [by_vid["VID_SINGLE_A"], by_vid["VID_COOC_GEM"]],
        seen_artists=set(),
        limit=2,
    )
    assert ranked[0].track.video_id == "VID_COOC_GEM"
    assert ranked[0].score > ranked[1].score
    assert ranked[0].to_queue_dict()["discovery_route"].startswith("共起2シード")


# ---------------------------------------------------------------------------
# Test 4: Auto-calibration from YouTube + Spotify + PES/Emily, ±0.02 single-step bound,
#         strict station isolation, Stage 3 artist cooldown, & --reset-tracks preservation
# ---------------------------------------------------------------------------
def test_calibration_bounded_slow_learning_station_isolation_and_reset_tracks(tmp_path: Path) -> None:
    db = RadioDatabase(tmp_path / "radio.db")
    stations = default_theory_stations()
    assert len(stations) == 5

    # Deterministic synthetic per-track Deezer/MIR vectors so calibration measures real variance & centroids
    class DeterministicMIR(AudioMIRExtractor):
        def extract_a1_c4_for_track(self, artist, title, **kwargs):
            low = f"{artist} {title}".lower()
            if any(k in low for k in ("bbno", "kyle", "migos", "drake")):
                # Tightly consistent high sub-bass (a4=0.88) and syncopation (a3=0.82)
                vec = np.array([0.45, 0.78, 0.82, 0.88, 0.42, 0.46, 0.48, 0.44, 0.40, 0.25, 0.82, 0.64])
                return vec, 114.0, None, "mir_pcm"
            if any(k in low for k in ("avicii", "calvin", "notion", "tiesto")):
                vec = np.array([0.58, 0.94, 0.70, 0.74, 0.66, 0.56, 0.52, 0.56, 0.64, 0.36, 0.90, 0.56])
                return vec, 128.0, None, "mir_pcm"
            if any(k in low for k in ("imperatrice", "phoenix", "royel", "francescoli", "temper trap")):
                vec = np.array([0.48, 0.72, 0.62, 0.48, 0.62, 0.74, 0.62, 0.72, 0.48, 0.22, 0.66, 0.64])
                return vec, 118.0, None, "mir_pcm"
            # Mild track-specific variation so variance is non-zero
            h = (abs(hash(low)) % 100) / 1000.0
            vec = np.clip(np.array([0.50 + h, 0.70, 0.60 - h, 0.55, 0.52, 0.58, 0.50, 0.60, 0.45 + h, 0.25, 0.70, 0.62]), 0.0, 1.0)
            return vec, 120.0, None, "mir_pcm"

    ref_now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
    yt_records = [
        {"artist": "bbno$", "title": "edamame", "video_id": "YT_BBNO_1", "played_at": (ref_now - timedelta(days=2)).isoformat()},
        {"artist": "bbno$", "title": "edamame", "video_id": "YT_BBNO_1", "played_at": (ref_now - timedelta(days=1)).isoformat()},
        {"artist": "Avicii", "title": "Gonna Love Ya", "video_id": "YT_AVICII_1", "played_at": (ref_now - timedelta(days=3)).isoformat()},
        {"artist": "Royel Otis", "title": "Oysters In My Pocket", "video_id": "YT_ROYEL_1", "played_at": (ref_now - timedelta(days=4)).isoformat()},
    ]

    mir = DeterministicMIR(db=db)
    summary = calibrate_all_stations(
        db,
        stations=stations,
        takeout_records=yt_records,
        mir_extractor=mir,
        allow_network=False,
        reference_time=ref_now,
    )
    assert summary["status"] == "ok"
    assert db.get_station_profile("__cult_chic_anchor__") is not None

    prof_us = db.get_station_profile("us_hiphop_bouncy")
    prof_edm = db.get_station_profile("edm_dance_party")
    assert prof_us is not None and prof_edm is not None
    assert len(prof_us["centroid"]) == MIR_DIM
    assert len(prof_us["axis_weights"]) == MIR_DIM
    assert prof_us["sample_counts"]["youtube"] >= 1
    assert prof_us["sample_counts"]["spotify"] >= 20
    # US Hip-Hop and EDM calibrated centroids genuinely differ in measured acoustics
    assert not np.allclose(prof_us["centroid"], prof_edm["centroid"])

    by_id = {s.station_id: s for s in stations}
    bandit_us = ContextualBanditRecommender(
        station_id="us_hiphop_bouncy",
        seed_tags=by_id["us_hiphop_bouncy"].seed_tags,
        model_path=tmp_path / "b_us.pkl",
        station_config=by_id["us_hiphop_bouncy"],
        mir_extractor=mir,
        db=db,
    )
    bandit_edm = ContextualBanditRecommender(
        station_id="edm_dance_party",
        seed_tags=by_id["edm_dance_party"].seed_tags,
        model_path=tmp_path / "b_edm.pkl",
        station_config=by_id["edm_dance_party"],
        mir_extractor=mir,
        db=db,
    )

    us_slow_before = bandit_us.get_slow_weights()
    us_eff_before = bandit_us.get_effective_weights()
    edm_slow_before = bandit_edm.get_slow_weights()
    edm_eff_before = bandit_edm.get_effective_weights()

    extreme_track = TrackEntity.build(
        artist="Extreme Synth",
        title="Extreme Test",
        video_id="VID_EXT_001",
        mir_vector=[1.0] * MIR_DIM,
    )
    # 1. Single GOOD (+1.0) on US station moves slow & effective weights by AT MOST ±0.02
    bandit_us.update(extreme_track, event_type="GOOD")
    us_slow_after_good = bandit_us.get_slow_weights()
    us_eff_after_good = bandit_us.get_effective_weights()
    assert float(np.max(np.abs(us_slow_after_good - us_slow_before))) <= 0.02 + 1e-9
    assert float(np.max(np.abs(us_eff_after_good - us_eff_before))) <= 0.02 + 1e-9
    assert float(np.max(np.abs(us_slow_after_good - us_slow_before))) > 0.0

    # 2. Single BAD (-1.0) on US station also bounded strictly within ±0.02
    bandit_us.update(extreme_track, event_type="BAD")
    us_slow_after_bad = bandit_us.get_slow_weights()
    assert float(np.max(np.abs(us_slow_after_bad - us_slow_after_good))) <= 0.02 + 1e-9

    # 3. Complete per-station isolation: EDM station weights (in-memory & SQLite) are 100% untouched!
    assert np.allclose(bandit_edm.get_slow_weights(), edm_slow_before)
    assert np.allclose(bandit_edm.get_effective_weights(), edm_eff_before)
    assert np.allclose(db.get_station_profile("edm_dance_party")["slow_weights"], prof_edm["slow_weights"])

    # 4. Implicit rewards match spec exactly (+0.35 for >=70% complete, -0.40 for <15s skip)
    assert ContextualBanditRecommender.compute_theory_reward("COMPLETE", listen_seconds=140, duration_seconds=180) == 0.35
    assert ContextualBanditRecommender.compute_theory_reward("SKIP", listen_seconds=10.0, duration_seconds=180) == -0.40
    assert ContextualBanditRecommender.compute_theory_reward("EARLY_SKIP", listen_seconds=8.0, duration_seconds=180) == -0.40

    # 5. Stage 3 Re-Ranking: recent 5-track artist cooldown pushes repeated artists behind fresh artists
    cands = [
        TrackEntity.build(artist="RecentArtist", title="Repeat Song", video_id="VID_REP_01", cooccurrence_count=4, view_count=5_000_000),
        TrackEntity.build(artist="FreshArtist1", title="Fresh Song 1", video_id="VID_FR_001", cooccurrence_count=2, view_count=5_000_000),
        TrackEntity.build(artist="FreshArtist2", title="Fresh Song 2", video_id="VID_FR_002", cooccurrence_count=2, view_count=5_000_000),
    ]
    ranked = bandit_us.rank_candidates(
        cands,
        seen_artists=set(),
        recent_artists=["recentartist"],
        limit=3,
    )
    assert ranked[0].track.artist != "RecentArtist"
    assert ranked[1].track.artist != "RecentArtist"

    # 6. CLI `python -m src.digger.pool --reset-tracks` deletes unrated tracks ONLY and preserves playlists & Good feedback
    good_track = TrackEntity.build(artist="Saved Artist", title="Good Gem", video_id="VID_SAVED_1", view_count=2_000_000)
    junk_track = TrackEntity.build(artist="Old Junk", title="Unrated Slop", video_id="VID_JUNK_01", view_count=100)
    db.upsert_track("us_hiphop_bouncy", good_track)
    db.upsert_track("us_hiphop_bouncy", junk_track)
    db.record_feedback("us_hiphop_bouncy", good_track, "GOOD", 1.0)
    db.add_track_to_playlist("pl_favorites", video_id="VID_SAVED_1", track_id=good_track.track_id, artist="Saved Artist", title="Good Gem")

    rc = pool_cli_main(["--db", str(tmp_path / "radio.db"), "--reset-tracks"])
    assert rc == 0
    assert db.get_track(video_id="VID_JUNK_01") is None, "Unrated legacy track must be purged"
    assert db.get_track(video_id="VID_SAVED_1") is not None, "Good-rated/playlist track must be preserved"
    assert db.count_feedback("us_hiphop_bouncy") == 1
    assert db.get_playlist_tracks("pl_favorites")["track_count"] == 1


def test_calibrate_profile_cli_and_web_ui_mir_indicators(tmp_path: Path) -> None:
    from src.recommender.calibrate_profile import main as cal_cli_main

    db_path = tmp_path / "cal_cli.db"
    rc = cal_cli_main(["--db", str(db_path), "--takeout", str(tmp_path / "none.json"), "--offline"])
    assert rc == 0
    db = RadioDatabase(db_path)
    for sid in ("us_hiphop_bouncy", "japan_hiphop_street", "edm_dance_party", "melodic_rap_alt", "indie_groove_cult"):
        prof = db.get_station_profile(sid)
        assert prof is not None and len(prof["centroid"]) == MIR_DIM

    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "発掘ルート" in html
    assert "知名度スイートスポット" in html
    assert "4レイヤー安定嗜好マッチ" in html
    assert "IMPLICIT_COMPLETE_RATIO = 0.70" in html
    assert "IMPLICIT_EARLY_SKIP_SEC = 15.0" in html
    assert "EARLY_SKIP" in html

