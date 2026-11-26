"""Deep Music Theory DNA profile, 10-dim feature engine, and online bandit learning tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.db import RadioDatabase
from src.digger.history_importer import default_theory_stations
from src.models import TrackEntity
from src.radio_server import create_app
from src.recommender.bandit import ContextualBanditRecommender
from src.recommender.theory_features import (
    DEFAULT_THEORY_PROFILE_PATH,
    AudioTheorySignal,
    TheoryFeatureExtractor,
    load_music_theory_profile,
    merge_audio_theory_signal,
)


def test_music_theory_profile_yaml_defines_6_axes_and_5_exact_stations() -> None:
    profile = load_music_theory_profile(DEFAULT_THEORY_PROFILE_PATH)
    assert set(profile["axes"].keys()) == {"T1", "T2", "T3", "T4", "T5", "T6"}

    stations = default_theory_stations()
    assert [s.station_id for s in stations] == [
        "us_hiphop_bouncy",
        "japan_hiphop_street",
        "edm_dance_party",
        "melodic_rap_alt",
        "indie_groove_hooks",
    ]

    by_id = {s.station_id: s for s in stations}
    assert by_id["us_hiphop_bouncy"].target_theory_vector == {
        "T1": 0.55, "T2": 0.90, "T3": 0.78, "T4": 0.50, "T5": 0.88,
    }
    assert by_id["japan_hiphop_street"].target_theory_vector == {
        "T1": 0.65, "T2": 0.75, "T3": 0.82, "T4": 0.72, "T5": 0.85,
    }
    assert by_id["edm_dance_party"].target_theory_vector == {
        "T1": 0.92, "T2": 0.88, "T3": 0.75, "T4": 0.85, "T5": 0.95,
    }
    assert by_id["melodic_rap_alt"].target_theory_vector == {
        "T1": 0.78, "T2": 0.62, "T3": 0.70, "T4": 0.96, "T5": 0.84,
    }
    assert by_id["indie_groove_hooks"].target_theory_vector == {
        "T1": 0.88, "T2": 0.86, "T3": 0.85, "T4": 0.80, "T5": 0.86,
    }
    for st in stations:
        assert st.bpm_zones
        assert st.theory_archetypes
        assert st.spotify_anchor_seeds
        assert st.discovery_queries


def test_theory_feature_extractor_produces_10_dim_vector_and_merges_dsp_signal() -> None:
    extractor = TheoryFeatureExtractor()
    stations = {s.station_id: s for s in default_theory_stations()}
    edm_station = stations["edm_dance_party"]
    edm_station.weighted_seeds = [{"artist": "NOTION", "weight": 1.0}]

    ukg_gem = TrackEntity.build(
        artist="Underground Producer",
        title="Midnight Groove (UKG 2-step Remix)",
        tags=["uk garage", "house"],
        view_count=12_000,
        duration_seconds=195,
        video_id="VID_UKG_1",
        source="watch_playlist",
        seed_artist="NOTION",
        seed_archetype="ukg_2step_drive",
    )
    vec = extractor.extract_vector(ukg_gem, station=edm_station, hour=22)
    assert vec.shape == (10,)
    assert vec.dtype == np.float64
    # High T2 tresillo/2-step, high history affinity from NOTION lineage, high hidden gem score
    assert vec[1] >= 0.85
    assert vec[5] >= 0.90
    assert vec[6] >= 0.85
    # tod_sin^2 + tod_cos^2 == 1
    assert abs((vec[8] ** 2 + vec[9] ** 2) - 1.0) < 1e-6

    explanation = extractor.build_explanation(vec, station=edm_station, track=ukg_gem)
    assert explanation["summary"]
    assert len(explanation["top_axes"]) == 2
    assert "T6" in explanation["axes"]

    # AudioTheorySignal hook blends smoothly by confidence
    dsp = AudioTheorySignal(
        bpm=126.0,
        key_ambiguity=0.95,
        tresillo_autocorr=0.98,
        harmonic_tension=0.80,
        valence_contrast=0.85,
        hook_onset_sec=8.0,
        confidence=0.8,
    )
    merged = merge_audio_theory_signal([0.4, 0.4, 0.4, 0.4, 0.4], dsp)
    assert len(merged) == 5
    assert merged[1] > 0.85
    assert merged[4] > 0.80


def test_theory_bandit_learns_from_signed_good_bad_rewards_and_exposes_profile_api(tmp_path: Path) -> None:
    """GOOD (+1.0) / BAD (-1.0) on 10-dim theory vectors shifts T1-T5 profile and API responses."""
    assert ContextualBanditRecommender.compute_theory_reward("GOOD") == 1.0
    assert ContextualBanditRecommender.compute_theory_reward("BAD") == -1.0

    stations_yaml = tmp_path / "stations.yaml"
    db_path = tmp_path / "radio.db"
    model_dir = tmp_path / "models"

    app = create_app(
        db_path=db_path,
        stations_path=stations_yaml,
        model_dir=model_dir,
        auto_dig=False,
    )
    db: RadioDatabase = app.state.runtime.db
    client = TestClient(app)

    # Seed EDM station with 12 tagless watch_playlist tracks differentiated by theory archetypes
    for i in range(6):
        db.upsert_track(
            "edm_dance_party",
            TrackEntity.build(
                artist=f"UKGArtist{i}",
                title=f"Bounce Drive (UKG 2-step Remix) #{i}",
                tags=[],
                view_count=15_000,
                duration_seconds=190,
                video_id=f"VID_UKG_{i}",
                source="watch_playlist",
                seed_archetype="ukg_2step_drive",
            ),
        )
        db.upsert_track(
            "edm_dance_party",
            TrackEntity.build(
                artist=f"SadGuitar{i}",
                title=f"Slow Acoustic Ballad #{i}",
                tags=[],
                view_count=420_000,
                duration_seconds=360,
                video_id=f"VID_BAL_{i}",
                source="watch_playlist",
                seed_archetype="valence_contrast_major_emo",
            ),
        )

    # Train online via /feedback (GOOD on UKG, BAD on Ballads)
    for i in range(4):
        r1 = client.post(
            "/feedback",
            json={"station_id": "edm_dance_party", "video_id": f"VID_UKG_{i}", "event_type": "GOOD"},
        )
        assert r1.status_code == 200
        assert r1.json()["theory_reward"] == 1.0

        r2 = client.post(
            "/feedback",
            json={"station_id": "edm_dance_party", "video_id": f"VID_BAL_{i}", "event_type": "BAD"},
        )
        assert r2.status_code == 200
        assert r2.json()["theory_reward"] == -1.0

    # Verify /api/stations/{id}/next and /batch return theory_explanation + top 2 matched axes
    next_res = client.get("/api/stations/edm_dance_party/next")
    assert next_res.status_code == 200
    next_track = next_res.json()["track"]
    assert next_track is not None
    assert len(next_track["theory_vector"]) == 10
    assert len(next_track["theory_explanation"]["top_axes"]) == 2
    assert next_track["artist"].startswith("UKGArtist")

    batch_res = client.get("/api/stations/edm_dance_party/batch", params={"count": 3})
    assert batch_res.status_code == 200
    assert len(batch_res.json()["items"]) >= 2

    # Verify GET /api/profile/theory reflects learned updates
    prof_res = client.get("/api/profile/theory", params={"station_id": "edm_dance_party"})
    assert prof_res.status_code == 200
    prof = prof_res.json()
    assert prof["updates"] == 8
    assert len(prof["axes"]) == 5
    assert set(prof["scores"].keys()) == {"T1", "T2", "T3", "T4", "T5"}

    # Global aggregated profile also works without station_id
    global_prof = client.get("/api/profile/theory").json()
    assert global_prof["updates"] == 8
    assert len(global_prof["axes"]) == 5
