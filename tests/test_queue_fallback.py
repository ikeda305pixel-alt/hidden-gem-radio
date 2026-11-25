"""Queue fallback and filtering tests (evidence_environment: FIXTURE, counts_toward_l5_certification: false)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.db import RadioDatabase
from src.models import TrackEntity
from src.radio_server import create_app


def _write_station_yaml(path: Path) -> None:
    path.write_text(
        """stations:
  - station_id: test_fm
    name: Test Gem FM
    seed_tags:
      - folktronica
      - indietronica
    seed_artists:
      - Oberhofer
    max_popularity_views: 500000
    exploration_rate: 0.8
""",
        encoding="utf-8",
    )


def test_unplayable_video_is_flagged_and_queue_advances_seamlessly(tmp_path: Path) -> None:
    """An unplayable dummy videoId in the queue must not stall playback and must be excluded afterward."""
    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    db = RadioDatabase(db_path)
    tracks = [
        TrackEntity(
            track_id="trk_bad_embed",
            canonical_key="artist0::broken track",
            artist="Artist0",
            title="Broken Embed Track",
            tags=["folktronica"],
            release_year=2020,
            view_count=12_000,
            video_id="DUMMY_UNPLAYABLE_01",
        ),
        TrackEntity(
            track_id="trk_good_1",
            canonical_key="artist1::hidden gem one",
            artist="Artist1",
            title="Hidden Gem One",
            tags=["folktronica"],
            release_year=2021,
            view_count=18_000,
            video_id="VALID_VIDEO_001",
        ),
        TrackEntity(
            track_id="trk_good_2",
            canonical_key="artist2::hidden gem two",
            artist="Artist2",
            title="Hidden Gem Two",
            tags=["indietronica"],
            release_year=2019,
            view_count=42_000,
            video_id="VALID_VIDEO_002",
        ),
        TrackEntity(
            track_id="trk_over_views",
            canonical_key="megastar::viral hit",
            artist="MegaStar",
            title="Viral Hit",
            tags=["folktronica"],
            release_year=2022,
            view_count=9_500_000,
            video_id="VIRAL_VIDEO_999",
        ),
    ]
    for t in tracks:
        db.upsert_track("test_fm", t)

    app = create_app(
        db_path=db_path,
        stations_path=stations_path,
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    client = TestClient(app)

    # Initial prefetch queue should never contain the viral track (> 500k views)
    res = client.get("/queue/next", params={"station_id": "test_fm", "count": 3})
    assert res.status_code == 200
    items = res.json()["items"]
    vids = [item["video_id"] for item in items]
    assert "VIRAL_VIDEO_999" not in vids
    assert "DUMMY_UNPLAYABLE_01" in vids

    # Simulate YouTube IFrame ERROR 150 (embed restricted) for dummy videoId
    fb = client.post(
        "/feedback",
        json={
            "station_id": "test_fm",
            "track_id": "trk_bad_embed",
            "video_id": "DUMMY_UNPLAYABLE_01",
            "event_type": "UNPLAYABLE",
            "yt_error_code": 150,
        },
    )
    assert fb.status_code == 200
    body = fb.json()
    assert body["unplayable_marked"] is True
    assert body["replacement"] is not None
    assert body["replacement"]["video_id"] != "DUMMY_UNPLAYABLE_01"

    # Subsequent queue requests must exclude the unplayable track
    res2 = client.get("/queue/next", params={"station_id": "test_fm", "count": 3})
    assert res2.status_code == 200
    vids2 = [item["video_id"] for item in res2.json()["items"]]
    assert "DUMMY_UNPLAYABLE_01" not in vids2
    assert set(vids2) == {"VALID_VIDEO_001", "VALID_VIDEO_002"}


def test_bad_artist_excluded_for_30_days(tmp_path: Path) -> None:
    """Artists given BAD feedback must be excluded from subsequent queue pulls within 30 days."""
    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    db = RadioDatabase(db_path)
    db.upsert_track(
        "test_fm",
        TrackEntity(
            track_id="trk_a1",
            canonical_key="badartist::song 1",
            artist="BadArtist",
            title="Song 1",
            tags=["folktronica"],
            view_count=20_000,
            video_id="VID_BAD_1",
        ),
    )
    db.upsert_track(
        "test_fm",
        TrackEntity(
            track_id="trk_a2",
            canonical_key="badartist::song 2",
            artist="BadArtist",
            title="Song 2",
            tags=["folktronica"],
            view_count=25_000,
            video_id="VID_BAD_2",
        ),
    )
    db.upsert_track(
        "test_fm",
        TrackEntity(
            track_id="trk_ok",
            canonical_key="goodartist::song ok",
            artist="GoodArtist",
            title="Song OK",
            tags=["folktronica"],
            view_count=30_000,
            video_id="VID_OK_1",
        ),
    )

    app = create_app(
        db_path=db_path,
        stations_path=stations_path,
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    client = TestClient(app)

    fb = client.post(
        "/feedback",
        json={
            "station_id": "test_fm",
            "track_id": "trk_a1",
            "video_id": "VID_BAD_1",
            "event_type": "BAD",
            "listen_seconds": 6.0,
            "duration_seconds": 210.0,
        },
    )
    assert fb.status_code == 200

    res = client.get("/queue/next", params={"station_id": "test_fm", "count": 3})
    assert res.status_code == 200
    artists = [item["artist"] for item in res.json()["items"]]
    assert "BadArtist" not in artists
    assert "GoodArtist" in artists
