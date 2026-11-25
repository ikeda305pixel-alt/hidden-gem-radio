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


def test_duration_3600s_video_is_rejected_and_returns_none() -> None:
    """[M3] A 1-hour mix must NOT bypass duration filter via results[0] fallback."""
    from unittest.mock import MagicMock

    from src.adapters.youtube import YouTubeMusicAdapter

    fake_yt = MagicMock()
    fake_yt.search.return_value = [
        {
            "resultType": "video",
            "videoId": "LONG_MIX_3600",
            "title": "Artist - Full Album Mix 1 Hour",
            "duration_seconds": 3600,
            "views": "250K",
            "artists": [{"name": "Artist"}],
        }
    ]
    adapter = YouTubeMusicAdapter(client=fake_yt)
    assert adapter.resolve_track("Artist", "Some Song") is None


def test_watch_playlist_tracks_do_not_inherit_station_seed_tags() -> None:
    """[C2] Watch Playlist tracks must keep empty tags (no seed tag contamination) and source='watch_playlist'."""
    from unittest.mock import MagicMock

    from src.adapters.youtube import YouTubeMusicAdapter
    from src.recommender.bandit import ContextualBanditRecommender

    fake_yt = MagicMock()
    fake_yt.get_watch_playlist.return_value = {
        "tracks": [
            {
                "videoId": "WP_VID_001",
                "title": "Atomic Blue - You're Right",
                "artists": [{"name": "Atomic Blue"}],
                "views": "15K",
                "length": "4:16",
            }
        ]
    }
    adapter = YouTubeMusicAdapter(client=fake_yt)
    tracks = adapter.get_related_tracks("SEED_VID", seed_tags=["folktronica", "indietronica"], limit=10)
    assert len(tracks) == 1
    assert tracks[0].tags == []
    assert tracks[0].source == "watch_playlist"

    # Bandit reason must NOT display fabricated matched_tags
    bandit = ContextualBanditRecommender("test_fm", seed_tags=["folktronica", "indietronica"])
    ranked = bandit.rank_candidates(tracks, seen_artists=set(), limit=1)
    assert ranked[0].matched_tags == []
    assert "folktronica" not in ranked[0].to_queue_dict()["reason"]["summary"]


def test_transient_youtube_error_does_not_mark_unplayable(tmp_path: Path) -> None:
    """[M5] Transient YT error codes (2 / 5) must NOT permanently flag unplayable=1."""
    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    db = RadioDatabase(db_path)
    db.upsert_track(
        "test_fm",
        TrackEntity(
            track_id="trk_t1",
            canonical_key="a1::s1",
            artist="A1",
            title="S1",
            tags=["folktronica"],
            view_count=10_000,
            video_id="VID_T1",
        ),
    )
    db.upsert_track(
        "test_fm",
        TrackEntity(
            track_id="trk_t2",
            canonical_key="a2::s2",
            artist="A2",
            title="S2",
            tags=["folktronica"],
            view_count=12_000,
            video_id="VID_T2",
        ),
    )

    app = create_app(db_path=db_path, stations_path=stations_path, model_dir=tmp_path / "m", auto_dig=False)
    client = TestClient(app)

    res = client.post(
        "/feedback",
        json={
            "station_id": "test_fm",
            "track_id": "trk_t1",
            "video_id": "VID_T1",
            "event_type": "UNPLAYABLE",
            "yt_error_code": 5,
        },
    )
    assert res.status_code == 200
    assert res.json()["unplayable_marked"] is False
    assert res.json()["replacement"]["video_id"] == "VID_T2"
    assert db.get_track(track_id="trk_t1").unplayable is False


def test_queue_next_triggers_non_blocking_background_dig_when_unheard_low(tmp_path: Path) -> None:
    """[M1/M4] When unheard tracks exist but < threshold, /queue/next returns immediately and digs in background."""
    import threading
    import time

    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    db = RadioDatabase(db_path)
    for i in range(2):
        db.upsert_track(
            "test_fm",
            TrackEntity(
                track_id=f"trk_seed_{i}",
                canonical_key=f"seed{i}::song{i}",
                artist=f"Seed{i}",
                title=f"Song{i}",
                tags=["folktronica"],
                view_count=10_000,
                video_id=f"VID_SEED_{i}",
            ),
        )

    app = create_app(db_path=db_path, stations_path=stations_path, model_dir=tmp_path / "m", auto_dig=True)
    dig_started = threading.Event()
    dig_finished = threading.Event()

    class SlowFakeDigger:
        def dig_station(self, station, target_pool_size=60):
            dig_started.set()
            time.sleep(0.35)
            db.upsert_track(
                station.station_id,
                TrackEntity(
                    track_id="trk_bg_new",
                    canonical_key="bg::new song",
                    artist="BgArtist",
                    title="New Song",
                    tags=["folktronica"],
                    view_count=8_000,
                    video_id="VID_BG_NEW",
                ),
            )
            dig_finished.set()
            return {"station_id": station.station_id, "added": 1}

    app.state.runtime._digger = SlowFakeDigger()
    client = TestClient(app)

    t0 = time.monotonic()
    res = client.get("/queue/next", params={"station_id": "test_fm", "count": 2})
    elapsed = time.monotonic() - t0

    assert res.status_code == 200
    assert len(res.json()["items"]) == 2
    assert elapsed < 0.25, f"/queue/next blocked for {elapsed:.2f}s instead of returning immediately"
    assert dig_finished.wait(timeout=2.0), "Background dig did not complete"
    assert dig_started.is_set()
    assert db.get_track(track_id="trk_bg_new") is not None


def test_concurrent_cold_pool_requests_execute_only_one_dig(tmp_path: Path) -> None:
    """[M1] Two concurrent /queue/next requests on an empty pool must share a single dig."""
    import threading
    import time

    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    app = create_app(db_path=db_path, stations_path=stations_path, model_dir=tmp_path / "m", auto_dig=True)
    dig_calls = 0

    class CountingFakeDigger:
        def dig_station(self, station, target_pool_size=60):
            nonlocal dig_calls
            dig_calls += 1
            time.sleep(0.15)
            for i in range(6):
                app.state.runtime.db.upsert_track(
                    station.station_id,
                    TrackEntity(
                        track_id=f"trk_cold_{i}",
                        canonical_key=f"cold{i}::s{i}",
                        artist=f"Cold{i}",
                        title=f"S{i}",
                        tags=["folktronica"],
                        view_count=5_000,
                        video_id=f"VID_COLD_{i}",
                    ),
                )
            return {"station_id": station.station_id, "added": 6}

    app.state.runtime._digger = CountingFakeDigger()
    client = TestClient(app)
    counts: list[int] = []

    def req() -> None:
        res = client.get("/queue/next", params={"station_id": "test_fm", "count": 3})
        assert res.status_code == 200
        counts.append(len(res.json()["items"]))

    t1 = threading.Thread(target=req)
    t2 = threading.Thread(target=req)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert counts == [3, 3]
    assert dig_calls == 1, f"Expected single deduplicated dig, got {dig_calls}"


def test_digging_stations_flag_does_not_leak_when_exclude_triggers_background_dig(tmp_path: Path) -> None:
    """[Bug #2] 6 tracks in DB + 3 excluded must not leak _digging_stations; subsequent 2-unheard state must auto-dig."""
    import threading

    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    db = RadioDatabase(db_path)
    for i in range(6):
        db.upsert_track(
            "test_fm",
            TrackEntity(
                track_id=f"trk_leak_{i}",
                canonical_key=f"leak{i}::song{i}",
                artist=f"LeakArtist{i}",
                title=f"Song{i}",
                tags=["folktronica"],
                view_count=10_000,
                video_id=f"VID_LEAK_{i}",
            ),
        )

    app = create_app(db_path=db_path, stations_path=stations_path, model_dir=tmp_path / "m", auto_dig=True)
    dig_event = threading.Event()
    dig_calls = 0

    class TrackingFakeDigger:
        def dig_station(self, station, target_pool_size=60):
            nonlocal dig_calls
            dig_calls += 1
            db.upsert_track(
                station.station_id,
                TrackEntity(
                    track_id=f"trk_replenished_{dig_calls}",
                    canonical_key=f"rep{dig_calls}::song",
                    artist=f"RepArtist{dig_calls}",
                    title="Replenished Song",
                    tags=["folktronica"],
                    view_count=9_000,
                    video_id=f"VID_REP_{dig_calls}",
                ),
            )
            dig_event.set()
            return {"station_id": station.station_id, "added": 1}

    app.state.runtime._digger = TrackingFakeDigger()
    client = TestClient(app)

    # Step 1: 6 unheard in DB, but 3 excluded -> unheard after exclude is 3 (< 5) -> triggers background dig
    res1 = client.get(
        "/queue/next",
        params={"station_id": "test_fm", "count": 3, "exclude": "trk_leak_0,trk_leak_1,trk_leak_2"},
    )
    assert res1.status_code == 200
    assert dig_event.wait(timeout=2.0), "First background dig should complete"
    # Wait briefly for finally block to clear _digging_stations
    for _ in range(20):
        if "test_fm" not in app.state.runtime._digging_stations:
            break
        threading.Event().wait(0.02)
    assert "test_fm" not in app.state.runtime._digging_stations, "_digging_stations leaked 'test_fm'!"

    # Step 2: Mark all but 2 tracks as heard, clear event, and verify a new background dig still launches
    dig_event.clear()
    before_calls = dig_calls
    for i in range(5):
        client.post(
            "/feedback",
            json={"station_id": "test_fm", "track_id": f"trk_leak_{i}", "event_type": "COMPLETE"},
        )

    res2 = client.get("/queue/next", params={"station_id": "test_fm", "count": 3})
    assert res2.status_code == 200
    assert dig_event.wait(timeout=2.0), "Subsequent background dig failed to launch due to leaked flag"
    assert dig_calls > before_calls


def test_feedback_endpoint_rolls_back_db_row_when_bandit_update_raises(tmp_path: Path, monkeypatch) -> None:
    """[Bug #3] If bandit.update raises inside POST /feedback, the inserted feedback_events row must be deleted."""
    import pytest

    db_path = tmp_path / "radio.db"
    stations_path = tmp_path / "stations.yaml"
    _write_station_yaml(stations_path)

    db = RadioDatabase(db_path)
    db.upsert_track(
        "test_fm",
        TrackEntity(
            track_id="trk_fb_rb",
            canonical_key="artist::rb song",
            artist="ArtistRB",
            title="RB Song",
            tags=["folktronica"],
            view_count=10_000,
            video_id="VID_RB_1",
        ),
    )

    app = create_app(db_path=db_path, stations_path=stations_path, model_dir=tmp_path / "m", auto_dig=False)
    bandit = app.state.runtime.bandits["test_fm"]

    def _failing_update(*args, **kwargs):
        raise OSError("Simulated bandit persistence failure")

    monkeypatch.setattr(bandit, "update", _failing_update)
    client = TestClient(app, raise_server_exceptions=True)

    with pytest.raises(OSError, match="Simulated bandit persistence failure"):
        client.post(
            "/feedback",
            json={
                "station_id": "test_fm",
                "track_id": "trk_fb_rb",
                "video_id": "VID_RB_1",
                "event_type": "GOOD",
            },
        )

    assert db.count_feedback("test_fm") == 0, "Orphan feedback_events row remained after bandit.update failure"
    assert db.get_recently_heard_track_ids("test_fm") == set()
