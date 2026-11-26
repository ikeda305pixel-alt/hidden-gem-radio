"""Tests for 30-day unplayable blacklist (no BAD pollution), Good playlists, YouTube sync, and UI checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase, build_youtube_watch_videos_urls
from src.models import TrackEntity
from src.radio_server import create_app


def test_unplayable_endpoint_blacklists_30d_without_polluting_bad_learning(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    db: RadioDatabase = app.state.runtime.db
    db.upsert_track(
        "edm_dance_party",
        TrackEntity.build(
            artist="SameArtist",
            title="Stalled Track",
            view_count=12_000,
            video_id="STALLED_VID_01",
        ),
    )
    db.upsert_track(
        "edm_dance_party",
        TrackEntity.build(
            artist="SameArtist",
            title="Playable Track By Same Artist",
            view_count=15_000,
            video_id="PLAYABLE_VID_02",
        ),
    )

    client = TestClient(app)
    res = client.post(
        "/api/tracks/STALLED_VID_01/unplayable",
        json={"station_id": "edm_dance_party", "reason": "stall_watchdog_7s", "ttl_days": 30},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["bad_learning_polluted"] is False
    assert body["blacklisted_days"] == 30
    assert body["replacement"]["video_id"] == "PLAYABLE_VID_02"

    # Zero feedback events & zero bandit updates (BAD artist block was NOT triggered)
    assert db.count_feedback("edm_dance_party") == 0
    assert app.state.runtime.bandits["edm_dance_party"].n_updates == 0
    assert "STALLED_VID_01" in db.get_blacklisted_video_ids()

    # Subsequent batch must exclude STALLED_VID_01 but KEEP PLAYABLE_VID_02 (same artist not penalized!)
    batch = client.get("/api/stations/edm_dance_party/batch", params={"count": 3}).json()
    vids = [item["video_id"] for item in batch["items"]]
    assert "STALLED_VID_01" not in vids
    assert "PLAYABLE_VID_02" in vids


def test_playlists_crud_good_save_and_50_chunk_youtube_watch_urls(tmp_path: Path) -> None:
    # 120 unique video IDs -> 3 URLs (50 + 50 + 20)
    urls = build_youtube_watch_videos_urls([f"VID_{i:03d}" for i in range(120)], chunk_size=50)
    assert len(urls) == 3
    assert urls[0].startswith("https://www.youtube.com/watch_videos?video_ids=")
    assert len(urls[0].split("video_ids=")[1].split(",")) == 50
    assert len(urls[1].split("video_ids=")[1].split(",")) == 50
    assert len(urls[2].split("video_ids=")[1].split(",")) == 20

    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    db: RadioDatabase = app.state.runtime.db
    db.upsert_track(
        "pes_passion_groove",
        TrackEntity.build(
            artist="Royel Otis",
            title="Hidden Gem Groove",
            view_count=22_000,
            video_id="GEM_YT_001",
        ),
    )

    client = TestClient(app)

    # 4 default playlists exist
    pls = client.get("/api/playlists").json()["playlists"]
    names = {p["name"] for p in pls}
    assert {"Good お気に入り", "US & JP Hip-Hop", "Party & Drive", "Groove & Chill"}.issubset(names)

    # Create a custom playlist
    created = client.post("/api/playlists", json={"name": "Late Night Drive"}).json()["playlist"]
    custom_pid = created["playlist_id"]

    # GOOD feedback saves directly into selected playlist
    fb = client.post(
        "/feedback",
        json={
            "station_id": "pes_passion_groove",
            "video_id": "GEM_YT_001",
            "event_type": "GOOD",
            "playlist_id": custom_pid,
        },
    )
    assert fb.status_code == 200
    assert fb.json()["playlist_saved"]["playlist_id"] == custom_pid

    # Multi-add same track to another playlist
    add_res = client.post(
        "/api/playlists/pl_groove_chill/tracks",
        json={
            "video_id": "GEM_YT_001",
            "artist": "Royel Otis",
            "title": "Hidden Gem Groove",
            "station_id": "pes_passion_groove",
        },
    )
    assert add_res.status_code == 200

    detail = client.get(f"/api/playlists/{custom_pid}/tracks").json()
    assert detail["track_count"] == 1
    assert detail["youtube_watch_urls"] == ["https://www.youtube.com/watch_videos?video_ids=GEM_YT_001"]


def test_youtube_hybrid_playlist_sync_fallback_and_authenticated_paths(tmp_path: Path) -> None:
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=tmp_path / "stations.yaml",
        model_dir=tmp_path / "models",
        browser_auth_path=tmp_path / "missing_auth.json",
        auto_dig=False,
    )
    # Force an explicitly unauthenticated adapter (isolated from repo-level data/browser.json if any)
    app.state.runtime._yt_adapter = YouTubeMusicAdapter(
        browser_auth_path=tmp_path / "none.json",
        project_root=tmp_path,
    )
    client = TestClient(app)

    # Unauthenticated: safe fallback without error
    unauth = client.get("/api/youtube/playlists").json()
    assert unauth["authenticated"] is False
    assert unauth["playlists"] == []

    sync_unauth = client.post(
        "/api/youtube/sync",
        json={"video_id": "VID_ABC", "youtube_playlist_id": "PL_YT_1"},
    ).json()
    assert sync_unauth["authenticated"] is False
    assert sync_unauth["liked"] is False

    # Authenticated (mocked ytmusicapi client): lists playlists + calls rate_song('LIKE') & add_playlist_items
    mock_yt = MagicMock()
    mock_yt.get_library_playlists.return_value = [{"playlistId": "PL_YT_1", "title": "My Gems", "count": "12"}]
    app.state.runtime._yt_adapter = YouTubeMusicAdapter(client=mock_yt)

    auth_pls = client.get("/api/youtube/playlists").json()
    assert auth_pls["authenticated"] is True
    assert auth_pls["playlists"][0]["playlist_id"] == "PL_YT_1"

    sync_ok = client.post(
        "/api/youtube/sync",
        json={"video_id": "VID_ABC", "youtube_playlist_id": "PL_YT_1"},
    ).json()
    assert sync_ok["authenticated"] is True
    assert sync_ok["liked"] is True
    assert sync_ok["playlist_added"] is True
    mock_yt.rate_song.assert_called_once_with("VID_ABC", "LIKE")
    mock_yt.add_playlist_items.assert_called_once_with("PL_YT_1", ["VID_ABC"], duplicates=False)


def test_web_index_contains_watchdog_media_session_mobile_bar_and_dna_ui() -> None:
    html = Path("web/index.html").read_text(encoding="utf-8")
    assert "PLAYBACK_WATCHDOG_MS = 7000" in html
    assert "stall_watchdog_7s" in html
    assert "/unplayable" in html
    assert "navigator.mediaSession" in html
    assert "while (state.prefetchPromise)" in html
    assert "controls-bar" in html
    assert "YouTube でプレイリストを開く" in html
    assert "音楽理論DNAバー" in html
