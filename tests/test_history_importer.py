"""History importer tests (evidence_environment: FIXTURE, counts_toward_l5_certification: false)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from src.digger.history_importer import (
    build_stations_from_history,
    generate_stations_yaml,
    parse_takeout_watch_history,
    parse_ytmusic_history_and_likes,
)


def test_parse_takeout_watch_history_extracts_music_and_ignores_non_music(tmp_path: Path) -> None:
    """Google Takeout watch-history.json should extract music videos (EN/JP) and drop non-music."""
    takeout_path = tmp_path / "watch-history.json"
    entries = [
        {
            "header": "YouTube Music",
            "title": "Watched Oooo",
            "titleUrl": "https://music.youtube.com/watch?v=abc12345678",
            "subtitles": [{"name": "Oberhofer - Topic"}],
            "time": "2026-11-20T10:00:00Z",
        },
        {
            "header": "YouTube",
            "title": "Avicii - Silhouettes (Official Music Video) を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=def12345678",
            "subtitles": [{"name": "AviciiOfficialVEVO"}],
            "time": "2026-11-21T10:00:00Z",
        },
        {
            "header": "YouTube",
            "title": "Watched Four Tet - Lush (Official Audio)",
            "titleUrl": "https://www.youtube.com/watch?v=ghi12345678",
            "subtitles": [{"name": "Four Tet - Topic"}],
            "time": "2026-11-22T10:00:00Z",
        },
        {
            "header": "YouTube",
            "title": "Watched Python FastAPI Tutorial 2026 - Full Course",
            "titleUrl": "https://www.youtube.com/watch?v=nonmusic001",
            "subtitles": [{"name": "Code Academy"}],
            "time": "2026-11-22T12:00:00Z",
        },
        {
            "header": "YouTube",
            "title": "今日のニュース速報 を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=nonmusic002",
            "subtitles": [{"name": "Daily News Channel"}],
            "time": "2026-11-22T13:00:00Z",
        },
    ]
    takeout_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    records = parse_takeout_watch_history(takeout_path)
    artists = {r["artist"] for r in records}
    assert "Oberhofer" in artists
    assert "Avicii" in artists
    assert "Four Tet" in artists
    assert "Code Academy" not in artists
    assert "Daily News Channel" not in artists


def test_parse_ytmusic_auth_history_and_likes() -> None:
    """ytmusicapi get_history() + get_liked_songs() should combine and weight liked tracks higher."""
    mock_yt = MagicMock()
    mock_yt.get_history.return_value = [
        {"videoId": "v1", "title": "Song A", "artists": [{"name": "Bonobo"}]},
        {"videoId": "v2", "title": "Song B", "artists": [{"name": "Four Tet"}]},
    ]
    mock_yt.get_liked_songs.return_value = {
        "tracks": [
            {"videoId": "v3", "title": "Song C", "artists": [{"name": "Caribou"}]},
            {"videoId": "v1", "title": "Song A", "artists": [{"name": "Bonobo"}]},
        ]
    }

    records = parse_ytmusic_history_and_likes(mock_yt, liked_limit=50)
    artists = [r["artist"] for r in records]
    assert "Bonobo" in artists
    assert "Four Tet" in artists
    assert "Caribou" in artists
    liked_weights = [r["weight"] for r in records if r["source"] == "ytmusic_liked"]
    history_weights = [r["weight"] for r in records if r["source"] == "ytmusic_history"]
    assert max(liked_weights) > max(history_weights)


def test_generate_stations_yaml_falls_back_when_no_sources_exist(tmp_path: Path) -> None:
    """When neither browser.json nor watch-history.json exists, fallback to spec example seeds."""
    output_yaml = tmp_path / "stations.yaml"
    result = generate_stations_yaml(
        output_path=output_yaml,
        browser_auth_path=tmp_path / "missing_browser.json",
        takeout_path=tmp_path / "missing_watch-history.json",
        num_stations=3,
        offline=True,
    )
    assert output_yaml.exists()
    assert result["fallback_used"] is True
    loaded = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
    stations = loaded["stations"]
    assert 2 <= len(stations) <= 4
    all_seed_artists = {a for s in stations for a in s["seed_artists"]}
    all_seed_tags = {t for s in stations for t in s["seed_tags"]}
    assert "Avicii" in all_seed_artists
    assert "Oberhofer" in all_seed_artists
    assert "melodic progressive house" in all_seed_tags
    assert "folktronica" in all_seed_tags


def test_build_stations_from_history_generates_2_to_4_stations() -> None:
    """Clustered history records should yield 2 to 4 well-formed stations."""
    records = [
        {"artist": "Avicii", "title": "Levels", "weight": 3.0, "source": "takeout"},
        {"artist": "Kygo", "title": "Firestone", "weight": 2.0, "source": "takeout"},
        {"artist": "Oberhofer", "title": "Oooo", "weight": 3.0, "source": "takeout"},
        {"artist": "Four Tet", "title": "Lush", "weight": 2.5, "source": "takeout"},
        {"artist": "Bonobo", "title": "Cirrus", "weight": 2.5, "source": "takeout"},
        {"artist": "Phoebe Bridgers", "title": "Kyoto", "weight": 2.0, "source": "takeout"},
    ]
    fake_tag_lookup = {
        "Avicii": ["melodic progressive house", "electronic", "dance"],
        "Kygo": ["tropical house", "melodic progressive house", "electronic"],
        "Oberhofer": ["indietronica", "indie pop", "dream pop"],
        "Four Tet": ["folktronica", "idm", "indietronica"],
        "Bonobo": ["downtempo", "folktronica", "electronic"],
        "Phoebe Bridgers": ["indie folk", "indie rock", "singer-songwriter"],
    }
    stations = build_stations_from_history(
        records,
        num_stations=3,
        tag_resolver=lambda artist: fake_tag_lookup.get(artist, ["indie"]),
    )
    assert 2 <= len(stations) <= 4
    for st in stations:
        assert st.station_id
        assert st.name
        assert len(st.seed_tags) >= 1
        assert len(st.seed_artists) >= 1
        assert st.max_popularity_views > 0
        assert st.exploration_rate > 0


def test_japanese_takeout_filters_ads_tutorials_cooking_and_normalizes_topic_and_brackets(tmp_path: Path) -> None:
    """[M6] Japanese Takeout must reject tutorials/cooking/ads and normalize '- トピック' and '「...」'."""
    takeout_path = tmp_path / "watch-history.json"
    entries = [
        {
            "header": "YouTube",
            "title": "Python入門 - 第1回 変数とは を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=non1",
            "subtitles": [{"name": "プログラミング教室"}],
        },
        {
            "header": "YouTube",
            "title": "【料理】簡単パスタ - 10分レシピ を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=non2",
            "subtitles": [{"name": "Cooking Channel"}],
        },
        {
            "header": "YouTube",
            "title": "Some Brand Spring Sale を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=ad01",
            "subtitles": [{"name": "BrandOfficial"}],
            "details": [{"name": "From Google Ads"}],
        },
        {
            "header": "YouTube",
            "title": "Earplugs を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=mus1",
            "subtitles": [{"name": "Oberhofer - トピック"}],
        },
        {
            "header": "YouTube",
            "title": "「Levels」を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=mus2",
            "subtitles": [{"name": "Avicii - Topic"}],
        },
    ]
    takeout_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")

    records = parse_takeout_watch_history(takeout_path)
    by_artist = {r["artist"]: r["title"] for r in records}
    assert by_artist == {"Oberhofer": "Earplugs", "Avicii": "Levels"}


def test_invalid_browser_auth_exposes_warnings_in_summary_and_api(tmp_path: Path) -> None:
    """[M7] Expired/invalid browser.json must surface clear warnings in summary and /stations/import-history."""
    from fastapi.testclient import TestClient

    from src.radio_server import create_app

    bad_browser = tmp_path / "browser.json"
    bad_browser.write_text(json.dumps({"cookie": "SAPISID=expired"}), encoding="utf-8")
    out_yaml = tmp_path / "stations.yaml"

    summary = generate_stations_yaml(
        output_path=out_yaml,
        browser_auth_path=bad_browser,
        takeout_path=tmp_path / "missing.json",
        offline=True,
    )
    assert summary["fallback_used"] is True
    assert summary["warnings"], "Expected non-empty warnings for invalid/expired browser.json"
    assert any("browser.json" in w or "ytmusicapi" in w.lower() for w in summary["warnings"])

    app = create_app(
        db_path=tmp_path / "r.db",
        stations_path=out_yaml,
        model_dir=tmp_path / "m",
        auto_dig=False,
    )
    client = TestClient(app)
    res = client.post(
        "/stations/import-history",
        json={
            "browser_auth_path": str(bad_browser),
            "takeout_path": str(tmp_path / "missing.json"),
            "num_stations": 3,
            "offline": True,
        },
    )
    assert res.status_code == 200
    assert res.json()["warnings"], "API response must expose auth warnings"
