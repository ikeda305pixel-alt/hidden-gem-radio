"""Tests for frequency x recency YouTube history weighting, 5-station routing, and known-track skipping."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from src.db import RadioDatabase
from src.digger.history_importer import (
    classify_entry_to_station,
    compute_recency_frequency_weight,
    generate_stations_yaml,
)
from src.models import TrackEntity
from src.radio_server import create_app


def test_recency_frequency_weight_formula_exact() -> None:
    assert abs(compute_recency_frequency_weight(10, 0.0) - math.log1p(10)) < 1e-9
    expected = math.log1p(5) * math.exp(-60.0 / 120.0)
    assert abs(compute_recency_frequency_weight(5, 60.0) - expected) < 1e-9
    # Recent heavy rotation outweighs stale plays
    assert compute_recency_frequency_weight(8, 5.0) > compute_recency_frequency_weight(8, 200.0)


def test_classify_entry_routes_all_5_stations_accurately() -> None:
    assert classify_entry_to_station("BAD HOP", "Hood Gospel") == "japan_hiphop_street"
    assert classify_entry_to_station("ZORN", "My Life") == "japan_hiphop_street"
    assert classify_entry_to_station("bbno$", "edamame") == "us_hiphop_bouncy"
    assert classify_entry_to_station("Avicii", "Gonna Love Ya") == "edm_dance_party"
    assert classify_entry_to_station("Chrystal", "The Days (NOTION Remix)") == "edm_dance_party"
    assert classify_entry_to_station("Juice WRLD", "Lucid Dreams") == "melodic_rap_alt"
    assert classify_entry_to_station("Lil Peep", "Better Off (Dying)") == "melodic_rap_alt"
    assert classify_entry_to_station("Royel Otis", "Oysters In My Pocket") == "pes_passion_groove"
    assert classify_entry_to_station("L'Impératrice", "Agitations tropicales") == "parisian_chic_disco"
    assert classify_entry_to_station("imase", "NIGHT DANCER") == "jpop_citypop_groove"


def test_weighted_history_generates_5_stations_and_skips_heavy_rotation_originals(tmp_path: Path) -> None:
    ref_now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
    recent_ts = (ref_now - timedelta(days=3)).isoformat()
    old_ts = (ref_now - timedelta(days=240)).isoformat()

    entries = [
        # Recent heavy-rotation Japanese hip-hop (4 plays)
        *[
            {
                "header": "YouTube Music",
                "title": "Watched Hood Gospel",
                "titleUrl": "https://music.youtube.com/watch?v=KNOWN_BADHOP_1",
                "subtitles": [{"name": "BAD HOP - Topic"}],
                "time": recent_ts,
            }
            for _ in range(4)
        ],
        # Older Japanese hip-hop (1 play, 240 days ago) -> lower weight than BAD HOP
        {
            "header": "YouTube Music",
            "title": "Watched My Life",
            "titleUrl": "https://music.youtube.com/watch?v=KNOWN_ZORN_1",
            "subtitles": [{"name": "ZORN - Topic"}],
            "time": old_ts,
        },
        # EDM remix
        {
            "header": "YouTube",
            "title": "Watched Chrystal - The Days (NOTION Remix) (Official Audio)",
            "titleUrl": "https://www.youtube.com/watch?v=KNOWN_EDM_1",
            "subtitles": [{"name": "NOTION"}],
            "time": recent_ts,
        },
        # US Hip-Hop
        {
            "header": "YouTube",
            "title": "Watched bbno$ - edamame (Official Video)",
            "titleUrl": "https://www.youtube.com/watch?v=KNOWN_USHH_1",
            "subtitles": [{"name": "bbno$"}],
            "time": recent_ts,
        },
        # Melodic Rap
        {
            "header": "YouTube Music",
            "title": "Watched Lucid Dreams",
            "titleUrl": "https://music.youtube.com/watch?v=KNOWN_EMO_1",
            "subtitles": [{"name": "Juice WRLD - Topic"}],
            "time": recent_ts,
        },
        # Indie Groove
        {
            "header": "YouTube Music",
            "title": "Watched Oysters In My Pocket",
            "titleUrl": "https://music.youtube.com/watch?v=KNOWN_INDIE_1",
            "subtitles": [{"name": "Royel Otis - Topic"}],
            "time": recent_ts,
        },
    ]

    takeout_path = tmp_path / "watch-history.json"
    takeout_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    stations_yaml = tmp_path / "stations.yaml"

    summary = generate_stations_yaml(
        output_path=stations_yaml,
        browser_auth_path=tmp_path / "missing_browser.json",
        takeout_path=takeout_path,
        num_stations=5,
        offline=True,
        reference_time=ref_now,
    )
    assert summary["station_count"] == 5
    assert summary["fallback_used"] is False
    assert summary["known_video_ids_count"] == 6

    raw = yaml.safe_load(stations_yaml.read_text(encoding="utf-8"))
    assert "KNOWN_BADHOP_1" in raw["history_index"]["known_video_ids"]
    by_id = {s["station_id"]: s for s in raw["stations"]}

    jp_seeds = by_id["japan_hiphop_street"]["weighted_seeds"]
    assert jp_seeds[0]["artist"] == "BAD HOP"
    assert jp_seeds[0]["weight"] == 1.0
    assert jp_seeds[1]["artist"] == "ZORN"
    assert 0.0 < jp_seeds[1]["weight"] < 0.5

    # Verify RadioRuntime skips the known heavy-rotation song itself and serves hidden-gem neighbor instead
    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=stations_yaml,
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    db: RadioDatabase = app.state.runtime.db
    db.upsert_track(
        "japan_hiphop_street",
        TrackEntity.build(
            artist="BAD HOP",
            title="Hood Gospel",
            view_count=50_000,
            video_id="KNOWN_BADHOP_1",
            source="watch_playlist",
        ),
    )
    db.upsert_track(
        "japan_hiphop_street",
        TrackEntity.build(
            artist="Underground JP MC",
            title="Tokyo Midnight Groove",
            view_count=18_000,
            video_id="HIDDEN_GEM_JP_1",
            source="watch_playlist",
            seed_artist="BAD HOP",
            seed_archetype="melancholic_minor_loop",
        ),
    )

    client = TestClient(app)
    res = client.get("/api/stations/japan_hiphop_street/batch", params={"count": 3})
    assert res.status_code == 200
    vids = [item["video_id"] for item in res.json()["items"]]
    assert "KNOWN_BADHOP_1" not in vids
    assert "HIDDEN_GEM_JP_1" in vids
