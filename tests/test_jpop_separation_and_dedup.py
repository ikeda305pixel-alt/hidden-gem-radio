"""Regression tests for 6-station J-Pop separation, Known-Artist suppression, artist cap, and bilingual dedup.

evidence_environment: "FIXTURE"
counts_toward_l5_certification: false
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from fastapi.testclient import TestClient

from src.adapters.youtube import YouTubeMusicAdapter
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
    load_familiar_artists,
    normalize_artist,
    normalize_track_key,
)
from src.radio_server import create_app


def test_bilingual_furigana_and_romaji_suffix_deduplication(tmp_path: Path) -> None:
    """[Req 3 & 4.1] 'サザンオールスターズ - 栞のテーマ' matches 'SOUTHERN ALL STARS - 栞(しおり)のテーマ - Shiori no Theme'."""
    # Artist JP <-> EN canonical alias matching
    assert normalize_artist("サザンオールスターズ") == normalize_artist("SOUTHERN ALL STARS")
    assert normalize_artist("桑田佳祐") == normalize_artist("Keisuke Kuwata")
    assert normalize_artist("山下達郎") == normalize_artist("Tatsuro Yamashita")
    assert normalize_artist("藤井風") == normalize_artist("Fujii Kaze")
    assert normalize_artist("星野源") == normalize_artist("Gen Hoshino")
    assert normalize_artist("宇多田ヒカル") == normalize_artist("Hikaru Utada")

    # Furigana + Romaji suffix canonical key matching
    key_jp = normalize_track_key("サザンオールスターズ", "栞のテーマ")
    key_atv_dash = normalize_track_key("SOUTHERN ALL STARS", "栞(しおり)のテーマ - Shiori no Theme")
    key_atv_fullwidth = normalize_track_key("Southern All Stars", "栞（しおり）のテーマ / Shiori no Theme")
    assert key_jp == key_atv_dash
    assert key_jp == key_atv_fullwidth

    # Western titles with slashes or genre parentheses must NOT be mangled into collisions
    assert normalize_track_key("Radiohead", "Weird Fishes/Arpeggi") != normalize_track_key(
        "Radiohead", "Weird Fishes"
    )
    assert normalize_track_key("Artist", "Bounce (UKG Remix)") != normalize_track_key("Artist", "Bounce")

    # End-to-end: history has Japanese 'サザンオールスターズ - 栞のテーマ', candidate is English ATV with furigana+romaji
    ref_now = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
    takeout_path = tmp_path / "watch-history.json"
    takeout_entries = [
        {
            "header": "YouTube",
            "title": "サザンオールスターズ - 栞のテーマ (Official Video) を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=HIST_SAS_JP1",
            "subtitles": [{"name": "サザンオールスターズ Official"}],
            "time": (ref_now - timedelta(days=5)).isoformat(),
        },
        {
            "header": "YouTube",
            "title": "希望の轍 - Kibou no Wadachi を視聴しました",
            "titleUrl": "https://www.youtube.com/watch?v=HIST_SAS_TOPIC1",
            "subtitles": [{"name": "SOUTHERN ALL STARS - Topic"}],
            "time": (ref_now - timedelta(days=3)).isoformat(),
        },
    ]
    takeout_path.write_text(json.dumps(takeout_entries, ensure_ascii=False), encoding="utf-8")
    stations_yaml = tmp_path / "stations.yaml"

    generate_stations_yaml(
        output_path=stations_yaml,
        browser_auth_path=tmp_path / "missing_browser.json",
        takeout_path=takeout_path,
        num_stations=6,
        offline=True,
        reference_time=ref_now,
    )

    app = create_app(
        db_path=tmp_path / "radio.db",
        stations_path=stations_yaml,
        model_dir=tmp_path / "models",
        auto_dig=False,
    )
    db: RadioDatabase = app.state.runtime.db

    # Even if an existing DB row had a legacy un-normalized canonical_key, it must be caught as known!
    legacy_atv_row = TrackEntity(
        track_id="trk_legacy_sas_shiori",
        canonical_key="southern all stars::栞 しおり のテーマ shiori no theme",
        artist="SOUTHERN ALL STARS",
        title="栞(しおり)のテーマ - Shiori no Theme",
        view_count=250_000,
        video_id="ATV_SAS_SHIORI",
        source="watch_playlist",
    )
    reverse_jp_row = TrackEntity.build(
        artist="サザンオールスターズ",
        title="希望の轍",
        view_count=300_000,
        video_id="JP_SAS_KIBOU",
        source="watch_playlist",
    )
    fresh_citypop_gem = TrackEntity.build(
        artist="流線形",
        title="TOKYO SNIPER",
        view_count=85_000,
        video_id="GEM_CITYPOP_01",
        source="watch_playlist",
    )
    db.upsert_track("jpop_citypop_groove", legacy_atv_row)
    db.upsert_track("jpop_citypop_groove", reverse_jp_row)
    db.upsert_track("jpop_citypop_groove", fresh_citypop_gem)

    client = TestClient(app)
    res = client.get("/api/stations/jpop_citypop_groove/batch", params={"count": 3})
    assert res.status_code == 200
    vids = [item["video_id"] for item in res.json()["items"]]
    assert "ATV_SAS_SHIORI" not in vids
    assert "JP_SAS_KIBOU" not in vids
    assert "GEM_CITYPOP_01" in vids


def test_six_stations_strict_western_indie_vs_jpop_separation(tmp_path: Path) -> None:
    """[Req 1 & 4.2] 6 stations exist; Southern All Stars & Japanese tracks never enter indie_groove_cult."""
    stations = default_theory_stations()
    station_ids = [s.station_id for s in stations]
    assert station_ids == [
        "us_hiphop_bouncy",
        "japan_hiphop_street",
        "edm_dance_party",
        "melodic_rap_alt",
        "indie_groove_cult",
        "jpop_citypop_groove",
    ]

    by_id = {s.station_id: s for s in stations}
    indie = by_id["indie_groove_cult"]
    jpop = by_id["jpop_citypop_groove"]

    # indie_groove_cult must contain zero Japanese artists (e.g. imase, Southern All Stars)
    indie_blob = " ".join([*indie.seed_artists, *indie.spotify_anchor_seeds, *indie.pes_anchors, *indie.emily_anchors]).lower()
    assert "imase" not in indie_blob
    assert "southern all stars" not in indie_blob
    assert jpop.spotify_anchor_seeds

    # History classifier routes JP groove/pop (both Japanese script and English aliases) to jpop_citypop_groove
    assert classify_entry_to_station("SOUTHERN ALL STARS", "TSUNAMI") == "jpop_citypop_groove"
    assert classify_entry_to_station("サザンオールスターズ", "勝手にシンドバッド") == "jpop_citypop_groove"
    assert classify_entry_to_station("Keisuke Kuwata", "波乗りジョニー") == "jpop_citypop_groove"
    assert classify_entry_to_station("Tatsuro Yamashita", "Sparkle") == "jpop_citypop_groove"
    assert classify_entry_to_station("imase", "NIGHT DANCER") == "jpop_citypop_groove"
    assert classify_entry_to_station("Fujii Kaze", "Shinunoga E-Wa") == "jpop_citypop_groove"
    assert classify_entry_to_station("Vaundy", "odoriko") == "jpop_citypop_groove"
    assert classify_entry_to_station("Suchmos", "STAY TUNE") == "jpop_citypop_groove"
    assert classify_entry_to_station("King Gnu", "白日") == "jpop_citypop_groove"
    assert classify_entry_to_station("BAD HOP", "Hood Gospel") == "japan_hiphop_street"
    assert classify_entry_to_station("Royel Otis", "Oysters In My Pocket") == "indie_groove_cult"
    assert classify_entry_to_station("L'Imperatrice", "Matahari") == "indie_groove_cult"

    # Stage 1 dig on indie_groove_cult must block Japanese / Southern All Stars leaks even if returned by YTMusic
    class MixedWatchPlaylistYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
            _ = (query, filter, limit)
            return [
                {
                    "videoId": "SEED_ROYEL_1",
                    "title": "Oysters In My Pocket",
                    "resultType": "song",
                    "videoType": "MUSIC_VIDEO_TYPE_ATV",
                    "artists": [{"name": "Royel Otis", "id": "UC_ROYEL"}],
                    "album": {"name": "Sofa Kings"},
                    "duration_seconds": 165,
                }
            ]

        def get_watch_playlist(self, videoId: str, radio: bool = True, limit: int = 25) -> dict[str, Any]:
            _ = (videoId, radio, limit)
            return {
                "tracks": [
                    {
                        "videoId": "LEAK_SAS_EN",
                        "title": "Shiori no Theme",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "SOUTHERN ALL STARS", "id": "UC_SAS"}],
                        "album": {"name": "Ninkimono"},
                        "length": "4:30",
                        "views": "200K",
                    },
                    {
                        "videoId": "LEAK_JP_TITLE",
                        "title": "真夜中のドア",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Miki Matsubara", "id": "UC_MIKI"}],
                        "album": {"name": "Pocket Park"},
                        "length": "5:12",
                        "views": "400K",
                    },
                    {
                        "videoId": "KEEP_FRENCH_1",
                        "title": "Agitations tropicales",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "L'Imperatrice", "id": "UC_LIMP"}],
                        "album": {"name": "Matahari"},
                        "length": "4:08",
                        "views": "350K",
                    },
                ]
            }

        def get_artist(self, channel_id: str) -> dict[str, Any]:
            _ = channel_id
            return {}

    db = RadioDatabase(tmp_path / "indie.db")
    digger = CandidatePoolDigger(
        db=db,
        yt_adapter=YouTubeMusicAdapter(client=MixedWatchPlaylistYT(), check_embeddable=False),
    )
    contaminated_indie = StationConfig(
        station_id="indie_groove_cult",
        name="Indie Groove, PES Cult & Parisian Chic",
        seed_tags=["indie pop", "french pop"],
        seed_artists=["Royel Otis", "Southern All Stars"],  # Even if polluted seed was present, must strip!
        spotify_anchor_seeds=["Royel Otis - Oysters In My Pocket", "サザンオールスターズ - 希望の轍"],
        max_popularity_views=500_000_000,
    )
    digger.dig_station(contaminated_indie, target_pool_size=20)
    cands = db.get_eligible_candidates("indie_groove_cult", 500_000_000)
    cand_vids = {c.video_id for c in cands}
    assert "LEAK_SAS_EN" not in cand_vids
    assert "LEAK_JP_TITLE" not in cand_vids
    assert "KEEP_FRENCH_1" in cand_vids


def test_known_familiar_artists_suppression_and_max_2_tracks_per_artist_cap(tmp_path: Path) -> None:
    """[Req 2 & 4.3] Heavy-rotation/seed artists' self-tracks are suppressed; any single artist is capped at <=2 tracks."""
    ref_now = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)
    # Southern All Stars watched 5 times across 3 tracks -> must become known_familiar_artist
    entries = [
        {
            "header": "YouTube Music",
            "title": f"Watched {song}",
            "titleUrl": f"https://music.youtube.com/watch?v=SAS_HIST_{idx}",
            "subtitles": [{"name": "サザンオールスターズ - Topic"}],
            "time": (ref_now - timedelta(days=idx + 1)).isoformat(),
        }
        for idx, song in enumerate(["TSUNAMI", "勝手にシンドバッド", "真夏の果実", "TSUNAMI", "涙のキッス"])
    ]
    takeout_path = tmp_path / "watch-history.json"
    takeout_path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    stations_yaml = tmp_path / "stations.yaml"

    summary = generate_stations_yaml(
        output_path=stations_yaml,
        browser_auth_path=tmp_path / "missing.json",
        takeout_path=takeout_path,
        num_stations=6,
        offline=True,
        reference_time=ref_now,
    )
    assert summary["station_count"] == 6
    familiar = load_familiar_artists(stations_yaml)
    assert normalize_artist("SOUTHERN ALL STARS") in familiar

    raw_yaml = yaml.safe_load(stations_yaml.read_text(encoding="utf-8"))
    jpop_cfg = StationConfig.from_dict(
        next(s for s in raw_yaml["stations"] if s["station_id"] == "jpop_citypop_groove")
    )

    class CityPopGraphYT:
        def search(self, query: str, filter: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
            _ = (query, filter, limit)
            return [
                {
                    "videoId": "SEED_SAS_ANCHOR",
                    "title": "TSUNAMI",
                    "resultType": "song",
                    "videoType": "MUSIC_VIDEO_TYPE_ATV",
                    "artists": [{"name": "SOUTHERN ALL STARS", "id": "UC_SAS_MAIN"}],
                    "album": {"name": "Southern All Stars"},
                    "duration_seconds": 290,
                }
            ]

        def get_watch_playlist(self, videoId: str, radio: bool = True, limit: int = 25) -> dict[str, Any]:
            _ = (videoId, radio, limit)
            return {
                "tracks": [
                    # Unheard songs by Southern All Stars themselves (both EN and JP names) -> MUST BE SUPPRESSED!
                    {
                        "videoId": "SAS_UNHEARD_1",
                        "title": "HOTEL PACIFIC",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "SOUTHERN ALL STARS", "id": "UC_SAS_MAIN"}],
                        "album": {"name": " Killer Street"},
                        "length": "4:15",
                        "views": "800K",
                    },
                    {
                        "videoId": "SAS_UNHEARD_2",
                        "title": "ミス・ブランニュー・デイ",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "サザンオールスターズ", "id": "UC_SAS_MAIN"}],
                        "album": {"name": "人気者で行こう"},
                        "length": "4:50",
                        "views": "900K",
                    },
                    # 4 tracks by the same fresh City Pop artist " Piper " -> MUST BE CAPPED AT MAX 2!
                    *[
                        {
                            "videoId": f"PIPER_GEM_{i}",
                            "title": f"Summer Breeze #{i}",
                            "videoType": "MUSIC_VIDEO_TYPE_ATV",
                            "artists": [{"name": "Piper", "id": "UC_PIPER"}],
                            "album": {"name": "Summer Breeze"},
                            "length": "3:55",
                            "views": f"{120 + i * 10}K",
                        }
                        for i in range(1, 5)
                    ],
                    # Another distinct unknown artist -> MUST BE INCLUDED
                    {
                        "videoId": "KADOMATSU_NEIGHBOR_1",
                        "title": "Midnight Cruisin'",
                        "videoType": "MUSIC_VIDEO_TYPE_ATV",
                        "artists": [{"name": "Kingyo", "id": "UC_KINGYO"}],
                        "album": {"name": "City Lights"},
                        "length": "4:10",
                        "views": "95K",
                    },
                ]
            }

        def get_artist(self, channel_id: str) -> dict[str, Any]:
            _ = channel_id
            return {}

    db = RadioDatabase(tmp_path / "jpop.db")
    digger = CandidatePoolDigger(
        db=db,
        yt_adapter=YouTubeMusicAdapter(client=CityPopGraphYT(), check_embeddable=False),
    )
    dig_res = digger.dig_station(
        jpop_cfg,
        target_pool_size=20,
        known_familiar_artists=familiar,
    )
    assert dig_res.get("skipped_familiar_artist", 0) >= 2

    cands = db.get_eligible_candidates("jpop_citypop_groove", 500_000_000)
    cand_artists = [normalize_artist(c.artist) for c in cands]
    cand_vids = {c.video_id for c in cands}

    # Zero Southern All Stars self-tracks in candidate pool
    assert normalize_artist("SOUTHERN ALL STARS") not in cand_artists
    assert "SAS_UNHEARD_1" not in cand_vids
    assert "SAS_UNHEARD_2" not in cand_vids

    # Piper capped strictly at <= 2 tracks even if dig_station is invoked a second time!
    assert cand_artists.count(normalize_artist("Piper")) == 2
    assert "KADOMATSU_NEIGHBOR_1" in cand_vids

    digger.dig_station(jpop_cfg, target_pool_size=20, known_familiar_artists=familiar)
    cands_after_second_dig = db.get_eligible_candidates("jpop_citypop_groove", 500_000_000)
    artists_after = [normalize_artist(c.artist) for c in cands_after_second_dig]
    assert artists_after.count(normalize_artist("Piper")) <= 2
