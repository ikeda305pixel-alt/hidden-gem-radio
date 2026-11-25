"""End-to-end API flow tests (evidence_environment: FIXTURE, counts_toward_l5_certification: false)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.db import RadioDatabase
from src.models import TrackEntity
from src.radio_server import create_app


def test_e2e_stations_queue_prefetch_and_feedback_persistence(tmp_path: Path) -> None:
    """Verify full loop: /stations -> /queue/next (N=3) -> /feedback -> DB & model updated."""
    db_path = tmp_path / "e2e.db"
    stations_path = tmp_path / "stations.yaml"
    model_dir = tmp_path / "models"

    stations_path.write_text(
        """stations:
  - station_id: gem_radio_1
    name: Melodic & Folktronica Gems
    seed_tags:
      - melodic progressive house
      - folktronica
      - indietronica
    seed_artists:
      - Avicii
      - Oberhofer
    max_popularity_views: 500000
    exploration_rate: 1.0
""",
        encoding="utf-8",
    )

    db = RadioDatabase(db_path)
    for i in range(6):
        db.upsert_track(
            "gem_radio_1",
            TrackEntity(
                track_id=f"trk_{i}",
                canonical_key=f"artist {i}::gem track {i}",
                artist=f"Artist {i}",
                title=f"Gem Track {i}",
                tags=["folktronica" if i % 2 == 0 else "indietronica"],
                release_year=2015 + i,
                view_count=15_000 + i * 20_000,
                video_id=f"VID_E2E_{i:03d}",
                isrc=f"USGEM260000{i}",
            ),
        )

    app = create_app(
        db_path=db_path,
        stations_path=stations_path,
        model_dir=model_dir,
        auto_dig=False,
    )
    client = TestClient(app)

    # 1. GET / serves single HTML UI
    html_res = client.get("/")
    assert html_res.status_code == 200
    assert "YouTube" in html_res.text or "youtube.com/iframe_api" in html_res.text

    # 2. GET /stations
    st_res = client.get("/stations")
    assert st_res.status_code == 200
    stations = st_res.json()["stations"]
    assert len(stations) == 1
    assert stations[0]["station_id"] == "gem_radio_1"
    assert stations[0]["candidate_count"] == 6

    # 3. GET /queue/next with N=3 prefetch
    q_res = client.get("/queue/next", params={"station_id": "gem_radio_1", "count": 3})
    assert q_res.status_code == 200
    q_items = q_res.json()["items"]
    assert len(q_items) == 3
    first = q_items[0]
    assert first["reason"]["mode"] in {"探索", "活用"}
    assert "hidden_level" in first["reason"]

    # 4. POST /feedback
    fb_res = client.post(
        "/feedback",
        json={
            "station_id": "gem_radio_1",
            "track_id": first["track_id"],
            "video_id": first["video_id"],
            "event_type": "GOOD",
            "listen_seconds": 45.0,
            "duration_seconds": 210.0,
        },
    )
    assert fb_res.status_code == 200
    fb_body = fb_res.json()
    assert fb_body["reward"] == 1.0
    assert fb_body["model_updates"] == 1

    # Verify DB & persisted bandit weights
    assert db.count_feedback("gem_radio_1") == 1
    assert (model_dir / "bandit_gem_radio_1.pkl").exists()
