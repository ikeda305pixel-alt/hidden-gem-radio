"""FastAPI local radio server connecting Candidate Pool, MABWiser Bandit, and Web UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase
from src.digger.history_importer import generate_stations_yaml
from src.digger.pool import CandidatePoolDigger
from src.models import StationConfig, load_stations_yaml, normalize_artist
from src.recommender.bandit import ContextualBanditRecommender

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FeedbackRequest(BaseModel):
    station_id: str
    track_id: str | None = None
    video_id: str | None = None
    event_type: str = Field(..., description="GOOD | BAD | COMPLETE | MID_SKIP | EARLY_SKIP | SKIP | ENDED | UNPLAYABLE")
    listen_seconds: float | None = None
    duration_seconds: float | None = None
    yt_error_code: int | None = None


class ImportHistoryRequest(BaseModel):
    browser_auth_path: str = "config/browser.json"
    takeout_path: str = "data/watch-history.json"
    num_stations: int = Field(default=3, ge=2, le=4)
    offline: bool = False


class RadioRuntime:
    """Holds stations, SQLite DB, digger, and per-station Contextual Bandit instances."""

    def __init__(
        self,
        db_path: Path | str = PROJECT_ROOT / "data/radio.db",
        stations_path: Path | str = PROJECT_ROOT / "config/stations.yaml",
        model_dir: Path | str = PROJECT_ROOT / "data",
        browser_auth_path: Path | str = PROJECT_ROOT / "config/browser.json",
        web_index_path: Path | str = PROJECT_ROOT / "web/index.html",
        auto_dig: bool = True,
    ) -> None:
        self.stations_path = Path(stations_path)
        self.model_dir = Path(model_dir)
        self.browser_auth_path = Path(browser_auth_path)
        self.web_index_path = Path(web_index_path)
        self.auto_dig = auto_dig

        self.db = RadioDatabase(db_path)
        self._digger: CandidatePoolDigger | None = None
        self.stations: dict[str, StationConfig] = {}
        self.bandits: dict[str, ContextualBanditRecommender] = {}
        self.reload_stations()

    @property
    def digger(self) -> CandidatePoolDigger:
        if self._digger is None:
            yt = YouTubeMusicAdapter(browser_auth_path=self.browser_auth_path)
            self._digger = CandidatePoolDigger(self.db, yt_adapter=yt)
        return self._digger

    def reload_stations(self) -> None:
        if not self.stations_path.exists():
            generate_stations_yaml(
                output_path=self.stations_path,
                browser_auth_path=self.browser_auth_path,
                takeout_path=PROJECT_ROOT / "data/watch-history.json",
                num_stations=3,
                offline=True,
            )
        station_list = load_stations_yaml(self.stations_path)
        self.stations = {s.station_id: s for s in station_list}
        self.bandits = {
            s.station_id: ContextualBanditRecommender(
                station_id=s.station_id,
                seed_tags=s.seed_tags,
                exploration_rate=s.exploration_rate,
                model_path=self.model_dir / f"bandit_{s.station_id}.pkl",
            )
            for s in station_list
        }

    def get_station(self, station_id: str) -> StationConfig:
        station = self.stations.get(station_id)
        if station is None:
            raise HTTPException(status_code=404, detail=f"Unknown station_id: {station_id}")
        return station

    def next_queue(
        self,
        station_id: str,
        count: int = 3,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        station = self.get_station(station_id)
        candidates = self.db.get_eligible_candidates(
            station_id=station.station_id,
            max_popularity_views=station.max_popularity_views,
            exclude_track_ids=exclude_ids,
        )

        # Auto-dig Stage 1 candidate pool when running low
        if self.auto_dig and len(candidates) < max(count, 5):
            try:
                self.digger.dig_station(station, target_pool_size=60)
                candidates = self.db.get_eligible_candidates(
                    station_id=station.station_id,
                    max_popularity_views=station.max_popularity_views,
                    exclude_track_ids=exclude_ids,
                )
            except Exception as exc:
                logger.warning("Auto-digging failed for %s: %s", station.station_id, exc)

        bandit = self.bandits[station.station_id]
        ranked = bandit.rank_candidates(
            candidates,
            seen_artists=self.db.get_seen_artists(station.station_id),
            max_popularity_views=station.max_popularity_views,
            limit=count,
        )
        return [item.to_queue_dict() for item in ranked]


def create_app(
    *,
    db_path: Path | str = PROJECT_ROOT / "data/radio.db",
    stations_path: Path | str = PROJECT_ROOT / "config/stations.yaml",
    model_dir: Path | str = PROJECT_ROOT / "data",
    browser_auth_path: Path | str = PROJECT_ROOT / "config/browser.json",
    web_index_path: Path | str = PROJECT_ROOT / "web/index.html",
    auto_dig: bool = True,
) -> FastAPI:
    """Create configured FastAPI application."""
    runtime = RadioRuntime(
        db_path=db_path,
        stations_path=stations_path,
        model_dir=model_dir,
        browser_auth_path=browser_auth_path,
        web_index_path=web_index_path,
        auto_dig=auto_dig,
    )

    app = FastAPI(title="Hidden Gem Radio", version="0.1.0")
    app.state.runtime = runtime

    @app.get("/", response_class=HTMLResponse)
    def serve_ui() -> HTMLResponse:
        if not runtime.web_index_path.exists():
            return HTMLResponse("<h1>Hidden Gem Radio</h1><p>web/index.html not found.</p>", status_code=404)
        return HTMLResponse(runtime.web_index_path.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "station_count": len(runtime.stations)}

    @app.get("/stations")
    def list_stations() -> dict[str, Any]:
        payload = []
        for s in runtime.stations.values():
            bandit = runtime.bandits[s.station_id]
            entry = s.to_dict()
            entry["candidate_count"] = runtime.db.count_playable_candidates(
                s.station_id, s.max_popularity_views
            )
            entry["model_updates"] = bandit.n_updates
            entry["feedback_count"] = runtime.db.count_feedback(s.station_id)
            payload.append(entry)
        return {"stations": payload}

    @app.post("/stations/import-history")
    def import_history(req: ImportHistoryRequest) -> dict[str, Any]:
        summary = generate_stations_yaml(
            output_path=runtime.stations_path,
            browser_auth_path=req.browser_auth_path,
            takeout_path=req.takeout_path,
            num_stations=req.num_stations,
            offline=req.offline,
        )
        runtime.reload_stations()
        return summary

    @app.post("/stations/{station_id}/dig")
    def dig_station(station_id: str, target_pool_size: int = Query(default=60, ge=10, le=120)) -> dict[str, Any]:
        station = runtime.get_station(station_id)
        return runtime.digger.dig_station(station, target_pool_size=target_pool_size)

    @app.get("/queue/next")
    def queue_next(
        station_id: str | None = Query(default=None),
        count: int = Query(default=3, ge=1, le=10),
        exclude: str = Query(default="", description="Comma-separated track_ids or video_ids to omit"),
    ) -> dict[str, Any]:
        if not station_id:
            if not runtime.stations:
                raise HTTPException(status_code=404, detail="No stations configured")
            station_id = next(iter(runtime.stations.keys()))
        exclude_set = {item.strip() for item in exclude.split(",") if item.strip()}
        items = runtime.next_queue(station_id=station_id, count=count, exclude_ids=exclude_set)
        return {"station_id": station_id, "count": len(items), "items": items}

    @app.post("/feedback")
    def post_feedback(req: FeedbackRequest) -> dict[str, Any]:
        station = runtime.get_station(req.station_id)
        event = req.event_type.strip().upper()

        # Silent fallback path when YouTube IFrame Player fires onError (embed disabled / region block)
        if event == "UNPLAYABLE" or req.yt_error_code is not None:
            updated_rows = runtime.db.mark_unplayable(video_id=req.video_id, track_id=req.track_id)
            exclude = {x for x in (req.track_id, req.video_id) if x}
            replacements = runtime.next_queue(station.station_id, count=1, exclude_ids=exclude)
            return {
                "status": "ok",
                "event_type": "UNPLAYABLE",
                "unplayable_marked": updated_rows > 0,
                "replacement": replacements[0] if replacements else None,
            }

        track = runtime.db.get_track(track_id=req.track_id, video_id=req.video_id)
        if track is None:
            raise HTTPException(status_code=404, detail="Track not found for feedback")

        seen_artists = runtime.db.get_seen_artists(station.station_id)
        is_new_artist = normalize_artist(track.artist) not in seen_artists

        bandit = runtime.bandits[station.station_id]
        reward = bandit.update(
            track,
            event_type=event,
            listen_seconds=req.listen_seconds,
            duration_seconds=req.duration_seconds,
            is_new_artist=is_new_artist,
        )
        runtime.db.record_feedback(
            station_id=station.station_id,
            track=track,
            event_type=event,
            reward=reward,
            listen_seconds=req.listen_seconds,
            duration_seconds=req.duration_seconds,
        )

        return {
            "status": "ok",
            "station_id": station.station_id,
            "track_id": track.track_id,
            "event_type": event,
            "reward": reward,
            "model_updates": bandit.n_updates,
        }

    return app


app = create_app()
