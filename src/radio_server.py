"""FastAPI local radio server connecting Candidate Pool, MABWiser Bandit, and Web UI."""

from __future__ import annotations

import logging
import threading
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
_PERMANENT_YT_UNPLAYABLE_CODES = frozenset({100, 101, 150})


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
    """Holds stations, SQLite DB, digger, per-station dig locks, and Contextual Bandit instances."""

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
        self._digger_init_lock = threading.Lock()
        self._dig_state_lock = threading.Lock()
        self._dig_locks: dict[str, threading.Lock] = {}
        self._digging_stations: set[str] = set()
        self.stations: dict[str, StationConfig] = {}
        self.bandits: dict[str, ContextualBanditRecommender] = {}
        self.reload_stations()

    @property
    def digger(self) -> CandidatePoolDigger:
        if self._digger is None:
            with self._digger_init_lock:
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
        with self._dig_state_lock:
            for s in station_list:
                self._dig_locks.setdefault(s.station_id, threading.Lock())

    def get_station(self, station_id: str) -> StationConfig:
        station = self.stations.get(station_id)
        if station is None:
            raise HTTPException(status_code=404, detail=f"Unknown station_id: {station_id}")
        return station

    def _get_station_dig_lock(self, station_id: str) -> threading.Lock:
        with self._dig_state_lock:
            lock = self._dig_locks.get(station_id)
            if lock is None:
                lock = threading.Lock()
                self._dig_locks[station_id] = lock
            return lock

    def _run_station_dig(
        self,
        station: StationConfig,
        target_pool_size: int = 60,
        skip_if_unheard_gte: int = 0,
    ) -> dict[str, Any] | None:
        """Serialize digging per station and guarantee _digging_stations cleanup on every exit path."""
        lock = self._get_station_dig_lock(station.station_id)
        try:
            with lock:
                with self._dig_state_lock:
                    self._digging_stations.add(station.station_id)
                if skip_if_unheard_gte > 0:
                    existing = self.db.count_playable_candidates(
                        station.station_id,
                        station.max_popularity_views,
                    )
                    if existing >= skip_if_unheard_gte:
                        return {
                            "station_id": station.station_id,
                            "added": 0,
                            "total_playable": existing,
                            "skipped_duplicate_dig": True,
                        }
                try:
                    return self.digger.dig_station(station, target_pool_size=target_pool_size)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Auto-digging failed for %s: %s", station.station_id, exc)
                    return None
        finally:
            with self._dig_state_lock:
                self._digging_stations.discard(station.station_id)

    def _trigger_background_dig(
        self,
        station: StationConfig,
        target_pool_size: int = 60,
        min_unheard: int = 5,
    ) -> bool:
        """Start a single non-blocking background dig thread if one is not already active."""
        with self._dig_state_lock:
            if station.station_id in self._digging_stations:
                return False
            self._digging_stations.add(station.station_id)

        def _worker() -> None:
            try:
                self._run_station_dig(
                    station,
                    target_pool_size=target_pool_size,
                    skip_if_unheard_gte=min_unheard,
                )
            finally:
                with self._dig_state_lock:
                    self._digging_stations.discard(station.station_id)

        try:
            thread = threading.Thread(
                target=_worker,
                name=f"dig-{station.station_id}",
                daemon=True,
            )
            thread.start()
        except Exception:
            with self._dig_state_lock:
                self._digging_stations.discard(station.station_id)
            raise
        return True

    def next_queue(
        self,
        station_id: str,
        count: int = 3,
        exclude_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        station = self.get_station(station_id)
        min_unheard_threshold = max(count, 5)
        required_db_unheard = min_unheard_threshold + len(exclude_ids or ())

        # [M4] Judge depletion strictly on UNHEARD candidates
        unheard = self.db.get_eligible_candidates(
            station_id=station.station_id,
            max_popularity_views=station.max_popularity_views,
            exclude_track_ids=exclude_ids,
            include_heard_fallback=False,
        )

        if self.auto_dig and len(unheard) < min_unheard_threshold:
            if unheard:
                # [M1 / Bug #2] Return immediately from existing unheard pool; replenish asynchronously
                self._trigger_background_dig(
                    station,
                    target_pool_size=60,
                    min_unheard=required_db_unheard,
                )
            else:
                # Cold empty or fully heard pool: serialize a single dig across concurrent callers
                self._run_station_dig(
                    station,
                    target_pool_size=60,
                    skip_if_unheard_gte=required_db_unheard,
                )
                unheard = self.db.get_eligible_candidates(
                    station_id=station.station_id,
                    max_popularity_views=station.max_popularity_views,
                    exclude_track_ids=exclude_ids,
                    include_heard_fallback=False,
                )

        # [M4] Fall back to previously liked/completed tracks ONLY if unheard is still empty
        candidates = unheard
        if not candidates:
            candidates = self.db.get_eligible_candidates(
                station_id=station.station_id,
                max_popularity_views=station.max_popularity_views,
                exclude_track_ids=exclude_ids,
                include_heard_fallback=True,
            )

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
        result = runtime._run_station_dig(station, target_pool_size=target_pool_size)
        return result or {"station_id": station.station_id, "added": 0, "total_playable": 0}

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

        # Silent fallback path when YouTube IFrame Player fires onError
        if event == "UNPLAYABLE" or req.yt_error_code is not None:
            # [M5] Only permanently mark unplayable on embed/removal codes (100/101/150) or explicit UNPLAYABLE
            is_permanent = req.yt_error_code is None or req.yt_error_code in _PERMANENT_YT_UNPLAYABLE_CODES
            updated_rows = (
                runtime.db.mark_unplayable(video_id=req.video_id, track_id=req.track_id)
                if is_permanent
                else 0
            )
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
        reward = ContextualBanditRecommender.compute_reward(
            event,
            listen_seconds=req.listen_seconds,
            duration_seconds=req.duration_seconds,
        )

        # [C1 / Bug #3] Record feedback in SQLite FIRST, and roll back row if bandit.update fails
        feedback_id = runtime.db.record_feedback(
            station_id=station.station_id,
            track=track,
            event_type=event,
            reward=reward,
            listen_seconds=req.listen_seconds,
            duration_seconds=req.duration_seconds,
        )
        try:
            bandit.update(
                track,
                event_type=event,
                listen_seconds=req.listen_seconds,
                duration_seconds=req.duration_seconds,
                is_new_artist=is_new_artist,
            )
        except Exception:
            runtime.db.delete_feedback(feedback_id)
            raise

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
