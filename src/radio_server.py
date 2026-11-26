"""FastAPI local radio server connecting Candidate Pool, 10-dim Theory DNA Bandit, Playlists, and Web UI."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from src.adapters.deezer import DeezerClient
from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase, sanitize_video_id
from src.digger.history_importer import generate_stations_yaml
from src.digger.pool import CandidatePoolDigger
from src.models import (
    StationConfig,
    load_history_index,
    load_stations_yaml,
    normalize_artist,
)
from src.recommender.audio_mir import AudioMIRExtractor
from src.recommender.bandit import ContextualBanditRecommender
from src.recommender.theory_features import THEORY_AXIS_KEYS, TheoryFeatureExtractor

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PERMANENT_YT_UNPLAYABLE_CODES = frozenset({100, 101, 150})
_TRANSIENT_YT_ERROR_CODES = frozenset({2, 5})


class FeedbackRequest(BaseModel):
    station_id: str
    track_id: str | None = None
    video_id: str | None = None
    event_type: str = Field(..., description="GOOD | BAD | COMPLETE | MID_SKIP | EARLY_SKIP | SKIP | ENDED | UNPLAYABLE")
    listen_seconds: float | None = None
    duration_seconds: float | None = None
    yt_error_code: int | None = None
    playlist_id: str | None = None
    youtube_playlist_id: str | None = None


class ImportHistoryRequest(BaseModel):
    browser_auth_path: str = "config/browser.json"
    takeout_path: str = "data/watch-history.json"
    num_stations: int = Field(default=5, ge=2, le=5)
    offline: bool = False


class UnplayableRequest(BaseModel):
    station_id: str | None = None
    track_id: str | None = None
    reason: str = "playback_stall_or_error"
    yt_error_code: int | None = None
    ttl_days: int = Field(default=30, ge=1, le=365)


class CreatePlaylistRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    youtube_playlist_id: str | None = None


class AddPlaylistTrackRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    track_id: str | None = None
    artist: str = "Unknown Artist"
    title: str = "Untitled"
    station_id: str | None = None
    sync_youtube: bool = False
    youtube_playlist_id: str | None = None

    @field_validator("video_id")
    @classmethod
    def _validate_video_id(cls, value: str) -> str:
        clean = sanitize_video_id(value)
        if clean is None:
            raise ValueError("Invalid YouTube video_id format")
        return clean


class YouTubeSyncRequest(BaseModel):
    video_id: str = Field(..., min_length=1)
    youtube_playlist_id: str | None = None

    @field_validator("video_id")
    @classmethod
    def _validate_video_id(cls, value: str) -> str:
        clean = sanitize_video_id(value)
        if clean is None:
            raise ValueError("Invalid YouTube video_id format")
        return clean


class RadioRuntime:
    """Holds stations, SQLite DB, Stage 1 graph digger, 16-dim MIR extractor, and per-station Bandits."""

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
        self.theory_extractor = TheoryFeatureExtractor()
        self._deezer_client: DeezerClient | None = None
        self.mir_extractor = AudioMIRExtractor(db=self.db)
        self._yt_adapter: YouTubeMusicAdapter | None = None
        self._digger: CandidatePoolDigger | None = None
        self._digger_init_lock = threading.Lock()
        self._dig_state_lock = threading.Lock()
        self._dig_locks: dict[str, threading.Lock] = {}
        self._digging_stations: dict[str, set[object]] = {}
        self.stations: dict[str, StationConfig] = {}
        self.bandits: dict[str, ContextualBanditRecommender] = {}
        self.known_video_ids: set[str] = set()
        self.known_track_keys: set[str] = set()
        self.reload_stations()

    @property
    def yt_adapter(self) -> YouTubeMusicAdapter:
        if self._yt_adapter is None:
            with self._digger_init_lock:
                if self._yt_adapter is None:
                    self._yt_adapter = YouTubeMusicAdapter(
                        browser_auth_path=self.browser_auth_path,
                        project_root=PROJECT_ROOT,
                    )
        return self._yt_adapter

    @property
    def digger(self) -> CandidatePoolDigger:
        if self._digger is None:
            with self._digger_init_lock:
                if self._digger is None:
                    if self._yt_adapter is None:
                        self._yt_adapter = YouTubeMusicAdapter(
                            browser_auth_path=self.browser_auth_path,
                            project_root=PROJECT_ROOT,
                        )
                    if self._deezer_client is None:
                        self._deezer_client = DeezerClient(db=self.db)
                        self.mir_extractor.deezer = self._deezer_client
                    self._digger = CandidatePoolDigger(
                        self.db,
                        yt_adapter=self._yt_adapter,
                        deezer_client=self._deezer_client,
                        mir_extractor=self.mir_extractor,
                        enable_mir_enrichment=True,
                        max_mir_tracks_per_dig=15,
                    )
        return self._digger

    def reload_stations(self) -> None:
        if not self.stations_path.exists():
            generate_stations_yaml(
                output_path=self.stations_path,
                browser_auth_path=self.browser_auth_path,
                takeout_path=PROJECT_ROOT / "data/watch-history.json",
                num_stations=5,
                offline=True,
            )
        station_list = load_stations_yaml(self.stations_path)
        self.known_video_ids, self.known_track_keys = load_history_index(self.stations_path)
        self.stations = {s.station_id: s for s in station_list}
        self.bandits = {
            s.station_id: ContextualBanditRecommender(
                station_id=s.station_id,
                seed_tags=s.seed_tags,
                exploration_rate=s.exploration_rate,
                model_path=self.model_dir / f"bandit_{s.station_id}.pkl",
                station_config=s,
                theory_extractor=self.theory_extractor,
                mir_extractor=self.mir_extractor,
                db=self.db,
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

    def _add_dig_token(self, station_id: str, token: object) -> None:
        """Register an active/waiting dig token for station_id without overwriting other tokens."""
        with self._dig_state_lock:
            self._digging_stations.setdefault(station_id, set()).add(token)

    def _release_dig_token(self, station_id: str, token: object) -> None:
        """Remove a dig token from station_id and delete the station key once no tokens remain."""
        with self._dig_state_lock:
            tokens = self._digging_stations.get(station_id)
            if tokens is not None:
                tokens.discard(token)
                if not tokens:
                    del self._digging_stations[station_id]

    def _run_station_dig(
        self,
        station: StationConfig,
        target_pool_size: int = 60,
        skip_if_unheard_gte: int = 0,
        owner_token: object | None = None,
    ) -> dict[str, Any] | None:
        """Serialize digging per station and guarantee ownership-scoped token cleanup."""
        my_token = owner_token
        try:
            lock = self._get_station_dig_lock(station.station_id)
            with lock:
                if my_token is None:
                    my_token = object()
                self._add_dig_token(station.station_id, my_token)
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
                    # Pass known history sets when digger supports them (keeps test mock compatibility)
                    try:
                        return self.digger.dig_station(
                            station,
                            target_pool_size=target_pool_size,
                            known_video_ids=self.known_video_ids,
                            known_track_keys=self.known_track_keys,
                        )
                    except TypeError:
                        return self.digger.dig_station(station, target_pool_size=target_pool_size)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Auto-digging failed for %s: %s", station.station_id, exc)
                    return None
        finally:
            if my_token is not None:
                self._release_dig_token(station.station_id, my_token)

    def _trigger_background_dig(
        self,
        station: StationConfig,
        target_pool_size: int = 60,
        min_unheard: int = 5,
    ) -> bool:
        """Start a single non-blocking background dig thread if one is not already active."""
        with self._dig_state_lock:
            if self._digging_stations.get(station.station_id):
                return False
            token = object()
            self._digging_stations.setdefault(station.station_id, set()).add(token)

        def _worker() -> None:
            self._run_station_dig(
                station,
                target_pool_size=target_pool_size,
                skip_if_unheard_gte=min_unheard,
                owner_token=token,
            )

        try:
            thread = threading.Thread(
                target=_worker,
                name=f"dig-{station.station_id}",
                daemon=True,
            )
            thread.start()
        except Exception:
            self._release_dig_token(station.station_id, token)
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
        effective_exclude_ids = set(exclude_ids or set()) | self.known_video_ids
        required_db_unheard = min_unheard_threshold + len(exclude_ids or ())

        # [M4] Judge depletion strictly on UNHEARD candidates
        unheard = self.db.get_eligible_candidates(
            station_id=station.station_id,
            max_popularity_views=station.max_popularity_views,
            exclude_track_ids=effective_exclude_ids,
            include_heard_fallback=False,
            exclude_canonical_keys=self.known_track_keys,
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
                    exclude_track_ids=effective_exclude_ids,
                    include_heard_fallback=False,
                    exclude_canonical_keys=self.known_track_keys,
                )

        # [M4 / Audit #8] Fall back to previously liked/completed tracks ONLY if unheard is still empty,
        # while continuing to exclude already-known YouTube history video IDs and track keys
        candidates = unheard
        if not candidates:
            candidates = self.db.get_eligible_candidates(
                station_id=station.station_id,
                max_popularity_views=station.max_popularity_views,
                exclude_track_ids=effective_exclude_ids,
                include_heard_fallback=True,
                exclude_canonical_keys=self.known_track_keys,
            )

        bandit = self.bandits[station.station_id]
        ranked = bandit.rank_candidates(
            candidates,
            seen_artists=self.db.get_seen_artists(station.station_id),
            max_popularity_views=station.max_popularity_views,
            limit=count,
            recent_artists=self.db.get_recent_played_artists(station.station_id, limit=5),
        )
        return [item.to_queue_dict() for item in ranked]

    def get_theory_profile(self, station_id: str | None = None) -> dict[str, Any]:
        """Return learned 5-axis Deep Music Theory DNA profile for one station or aggregated globally."""
        if station_id:
            self.get_station(station_id)
            return self.bandits[station_id].get_learned_theory_profile()

        per_station = {sid: b.get_learned_theory_profile() for sid, b in self.bandits.items()}
        if not per_station:
            return {"station_id": None, "updates": 0, "scores": {}, "axes": [], "stations": {}}

        total_updates = sum(int(p["updates"]) for p in per_station.values())
        axes_agg: list[dict[str, Any]] = []
        scores_agg: dict[str, float] = {}

        for ax in THEORY_AXIS_KEYS:
            meta = self.theory_extractor.axes_meta.get(ax, {})
            # Weight stations by (1 + updates) so actively rated stations shape the global profile
            weighted_sum = 0.0
            weight_total = 0.0
            for prof in per_station.values():
                w = 1.0 + float(prof["updates"])
                weighted_sum += w * float(prof["scores"].get(ax, 0.65))
                weight_total += w
            avg_score = round(weighted_sum / max(1.0, weight_total), 3)
            scores_agg[ax] = avg_score
            axes_agg.append(
                {
                    "axis": ax,
                    "key": meta.get("key", ax.lower()),
                    "label_ja": meta.get("label_ja", ax),
                    "label_en": meta.get("label_en", ax),
                    "score": avg_score,
                }
            )

        return {
            "station_id": None,
            "updates": total_updates,
            "scores": scores_agg,
            "axes": axes_agg,
            "stations": per_station,
        }


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

    app = FastAPI(title="Hidden Gem Radio", version="0.2.0")
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

    @app.get("/api/stations/{station_id}/next")
    def api_station_next(
        station_id: str,
        exclude: str = Query(default="", description="Comma-separated track_ids or video_ids to omit"),
    ) -> dict[str, Any]:
        exclude_set = {item.strip() for item in exclude.split(",") if item.strip()}
        items = runtime.next_queue(station_id=station_id, count=1, exclude_ids=exclude_set)
        return {
            "station_id": station_id,
            "track": items[0] if items else None,
        }

    @app.get("/api/stations/{station_id}/batch")
    def api_station_batch(
        station_id: str,
        count: int = Query(default=3, ge=1, le=10),
        exclude: str = Query(default="", description="Comma-separated track_ids or video_ids to omit"),
    ) -> dict[str, Any]:
        exclude_set = {item.strip() for item in exclude.split(",") if item.strip()}
        items = runtime.next_queue(station_id=station_id, count=count, exclude_ids=exclude_set)
        return {"station_id": station_id, "count": len(items), "items": items}

    @app.get("/api/profile/theory")
    def api_theory_profile(station_id: str | None = Query(default=None)) -> dict[str, Any]:
        return runtime.get_theory_profile(station_id=station_id)

    @app.post("/api/tracks/{video_id}/unplayable")
    def api_mark_track_unplayable(
        video_id: str,
        req: UnplayableRequest | None = None,
    ) -> dict[str, Any]:
        """Blacklist an unplayable/stalled video_id for 30 days WITHOUT polluting BAD bandit learning.

        Transient YT player errors (codes 2, 5) return a replacement without blacklisting.
        Permanent YT errors (codes 100, 101, 150) also mark tracks.unplayable = 1.
        """
        clean_vid = sanitize_video_id(video_id)
        if clean_vid is None:
            raise HTTPException(status_code=400, detail="Invalid YouTube video_id format")

        body = req or UnplayableRequest()
        is_transient = body.yt_error_code in _TRANSIENT_YT_ERROR_CODES
        expires_at: float | None = None

        if not is_transient:
            expires_at = runtime.db.blacklist_unplayable(
                clean_vid,
                reason=body.reason,
                yt_error_code=body.yt_error_code,
                ttl_days=body.ttl_days,
            )
            if body.yt_error_code in _PERMANENT_YT_UNPLAYABLE_CODES:
                runtime.db.mark_unplayable(video_id=clean_vid, track_id=body.track_id)

        replacement = None
        if body.station_id and body.station_id in runtime.stations:
            exclude = {x for x in (clean_vid, body.track_id) if x}
            reps = runtime.next_queue(body.station_id, count=1, exclude_ids=exclude)
            replacement = reps[0] if reps else None

        return {
            "status": "ok",
            "video_id": clean_vid,
            "blacklisted": not is_transient,
            "blacklisted_days": 0 if is_transient else body.ttl_days,
            "expires_at": expires_at,
            "bad_learning_polluted": False,
            "replacement": replacement,
        }

    # --- Playlists & Hybrid YouTube Integration ---
    @app.get("/api/playlists")
    def api_list_playlists() -> dict[str, Any]:
        return {"playlists": runtime.db.list_playlists()}

    @app.post("/api/playlists")
    def api_create_playlist(req: CreatePlaylistRequest) -> dict[str, Any]:
        try:
            created = runtime.db.create_playlist(req.name, youtube_playlist_id=req.youtube_playlist_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok", "playlist": created}

    @app.get("/api/playlists/{playlist_id}/tracks")
    def api_get_playlist_tracks(playlist_id: str) -> dict[str, Any]:
        data = runtime.db.get_playlist_tracks(playlist_id)
        if data is None:
            raise HTTPException(status_code=404, detail=f"Unknown playlist_id: {playlist_id}")
        return data

    @app.post("/api/playlists/{playlist_id}/tracks")
    def api_add_playlist_track(playlist_id: str, req: AddPlaylistTrackRequest) -> dict[str, Any]:
        if not runtime.db.playlist_exists(playlist_id):
            raise HTTPException(status_code=404, detail=f"Unknown playlist_id: {playlist_id}")
        saved = runtime.db.add_track_to_playlist(
            playlist_id,
            video_id=req.video_id,
            track_id=req.track_id,
            artist=req.artist,
            title=req.title,
            station_id=req.station_id,
        )
        yt_sync: dict[str, Any] | None = None
        if req.sync_youtube:
            yt_sync = runtime.yt_adapter.sync_liked_track(
                req.video_id,
                youtube_playlist_id=req.youtube_playlist_id,
            )
        return {"status": "ok", "entry": saved, "youtube_sync": yt_sync}

    @app.delete("/api/playlists/{playlist_id}/tracks/{video_id}")
    def api_remove_playlist_track(playlist_id: str, video_id: str) -> dict[str, Any]:
        if not runtime.db.playlist_exists(playlist_id):
            raise HTTPException(status_code=404, detail=f"Unknown playlist_id: {playlist_id}")
        clean_vid = sanitize_video_id(video_id)
        if clean_vid is None:
            raise HTTPException(status_code=400, detail="Invalid YouTube video_id format")
        removed = runtime.db.remove_track_from_playlist(playlist_id, clean_vid)
        return {"status": "ok", "removed": removed > 0}

    @app.get("/api/youtube/playlists")
    def api_youtube_playlists() -> dict[str, Any]:
        return runtime.yt_adapter.list_user_playlists()

    @app.post("/api/youtube/sync")
    def api_youtube_sync(req: YouTubeSyncRequest) -> dict[str, Any]:
        return runtime.yt_adapter.sync_liked_track(
            req.video_id,
            youtube_playlist_id=req.youtube_playlist_id,
        )

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

        playlist_saved: dict[str, Any] | None = None
        yt_sync: dict[str, Any] | None = None
        if event == "GOOD" and track.video_id:
            if req.playlist_id and runtime.db.playlist_exists(req.playlist_id):
                playlist_saved = runtime.db.add_track_to_playlist(
                    req.playlist_id,
                    video_id=track.video_id,
                    track_id=track.track_id,
                    artist=track.artist,
                    title=track.title,
                    station_id=station.station_id,
                )
            if req.youtube_playlist_id and runtime.yt_adapter.is_authenticated:
                yt_sync = runtime.yt_adapter.sync_liked_track(
                    track.video_id,
                    youtube_playlist_id=req.youtube_playlist_id,
                )

        return {
            "status": "ok",
            "station_id": station.station_id,
            "track_id": track.track_id,
            "event_type": event,
            "reward": reward,
            "theory_reward": ContextualBanditRecommender.compute_theory_reward(
                event,
                listen_seconds=req.listen_seconds,
                duration_seconds=req.duration_seconds,
            ),
            "model_updates": bandit.n_updates,
            "playlist_saved": playlist_saved,
            "youtube_sync": yt_sync,
        }

    return app


app = create_app()
