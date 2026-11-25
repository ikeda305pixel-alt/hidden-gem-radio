"""Stage 1 Candidate Pool manager combining ListenBrainz/MusicBrainz + YouTube Music."""

from __future__ import annotations

import logging
from typing import Any

from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase
from src.digger.listenbrainz import ListenBrainzClient
from src.models import StationConfig, normalize_artist

logger = logging.getLogger(__name__)


class CandidatePoolDigger:
    """Collects 50-100 unheard hidden-gem tracks per station and caches them in SQLite."""

    def __init__(
        self,
        db: RadioDatabase,
        yt_adapter: YouTubeMusicAdapter | None = None,
        lb_client: ListenBrainzClient | None = None,
    ) -> None:
        self.db = db
        self.yt = yt_adapter or YouTubeMusicAdapter()
        self.lb = lb_client or ListenBrainzClient()

    def dig_station(
        self,
        station: StationConfig,
        target_pool_size: int = 60,
        max_lb_resolves: int = 18,
    ) -> dict[str, Any]:
        """Dig candidates from ListenBrainz + ytmusicapi Watch Playlists into SQLite."""
        blocked_artists = self.db.get_blocked_artists_30d()
        added = 0
        skipped_popular = 0
        skipped_blocked = 0
        seed_video_ids: list[str] = []

        # 1. Dig from ListenBrainz / MusicBrainz (platform-agnostic ISRC/MBID metadata)
        lb_tracks = self.lb.dig_candidates(
            seed_tags=station.seed_tags,
            seed_artists=station.seed_artists,
            target_count=max_lb_resolves * 2,
        )
        resolved_count = 0
        for track in lb_tracks:
            if resolved_count >= max_lb_resolves:
                break
            if normalize_artist(track.artist) in blocked_artists:
                skipped_blocked += 1
                continue

            resolution = self.yt.resolve_track(track.artist, track.title, isrc=track.isrc)
            if not resolution:
                continue
            resolved_count += 1

            if (
                resolution.view_count is not None
                and resolution.view_count > station.max_popularity_views
            ):
                skipped_popular += 1
                continue

            track.video_id = resolution.platform_track_id
            track.view_count = resolution.view_count
            if track.duration_seconds is None:
                track.duration_seconds = resolution.duration_seconds
            if track.release_year is None:
                track.release_year = resolution.release_year
            if not track.tags:
                track.tags = list(station.seed_tags)

            self.db.upsert_track(station.station_id, track)
            seed_video_ids.append(track.video_id)
            added += 1

        # 2. Resolve seed artists / seed tags on YouTube Music to obtain Watch Playlist anchors
        for artist in station.seed_artists[:3]:
            if len(seed_video_ids) >= 4:
                break
            res = self.yt.resolve_track(artist, "")
            if res and res.platform_track_id not in seed_video_ids:
                seed_video_ids.append(res.platform_track_id)

        if not seed_video_ids:
            for tag in station.seed_tags[:2]:
                res = self.yt.resolve_track(tag, "official audio")
                if res and res.platform_track_id not in seed_video_ids:
                    seed_video_ids.append(res.platform_track_id)

        # 3. Expand candidate pool rapidly via YouTube Music Watch Playlists (typically 25-50 tracks each)
        for seed_vid in seed_video_ids[:3]:
            if added >= target_pool_size:
                break
            related = self.yt.get_related_tracks(
                seed_vid,
                seed_tags=station.seed_tags,
                limit=30,
            )
            for track in related:
                if added >= target_pool_size:
                    break
                if normalize_artist(track.artist) in blocked_artists:
                    skipped_blocked += 1
                    continue
                if (
                    track.view_count is not None
                    and track.view_count > station.max_popularity_views
                ):
                    skipped_popular += 1
                    continue
                self.db.upsert_track(station.station_id, track)
                added += 1

        total_playable = self.db.count_playable_candidates(
            station.station_id,
            station.max_popularity_views,
        )
        return {
            "station_id": station.station_id,
            "added": added,
            "skipped_popular": skipped_popular,
            "skipped_blocked_30d": skipped_blocked,
            "total_playable": total_playable,
        }
