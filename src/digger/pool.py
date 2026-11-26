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
        max_lb_resolves: int = 16,
        known_video_ids: set[str] | None = None,
        known_track_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Dig hidden-gem neighbors of weighted history seeds, Spotify anchors, and discovery queries."""
        blocked_artists = self.db.get_blocked_artists_30d()
        blacklisted_vids = self.db.get_blacklisted_video_ids()
        skip_vids = set(known_video_ids or set()) | blacklisted_vids
        skip_keys = set(known_track_keys or set())

        added = 0
        skipped_popular = 0
        skipped_blocked = 0
        skipped_known = 0

        # Track (video_id, seed_artist, seed_archetype) anchors for Watch Playlist neighborhood digging
        seed_anchors: list[tuple[str, str | None, str | None]] = []
        seen_anchor_vids: set[str] = set()
        default_archetype = station.theory_archetypes[0] if station.theory_archetypes else None

        # Prioritize high-weight YouTube history seeds first, then station seed artists
        weighted_artists = [
            str(item.get("artist") or "").strip()
            for item in sorted(
                station.weighted_seeds,
                key=lambda x: float(x.get("weight", 0.0)),
                reverse=True,
            )
            if item.get("artist")
        ]
        prioritized_artists = list(dict.fromkeys([*weighted_artists[:4], *station.seed_artists]))

        # 1. Dig from ListenBrainz / MusicBrainz (platform-agnostic ISRC/MBID metadata)
        lb_tracks = self.lb.dig_candidates(
            seed_tags=station.seed_tags,
            seed_artists=prioritized_artists,
            target_count=max_lb_resolves * 2,
        )
        resolved_count = 0
        for idx, track in enumerate(lb_tracks):
            if resolved_count >= max_lb_resolves:
                break
            if normalize_artist(track.artist) in blocked_artists:
                skipped_blocked += 1
                continue
            if track.canonical_key in skip_keys:
                skipped_known += 1
                continue

            resolution = self.yt.resolve_track(
                track.artist,
                track.title,
                isrc=track.isrc,
                max_popularity_views=station.max_popularity_views,
            )
            if not resolution:
                continue
            resolved_count += 1

            if resolution.platform_track_id in skip_vids:
                skipped_known += 1
                continue
            if (
                resolution.view_count is not None
                and resolution.view_count > station.max_popularity_views
            ):
                skipped_popular += 1
                continue

            arch = (
                station.theory_archetypes[idx % len(station.theory_archetypes)]
                if station.theory_archetypes
                else None
            )
            track.video_id = resolution.platform_track_id
            track.view_count = resolution.view_count
            track.seed_archetype = track.seed_archetype or arch
            if track.duration_seconds is None:
                track.duration_seconds = resolution.duration_seconds
            if track.release_year is None:
                track.release_year = resolution.release_year

            self.db.upsert_track(station.station_id, track)
            if track.video_id not in seen_anchor_vids:
                seen_anchor_vids.add(track.video_id)
                seed_anchors.append((track.video_id, track.artist, track.seed_archetype))
            added += 1

        # 2. Resolve high-weight history seeds & Spotify anchor tracks to dig their Watch Playlist neighbors
        #    (The anchor song itself can be a famous heavy-rotation track, so we don't cap its views!)
        anchor_queries: list[tuple[str, str, str | None]] = []
        for item in station.weighted_seeds[:3]:
            w_artist = str(item.get("artist") or "").strip()
            w_title = str(item.get("top_title") or item.get("title") or "").strip()
            if w_artist:
                anchor_queries.append((w_artist, w_title, default_archetype))

        for idx, anchor in enumerate(station.spotify_anchor_seeds[:4]):
            arch = (
                station.theory_archetypes[idx % len(station.theory_archetypes)]
                if station.theory_archetypes
                else None
            )
            if " - " in anchor:
                a_left, a_right = anchor.split(" - ", 1)
                anchor_queries.append((a_left.strip(), a_right.strip(), arch))
            else:
                anchor_queries.append((anchor.strip(), "", arch))

        for artist in prioritized_artists[:3]:
            anchor_queries.append((artist, "", default_archetype))

        for q_artist, q_title, q_arch in anchor_queries:
            if len(seed_anchors) >= 5:
                break
            res = self.yt.resolve_track(q_artist, q_title)
            if res and res.platform_track_id not in seen_anchor_vids:
                seen_anchor_vids.add(res.platform_track_id)
                seed_anchors.append((res.platform_track_id, q_artist, q_arch))

        if not seed_anchors:
            for query in station.discovery_queries[:2] or station.seed_tags[:2]:
                res = self.yt.resolve_track(query, "")
                if res and res.platform_track_id not in seen_anchor_vids:
                    seen_anchor_vids.add(res.platform_track_id)
                    seed_anchors.append((res.platform_track_id, None, default_archetype))

        # 3. Expand hidden-gem neighborhood via Watch Playlists (excluding known heavy-rotation tracks)
        for seed_vid, seed_artist, seed_arch in seed_anchors[:4]:
            if added >= target_pool_size:
                break
            related = self.yt.get_related_tracks(
                seed_vid,
                limit=30,
                seed_artist=seed_artist,
                seed_archetype=seed_arch,
            )
            for track in related:
                if added >= target_pool_size:
                    break
                if track.video_id in skip_vids or track.canonical_key in skip_keys:
                    skipped_known += 1
                    continue
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
            "skipped_known_history": skipped_known,
            "total_playable": total_playable,
        }
