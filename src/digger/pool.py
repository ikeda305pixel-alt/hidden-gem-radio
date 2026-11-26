"""Stage 1 Candidate Generation: Collaborative Co-occurrence Graph & Official Official-Audio-Only Digger.

Completely eliminates generic genre-keyword searches from primary discovery.
Expands each station strictly from its own:
  1. YouTube watch-history weighted seeds
  2. Spotify curated anchor tracks
  3. PES 2011 / Emily in Paris curated anchors

via:
  - ytmusic.get_watch_playlist(videoId=..., radio=True, limit=25)
  - ytmusic.get_artist(browseId) -> related real artists -> songs #3..#15 deep cuts

Tracks appearing across multiple distinct seeds receive elevated `cooccurrence_count` and
human-readable `discovery_route` (e.g., `共起2シード: bbno$ × KYLE 関連`).

CLI:
    python -m src.digger.pool --reset-tracks
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from src.adapters.deezer import DeezerClient
from src.adapters.youtube import YouTubeMusicAdapter
from src.db import RadioDatabase
from src.digger.listenbrainz import ListenBrainzClient
from src.models import StationConfig, TrackEntity, normalize_artist
from src.recommender.audio_mir import AudioMIRExtractor

logger = logging.getLogger(__name__)


def _parse_anchor(anchor: str) -> tuple[str, str]:
    text = str(anchor or "").strip()
    if " - " in text:
        left, right = text.split(" - ", 1)
        return left.strip(), right.strip()
    return text, ""


def _format_discovery_route(sources: list[str], cooccurrence_count: int) -> str:
    short_names: list[str] = []
    seen_norm: set[str] = set()
    for raw in sources:
        cleaned = raw.replace(" 関連アーティスト", "").replace(" 関連", "").strip()
        norm = normalize_artist(cleaned) or cleaned.lower()
        if cleaned and norm not in seen_norm:
            seen_norm.add(norm)
            short_names.append(cleaned)

    if cooccurrence_count >= 2 and len(short_names) >= 2:
        joined = " × ".join(short_names[:3])
        return f"共起{cooccurrence_count}シード: {joined} 関連"
    if short_names:
        return f"{short_names[0]} 関連"
    return "協調グラフ発掘"


class CandidatePoolDigger:
    """Stage 1 Collaborative Graph Digger: official songs only, multi-seed co-occurrence, zero AI slop."""

    def __init__(
        self,
        db: RadioDatabase,
        yt_adapter: YouTubeMusicAdapter | None = None,
        lb_client: ListenBrainzClient | None = None,
        deezer_client: DeezerClient | None = None,
        mir_extractor: AudioMIRExtractor | None = None,
        enable_mir_enrichment: bool = False,
        max_mir_tracks_per_dig: int = 15,
    ) -> None:
        self.db = db
        self.yt = yt_adapter or YouTubeMusicAdapter()
        # Kept only as optional constructor arg for legacy callers; never used for generic genre-tag blasting
        self.lb = lb_client
        self.deezer = deezer_client
        self.mir_extractor = mir_extractor
        self.enable_mir_enrichment = enable_mir_enrichment
        self.max_mir_tracks_per_dig = max(0, int(max_mir_tracks_per_dig))

    def _build_station_seed_plan(
        self,
        station: StationConfig,
        max_seeds: int = 8,
    ) -> list[tuple[str, str, str, str | None]]:
        """Build ordered list of (artist, title, seed_label, archetype) exclusively from this station's seeds."""
        plan: list[tuple[str, str, str, str | None]] = []
        seen_keys: set[tuple[str, str]] = set()
        default_arch = station.theory_archetypes[0] if station.theory_archetypes else None

        def _add(artist: str, title: str, label: str, arch: str | None) -> None:
            a = artist.strip()
            t = title.strip()
            if not a:
                return
            k = (a.lower(), t.lower())
            if k in seen_keys:
                return
            seen_keys.add(k)
            plan.append((a, t, label.strip() or a, arch))

        # 1. Highest-weight YouTube watch history seeds for this station
        for item in sorted(
            station.weighted_seeds,
            key=lambda x: float(x.get("weight", 0.0)),
            reverse=True,
        )[:4]:
            w_artist = str(item.get("artist") or "").strip()
            w_title = str(item.get("top_title") or item.get("title") or "").strip()
            _add(w_artist, w_title, w_artist, default_arch)

        # 2. Curated Spotify anchor tracks for this station
        for idx, anchor in enumerate(station.spotify_anchor_seeds[:6]):
            a, t = _parse_anchor(anchor)
            arch = (
                station.theory_archetypes[idx % len(station.theory_archetypes)]
                if station.theory_archetypes
                else None
            )
            _add(a, t, a, arch)

        # 3. PES 2011 & Emily in Paris anchors for this station (cult/chic crossover seeds)
        for anchor in [*station.pes_anchors[:2], *station.emily_anchors[:2]]:
            a, t = _parse_anchor(anchor)
            _add(a, t, a, default_arch)

        # 4. Fallback to explicit station seed_artists if no song anchors exist (e.g. minimal unit-test station)
        if not plan:
            for artist in station.seed_artists[:4]:
                _add(artist, "", artist, default_arch)

        return plan[:max_seeds]

    def dig_station(
        self,
        station: StationConfig,
        target_pool_size: int = 60,
        max_lb_resolves: int = 16,
        known_video_ids: set[str] | None = None,
        known_track_keys: set[str] | None = None,
        known_familiar_artists: set[str] | None = None,
        max_seeds: int = 8,
        max_alt_searches_per_dig: int = 12,
        max_tracks_per_artist_in_pool: int = 2,
    ) -> dict[str, Any]:
        """Run Stage 1 Collaborative Co-occurrence Graph digging strictly within station's own seed graph."""
        from src.models import is_japanese_music_entity

        _ = max_lb_resolves  # Preserved for signature compatibility
        blocked_artists = self.db.get_blocked_artists_30d()
        blacklisted_vids = set(self.db.get_blacklisted_video_ids())
        known_vids = set(known_video_ids or set())
        skip_keys = set(known_track_keys or set())
        familiar_artists = {
            normalize_artist(a) for a in (known_familiar_artists or set()) if normalize_artist(a)
        }
        is_western_only = station.region_policy == "western_only"
        embed_check_enabled = bool(getattr(self.yt, "embed_check_enabled", False))
        pool_artist_cap = max(1, int(max_tracks_per_artist_in_pool))
        pool_artist_counts = self.db.count_unheard_by_artist(
            station.station_id,
            station.max_popularity_views,
        )

        seed_plan = self._build_station_seed_plan(station, max_seeds=max_seeds)
        if is_western_only:
            seed_plan = [
                row
                for row in seed_plan
                if not is_japanese_music_entity(row[0], row[1])
            ]

        # Resolve official seed songs (and artist browseIds) via filter="songs"
        resolved_seeds: list[tuple[str, str, str | None, str | None]] = []
        seen_seed_vids: set[str] = set()
        for q_artist, q_title, seed_label, q_arch in seed_plan:
            res = self.yt.resolve_track(q_artist, q_title)
            if not res or not res.platform_track_id:
                continue
            if res.platform_track_id in seen_seed_vids:
                continue
            seen_seed_vids.add(res.platform_track_id)
            resolved_seeds.append((res.platform_track_id, seed_label, res.artist_browse_id, q_arch))

        # Aggregate candidates across distinct seed artists (prevents same-artist self-co-occurrence)
        graph_candidates: dict[str, TrackEntity] = {}
        seed_hit_sets: dict[str, set[str]] = {}
        source_labels: dict[str, list[str]] = {}
        seen_artist_browse_ids: set[str] = set()

        skipped_popular = 0
        skipped_blocked = 0
        skipped_known = 0
        skipped_region_mismatch = 0
        skipped_familiar_artist = 0
        skipped_artist_cap = 0

        def _ingest(tracks: list[TrackEntity], seed_artist_key: str, label: str) -> None:
            nonlocal skipped_popular, skipped_blocked, skipped_known, skipped_region_mismatch, skipped_familiar_artist
            for tr in tracks:
                if not tr.video_id:
                    continue
                if tr.video_id in known_vids or tr.video_id in seen_seed_vids or tr.canonical_key in skip_keys:
                    skipped_known += 1
                    continue
                # If already blacklisted (e.g. legacy yt_error_150), only retain when embed_check_enabled
                # so the replacement pass below can rescue it via OMV / Official Lyric Video.
                if tr.video_id in blacklisted_vids and not embed_check_enabled:
                    skipped_known += 1
                    continue
                a_norm = normalize_artist(tr.artist)
                if a_norm in blocked_artists:
                    skipped_blocked += 1
                    continue
                if is_western_only and is_japanese_music_entity(tr.artist, tr.title):
                    skipped_region_mismatch += 1
                    continue
                if a_norm and a_norm in familiar_artists:
                    skipped_familiar_artist += 1
                    continue
                if tr.view_count is not None and tr.view_count > station.max_popularity_views:
                    skipped_popular += 1
                    continue

                dedup_key = tr.video_id
                if dedup_key not in graph_candidates:
                    graph_candidates[dedup_key] = tr
                    seed_hit_sets[dedup_key] = {seed_artist_key}
                    source_labels[dedup_key] = [label]
                else:
                    seed_hit_sets[dedup_key].add(seed_artist_key)
                    if label not in source_labels[dedup_key]:
                        source_labels[dedup_key].append(label)
                    existing = graph_candidates[dedup_key]
                    if existing.view_count is None and tr.view_count is not None:
                        existing.view_count = tr.view_count
                    if existing.duration_seconds is None and tr.duration_seconds is not None:
                        existing.duration_seconds = tr.duration_seconds

        for seed_vid, seed_label, artist_browse_id, seed_arch in resolved_seeds:
            # Unify co-occurrence key by normalized seed artist so 2 songs by the same artist never self-co-occur
            artist_key = f"artist:{normalize_artist(seed_label) or seed_label.lower()}"

            # 1. Watch Playlist Radio neighborhood (official songs only, slop filtered in adapter)
            related = self.yt.get_related_tracks(
                seed_vid,
                limit=25,
                seed_artist=seed_label,
                seed_archetype=seed_arch,
            )
            _ingest(related, artist_key, f"{seed_label} 関連")

            # 2. Related real artists -> deep cuts (songs #3..#15), deduplicated per artist_browse_id
            if artist_browse_id and artist_browse_id not in seen_artist_browse_ids:
                seen_artist_browse_ids.add(artist_browse_id)
                deep_cuts = self.yt.get_artist_related_deep_cuts(
                    artist_browse_id,
                    seed_label=seed_label,
                    seed_archetype=seed_arch,
                    max_related_artists=3,
                )
                _ingest(deep_cuts, artist_key, f"{seed_label} 関連アーティスト")

        # Prioritize multi-seed co-occurring gems first, replacing embed-disabled videos with OMV / Official Lyric Video
        ranked_graph = sorted(
            graph_candidates.items(),
            key=lambda kv: (len(seed_hit_sets[kv[0]]), kv[1].view_count or 0),
            reverse=True,
        )

        added = 0
        cooccurring_count = 0
        mir_enriched_count = 0
        skipped_unembeddable = 0
        replaced_omv = 0
        replaced_lyric = 0
        alt_searches_used = 0
        alt_search_cap = max(0, int(max_alt_searches_per_dig))
        persisted_vids: set[str] = set()
        persisted_keys: set[str] = set()
        alt_finder = getattr(self.yt, "find_embeddable_alternative", None)

        for vid_key, track in ranked_graph:
            if added >= target_pool_size:
                break
            if track.canonical_key in persisted_keys:
                continue
            a_norm = normalize_artist(track.artist)
            if a_norm and pool_artist_counts.get(a_norm, 0) >= pool_artist_cap:
                skipped_artist_cap += 1
                continue

            replacement_tag = ""
            is_unembeddable = (track.video_id in blacklisted_vids) or (
                embed_check_enabled and not self.yt.is_video_embeddable(track.video_id)
            )
            if is_unembeddable:
                if track.video_id not in blacklisted_vids:
                    self.db.blacklist_unplayable(
                        track.video_id,
                        reason="oembed_embed_disabled",
                        yt_error_code=150,
                        ttl_days=30,
                    )
                    blacklisted_vids.add(track.video_id)

                alt_result = None
                if callable(alt_finder) and alt_searches_used < alt_search_cap:
                    alt_searches_used += 1
                    alt_result = alt_finder(
                        track.artist,
                        track.title,
                        exclude_video_ids=known_vids | blacklisted_vids | seen_seed_vids | persisted_vids,
                    )

                if alt_result is None:
                    skipped_unembeddable += 1
                    continue

                alt_track, alt_kind = alt_result
                if not alt_track.video_id or alt_track.video_id in persisted_vids:
                    skipped_unembeddable += 1
                    continue

                track = TrackEntity(
                    track_id=track.track_id,
                    canonical_key=track.canonical_key,
                    isrc=track.isrc,
                    mbid=track.mbid,
                    artist=track.artist,
                    title=track.title,
                    tags=list(track.tags),
                    release_year=track.release_year,
                    view_count=alt_track.view_count if alt_track.view_count is not None else track.view_count,
                    duration_seconds=(
                        alt_track.duration_seconds
                        if alt_track.duration_seconds is not None
                        else track.duration_seconds
                    ),
                    video_id=alt_track.video_id,
                    source=alt_track.source,
                    seed_artist=track.seed_artist,
                    seed_archetype=track.seed_archetype,
                )
                if alt_kind == "omv":
                    replaced_omv += 1
                    replacement_tag = " [公式MV代替]"
                else:
                    replaced_lyric += 1
                    replacement_tag = " [公式Lyric代替]"

            cooc = max(1, len(seed_hit_sets[vid_key]))
            sources = source_labels[vid_key]
            track.cooccurrence_count = cooc
            track.discovery_sources = sources
            track.discovery_route = f"{_format_discovery_route(sources, cooc)}{replacement_tag}"
            if cooc >= 2:
                cooccurring_count += 1

            # Bounded Stage 2 MIR enrichment (top co-occurring tracks first, capped at max_mir_tracks_per_dig)
            if (
                self.enable_mir_enrichment
                and self.mir_extractor is not None
                and mir_enriched_count < self.max_mir_tracks_per_dig
            ):
                vec16, est_bpm, _src = self.mir_extractor.extract_16d_vector(track, allow_network=True)
                track.mir_vector = [round(float(x), 5) for x in vec16]
                track.deezer_bpm = est_bpm  # None if unmeasured; never fake 118.0
                cached_meta = self.db.get_audio_feature_cache(
                    canonical_key=track.canonical_key,
                    video_id=track.video_id,
                )
                if cached_meta is not None and cached_meta.get("deezer_id") is not None:
                    track.deezer_id = int(cached_meta["deezer_id"])
                    if cached_meta.get("gain") is not None:
                        track.deezer_gain = float(cached_meta["gain"])
                    if cached_meta.get("rank") is not None:
                        track.deezer_rank = int(cached_meta["rank"])
                mir_enriched_count += 1

            self.db.upsert_track(station.station_id, track)
            persisted_vids.add(track.video_id)
            persisted_keys.add(track.canonical_key)
            if a_norm:
                pool_artist_counts[a_norm] = pool_artist_counts.get(a_norm, 0) + 1
            added += 1

        total_playable = self.db.count_playable_candidates(
            station.station_id,
            station.max_popularity_views,
            exclude_track_ids=known_vids,
            exclude_canonical_keys=skip_keys,
        )
        return {
            "station_id": station.station_id,
            "added": added,
            "cooccurring_added": cooccurring_count,
            "mir_enriched": mir_enriched_count,
            "skipped_popular": skipped_popular,
            "skipped_blocked_30d": skipped_blocked,
            "skipped_known_history": skipped_known,
            "skipped_region_mismatch": skipped_region_mismatch,
            "skipped_familiar_artist": skipped_familiar_artist,
            "skipped_artist_cap": skipped_artist_cap,
            "skipped_unembeddable": skipped_unembeddable,
            "replaced_omv": replaced_omv,
            "replaced_lyric": replaced_lyric,
            "alt_searches_used": alt_searches_used,
            "total_playable": total_playable,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hidden Gem Radio Stage 1 Pool Manager (purge unrated legacy tracks or dig fresh candidates)."
    )
    parser.add_argument("--db", default="data/radio.db", help="SQLite database path")
    parser.add_argument(
        "--reset-tracks",
        action="store_true",
        help="Purge unrated legacy candidate tracks while preserving playlists and Good/BAD feedback history.",
    )
    args = parser.parse_args(argv)

    db = RadioDatabase(args.db)
    if args.reset_tracks:
        summary = db.reset_unrated_tracks()
        print(json.dumps({"status": "ok", "action": "reset_tracks", **summary}, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
