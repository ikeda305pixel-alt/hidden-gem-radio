"""SQLite persistence for tracks, station candidate pools, play history, and feedback."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.models import (
    LEGACY_STATION_ID_MAP,
    TrackEntity,
    is_japanese_artist_or_text,
    normalize_artist,
    normalize_track_key,
)

THIRTY_DAYS_SECONDS = 30 * 24 * 60 * 60
VALID_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
WESTERN_ONLY_STATION_IDS = frozenset(
    {
        "us_hiphop_bouncy",
        "edm_dance_party",
        "melodic_rap_alt",
        "indie_groove_cult",
    }
)


DEFAULT_PLAYLISTS: tuple[tuple[str, str], ...] = (
    ("pl_favorites", "Good お気に入り"),
    ("pl_hiphop", "US & JP Hip-Hop"),
    ("pl_party_drive", "Party & Drive"),
    ("pl_groove_chill", "Groove & Chill"),
)


def sanitize_video_id(raw: str | None) -> str | None:
    """Extract and strictly validate a YouTube video_id against ^[A-Za-z0-9_-]{6,32}$."""
    if not raw:
        return None
    candidate = str(raw).strip()
    if not candidate:
        return None
    if "youtube.com" in candidate or "youtu.be" in candidate:
        try:
            parsed = urlparse(candidate)
            qs = parse_qs(parsed.query)
            if qs.get("v"):
                candidate = str(qs["v"][0]).strip()
            elif "youtu.be" in parsed.netloc and parsed.path.strip("/"):
                candidate = parsed.path.strip("/").split("/")[0].strip()
        except ValueError:
            return None
    if not VALID_VIDEO_ID_RE.fullmatch(candidate):
        return None
    return candidate


def build_youtube_watch_videos_urls(video_ids: list[str], chunk_size: int = 50) -> list[str]:
    """Generate auth-free YouTube anonymous playlist URLs chunked in groups of 50 sanitized video IDs."""
    cleaned = [sanitized for vid in video_ids if (sanitized := sanitize_video_id(vid)) is not None]
    unique = list(dict.fromkeys(cleaned))
    if not unique:
        return []
    size = max(1, int(chunk_size))
    return [
        "https://www.youtube.com/watch_videos?video_ids=" + ",".join(unique[i : i + size])
        for i in range(0, len(unique), size)
    ]


class RadioDatabase:
    """Lightweight SQLite repository for Hidden Gem Radio."""

    def __init__(self, db_path: Path | str = Path("data/radio.db")) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        now = time.time()
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    track_id TEXT PRIMARY KEY,
                    canonical_key TEXT NOT NULL UNIQUE,
                    isrc TEXT,
                    mbid TEXT,
                    artist TEXT NOT NULL,
                    artist_norm TEXT NOT NULL,
                    title TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    release_year INTEGER,
                    view_count INTEGER,
                    duration_seconds INTEGER,
                    video_id TEXT,
                    unplayable INTEGER NOT NULL DEFAULT 0,
                    source TEXT NOT NULL DEFAULT 'unknown',
                    seed_artist TEXT,
                    seed_archetype TEXT,
                    cooccurrence_count INTEGER NOT NULL DEFAULT 1,
                    discovery_sources_json TEXT NOT NULL DEFAULT '[]',
                    discovery_route TEXT,
                    deezer_id INTEGER,
                    deezer_bpm REAL,
                    deezer_gain REAL,
                    deezer_rank INTEGER,
                    mir_vector_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_tracks_video_id ON tracks(video_id);
                CREATE INDEX IF NOT EXISTS idx_tracks_artist_norm ON tracks(artist_norm);
                CREATE INDEX IF NOT EXISTS idx_tracks_isrc ON tracks(isrc);
                CREATE INDEX IF NOT EXISTS idx_tracks_mbid ON tracks(mbid);

                CREATE TABLE IF NOT EXISTS station_candidates (
                    station_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    added_at REAL NOT NULL,
                    PRIMARY KEY (station_id, track_id),
                    FOREIGN KEY (track_id) REFERENCES tracks(track_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    station_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    video_id TEXT,
                    artist_norm TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    reward REAL NOT NULL,
                    listen_seconds REAL,
                    duration_seconds REAL,
                    created_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_feedback_station ON feedback_events(station_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_feedback_artist_event
                    ON feedback_events(artist_norm, event_type, created_at);

                CREATE TABLE IF NOT EXISTS unplayable_blacklist (
                    video_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL DEFAULT 'playback_error',
                    yt_error_code INTEGER,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON unplayable_blacklist(expires_at);

                CREATE TABLE IF NOT EXISTS playlists (
                    playlist_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    youtube_playlist_id TEXT,
                    created_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS playlist_tracks (
                    playlist_id TEXT NOT NULL,
                    video_id TEXT NOT NULL,
                    track_id TEXT,
                    artist TEXT NOT NULL,
                    title TEXT NOT NULL,
                    station_id TEXT,
                    added_at REAL NOT NULL,
                    PRIMARY KEY (playlist_id, video_id),
                    FOREIGN KEY (playlist_id) REFERENCES playlists(playlist_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_playlist_tracks_added
                    ON playlist_tracks(playlist_id, added_at DESC);

                CREATE TABLE IF NOT EXISTS audio_features_cache (
                    canonical_key TEXT PRIMARY KEY,
                    video_id TEXT,
                    deezer_id INTEGER,
                    artist TEXT NOT NULL,
                    title TEXT NOT NULL,
                    bpm REAL,
                    gain REAL,
                    rank INTEGER,
                    duration INTEGER,
                    preview_url TEXT,
                    mir_a1_c4_json TEXT,
                    analysis_source TEXT NOT NULL DEFAULT 'deezer_metadata',
                    updated_at REAL NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_audio_cache_video_id ON audio_features_cache(video_id);

                CREATE TABLE IF NOT EXISTS station_profiles (
                    station_id TEXT PRIMARY KEY,
                    centroid_json TEXT NOT NULL,
                    axis_weights_json TEXT NOT NULL,
                    slow_weights_json TEXT NOT NULL,
                    sample_counts_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                );
                """
            )
            # Safe idempotent migration for pre-existing radio.db files
            cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(tracks)").fetchall()}
            migrations = (
                ("seed_artist", "ALTER TABLE tracks ADD COLUMN seed_artist TEXT;"),
                ("seed_archetype", "ALTER TABLE tracks ADD COLUMN seed_archetype TEXT;"),
                ("cooccurrence_count", "ALTER TABLE tracks ADD COLUMN cooccurrence_count INTEGER NOT NULL DEFAULT 1;"),
                ("discovery_sources_json", "ALTER TABLE tracks ADD COLUMN discovery_sources_json TEXT NOT NULL DEFAULT '[]';"),
                ("discovery_route", "ALTER TABLE tracks ADD COLUMN discovery_route TEXT;"),
                ("deezer_id", "ALTER TABLE tracks ADD COLUMN deezer_id INTEGER;"),
                ("deezer_bpm", "ALTER TABLE tracks ADD COLUMN deezer_bpm REAL;"),
                ("deezer_gain", "ALTER TABLE tracks ADD COLUMN deezer_gain REAL;"),
                ("deezer_rank", "ALTER TABLE tracks ADD COLUMN deezer_rank INTEGER;"),
                ("mir_vector_json", "ALTER TABLE tracks ADD COLUMN mir_vector_json TEXT;"),
            )
            for col_name, ddl in migrations:
                if col_name not in cols:
                    conn.execute(ddl)

            for pl_id, pl_name in DEFAULT_PLAYLISTS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO playlists (playlist_id, name, youtube_playlist_id, created_at)
                    VALUES (?, ?, NULL, ?)
                    """,
                    (pl_id, pl_name, now),
                )

            # Safe idempotent migration for deprecated station IDs (e.g. indie_groove_hooks -> indie_groove_cult)
            for old_id, new_id in LEGACY_STATION_ID_MAP.items():
                conn.execute(
                    """
                    INSERT OR IGNORE INTO station_candidates (station_id, track_id, added_at)
                    SELECT ?, track_id, added_at FROM station_candidates WHERE station_id = ?
                    """,
                    (new_id, old_id),
                )
                conn.execute("DELETE FROM station_candidates WHERE station_id = ?", (old_id,))
                conn.execute(
                    "UPDATE feedback_events SET station_id = ? WHERE station_id = ?",
                    (new_id, old_id),
                )
                conn.execute(
                    "UPDATE playlist_tracks SET station_id = ? WHERE station_id = ?",
                    (new_id, old_id),
                )
                has_new_prof = conn.execute(
                    "SELECT 1 FROM station_profiles WHERE station_id = ? LIMIT 1",
                    (new_id,),
                ).fetchone()
                if has_new_prof is None:
                    conn.execute(
                        "UPDATE station_profiles SET station_id = ? WHERE station_id = ?",
                        (new_id, old_id),
                    )
                else:
                    conn.execute("DELETE FROM station_profiles WHERE station_id = ?", (old_id,))

    def upsert_track(self, station_id: str, track: TrackEntity) -> str:
        """Insert or merge a platform-agnostic track and associate it with station_id."""
        now = time.time()
        artist_norm = normalize_artist(track.artist)
        tags_json = json.dumps(sorted({t.lower() for t in track.tags if t}), ensure_ascii=False)
        disc_sources_json = json.dumps(list(track.discovery_sources), ensure_ascii=False)
        mir_json = json.dumps(track.mir_vector) if track.mir_vector is not None else None

        with self._connect() as conn:
            # Match by canonical_key, ISRC, or MBID to keep entities deduplicated across platforms
            existing = conn.execute(
                """
                SELECT track_id, tags_json, isrc, mbid, release_year, view_count,
                       duration_seconds, video_id, unplayable, source, seed_artist, seed_archetype,
                       cooccurrence_count, discovery_sources_json, discovery_route,
                       deezer_id, deezer_bpm, deezer_gain, deezer_rank, mir_vector_json
                FROM tracks
                WHERE track_id = ?
                   OR canonical_key = ?
                   OR (? IS NOT NULL AND isrc = ?)
                   OR (? IS NOT NULL AND mbid = ?)
                LIMIT 1
                """,
                (track.track_id, track.canonical_key, track.isrc, track.isrc, track.mbid, track.mbid),
            ).fetchone()

            if existing is None:
                resolved_id = track.track_id
                conn.execute(
                    """
                    INSERT INTO tracks (
                        track_id, canonical_key, isrc, mbid, artist, artist_norm, title,
                        tags_json, release_year, view_count, duration_seconds, video_id,
                        unplayable, source, seed_artist, seed_archetype,
                        cooccurrence_count, discovery_sources_json, discovery_route,
                        deezer_id, deezer_bpm, deezer_gain, deezer_rank, mir_vector_json,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_id,
                        track.canonical_key,
                        track.isrc,
                        track.mbid,
                        track.artist,
                        artist_norm,
                        track.title,
                        tags_json,
                        track.release_year,
                        track.view_count,
                        track.duration_seconds,
                        track.video_id,
                        1 if track.unplayable else 0,
                        track.source,
                        track.seed_artist,
                        track.seed_archetype,
                        max(1, int(track.cooccurrence_count)),
                        disc_sources_json,
                        track.discovery_route,
                        track.deezer_id,
                        track.deezer_bpm,
                        track.deezer_gain,
                        track.deezer_rank,
                        mir_json,
                        now,
                        now,
                    ),
                )
            else:
                resolved_id = str(existing["track_id"])
                merged_tags = sorted(set(json.loads(existing["tags_json"] or "[]")) | set(track.tags))
                prev_sources = json.loads(existing["discovery_sources_json"] or "[]")
                merged_sources = list(dict.fromkeys([*prev_sources, *track.discovery_sources]))
                merged_cooc = max(int(existing["cooccurrence_count"] or 1), int(track.cooccurrence_count or 1), len(merged_sources) or 1)
                video_replaced = bool(
                    track.video_id
                    and track.video_id != existing["video_id"]
                    and not track.unplayable
                )
                new_unplayable = 0 if video_replaced else int(existing["unplayable"] or 0)
                new_source = track.source if video_replaced else str(existing["source"] or track.source)
                conn.execute(
                    """
                    UPDATE tracks
                    SET isrc = COALESCE(isrc, ?),
                        mbid = COALESCE(mbid, ?),
                        tags_json = ?,
                        release_year = COALESCE(release_year, ?),
                        view_count = COALESCE(?, view_count),
                        duration_seconds = COALESCE(?, duration_seconds),
                        video_id = COALESCE(?, video_id),
                        unplayable = ?,
                        source = ?,
                        seed_artist = COALESCE(?, seed_artist),
                        seed_archetype = COALESCE(?, seed_archetype),
                        cooccurrence_count = ?,
                        discovery_sources_json = ?,
                        discovery_route = COALESCE(?, discovery_route),
                        deezer_id = COALESCE(?, deezer_id),
                        deezer_bpm = COALESCE(?, deezer_bpm),
                        deezer_gain = COALESCE(?, deezer_gain),
                        deezer_rank = COALESCE(?, deezer_rank),
                        mir_vector_json = COALESCE(?, mir_vector_json),
                        updated_at = ?
                    WHERE track_id = ?
                    """,
                    (
                        track.isrc,
                        track.mbid,
                        json.dumps(merged_tags, ensure_ascii=False),
                        track.release_year,
                        track.view_count,
                        track.duration_seconds,
                        track.video_id,
                        new_unplayable,
                        new_source,
                        track.seed_artist,
                        track.seed_archetype,
                        merged_cooc,
                        json.dumps(merged_sources, ensure_ascii=False),
                        track.discovery_route,
                        track.deezer_id,
                        track.deezer_bpm,
                        track.deezer_gain,
                        track.deezer_rank,
                        mir_json,
                        now,
                        resolved_id,
                    ),
                )

            conn.execute(
                """
                INSERT OR IGNORE INTO station_candidates (station_id, track_id, added_at)
                VALUES (?, ?, ?)
                """,
                (station_id, resolved_id, now),
            )
            return resolved_id

    def get_track(self, track_id: str | None = None, video_id: str | None = None) -> TrackEntity | None:
        """Fetch a single track by track_id or video_id."""
        with self._connect() as conn:
            if track_id:
                row = conn.execute("SELECT * FROM tracks WHERE track_id = ? LIMIT 1", (track_id,)).fetchone()
            elif video_id:
                row = conn.execute("SELECT * FROM tracks WHERE video_id = ? LIMIT 1", (video_id,)).fetchone()
            else:
                return None
            return self._row_to_track(row) if row else None

    def mark_unplayable(self, video_id: str | None = None, track_id: str | None = None) -> int:
        """Flag a video_id / track_id as unplayable so it is skipped automatically."""
        now = time.time()
        with self._connect() as conn:
            if video_id and track_id:
                cur = conn.execute(
                    "UPDATE tracks SET unplayable = 1, updated_at = ? WHERE video_id = ? OR track_id = ?",
                    (now, video_id, track_id),
                )
            elif video_id:
                cur = conn.execute(
                    "UPDATE tracks SET unplayable = 1, updated_at = ? WHERE video_id = ?",
                    (now, video_id),
                )
            elif track_id:
                cur = conn.execute(
                    "UPDATE tracks SET unplayable = 1, updated_at = ? WHERE track_id = ?",
                    (now, track_id),
                )
            else:
                return 0
            return cur.rowcount

    def blacklist_unplayable(
        self,
        video_id: str,
        *,
        reason: str = "playback_error",
        yt_error_code: int | None = None,
        ttl_days: int = 30,
        now: float | None = None,
    ) -> float:
        """Blacklist a video_id for 30 days in SQLite without affecting BAD bandit feedback."""
        ts = time.time() if now is None else float(now)
        expires_at = ts + max(1, int(ttl_days)) * 24 * 60 * 60
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO unplayable_blacklist (video_id, reason, yt_error_code, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    reason = excluded.reason,
                    yt_error_code = excluded.yt_error_code,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (video_id.strip(), reason, yt_error_code, ts, expires_at),
            )
        return expires_at

    def get_blacklisted_video_ids(self, now: float | None = None) -> set[str]:
        """Return currently active (unexpired) blacklisted video IDs."""
        ts = time.time() if now is None else float(now)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT video_id FROM unplayable_blacklist WHERE expires_at > ?",
                (ts,),
            ).fetchall()
            return {str(r["video_id"]) for r in rows}

    def list_playlists(self) -> list[dict[str, object]]:
        """Return all playlists with track counts and 50-track chunked YouTube watch_videos URLs."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT p.playlist_id, p.name, p.youtube_playlist_id, p.created_at,
                       COUNT(pt.video_id) AS track_count
                FROM playlists p
                LEFT JOIN playlist_tracks pt ON pt.playlist_id = p.playlist_id
                GROUP BY p.playlist_id
                ORDER BY p.created_at ASC, p.playlist_id ASC
                """
            ).fetchall()
            result: list[dict[str, object]] = []
            for row in rows:
                pid = str(row["playlist_id"])
                vid_rows = conn.execute(
                    "SELECT video_id FROM playlist_tracks WHERE playlist_id = ? ORDER BY added_at DESC",
                    (pid,),
                ).fetchall()
                vids = [str(v["video_id"]) for v in vid_rows if v["video_id"]]
                result.append(
                    {
                        "playlist_id": pid,
                        "name": str(row["name"]),
                        "youtube_playlist_id": row["youtube_playlist_id"],
                        "track_count": int(row["track_count"]),
                        "youtube_watch_urls": build_youtube_watch_videos_urls(vids, chunk_size=50),
                    }
                )
            return result

    def create_playlist(self, name: str, youtube_playlist_id: str | None = None) -> dict[str, object]:
        """Create a new custom playlist."""
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Playlist name must not be empty")
        now = time.time()
        pid = f"pl_{int(now * 1000)}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO playlists (playlist_id, name, youtube_playlist_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (pid, clean_name, youtube_playlist_id, now),
            )
        return {
            "playlist_id": pid,
            "name": clean_name,
            "youtube_playlist_id": youtube_playlist_id,
            "track_count": 0,
            "youtube_watch_urls": [],
        }

    def playlist_exists(self, playlist_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM playlists WHERE playlist_id = ? LIMIT 1",
                (playlist_id,),
            ).fetchone()
            return row is not None

    def add_track_to_playlist(
        self,
        playlist_id: str,
        *,
        video_id: str,
        artist: str,
        title: str,
        track_id: str | None = None,
        station_id: str | None = None,
    ) -> dict[str, object]:
        """Add or update a track in a user playlist."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO playlist_tracks (playlist_id, video_id, track_id, artist, title, station_id, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(playlist_id, video_id) DO UPDATE SET
                    track_id = COALESCE(excluded.track_id, playlist_tracks.track_id),
                    artist = excluded.artist,
                    title = excluded.title,
                    station_id = COALESCE(excluded.station_id, playlist_tracks.station_id),
                    added_at = excluded.added_at
                """,
                (playlist_id, video_id.strip(), track_id, artist.strip(), title.strip(), station_id, now),
            )
        return {
            "playlist_id": playlist_id,
            "video_id": video_id.strip(),
            "track_id": track_id,
            "artist": artist.strip(),
            "title": title.strip(),
            "station_id": station_id,
            "added_at": now,
        }

    def remove_track_from_playlist(self, playlist_id: str, video_id: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM playlist_tracks WHERE playlist_id = ? AND video_id = ?",
                (playlist_id, video_id.strip()),
            )
            return cur.rowcount

    def get_playlist_tracks(self, playlist_id: str) -> dict[str, object] | None:
        with self._connect() as conn:
            pl = conn.execute(
                "SELECT playlist_id, name, youtube_playlist_id FROM playlists WHERE playlist_id = ?",
                (playlist_id,),
            ).fetchone()
            if pl is None:
                return None
            rows = conn.execute(
                """
                SELECT video_id, track_id, artist, title, station_id, added_at
                FROM playlist_tracks
                WHERE playlist_id = ?
                ORDER BY added_at DESC
                """,
                (playlist_id,),
            ).fetchall()
            tracks = [
                {
                    "video_id": str(r["video_id"]),
                    "track_id": r["track_id"],
                    "artist": str(r["artist"]),
                    "title": str(r["title"]),
                    "station_id": r["station_id"],
                    "added_at": float(r["added_at"]),
                }
                for r in rows
            ]
            vids = [t["video_id"] for t in tracks]
            return {
                "playlist_id": str(pl["playlist_id"]),
                "name": str(pl["name"]),
                "youtube_playlist_id": pl["youtube_playlist_id"],
                "track_count": len(tracks),
                "tracks": tracks,
                "youtube_watch_urls": build_youtube_watch_videos_urls(vids, chunk_size=50),
            }

    def record_feedback(
        self,
        station_id: str,
        track: TrackEntity,
        event_type: str,
        reward: float,
        listen_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> int:
        """Persist user feedback event and return its inserted feedback_id."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO feedback_events (
                    station_id, track_id, video_id, artist_norm, event_type,
                    reward, listen_seconds, duration_seconds, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    station_id,
                    track.track_id,
                    track.video_id,
                    normalize_artist(track.artist),
                    event_type.upper(),
                    float(reward),
                    listen_seconds,
                    duration_seconds,
                    time.time(),
                ),
            )
            return int(cur.lastrowid or 0)

    def delete_feedback(self, feedback_id: int) -> None:
        """Remove a feedback row by ID (compensating rollback when downstream model update fails)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM feedback_events WHERE id = ?", (int(feedback_id),))

    def get_blocked_artists_30d(self, now: float | None = None) -> set[str]:
        """Return normalized artist names rated BAD within the last 30 days."""
        cutoff = (now if now is not None else time.time()) - THIRTY_DAYS_SECONDS
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT artist_norm
                FROM feedback_events
                WHERE event_type = 'BAD' AND created_at >= ?
                """,
                (cutoff,),
            ).fetchall()
            return {normalize_artist(str(r["artist_norm"])) for r in rows if r["artist_norm"]}

    def get_seen_artists(self, station_id: str | None = None) -> set[str]:
        """Return normalized artist names that already received playback feedback."""
        with self._connect() as conn:
            if station_id:
                rows = conn.execute(
                    "SELECT DISTINCT artist_norm FROM feedback_events WHERE station_id = ?",
                    (station_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT DISTINCT artist_norm FROM feedback_events").fetchall()
            return {normalize_artist(str(r["artist_norm"])) for r in rows if r["artist_norm"]}

    def get_recently_heard_track_ids(self, station_id: str, limit: int = 200) -> set[str]:
        """Return track IDs already played/rated on this station to prioritize unheard tracks."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT track_id
                FROM feedback_events
                WHERE station_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (station_id, limit),
            ).fetchall()
            return {str(r["track_id"]) for r in rows}

    def get_skipped_or_bad_track_ids(self, station_id: str) -> set[str]:
        """Return track IDs explicitly rejected (BAD or EARLY_SKIP) on this station."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT track_id
                FROM feedback_events
                WHERE station_id = ? AND event_type IN ('BAD', 'EARLY_SKIP')
                """,
                (station_id,),
            ).fetchall()
            return {str(r["track_id"]) for r in rows}

    def get_eligible_candidates(
        self,
        station_id: str,
        max_popularity_views: int,
        exclude_track_ids: set[str] | None = None,
        include_heard_fallback: bool = True,
        exclude_canonical_keys: set[str] | None = None,
        now: float | None = None,
    ) -> list[TrackEntity]:
        """Return playable candidate tracks matching popularity cap, 30-day BAD artist filter, and blacklist."""
        blocked_artists = self.get_blocked_artists_30d(now=now)
        blacklisted_vids = self.get_blacklisted_video_ids(now=now)
        heard_ids = self.get_recently_heard_track_ids(station_id)
        rejected_ids = self.get_skipped_or_bad_track_ids(station_id) if include_heard_fallback else set()
        excluded = set(exclude_track_ids or set())
        excluded_keys = set(exclude_canonical_keys or set())
        western_only = station_id in WESTERN_ONLY_STATION_IDS

        with self._connect() as conn:
            heard_keys: set[str] = set()
            rejected_keys: set[str] = set()
            if heard_ids:
                ph = ",".join("?" for _ in heard_ids)
                h_rows = conn.execute(
                    f"SELECT artist, title FROM tracks WHERE track_id IN ({ph})",
                    tuple(heard_ids),
                ).fetchall()
                heard_keys = {normalize_track_key(str(r["artist"]), str(r["title"])) for r in h_rows}
            if rejected_ids:
                ph_r = ",".join("?" for _ in rejected_ids)
                r_rows = conn.execute(
                    f"SELECT artist, title FROM tracks WHERE track_id IN ({ph_r})",
                    tuple(rejected_ids),
                ).fetchall()
                rejected_keys = {normalize_track_key(str(r["artist"]), str(r["title"])) for r in r_rows}

            rows = conn.execute(
                """
                SELECT t.*
                FROM tracks t
                INNER JOIN station_candidates sc ON sc.track_id = t.track_id
                WHERE sc.station_id = ?
                  AND t.unplayable = 0
                  AND t.video_id IS NOT NULL
                  AND t.video_id != ''
                  AND (t.view_count IS NULL OR t.view_count <= ?)
                ORDER BY sc.added_at DESC
                """,
                (station_id, int(max_popularity_views)),
            ).fetchall()

        unheard: list[TrackEntity] = []
        heard_non_bad: list[TrackEntity] = []
        for row in rows:
            track = self._row_to_track(row)
            if track.track_id in excluded or (track.video_id and track.video_id in excluded):
                continue
            if track.video_id and track.video_id in blacklisted_vids:
                continue
            if western_only and is_japanese_artist_or_text(track.artist, track.title):
                continue
            dyn_key = normalize_track_key(track.artist, track.title)
            if excluded_keys and (track.canonical_key in excluded_keys or dyn_key in excluded_keys):
                continue
            if normalize_artist(track.artist) in blocked_artists:
                continue
            if track.track_id not in heard_ids and dyn_key not in heard_keys:
                unheard.append(track)
            elif include_heard_fallback and track.track_id not in rejected_ids and dyn_key not in rejected_keys:
                heard_non_bad.append(track)

        # Prefer strictly unheard candidates; fall back to previously liked/completed tracks only if exhausted
        return unheard if unheard else heard_non_bad

    def get_unheard_artist_counts(
        self,
        station_id: str,
        max_popularity_views: int,
        *,
        exclude_track_ids: set[str] | None = None,
        exclude_canonical_keys: set[str] | None = None,
    ) -> dict[str, int]:
        """Count unheard eligible candidate tracks per normalized artist in a station's pool."""
        counts: dict[str, int] = {}
        cands = self.get_eligible_candidates(
            station_id,
            max_popularity_views,
            exclude_track_ids=exclude_track_ids,
            include_heard_fallback=False,
            exclude_canonical_keys=exclude_canonical_keys,
        )
        for tr in cands:
            a_norm = normalize_artist(tr.artist)
            if a_norm:
                counts[a_norm] = counts.get(a_norm, 0) + 1
        return counts

    def count_playable_candidates(
        self,
        station_id: str,
        max_popularity_views: int,
        *,
        exclude_track_ids: set[str] | None = None,
        exclude_canonical_keys: set[str] | None = None,
    ) -> int:
        return len(
            self.get_eligible_candidates(
                station_id,
                max_popularity_views,
                exclude_track_ids=exclude_track_ids,
                include_heard_fallback=False,
                exclude_canonical_keys=exclude_canonical_keys,
            )
        )

    def count_feedback(self, station_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM feedback_events WHERE station_id = ?",
                (station_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def get_recent_played_artists(self, station_id: str, limit: int = 5) -> list[str]:
        """Return normalized artist names of the most recent N played/rated tracks on this station."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT artist_norm
                FROM feedback_events
                WHERE station_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (station_id, max(1, int(limit))),
            ).fetchall()
            return [str(r["artist_norm"]) for r in rows if r["artist_norm"]]

    def get_audio_feature_cache(
        self,
        *,
        canonical_key: str | None = None,
        video_id: str | None = None,
    ) -> dict[str, object] | None:
        """Fetch cached Deezer metadata & 12-dim MIR features from SQLite."""
        with self._connect() as conn:
            row = None
            if canonical_key:
                row = conn.execute(
                    "SELECT * FROM audio_features_cache WHERE canonical_key = ? LIMIT 1",
                    (canonical_key,),
                ).fetchone()
            if row is None and video_id:
                row = conn.execute(
                    "SELECT * FROM audio_features_cache WHERE video_id = ? LIMIT 1",
                    (video_id,),
                ).fetchone()
            if row is None:
                return None
            raw_mir = row["mir_a1_c4_json"]
            return {
                "canonical_key": str(row["canonical_key"]),
                "video_id": row["video_id"],
                "deezer_id": row["deezer_id"],
                "artist": str(row["artist"]),
                "title": str(row["title"]),
                "bpm": row["bpm"],
                "gain": row["gain"],
                "rank": row["rank"],
                "duration": row["duration"],
                "preview_url": row["preview_url"],
                "mir_a1_c4": list(json.loads(raw_mir)) if raw_mir else None,
                "analysis_source": str(row["analysis_source"]),
                "updated_at": float(row["updated_at"]),
            }

    def upsert_audio_feature_cache(
        self,
        *,
        canonical_key: str,
        video_id: str | None = None,
        deezer_id: int | None = None,
        artist: str = "",
        title: str = "",
        bpm: float | None = None,
        gain: float | None = None,
        rank: int | None = None,
        duration: int | None = None,
        preview_url: str | None = None,
        mir_a1_c4: list[float] | None = None,
        analysis_source: str = "deezer_metadata",
    ) -> None:
        """Store or update Deezer metadata and measured 12-dim MIR vector in SQLite cache."""
        now = time.time()
        mir_json = json.dumps([float(x) for x in mir_a1_c4]) if mir_a1_c4 is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audio_features_cache (
                    canonical_key, video_id, deezer_id, artist, title, bpm, gain, rank,
                    duration, preview_url, mir_a1_c4_json, analysis_source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(canonical_key) DO UPDATE SET
                    video_id = COALESCE(excluded.video_id, audio_features_cache.video_id),
                    deezer_id = COALESCE(excluded.deezer_id, audio_features_cache.deezer_id),
                    artist = COALESCE(NULLIF(excluded.artist, ''), audio_features_cache.artist),
                    title = COALESCE(NULLIF(excluded.title, ''), audio_features_cache.title),
                    bpm = COALESCE(excluded.bpm, audio_features_cache.bpm),
                    gain = COALESCE(excluded.gain, audio_features_cache.gain),
                    rank = COALESCE(excluded.rank, audio_features_cache.rank),
                    duration = COALESCE(excluded.duration, audio_features_cache.duration),
                    preview_url = COALESCE(excluded.preview_url, audio_features_cache.preview_url),
                    mir_a1_c4_json = COALESCE(excluded.mir_a1_c4_json, audio_features_cache.mir_a1_c4_json),
                    analysis_source = excluded.analysis_source,
                    updated_at = excluded.updated_at
                """,
                (
                    canonical_key,
                    video_id,
                    deezer_id,
                    artist,
                    title,
                    bpm,
                    gain,
                    rank,
                    duration,
                    preview_url,
                    mir_json,
                    analysis_source,
                    now,
                ),
            )

    def get_station_profile(self, station_id: str) -> dict[str, object] | None:
        """Load calibrated 16-dim centroid, consistency axis_weights, and slow_weights for station_id."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM station_profiles WHERE station_id = ? LIMIT 1",
                (station_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "station_id": str(row["station_id"]),
                "centroid": list(json.loads(row["centroid_json"])),
                "axis_weights": list(json.loads(row["axis_weights_json"])),
                "slow_weights": list(json.loads(row["slow_weights_json"])),
                "sample_counts": dict(json.loads(row["sample_counts_json"] or "{}")),
                "updated_at": float(row["updated_at"]),
            }

    def upsert_station_profile(
        self,
        *,
        station_id: str,
        centroid: list[float],
        axis_weights: list[float],
        slow_weights: list[float],
        sample_counts: dict[str, int] | None = None,
    ) -> None:
        """Persist calibrated 16-dim station centroid, axis_weights, and slow_weights to SQLite."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO station_profiles (
                    station_id, centroid_json, axis_weights_json, slow_weights_json,
                    sample_counts_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(station_id) DO UPDATE SET
                    centroid_json = excluded.centroid_json,
                    axis_weights_json = excluded.axis_weights_json,
                    slow_weights_json = excluded.slow_weights_json,
                    sample_counts_json = excluded.sample_counts_json,
                    updated_at = excluded.updated_at
                """,
                (
                    station_id,
                    json.dumps([float(x) for x in centroid]),
                    json.dumps([float(x) for x in axis_weights]),
                    json.dumps([float(x) for x in slow_weights]),
                    json.dumps(sample_counts or {}),
                    now,
                ),
            )

    def update_station_slow_weights(
        self,
        *,
        station_id: str,
        slow_weights: list[float],
        default_centroid: list[float],
        default_axis_weights: list[float],
    ) -> None:
        """Persist only slow_weights for station_id without overwriting externally calibrated centroid/axis_weights."""
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO station_profiles (
                    station_id, centroid_json, axis_weights_json, slow_weights_json,
                    sample_counts_json, updated_at
                ) VALUES (?, ?, ?, ?, '{}', ?)
                ON CONFLICT(station_id) DO UPDATE SET
                    slow_weights_json = excluded.slow_weights_json
                """,
                (
                    station_id,
                    json.dumps([float(x) for x in default_centroid]),
                    json.dumps([float(x) for x in default_axis_weights]),
                    json.dumps([float(x) for x in slow_weights]),
                    now,
                ),
            )

    def reset_unrated_tracks(self) -> dict[str, int]:
        """Purge unrated legacy candidate tracks while preserving playlists, Good/BAD feedback, and rated tracks."""
        with self._connect() as conn:
            before_tracks = int(conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
            # Protect tracks referenced by feedback_events OR playlist_tracks (by track_id OR video_id)
            cur_links = conn.execute(
                """
                DELETE FROM station_candidates
                WHERE track_id NOT IN (SELECT DISTINCT track_id FROM feedback_events WHERE track_id IS NOT NULL)
                  AND track_id NOT IN (SELECT DISTINCT track_id FROM playlist_tracks WHERE track_id IS NOT NULL)
                  AND track_id NOT IN (
                      SELECT t.track_id FROM tracks t
                      WHERE t.video_id IS NOT NULL
                        AND t.video_id IN (SELECT DISTINCT video_id FROM playlist_tracks WHERE video_id IS NOT NULL)
                  )
                """
            )
            cur_tracks = conn.execute(
                """
                DELETE FROM tracks
                WHERE track_id NOT IN (SELECT DISTINCT track_id FROM feedback_events WHERE track_id IS NOT NULL)
                  AND track_id NOT IN (SELECT DISTINCT track_id FROM playlist_tracks WHERE track_id IS NOT NULL)
                  AND (video_id IS NULL OR video_id NOT IN (SELECT DISTINCT video_id FROM playlist_tracks WHERE video_id IS NOT NULL))
                """
            )
            remaining_tracks = int(conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0])
            preserved_playlists = int(conn.execute("SELECT COUNT(*) FROM playlist_tracks").fetchone()[0])
            preserved_feedback = int(conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0])
            return {
                "before_tracks": before_tracks,
                "deleted_tracks": cur_tracks.rowcount,
                "deleted_candidate_links": cur_links.rowcount,
                "remaining_tracks": remaining_tracks,
                "preserved_playlist_tracks": preserved_playlists,
                "preserved_feedback_events": preserved_feedback,
            }

    @staticmethod
    def _row_to_track(row: sqlite3.Row) -> TrackEntity:
        keys = set(row.keys())
        raw_sources = row["discovery_sources_json"] if "discovery_sources_json" in keys else "[]"
        raw_mir = row["mir_vector_json"] if "mir_vector_json" in keys else None
        return TrackEntity(
            track_id=str(row["track_id"]),
            canonical_key=str(row["canonical_key"]),
            isrc=row["isrc"],
            mbid=row["mbid"],
            artist=str(row["artist"]),
            title=str(row["title"]),
            tags=list(json.loads(row["tags_json"] or "[]")),
            release_year=row["release_year"],
            view_count=row["view_count"],
            duration_seconds=row["duration_seconds"],
            video_id=row["video_id"],
            unplayable=bool(row["unplayable"]),
            source=str(row["source"]),
            seed_artist=row["seed_artist"] if "seed_artist" in keys else None,
            seed_archetype=row["seed_archetype"] if "seed_archetype" in keys else None,
            cooccurrence_count=int(row["cooccurrence_count"]) if "cooccurrence_count" in keys and row["cooccurrence_count"] is not None else 1,
            discovery_sources=list(json.loads(raw_sources or "[]")),
            discovery_route=row["discovery_route"] if "discovery_route" in keys else None,
            deezer_id=row["deezer_id"] if "deezer_id" in keys else None,
            deezer_bpm=row["deezer_bpm"] if "deezer_bpm" in keys else None,
            deezer_gain=row["deezer_gain"] if "deezer_gain" in keys else None,
            deezer_rank=row["deezer_rank"] if "deezer_rank" in keys else None,
            mir_vector=list(json.loads(raw_mir)) if raw_mir else None,
        )
