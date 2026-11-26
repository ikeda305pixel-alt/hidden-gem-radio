"""SQLite persistence for tracks, station candidate pools, play history, and feedback."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.models import TrackEntity, normalize_artist

THIRTY_DAYS_SECONDS = 30 * 24 * 60 * 60


DEFAULT_PLAYLISTS: tuple[tuple[str, str], ...] = (
    ("pl_favorites", "Good お気に入り"),
    ("pl_hiphop", "US & JP Hip-Hop"),
    ("pl_party_drive", "Party & Drive"),
    ("pl_groove_chill", "Groove & Chill"),
)


def build_youtube_watch_videos_urls(video_ids: list[str], chunk_size: int = 50) -> list[str]:
    """Generate auth-free YouTube anonymous playlist URLs chunked in groups of 50 video IDs."""
    cleaned = [vid.strip() for vid in video_ids if vid and vid.strip()]
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
                """
            )
            # Safe idempotent migration for pre-existing radio.db files missing lineage columns
            cols = {str(r["name"]) for r in conn.execute("PRAGMA table_info(tracks)").fetchall()}
            if "seed_artist" not in cols:
                conn.execute("ALTER TABLE tracks ADD COLUMN seed_artist TEXT;")
            if "seed_archetype" not in cols:
                conn.execute("ALTER TABLE tracks ADD COLUMN seed_archetype TEXT;")

            for pl_id, pl_name in DEFAULT_PLAYLISTS:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO playlists (playlist_id, name, youtube_playlist_id, created_at)
                    VALUES (?, ?, NULL, ?)
                    """,
                    (pl_id, pl_name, now),
                )

    def upsert_track(self, station_id: str, track: TrackEntity) -> str:
        """Insert or merge a platform-agnostic track and associate it with station_id."""
        now = time.time()
        artist_norm = normalize_artist(track.artist)
        tags_json = json.dumps(sorted({t.lower() for t in track.tags if t}), ensure_ascii=False)

        with self._connect() as conn:
            # Match by canonical_key, ISRC, or MBID to keep entities deduplicated across platforms
            existing = conn.execute(
                """
                SELECT track_id, tags_json, isrc, mbid, release_year, view_count,
                       duration_seconds, video_id, unplayable, seed_artist, seed_archetype
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
                        unplayable, source, seed_artist, seed_archetype, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        now,
                        now,
                    ),
                )
            else:
                resolved_id = str(existing["track_id"])
                merged_tags = sorted(set(json.loads(existing["tags_json"] or "[]")) | set(track.tags))
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
                        seed_artist = COALESCE(?, seed_artist),
                        seed_archetype = COALESCE(?, seed_archetype),
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
                        track.seed_artist,
                        track.seed_archetype,
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
            conn.execute(
                "UPDATE tracks SET unplayable = 1, updated_at = ? WHERE video_id = ?",
                (ts, video_id.strip()),
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
            return {str(r["artist_norm"]) for r in rows}

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
            return {str(r["artist_norm"]) for r in rows}

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
    ) -> list[TrackEntity]:
        """Return playable candidate tracks matching popularity cap, 30-day BAD artist filter, and blacklist."""
        blocked_artists = self.get_blocked_artists_30d()
        blacklisted_vids = self.get_blacklisted_video_ids()
        heard_ids = self.get_recently_heard_track_ids(station_id)
        rejected_ids = self.get_skipped_or_bad_track_ids(station_id) if include_heard_fallback else set()
        excluded = set(exclude_track_ids or set())
        excluded_keys = set(exclude_canonical_keys or set())

        with self._connect() as conn:
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
            if excluded_keys and track.canonical_key in excluded_keys:
                continue
            if normalize_artist(track.artist) in blocked_artists:
                continue
            if track.track_id not in heard_ids:
                unheard.append(track)
            elif include_heard_fallback and track.track_id not in rejected_ids:
                heard_non_bad.append(track)

        # Prefer strictly unheard candidates; fall back to previously liked/completed tracks only if exhausted
        return unheard if unheard else heard_non_bad

    def count_playable_candidates(self, station_id: str, max_popularity_views: int) -> int:
        return len(self.get_eligible_candidates(station_id, max_popularity_views, include_heard_fallback=False))

    def count_feedback(self, station_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM feedback_events WHERE station_id = ?",
                (station_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    @staticmethod
    def _row_to_track(row: sqlite3.Row) -> TrackEntity:
        keys = set(row.keys())
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
        )
