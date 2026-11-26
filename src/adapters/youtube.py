"""YouTube Music adapter powered by ytmusicapi (Phase 1 MVP)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ytmusicapi import YTMusic

from src.adapters.base import MusicPlatformAdapter
from src.models import PlaybackResolution, TrackEntity, normalize_text

logger = logging.getLogger(__name__)

_VIEW_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([KMB])?", re.IGNORECASE)
_MIN_SONG_SECONDS = 60
_MAX_SONG_SECONDS = 660  # Reject full album mixes / hour-long DJ sets & podcasts


def parse_view_count(raw: Any) -> int | None:
    """Parse ytmusicapi view strings like '452', '15K', '1.4M views', '2.1B'."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).replace(",", "").strip()
    match = _VIEW_RE.search(text)
    if not match:
        return None
    num = float(match.group(1))
    suffix = (match.group(2) or "").upper()
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}.get(suffix, 1)
    return int(num * multiplier)


def parse_duration_seconds(raw_seconds: Any = None, raw_length: str | None = None) -> int | None:
    """Parse duration from integer seconds or 'M:SS' / 'H:MM:SS' strings."""
    if isinstance(raw_seconds, (int, float)) and raw_seconds > 0:
        return int(raw_seconds)
    if not raw_length:
        return None
    parts = [p.strip() for p in str(raw_length).split(":")]
    if not all(p.isdigit() for p in parts):
        return None
    nums = [int(p) for p in parts]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    return None


def _extract_artist_name(item: dict[str, Any], fallback_title: str = "") -> str:
    artists = item.get("artists") or []
    names = [str(a.get("name", "")).strip() for a in artists if isinstance(a, dict) and a.get("name")]
    if names:
        return names[0]
    if " - " in fallback_title:
        return fallback_title.split(" - ", 1)[0].strip()
    return "Unknown Artist"


def _clean_video_title(title: str, artist: str) -> str:
    """Strip duplicated 'Artist - ' prefix from YouTube Music video titles."""
    prefix = f"{artist} - "
    if title.lower().startswith(prefix.lower()):
        return title[len(prefix) :].strip()
    return title.strip()


def resolve_youtube_auth_file(
    explicit_path: Path | str | None = None,
    project_root: Path | None = None,
) -> Path | None:
    """Locate an existing ytmusicapi auth file (data/browser.json, data/oauth.json, config/browser.json)."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    root = project_root or Path(__file__).resolve().parent.parent.parent
    candidates.extend(
        [
            root / "data" / "browser.json",
            root / "data" / "oauth.json",
            root / "config" / "browser.json",
        ]
    )
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


class YouTubeMusicAdapter(MusicPlatformAdapter):
    """Resolves tracks to YouTube videoIds, expands pools via Watch Playlists, and syncs user playlists."""

    platform_name = "youtube_music"

    def __init__(
        self,
        browser_auth_path: Path | str | None = None,
        client: YTMusic | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._songs_filter_enabled = False
        self._authenticated = False
        self._auth_source: str | None = None
        self._auth_error: str | None = None

        if client is not None:
            self._yt = client
            self._songs_filter_enabled = True
            self._authenticated = True
            self._auth_source = "injected_client"
        else:
            auth_file = resolve_youtube_auth_file(browser_auth_path, project_root=project_root)
            if auth_file is not None:
                try:
                    self._yt = YTMusic(str(auth_file))
                    self._songs_filter_enabled = True
                    self._authenticated = True
                    self._auth_source = str(auth_file)
                except Exception as exc:  # noqa: BLE001
                    self._auth_error = str(exc)
                    logger.warning(
                        "Failed to load ytmusicapi auth %s (%s); falling back to anonymous mode",
                        auth_file,
                        exc,
                    )
                    self._yt = YTMusic()
            else:
                self._yt = YTMusic()

    @property
    def client(self) -> YTMusic:
        return self._yt

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated

    @property
    def auth_source(self) -> str | None:
        return self._auth_source

    @property
    def auth_error(self) -> str | None:
        return self._auth_error

    def list_user_playlists(self, limit: int = 50) -> dict[str, Any]:
        """List authenticated YouTube Music library playlists, or safely return unauthenticated fallback."""
        if not self._authenticated:
            return {
                "authenticated": False,
                "auth_source": None,
                "playlists": [],
                "reason": self._auth_error or "No data/browser.json or data/oauth.json configured",
            }
        try:
            raw = self._yt.get_library_playlists(limit=limit) or []
            playlists = [
                {
                    "playlist_id": str(item.get("playlistId") or ""),
                    "title": str(item.get("title") or "Untitled"),
                    "count": item.get("count"),
                }
                for item in raw
                if isinstance(item, dict) and item.get("playlistId")
            ]
            return {
                "authenticated": True,
                "auth_source": self._auth_source,
                "playlists": playlists,
                "reason": None,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("ytmusicapi get_library_playlists failed: %s", exc)
            return {
                "authenticated": False,
                "auth_source": self._auth_source,
                "playlists": [],
                "reason": str(exc),
            }

    def sync_liked_track(
        self,
        video_id: str,
        *,
        youtube_playlist_id: str | None = None,
    ) -> dict[str, Any]:
        """Sync a Good-rated track to YouTube Music (LIKE + optional add_playlist_items)."""
        if not self._authenticated:
            return {
                "authenticated": False,
                "liked": False,
                "playlist_added": False,
                "reason": self._auth_error or "No data/browser.json or data/oauth.json configured",
            }
        liked = False
        playlist_added = False
        errors: list[str] = []

        try:
            self._yt.rate_song(video_id, "LIKE")
            liked = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("ytmusicapi rate_song(%s, LIKE) failed: %s", video_id, exc)
            errors.append(f"rate_song: {exc}")

        if youtube_playlist_id:
            try:
                self._yt.add_playlist_items(youtube_playlist_id, [video_id], duplicates=False)
                playlist_added = True
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ytmusicapi add_playlist_items(%s, %s) failed: %s",
                    youtube_playlist_id,
                    video_id,
                    exc,
                )
                errors.append(f"add_playlist_items: {exc}")

        return {
            "authenticated": True,
            "liked": liked,
            "playlist_added": playlist_added,
            "youtube_playlist_id": youtube_playlist_id,
            "errors": errors,
        }

    def search_playable(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Search YouTube Music with graceful fallback when anonymous 'songs' filter returns empty."""
        filters = ("songs", "videos", None) if self._songs_filter_enabled else ("videos", None)
        candidates: list[dict[str, Any]] = []
        for search_filter in filters:
            try:
                rows = self._yt.search(query, filter=search_filter, limit=limit)
            except Exception as exc:  # noqa: BLE001
                logger.warning("ytmusicapi search failed (query=%r, filter=%r): %s", query, search_filter, exc)
                continue

            playable = [
                r
                for r in (rows or [])
                if isinstance(r, dict)
                and r.get("videoId")
                and r.get("resultType") in {"song", "video", None}
            ]
            if playable:
                candidates = playable
                break
        return candidates

    def resolve_track(
        self,
        artist: str,
        title: str,
        isrc: str | None = None,
        max_popularity_views: int | None = None,
    ) -> PlaybackResolution | None:
        """Resolve (artist, title) into a playable YouTube videoId respecting duration & view limits."""
        query = f"{artist} {title}".strip()
        results = self.search_playable(query, limit=6)
        if not results:
            return None

        target_artist = normalize_text(artist)
        target_title = normalize_text(title)
        best_item: dict[str, Any] | None = None
        best_score = -10_000

        for item in results:
            duration = parse_duration_seconds(item.get("duration_seconds"), item.get("duration"))
            if duration is not None and not (_MIN_SONG_SECONDS <= duration <= _MAX_SONG_SECONDS):
                continue

            views = parse_view_count(item.get("views"))
            if max_popularity_views is not None and views is not None and views > max_popularity_views:
                continue

            raw_title = str(item.get("title") or "")
            found_artist = _extract_artist_name(item, raw_title)
            norm_title = normalize_text(raw_title)
            norm_artist = normalize_text(found_artist)

            score = 0
            if item.get("resultType") == "song":
                score += 4
            if target_title and (target_title in norm_title or norm_title in target_title):
                score += 4
            if target_artist and (target_artist in norm_artist or target_artist in norm_title):
                score += 3
            if "full album" in raw_title.lower() or "playlist" in raw_title.lower() or "1 hour" in raw_title.lower():
                score -= 5

            if score > best_score:
                best_score = score
                best_item = item

        # [M3] Never bypass duration/view filters by falling back to results[0]
        if best_item is None:
            return None

        raw_title = str(best_item.get("title") or title)
        found_artist = _extract_artist_name(best_item, raw_title)
        year_raw = best_item.get("year")
        release_year = int(year_raw) if isinstance(year_raw, (int, str)) and str(year_raw).isdigit() else None

        return PlaybackResolution(
            platform=self.platform_name,
            platform_track_id=str(best_item["videoId"]),
            title=_clean_video_title(raw_title, found_artist) or title,
            artist=found_artist or artist,
            view_count=parse_view_count(best_item.get("views")),
            duration_seconds=parse_duration_seconds(best_item.get("duration_seconds"), best_item.get("duration")),
            release_year=release_year,
        )

    def get_related_tracks(
        self,
        platform_track_id: str,
        seed_tags: list[str] | None = None,
        limit: int = 25,
        seed_artist: str | None = None,
        seed_archetype: str | None = None,
    ) -> list[TrackEntity]:
        """Collect related candidate tracks via YouTube Music Watch Playlist without fabricating seed tags."""
        _ = seed_tags  # Intentionally unused: Watch Playlist tracks carry no verified genre tags
        try:
            watch = self._yt.get_watch_playlist(videoId=platform_track_id, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ytmusicapi get_watch_playlist failed for %s: %s", platform_track_id, exc)
            return []

        entities: list[TrackEntity] = []
        for item in watch.get("tracks", []) or []:
            vid = item.get("videoId")
            raw_title = str(item.get("title") or "").strip()
            if not vid or not raw_title:
                continue
            duration = parse_duration_seconds(item.get("duration_seconds"), item.get("length"))
            if duration is not None and not (_MIN_SONG_SECONDS <= duration <= _MAX_SONG_SECONDS):
                continue

            artist = _extract_artist_name(item, raw_title)
            title = _clean_video_title(raw_title, artist)
            year_raw = item.get("year")
            release_year = int(year_raw) if isinstance(year_raw, (int, str)) and str(year_raw).isdigit() else None

            entities.append(
                TrackEntity.build(
                    artist=artist,
                    title=title,
                    tags=[],  # [C2] Never copy station seed_tags onto untagged Watch Playlist tracks
                    release_year=release_year,
                    view_count=parse_view_count(item.get("views")),
                    duration_seconds=duration,
                    video_id=str(vid),
                    source="watch_playlist",
                    seed_artist=seed_artist,
                    seed_archetype=seed_archetype,
                )
            )
        return entities
