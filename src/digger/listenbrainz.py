"""ListenBrainz and MusicBrainz metadata client using verified public endpoints."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from src.models import TrackEntity

logger = logging.getLogger(__name__)

USER_AGENT = "HiddenGemRadio/0.1 (https://github.com/ikeda305pixel-alt/hidden-gem-radio)"
MB_BASE = "https://musicbrainz.org/ws/2"
LB_BASE = "https://api.listenbrainz.org/1"


class ListenBrainzClient:
    """Fetches hidden-gem recording candidates and artist tags from ListenBrainz / MusicBrainz."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=timeout,
            follow_redirects=True,
        )
        self._mb_lock = threading.Lock()
        self._last_mb_call = 0.0

    def close(self) -> None:
        self._client.close()

    def _mb_rate_limit(self) -> None:
        """Respect MusicBrainz 1 req/sec rate limit safely across concurrent threads."""
        with self._mb_lock:
            elapsed = time.monotonic() - self._last_mb_call
            if elapsed < 1.05:
                time.sleep(1.05 - elapsed)
            self._last_mb_call = time.monotonic()

    def _get_json(self, url: str, params: dict[str, Any] | None = None, is_mb: bool = False) -> Any:
        if is_mb:
            self._mb_rate_limit()
        try:
            resp = self._client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning("API returned %s for %s", resp.status_code, url)
                return None
            return resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("API request failed for %s: %s", url, exc)
            return None

    def lookup_artist(self, artist_name: str) -> tuple[str | None, list[str]]:
        """Resolve an artist name via MusicBrainz into (artist_mbid, top_tags)."""
        data = self._get_json(
            f"{MB_BASE}/artist/",
            params={"query": f'artist:"{artist_name}"', "fmt": "json", "limit": 1},
            is_mb=True,
        )
        artists = (data or {}).get("artists") or []
        if not artists:
            return None, []
        top = artists[0]
        mbid = top.get("id")
        raw_tags = top.get("tags") or []
        sorted_tags = sorted(raw_tags, key=lambda t: int(t.get("count", 0)), reverse=True)
        tags = [str(t["name"]).strip().lower() for t in sorted_tags if t.get("name")]
        return mbid, tags[:8]

    def fetch_tag_recording_mbids(
        self,
        tag: str,
        count: int = 20,
        pop_begin: int = 0,
        pop_end: int = 60,
    ) -> list[str]:
        """Query ListenBrainz LB-Radio tag endpoint biased toward low-to-mid popularity gems."""
        data = self._get_json(
            f"{LB_BASE}/lb-radio/tags",
            params={
                "tag": tag,
                "pop_begin": pop_begin,
                "pop_end": pop_end,
                "count": count,
            },
        )
        if not isinstance(data, list):
            return []
        return [str(item["recording_mbid"]) for item in data if isinstance(item, dict) and item.get("recording_mbid")]

    def fetch_similar_artist_recording_mbids(
        self,
        artist_mbid: str,
        max_similar_artists: int = 8,
        max_recordings_per_artist: int = 4,
        pop_begin: int = 0,
        pop_end: int = 60,
    ) -> list[str]:
        """Query ListenBrainz LB-Radio artist endpoint for similar-artist hidden gems."""
        data = self._get_json(
            f"{LB_BASE}/lb-radio/artist/{artist_mbid}",
            params={
                "mode": "easy",
                "max_similar_artists": max_similar_artists,
                "max_recordings_per_artist": max_recordings_per_artist,
                "pop_begin": pop_begin,
                "pop_end": pop_end,
            },
        )
        if not isinstance(data, dict):
            return []
        mbids: list[str] = []
        for recordings in data.values():
            if not isinstance(recordings, list):
                continue
            for rec in recordings:
                if isinstance(rec, dict) and rec.get("recording_mbid"):
                    mbids.append(str(rec["recording_mbid"]))
        return mbids

    def fetch_recordings_metadata(self, recording_mbids: list[str], extra_tags: list[str] | None = None) -> list[TrackEntity]:
        """Bulk-resolve recording MBIDs into platform-agnostic TrackEntity objects (with ISRC & tags)."""
        unique_mbids = list(dict.fromkeys(m for m in recording_mbids if m))
        tracks: list[TrackEntity] = []
        batch_size = 25

        for idx in range(0, len(unique_mbids), batch_size):
            batch = unique_mbids[idx : idx + batch_size]
            data = self._get_json(
                f"{LB_BASE}/metadata/recording/",
                params={
                    "recording_mbids": ",".join(batch),
                    "inc": "artist tag release",
                },
            )
            if not isinstance(data, dict):
                continue

            for mbid, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                rec = entry.get("recording") or {}
                rel = entry.get("release") or {}
                artist_block = entry.get("artist") or {}
                artists = artist_block.get("artists") or []

                title = str(rec.get("name") or "").strip()
                artist_name = ""
                if artists and isinstance(artists[0], dict):
                    artist_name = str(artists[0].get("name") or "").strip()
                if not artist_name:
                    artist_name = str(rel.get("album_artist_name") or "").strip()
                if not title or not artist_name:
                    continue

                isrcs = rec.get("isrcs") or []
                isrc = str(isrcs[0]).strip() if isinstance(isrcs, list) and isrcs else None

                release_year: int | None = None
                if isinstance(rel.get("year"), int):
                    release_year = rel["year"]
                else:
                    first_date = str(rec.get("first_release_date") or "")
                    if len(first_date) >= 4 and first_date[:4].isdigit():
                        release_year = int(first_date[:4])

                length_ms = rec.get("length")
                duration_sec = int(length_ms // 1000) if isinstance(length_ms, (int, float)) and length_ms > 0 else None

                tags_block = entry.get("tag") or {}
                collected_tags: list[str] = list(extra_tags or [])
                for scope in ("recording", "release_group", "artist"):
                    for tag_obj in tags_block.get(scope) or []:
                        if isinstance(tag_obj, dict) and tag_obj.get("tag"):
                            collected_tags.append(str(tag_obj["tag"]).strip().lower())

                tracks.append(
                    TrackEntity.build(
                        artist=artist_name,
                        title=title,
                        tags=collected_tags[:12],
                        isrc=isrc,
                        mbid=mbid,
                        release_year=release_year,
                        duration_seconds=duration_sec,
                        source="listenbrainz",
                    )
                )

        return tracks

    def dig_candidates(
        self,
        seed_tags: list[str],
        seed_artists: list[str],
        target_count: int = 40,
    ) -> list[TrackEntity]:
        """Discover hidden-gem recording candidates from tags and similar artists."""
        mbid_to_seed_tags: dict[str, set[str]] = {}

        for tag in seed_tags[:4]:
            for mbid in self.fetch_tag_recording_mbids(tag, count=15):
                mbid_to_seed_tags.setdefault(mbid, set()).add(tag.lower())

        for artist in seed_artists[:3]:
            artist_mbid, artist_tags = self.lookup_artist(artist)
            if not artist_mbid:
                continue
            for mbid in self.fetch_similar_artist_recording_mbids(
                artist_mbid,
                max_similar_artists=6,
                max_recordings_per_artist=3,
            ):
                mbid_to_seed_tags.setdefault(mbid, set()).update(artist_tags[:3])

        selected_mbids = list(mbid_to_seed_tags.keys())[:target_count]
        if not selected_mbids:
            return []

        entities = self.fetch_recordings_metadata(selected_mbids)
        for entity in entities:
            if entity.mbid and entity.mbid in mbid_to_seed_tags:
                entity.tags = sorted(set(entity.tags) | mbid_to_seed_tags[entity.mbid])
        return entities
