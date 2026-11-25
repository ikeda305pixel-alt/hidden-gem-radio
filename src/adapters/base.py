"""Platform-agnostic music playback/resolution adapter interfaces.

Phase 1 (MVP): YouTubeMusicAdapter (src/adapters/youtube.py)
Phase 2: SpotifyAdapter (OAuth + Web Playback SDK / Playlist Sync — NO deprecated
         /v1/recommendations or /v1/audio-features endpoints)
Phase 3: AppleMusicAdapter (Apple MusicKit JS + Catalog ISRC resolution)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.models import PlaybackResolution, TrackEntity


class MusicPlatformAdapter(ABC):
    """Abstract contract isolating platform-specific search & playback resolution."""

    platform_name: str = "abstract"

    @abstractmethod
    def resolve_track(
        self,
        artist: str,
        title: str,
        isrc: str | None = None,
    ) -> PlaybackResolution | None:
        """Resolve a platform-agnostic track (ISRC / artist + title) to a playable ID."""

    @abstractmethod
    def get_related_tracks(
        self,
        platform_track_id: str,
        seed_tags: list[str] | None = None,
        limit: int = 25,
    ) -> list[TrackEntity]:
        """Expand candidate pool from a resolved platform track ID."""


class SpotifyAdapter(MusicPlatformAdapter):
    """Phase 2 extension point: Spotify OAuth + Web Playback SDK / Playlist Sync.

    Uses ISRC / catalog search (`GET /v1/search?q=isrc:{isrc}&type=track`) and
    playlist sync endpoints only. Does NOT rely on deprecated `/v1/recommendations`
    or `/v1/audio-features`.
    """

    platform_name = "spotify"

    def resolve_track(
        self,
        artist: str,
        title: str,
        isrc: str | None = None,
    ) -> PlaybackResolution | None:
        raise NotImplementedError("Phase 2: Configure Spotify OAuth + Web Playback SDK adapter.")

    def get_related_tracks(
        self,
        platform_track_id: str,
        seed_tags: list[str] | None = None,
        limit: int = 25,
    ) -> list[TrackEntity]:
        raise NotImplementedError("Phase 2: Related digging remains powered by ListenBrainz/MusicBrainz.")


class AppleMusicAdapter(MusicPlatformAdapter):
    """Phase 3 extension point: Apple MusicKit JS catalog ISRC resolution."""

    platform_name = "apple_music"

    def resolve_track(
        self,
        artist: str,
        title: str,
        isrc: str | None = None,
    ) -> PlaybackResolution | None:
        raise NotImplementedError("Phase 3: Configure Apple MusicKit JS developer token adapter.")

    def get_related_tracks(
        self,
        platform_track_id: str,
        seed_tags: list[str] | None = None,
        limit: int = 25,
    ) -> list[TrackEntity]:
        raise NotImplementedError("Phase 3: Related digging remains powered by ListenBrainz/MusicBrainz.")
