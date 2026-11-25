"""Platform adapters keeping YouTube, Spotify (Phase 2), and Apple Music (Phase 3) isolated."""

from src.adapters.base import AppleMusicAdapter, MusicPlatformAdapter, SpotifyAdapter
from src.adapters.youtube import YouTubeMusicAdapter

__all__ = [
    "AppleMusicAdapter",
    "MusicPlatformAdapter",
    "SpotifyAdapter",
    "YouTubeMusicAdapter",
]
