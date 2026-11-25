"""Platform-agnostic core models and normalization helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_NON_ALNUM_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_PAREN_NOISE_RE = re.compile(
    r"\s*[\(\[\{](?:official\s*(?:music\s*)?(?:video|audio|lyric\s*video|visualizer)|"
    r"hd|hq|4k|lyrics?|audio|mv|music\s*video|remaster(?:ed)?(?:\s*\d{4})?)[\)\]\}]",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    """Normalize text for platform-agnostic deduplication."""
    cleaned = _PAREN_NOISE_RE.sub("", value or "")
    cleaned = cleaned.lower().strip()
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def normalize_artist(artist: str) -> str:
    """Normalize artist name for deduplication and 30-day BAD artist exclusion."""
    return normalize_text(artist)


def normalize_track_key(artist: str, title: str) -> str:
    """Create canonical platform-agnostic key from normalized (artist, title)."""
    return f"{normalize_artist(artist)}::{normalize_text(title)}"


def make_track_id(
    artist: str,
    title: str,
    isrc: str | None = None,
    mbid: str | None = None,
) -> str:
    """Deterministic platform-agnostic track ID preferring ISRC > MBID > normalized key."""
    if isrc:
        seed = f"isrc:{isrc.strip().upper()}"
    elif mbid:
        seed = f"mbid:{mbid.strip().lower()}"
    else:
        seed = f"key:{normalize_track_key(artist, title)}"
    digest = hashlib.sha1(seed.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return f"trk_{digest}"


@dataclass(slots=True)
class StationConfig:
    """Configuration for a single radio station."""

    station_id: str
    name: str
    seed_tags: list[str]
    seed_artists: list[str]
    max_popularity_views: int = 500_000
    exploration_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "seed_tags": list(self.seed_tags),
            "seed_artists": list(self.seed_artists),
            "max_popularity_views": int(self.max_popularity_views),
            "exploration_rate": float(self.exploration_rate),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StationConfig:
        return cls(
            station_id=str(data["station_id"]).strip(),
            name=str(data["name"]).strip(),
            seed_tags=[str(t).strip().lower() for t in data.get("seed_tags", []) if str(t).strip()],
            seed_artists=[str(a).strip() for a in data.get("seed_artists", []) if str(a).strip()],
            max_popularity_views=int(data.get("max_popularity_views", 500_000)),
            exploration_rate=float(data.get("exploration_rate", 1.0)),
        )


@dataclass(slots=True)
class TrackEntity:
    """Platform-agnostic song entity keyed by ISRC, MBID, and normalized (artist, title)."""

    track_id: str
    canonical_key: str
    artist: str
    title: str
    tags: list[str] = field(default_factory=list)
    isrc: str | None = None
    mbid: str | None = None
    release_year: int | None = None
    view_count: int | None = None
    duration_seconds: int | None = None
    video_id: str | None = None
    unplayable: bool = False
    source: str = "unknown"

    @classmethod
    def build(
        cls,
        artist: str,
        title: str,
        tags: list[str] | None = None,
        isrc: str | None = None,
        mbid: str | None = None,
        release_year: int | None = None,
        view_count: int | None = None,
        duration_seconds: int | None = None,
        video_id: str | None = None,
        source: str = "unknown",
    ) -> TrackEntity:
        clean_artist = (artist or "Unknown Artist").strip()
        clean_title = (title or "Untitled").strip()
        canonical_key = normalize_track_key(clean_artist, clean_title)
        track_id = make_track_id(clean_artist, clean_title, isrc=isrc, mbid=mbid)
        norm_tags = sorted({t.strip().lower() for t in (tags or []) if t and t.strip()})
        return cls(
            track_id=track_id,
            canonical_key=canonical_key,
            artist=clean_artist,
            title=clean_title,
            tags=norm_tags,
            isrc=isrc.strip().upper() if isrc else None,
            mbid=mbid.strip().lower() if mbid else None,
            release_year=release_year,
            view_count=view_count,
            duration_seconds=duration_seconds,
            video_id=video_id,
            unplayable=False,
            source=source,
        )


@dataclass(slots=True)
class PlaybackResolution:
    """Platform-specific playback mapping returned by an adapter."""

    platform: str
    platform_track_id: str
    title: str
    artist: str
    view_count: int | None = None
    duration_seconds: int | None = None
    release_year: int | None = None


@dataclass(slots=True)
class RankedCandidate:
    """Track paired with its Contextual Bandit score and 1-line explanation."""

    track: TrackEntity
    score: float
    exploit_score: float
    explore_bonus: float
    mode: str  # "活用" or "探索"
    hidden_level: str
    matched_tags: list[str]

    def to_queue_dict(self) -> dict[str, Any]:
        views_label = (
            f"{self.track.view_count:,} views" if self.track.view_count is not None else "views不明"
        )
        tags_str = ", ".join(self.matched_tags[:3]) if self.matched_tags else "新規発掘"
        summary = f"{self.mode} | 隠れ度:{self.hidden_level} ({views_label}) | タグ:{tags_str}"
        return {
            "track_id": self.track.track_id,
            "video_id": self.track.video_id,
            "artist": self.track.artist,
            "title": self.track.title,
            "tags": self.track.tags,
            "isrc": self.track.isrc,
            "mbid": self.track.mbid,
            "release_year": self.track.release_year,
            "view_count": self.track.view_count,
            "duration_seconds": self.track.duration_seconds,
            "source": self.track.source,
            "score": round(float(self.score), 4),
            "reason": {
                "mode": self.mode,
                "exploit_score": round(float(self.exploit_score), 4),
                "explore_bonus": round(float(self.explore_bonus), 4),
                "hidden_level": self.hidden_level,
                "matched_tags": self.matched_tags[:3],
                "summary": summary,
            },
        }


def load_stations_yaml(path: Path) -> list[StationConfig]:
    """Load StationConfig list from YAML file."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("stations", []) if isinstance(raw, dict) else []
    return [StationConfig.from_dict(item) for item in entries if isinstance(item, dict)]


def save_stations_yaml(path: Path, stations: list[StationConfig], meta: dict[str, Any] | None = None) -> None:
    """Write StationConfig list to YAML file cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if meta:
        payload["meta"] = meta
    payload["stations"] = [s.to_dict() for s in stations]
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")
