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


LEGACY_STATION_ID_MAP: dict[str, str] = {
    "indie_groove_hooks": "indie_groove_cult",
}


def normalize_station_id(station_id: str) -> str:
    """Map deprecated station IDs (e.g. indie_groove_hooks) to their canonical successor."""
    clean = str(station_id or "").strip()
    return LEGACY_STATION_ID_MAP.get(clean, clean)


@dataclass(slots=True)
class StationConfig:
    """Configuration for a single radio station."""

    station_id: str
    name: str
    seed_tags: list[str]
    seed_artists: list[str]
    max_popularity_views: int = 500_000_000
    exploration_rate: float = 1.0
    target_theory_vector: dict[str, float] = field(default_factory=dict)
    bpm_zones: list[str] = field(default_factory=list)
    theory_archetypes: list[str] = field(default_factory=list)
    spotify_anchor_seeds: list[str] = field(default_factory=list)
    pes_anchors: list[str] = field(default_factory=list)
    emily_anchors: list[str] = field(default_factory=list)
    discovery_queries: list[str] = field(default_factory=list)
    weighted_seeds: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "station_id": self.station_id,
            "name": self.name,
            "seed_tags": list(self.seed_tags),
            "seed_artists": list(self.seed_artists),
            "max_popularity_views": int(self.max_popularity_views),
            "exploration_rate": float(self.exploration_rate),
        }
        if self.target_theory_vector:
            payload["target_theory_vector"] = {
                str(k): round(float(v), 4) for k, v in self.target_theory_vector.items()
            }
        if self.bpm_zones:
            payload["bpm_zones"] = list(self.bpm_zones)
        if self.theory_archetypes:
            payload["theory_archetypes"] = list(self.theory_archetypes)
        if self.spotify_anchor_seeds:
            payload["spotify_anchor_seeds"] = list(self.spotify_anchor_seeds)
        if self.pes_anchors:
            payload["pes_anchors"] = list(self.pes_anchors)
        if self.emily_anchors:
            payload["emily_anchors"] = list(self.emily_anchors)
        if self.discovery_queries:
            payload["discovery_queries"] = list(self.discovery_queries)
        if self.weighted_seeds:
            payload["weighted_seeds"] = list(self.weighted_seeds)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StationConfig:
        raw_id = str(data["station_id"]).strip()
        station_id = normalize_station_id(raw_id)
        is_legacy = raw_id in LEGACY_STATION_ID_MAP
        raw_target = data.get("target_theory_vector") or {}
        target_vec = (
            {str(k).upper(): float(v) for k, v in raw_target.items()}
            if isinstance(raw_target, dict)
            else {}
        )
        max_views = int(data.get("max_popularity_views", 500_000_000))
        if is_legacy and max_views == 500_000:
            max_views = 500_000_000
        return cls(
            station_id=station_id,
            name=str(data["name"]).strip(),
            seed_tags=[str(t).strip().lower() for t in data.get("seed_tags", []) if str(t).strip()],
            seed_artists=[str(a).strip() for a in data.get("seed_artists", []) if str(a).strip()],
            max_popularity_views=max_views,
            exploration_rate=float(data.get("exploration_rate", 1.0)),
            target_theory_vector=target_vec,
            bpm_zones=[str(z).strip() for z in data.get("bpm_zones", []) if str(z).strip()],
            theory_archetypes=[str(a).strip() for a in data.get("theory_archetypes", []) if str(a).strip()],
            spotify_anchor_seeds=[str(s).strip() for s in data.get("spotify_anchor_seeds", []) if str(s).strip()],
            pes_anchors=[str(s).strip() for s in data.get("pes_anchors", []) if str(s).strip()],
            emily_anchors=[str(s).strip() for s in data.get("emily_anchors", []) if str(s).strip()],
            discovery_queries=[str(q).strip() for q in data.get("discovery_queries", []) if str(q).strip()],
            weighted_seeds=[dict(item) for item in data.get("weighted_seeds", []) if isinstance(item, dict)],
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
    seed_artist: str | None = None
    seed_archetype: str | None = None
    cooccurrence_count: int = 1
    discovery_sources: list[str] = field(default_factory=list)
    discovery_route: str | None = None
    deezer_id: int | None = None
    deezer_bpm: float | None = None
    deezer_gain: float | None = None
    deezer_rank: int | None = None
    mir_vector: list[float] | None = None

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
        seed_artist: str | None = None,
        seed_archetype: str | None = None,
        cooccurrence_count: int = 1,
        discovery_sources: list[str] | None = None,
        discovery_route: str | None = None,
        deezer_id: int | None = None,
        deezer_bpm: float | None = None,
        deezer_gain: float | None = None,
        deezer_rank: int | None = None,
        mir_vector: list[float] | None = None,
    ) -> TrackEntity:
        clean_artist = (artist or "Unknown Artist").strip()
        clean_title = (title or "Untitled").strip()
        canonical_key = normalize_track_key(clean_artist, clean_title)
        track_id = make_track_id(clean_artist, clean_title, isrc=isrc, mbid=mbid)
        norm_tags = sorted({t.strip().lower() for t in (tags or []) if t and t.strip()})
        sources = list(dict.fromkeys(str(s).strip() for s in (discovery_sources or []) if str(s).strip()))
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
            seed_artist=seed_artist.strip() if seed_artist else None,
            seed_archetype=seed_archetype.strip() if seed_archetype else None,
            cooccurrence_count=max(1, int(cooccurrence_count)),
            discovery_sources=sources,
            discovery_route=discovery_route.strip() if discovery_route else None,
            deezer_id=deezer_id,
            deezer_bpm=float(deezer_bpm) if deezer_bpm is not None else None,
            deezer_gain=float(deezer_gain) if deezer_gain is not None else None,
            deezer_rank=int(deezer_rank) if deezer_rank is not None else None,
            mir_vector=[float(v) for v in mir_vector] if mir_vector is not None else None,
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
    artist_browse_id: str | None = None


@dataclass(slots=True)
class RankedCandidate:
    """Track paired with its MIR/Bandit score, discovery route, 4-layer match, and summary."""

    track: TrackEntity
    score: float
    exploit_score: float
    explore_bonus: float
    mode: str  # "活用" or "探索"
    hidden_level: str
    matched_tags: list[str]
    theory_vector: list[float] | None = None
    theory_explanation: dict[str, Any] | None = None
    mir_vector: list[float] | None = None
    mir_layers: dict[str, Any] | None = None
    measured_bpm: float | None = None
    sweetspot_score: float | None = None

    def to_queue_dict(self) -> dict[str, Any]:
        views_label = (
            f"{self.track.view_count:,} views" if self.track.view_count is not None else "views不明"
        )
        route_str = self.track.discovery_route or (
            f"シード関連: {self.track.seed_artist}" if self.track.seed_artist else "協調グラフ発掘"
        )
        tags_str = ", ".join(self.matched_tags[:3]) if self.matched_tags else route_str
        summary = f"{self.mode} | 隠れ度:{self.hidden_level} ({views_label}) | {tags_str}"
        payload: dict[str, Any] = {
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
            "seed_artist": self.track.seed_artist,
            "seed_archetype": self.track.seed_archetype,
            "cooccurrence_count": self.track.cooccurrence_count,
            "discovery_sources": list(self.track.discovery_sources),
            "discovery_route": route_str,
            "measured_bpm": round(float(self.measured_bpm), 1) if self.measured_bpm else None,
            "sweetspot_score": round(float(self.sweetspot_score), 3) if self.sweetspot_score is not None else None,
            "score": round(float(self.score), 4),
            "reason": {
                "mode": self.mode,
                "exploit_score": round(float(self.exploit_score), 4),
                "explore_bonus": round(float(self.explore_bonus), 4),
                "hidden_level": self.hidden_level,
                "matched_tags": self.matched_tags[:3],
                "discovery_route": route_str,
                "summary": summary,
            },
        }
        if self.theory_vector is not None:
            payload["theory_vector"] = [round(float(v), 4) for v in self.theory_vector]
        if self.theory_explanation is not None:
            payload["theory_explanation"] = self.theory_explanation
        if self.mir_vector is not None:
            payload["mir_vector"] = [round(float(v), 4) for v in self.mir_vector]
        if self.mir_layers is not None:
            payload["mir_layers"] = self.mir_layers
        return payload


def load_stations_yaml(path: Path) -> list[StationConfig]:
    """Load StationConfig list from YAML file."""
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("stations", []) if isinstance(raw, dict) else []
    return [StationConfig.from_dict(item) for item in entries if isinstance(item, dict)]


def load_history_index(path: Path) -> tuple[set[str], set[str]]:
    """Load known_video_ids and known_track_keys from stations.yaml history_index section."""
    if not path.exists():
        return set(), set()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return set(), set()
    idx = raw.get("history_index") or {}
    if not isinstance(idx, dict):
        return set(), set()
    vids = {str(v).strip() for v in idx.get("known_video_ids", []) if str(v).strip()}
    keys = {str(k).strip() for k in idx.get("known_track_keys", []) if str(k).strip()}
    return vids, keys


def save_stations_yaml(
    path: Path,
    stations: list[StationConfig],
    meta: dict[str, Any] | None = None,
    history_index: dict[str, Any] | None = None,
) -> None:
    """Write StationConfig list (plus optional meta and history_index) to YAML cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if meta:
        payload["meta"] = meta
    if history_index:
        payload["history_index"] = history_index
    payload["stations"] = [s.to_dict() for s in stations]
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")
