"""Platform-agnostic core models and normalization helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
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
_JP_SCRIPT_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf]")
_FURIGANA_PAREN_RE = re.compile(
    r"(?<=[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf])"
    r"\s*[\(\[\{【][\u3040-\u309f\u30a0-\u30ffー・\s]+[\)\]\}】]\s*"
)
_BILINGUAL_SUFFIX_RE = re.compile(
    r"(?:\s+[-–—|]\s+|\s*/\s*)([A-Za-z0-9\s'’\".,!?&~()\-]+)$"
)
_BILINGUAL_PAREN_SUFFIX_RE = re.compile(
    r"\s*\(([A-Za-z0-9\s'’\".,!?&\-]+)\)\s*$"
)
_VERSION_KEEP_RE = re.compile(
    r"\b(?:remix|mix|ver\.?|version|live|inst\.?|instrumental|acoustic|cover|"
    r"edit|dub|vip|bootleg|flip|rework|feat\.?|ft\.?|prod\.?)\b",
    re.IGNORECASE,
)
_ARTIST_SPLIT_RE = re.compile(
    r"\s*(?:,|&|\bfeat\.?\b|\bft\.?\b|\bwith\b|\bx\b|×|/)\s*",
    re.IGNORECASE,
)

# Canonicalize Japanese <-> English/Romaji artist aliases so history and official ATV metadata match.
JP_EN_ARTIST_ALIASES: dict[str, str] = {
    "サザンオールスターズ": "southern all stars",
    "サザン": "southern all stars",
    "southern all stars": "southern all stars",
    "桑田佳祐": "keisuke kuwata",
    "kuwata keisuke": "keisuke kuwata",
    "keisuke kuwata": "keisuke kuwata",
    "山下達郎": "tatsuro yamashita",
    "yamashita tatsuro": "tatsuro yamashita",
    "tatsuro yamashita": "tatsuro yamashita",
    "竹内まりや": "mariya takeuchi",
    "takeuchi mariya": "mariya takeuchi",
    "mariya takeuchi": "mariya takeuchi",
    "大貫妙子": "taeko onuki",
    "onuki taeko": "taeko onuki",
    "taeko onuki": "taeko onuki",
    "角松敏生": "toshiki kadomatsu",
    "kadomatsu toshiki": "toshiki kadomatsu",
    "toshiki kadomatsu": "toshiki kadomatsu",
    "松原みき": "miki matsubara",
    "matsubara miki": "miki matsubara",
    "miki matsubara": "miki matsubara",
    "杏里": "anri",
    "大橋純子": "junko ohashi",
    "八神純子": "junko yagami",
    "菊池桃子": "momoko kikuchi",
    "杉山清貴": "kiyotaka sugiyama",
    "松任谷由実": "yumi matsutoya",
    "荒井由実": "yumi matsutoya",
    "yumi arai": "yumi matsutoya",
    "大滝詠一": "eiichi ohtaki",
    "大瀧詠一": "eiichi ohtaki",
    "細野晴臣": "haruomi hosono",
    "坂本龍一": "ryuichi sakamoto",
    "高橋幸宏": "yukihiro takahashi",
    "佐藤博": "hiroshi sato",
    "吉田美奈子": "minako yoshida",
    "松下誠": "makoto matsushita",
    "高中正義": "masayoshi takanaka",
    "流線形": "ryusenkei",
    "一十三十一": "hitomitoi",
    "キリンジ": "kirinji",
    "オリジナル ラヴ": "original love",
    "オリジナルラブ": "original love",
    "フィッシュマンズ": "fishmans",
    "坂本慎太郎": "shintaro sakamoto",
    "青葉市子": "ichiko aoba",
    "藤井風": "fujii kaze",
    "藤井 風": "fujii kaze",
    "kaze fujii": "fujii kaze",
    "fujii kaze": "fujii kaze",
    "星野源": "gen hoshino",
    "星野 源": "gen hoshino",
    "hoshino gen": "gen hoshino",
    "gen hoshino": "gen hoshino",
    "宇多田ヒカル": "hikaru utada",
    "utada hikaru": "hikaru utada",
    "hikaru utada": "hikaru utada",
    "米津玄師": "kenshi yonezu",
    "yonezu kenshi": "kenshi yonezu",
    "kenshi yonezu": "kenshi yonezu",
    "椎名林檎": "shiina ringo",
    "ringo sheena": "shiina ringo",
    "sheena ringo": "shiina ringo",
    "東京事変": "tokyo incidents",
    "official髭男dism": "official hige dandism",
    "スピッツ": "spitz",
    "サカナクション": "sakanaction",
    "あいみょん": "aimyon",
    "ずっと真夜中でいいのに": "zutomayo",
    "ヨルシカ": "yorushika",
    "ちゃんみな": "chanmina",
    "千葉雄喜": "yuki chiba",
    "ellow bucks": "yellow bucks",
    "変態紳士クラブ": "hentai shinshi club",
    "舐達麻": "namedaruma",
}

KNOWN_JAPANESE_ARTISTS_NORM: frozenset[str] = frozenset(
    {
        *JP_EN_ARTIST_ALIASES.values(),
        # J-Pop / City Pop / J-Groove / J-Rock (Latin-script names)
        "imase",
        "vaundy",
        "suchmos",
        "nulbarich",
        "kroi",
        "lucky kilimanjaro",
        "sirup",
        "kan sano",
        "tendre",
        "iri",
        "chelmico",
        "tofubeats",
        "piper",
        "casiopea",
        "t square",
        "yellow magic orchestra",
        "ymo",
        "omega tribe",
        "1986 omega tribe",
        "sugar babe",
        "tin pan alley",
        "pizzicato five",
        "cornelius",
        "cero",
        "yogee new waves",
        "never young beach",
        "toconoma",
        "ovall",
        "lamp",
        "king gnu",
        "millennium parade",
        "yoasobi",
        "mrs green apple",
        "back number",
        "mr children",
        "bump of chicken",
        "radwimps",
        "one ok rock",
        "perfume",
        "capsule",
        "ado",
        "yama",
        "creepy nuts",
        # Japanese Hip-Hop (Latin-script names)
        "bad hop",
        "yzerr",
        "t pablow",
        "benjazzy",
        "bark",
        "vingo",
        "tiji jojo",
        "yellow pato",
        "g kid",
        "kandytown",
        "keiju",
        "io",
        "ryohu",
        "gottz",
        "mud",
        "jjj",
        "febb",
        "kid fresino",
        "flashbacks",
        "awich",
        "jp the wavy",
        "lex",
        "lana",
        "watson",
        "ralph",
        "jumadiba",
        "kzm",
        "eyden",
        "bonbero",
        "hideyoshi",
        "ozworld",
        "aklo",
        "salu",
        "kohh",
        "zorn",
        "anarchy",
        "ak 69",
        "zeebra",
        "kreva",
        "rhymester",
        "buddha brand",
        "seeda",
        "norikiyo",
        "issugi",
        "5lack",
        "punpee",
        "stuts",
        "bim",
        "vava",
        "omsb",
        "campanella",
        "tohji",
        "gummyboy",
        "elle teresa",
        "sky hi",
        "yellow bucks",
        "dj ryow",
        "red eye",
        "mc tyson",
        "jin dogg",
        "wilywnka",
        "vigorman",
        "basi",
        "chico carlito",
        "sweet william",
        "daichi yamamoto",
        "skaai",
        "dodo",
        "kaneee",
        "kohjiya",
    }
)


def _normalize_basic(value: str) -> str:
    """Lowercase NFKC alphanumeric normalization."""
    cleaned = unicodedata.normalize("NFKC", value or "")
    cleaned = _PAREN_NOISE_RE.sub("", cleaned)
    cleaned = cleaned.lower().strip()
    cleaned = _NON_ALNUM_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def normalize_text(value: str) -> str:
    """Normalize text for platform-agnostic deduplication."""
    return _normalize_basic(value)


def normalize_artist(artist: str) -> str:
    """Normalize artist name for deduplication, JP<->EN alias matching, and suppression."""
    norm = _normalize_basic(artist)
    return JP_EN_ARTIST_ALIASES.get(norm, norm)


def split_normalized_artists(artist: str) -> set[str]:
    """Return normalized full artist plus individual collaboration tokens."""
    out: set[str] = set()
    full = normalize_artist(artist)
    if full:
        out.add(full)
    for part in _ARTIST_SPLIT_RE.split(unicodedata.normalize("NFKC", artist or "")):
        norm_part = normalize_artist(part)
        if norm_part:
            out.add(norm_part)
    return out


def contains_japanese_script(text: str) -> bool:
    """Return True if text contains Hiragana, Katakana, or Kanji characters."""
    return bool(_JP_SCRIPT_RE.search(unicodedata.normalize("NFKC", text or "")))


def is_japanese_artist_or_text(artist: str, title: str = "") -> bool:
    """Return True if artist or title is Japanese (script or known Latin-script JP artist)."""
    if contains_japanese_script(artist) or contains_japanese_script(title):
        return True
    tokens = split_normalized_artists(artist)
    return bool(tokens & KNOWN_JAPANESE_ARTISTS_NORM)


def normalize_title_for_key(title: str) -> str:
    """Normalize track title, stripping furigana and bilingual romaji suffixes on JP titles."""
    cleaned = unicodedata.normalize("NFKC", title or "").strip()
    cleaned = _PAREN_NOISE_RE.sub("", cleaned).strip()
    if _JP_SCRIPT_RE.search(cleaned):
        without_furigana = _FURIGANA_PAREN_RE.sub("", cleaned).strip()
        if without_furigana:
            cleaned = without_furigana
        m_suffix = _BILINGUAL_SUFFIX_RE.search(cleaned)
        if m_suffix:
            rhs = m_suffix.group(1).strip()
            lhs = cleaned[: m_suffix.start()].strip()
            if (
                lhs
                and _JP_SCRIPT_RE.search(lhs)
                and re.search(r"[A-Za-z]", rhs)
                and not _VERSION_KEEP_RE.search(rhs)
            ):
                cleaned = lhs
        m_paren = _BILINGUAL_PAREN_SUFFIX_RE.search(cleaned)
        if m_paren:
            rhs = m_paren.group(1).strip()
            lhs = cleaned[: m_paren.start()].strip()
            if (
                lhs
                and _JP_SCRIPT_RE.search(lhs)
                and re.search(r"[A-Za-z]", rhs)
                and not _VERSION_KEEP_RE.search(rhs)
            ):
                cleaned = lhs
    return _normalize_basic(cleaned)


def normalize_track_key(artist: str, title: str) -> str:
    """Create canonical platform-agnostic key from normalized (artist, title)."""
    return f"{normalize_artist(artist)}::{normalize_title_for_key(title)}"


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
    keys: set[str] = set()
    for raw_key in idx.get("known_track_keys", []):
        k = str(raw_key).strip()
        if not k:
            continue
        keys.add(k)
        if "::" in k:
            art_part, ttl_part = k.split("::", 1)
            keys.add(normalize_track_key(art_part, ttl_part))
    return vids, keys


def load_familiar_artists(path: Path) -> set[str]:
    """Load normalized known_familiar_artists from stations.yaml history_index section."""
    if not path.exists():
        return set()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return set()
    idx = raw.get("history_index") or {}
    if not isinstance(idx, dict):
        return set()
    familiar: set[str] = set()
    for item in idx.get("known_familiar_artists", []):
        norm = normalize_artist(str(item))
        if norm:
            familiar.add(norm)
    return familiar


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
