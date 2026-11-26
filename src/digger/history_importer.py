"""Auto-generate config/stations.yaml from YouTube / YouTube Music watch history & likes.

Supported sources:
1. ytmusicapi browser/oauth auth (`data/browser.json`, `data/oauth.json`, `config/browser.json`)
   -> get_history() & get_liked_songs()
2. Google Takeout `watch-history.json` (`data/watch-history.json`)
   -> music video extractor with recency x frequency weighting:
      weight = math.log1p(play_count) * math.exp(-days_ago / 120.0)

Routes history entries into the 5 Deep Music Theory DNA stations, attaches normalized
`weighted_seeds` (0.0-1.0) per station, and records `known_video_ids` / `known_track_keys`
so heavy-rotation tracks themselves are skipped while their hidden-gem neighborhoods are dug first.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ytmusicapi import YTMusic

from src.adapters.youtube import resolve_youtube_auth_file
from src.digger.listenbrainz import ListenBrainzClient
from src.models import (
    StationConfig,
    normalize_artist,
    normalize_track_key,
    save_stations_yaml,
)
from src.recommender.theory_features import (
    DEFAULT_THEORY_PROFILE_PATH,
    load_music_theory_profile,
)

logger = logging.getLogger(__name__)

# Noise / overly generic tags to avoid using as station clustering anchors
_GENERIC_TAGS = {
    "seen live",
    "favorites",
    "favourite",
    "english",
    "british",
    "american",
    "swedish",
    "japanese",
    "uk",
    "usa",
    "music",
    "male vocalists",
    "female vocalists",
    "2000s",
    "2010s",
    "2020s",
}

_TAKEOUT_PREFIX_EN = re.compile(r"^Watched\s+", re.IGNORECASE)
_TAKEOUT_SUFFIX_JP = re.compile(r"\s*を視聴しました\s*$")
_NON_MUSIC_TITLE_RE = re.compile(
    r"\b(?:tutorial|full course|news|podcast|gameplay|walkthrough|review|unboxing|"
    r"vlog|reaction|trailer|episode\s*\d+|lecture|interview|highlights|shorts|recipe|cooking)\b|"
    r"(?:ニュース|速報|解説|講座|入門|第\d+回|実況|レビュー|開封|切り抜き|まとめ|生配信|雑談|料理|レシピ)",
    re.IGNORECASE,
)
_MUSIC_SIGNAL_TITLE_RE = re.compile(
    r"(?:\b(?:official\s*(?:(?:hd|4k|music|lyric|audio|visuali[sz]er)\s*)*(?:video|audio|mv|m/v|lyric|visuali[sz]er)|"
    r"music\s*video|lyric\s*video|feat\.?|ft\.?|remix|acoustic|mv|full\s*song|topic|a\s+colors\s+show|boiler\s+room)\b|"
    r"\bM/V\b|"
    r"[\(\[\{]\s*(?:official\s*)?(?:audio|lyrics?|original\s+mix|remix|extended\s+mix|live\s+(?:on|at)\s+[^\)\]\}]*)\s*[\)\]\}])",
    re.IGNORECASE,
)
_MUSIC_CHANNEL_RE = re.compile(
    r"(?:\s*-\s*(?:topic|トピック)$|vevo$|records?$|music$|official\s*(?:music|artist)|"
    r"sounds?$|recordings?$|^\s*(?:kexp|boiler\s*room|ninja\s*tune|colors|colorsxstudios|sub\s*pop|"
    r"warp\s*records|xl\s*recordings|domino\s*recording\s*co\.?|4ad|matador\s*records|spinnin'\s*records|"
    r"armada\s*music|anjunadeep|anjunabeats|ultra\s*records|monstercat|ghostly\s*international)\s*$)",
    re.IGNORECASE,
)
_CHANNEL_CLEAN_RE = re.compile(
    r"(?:\s*-\s*(?:topic|トピック)$|\s*official\s*vevo$|\s*vevo$|\s*official\s*(?:music|artist|channel)$|\s*official$)",
    re.IGNORECASE,
)
_JP_BRACKET_TITLE_RE = re.compile(r"^[「『](.+?)[」』]$")
_AD_DETAIL_RE = re.compile(r"(?:google\s*ads|広告)", re.IGNORECASE)
_JP_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")

# 5-station classification lexicons
_JP_HIPHOP_ARTISTS = {
    "bad hop", "yzerr", "t pablow", "ak 69", "zorn", "creepy nuts", "awich",
    "jp the wavy", "lex", "ralph", "jjj", "kid fresino", "bim", "kzm", "tohji",
    "watson", "lana", "chanmina", "ちゃんみな", "梅田サイファー", "舐達麻", "般若",
    "kohh", "千葉雄喜", "yellow bucks", "eyden", "bonbero", "c o s a", "5lack",
    "punpee", "stuts", "仙人掌", "kandytown", "keiju", "io", "ozworld", "red eye",
    "jin dogg", "guca owl", "rykeydaddydirty", "anarchy", "norikiyo",
    "salu", "kohjiya", "dahlia", "mari", "百足", "韻マン", "mcニガリ", "r指定",
}
_JP_HIPHOP_KW_RE = re.compile(
    r"(?:ラップ|ヒップホップ|フリースタイル|サイファー|ストリート|ビーフ|ラップスタア|"
    r"\b(?:j-?rap|japanese\s*hip\s*hop|cypher|freestyle|hood|drill|trap|mc\s*battle|03-?\s*performance)\b)",
    re.IGNORECASE,
)
_US_HIPHOP_ARTISTS = {
    "bbno", "migos", "j cole", "lil durk", "kyle", "macklemore", "lupe fiasco",
    "drake", "eminem", "kendrick lamar", "travis scott", "future", "21 savage",
    "metro boomin", "tyler the creator", "jid", "denzel curry", "a ap rocky",
    "asap rocky", "playboi carti", "gunna", "lil baby", "baby keem", "amine",
    "yung gravy", "snoop dogg", "kanye west", "pusha t", "joey bada", "jack harlow",
    "da baby", "dababy", "kodak black", "roddy ricch", "lil wayne", "50 cent",
    "outkast", "anderson paak", "mac miller", "wiz khalifa", "logic", "big sean",
}
_US_HIPHOP_KW_RE = re.compile(
    r"\b(?:hip\s*hop|rap|trap|drill|808|bouncy\s*rap|boom\s*bap|freestyle|cypher|gangsta)\b",
    re.IGNORECASE,
)
_EDM_ARTISTS = {
    "avicii", "david guetta", "calvin harris", "chrystal", "notion", "saint jhn",
    "imanbek", "tiesto", "tiësto", "major lazer", "jax jones", "black eyed peas",
    "madism", "katy perry", "kygo", "zedd", "martin garrix", "alesso", "fred again",
    "bicep", "disclosure", "four tet", "skrillex", "daft punk", "swedish house mafia",
    "marshmello", "fisher", "peggy gou", "kaytranada", "sub focus", "chase status",
    "bonobo", "caribou", "flume", "odesza", "illenium", "porter robinson", "madeon",
    "galantis", "alan walker", "david guetta avicii", "diplo", "dom dolla", "john summit",
    "barry can t swim", "sammy virji", "interplanetary criminal",
}
_EDM_KW_RE = re.compile(
    r"\b(?:remix|edm|house|ukg|uk\s*garage|2-?step|techno|trance|dnb|drum\s*and\s*bass|"
    r"dubstep|club\s*mix|original\s*mix|extended\s*mix|boiler\s*room|dj\s*set|dance\s*pop|progressive\s*house)\b",
    re.IGNORECASE,
)
_MELODIC_RAP_ALT_ARTISTS = {
    "lil peep", "juice wrld", "post malone", "the kid laroi", "24kgoldn", "blackbear",
    "iann dior", "mgk", "machine gun kelly", "jeremy zucker", "lil uzi vert",
    "xxxtentacion", "trippie redd", "nf", "lauv", "bazzi", "joji", "oliver tree",
    "d4vd", "powfu", "yungblud", "glaive", "brakence", "lil tecca", "polo g",
    "kid cudi", "chase atlantic", "billie eilish", "conan gray", "chelsea cutler",
    "alec benjamin", "tate mcrae", "jvke", "g eazy",
}
_MELODIC_RAP_ALT_KW_RE = re.compile(
    r"\b(?:emo\s*rap|melodic\s*rap|alt-?pop|pop\s*punk|sad\s*trap|emo|heartbreak|soundcloud\s*rap)\b",
    re.IGNORECASE,
)
_INDIE_GROOVE_ARTISTS = {
    "royel otis", "oberhofer", "foster the people", "fitz and the tantrums", "hozier",
    "imase", "earth wind fire", "surfaces", "dominic fike", "dope lemon", "phoenix",
    "bastille", "walk the moon", "tame impala", "parcels", "vulfpeck", "khruangbin",
    "two door cinema club", "vampire weekend", "mgmt", "jungle", "l imperatrice",
    "fujii kaze", "藤井風", "vaundy", "official髭男dism", "星野源", "suchmos",
    "nulbarich", "kroi", "lucky kilimanjaro", "sirup", "bruno mars", "silk sonic",
    "mac demarco", "men i trust", "clairo", "rex orange county", "still woozy",
    "glass animals", "metronomy", "the 1975", "arctic monkeys", "cage the elephant",
    "the strokes", "the shins", "phoebe bridgers",
}


def compute_recency_frequency_weight(play_count: int, days_ago: float) -> float:
    """Compute YouTube history seed weight: log1p(play_count) * exp(-days_ago / 120.0)."""
    count = max(0, int(play_count))
    age_days = max(0.0, float(days_ago))
    return float(math.log1p(count) * math.exp(-age_days / 120.0))


def default_theory_stations(
    profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH,
) -> list[StationConfig]:
    """Load the 5 Deep Music Theory DNA stations from config/music_theory_profile.yaml."""
    profile = load_music_theory_profile(profile_path)
    raw_stations = profile.get("stations") or []
    stations = [StationConfig.from_dict(item) for item in raw_stations if isinstance(item, dict)]
    if stations:
        return stations
    return default_fallback_stations()


def default_fallback_stations() -> list[StationConfig]:
    """Fallback stations prioritizing the 5 Deep Music Theory DNA stations."""
    profile_stations = load_music_theory_profile(DEFAULT_THEORY_PROFILE_PATH).get("stations") or []
    if profile_stations:
        return [StationConfig.from_dict(item) for item in profile_stations if isinstance(item, dict)]
    return [
        StationConfig(
            station_id="edm_dance_party",
            name="EDM & Dance Party",
            seed_tags=["melodic progressive house", "uk garage", "dance pop"],
            seed_artists=["Avicii", "Calvin Harris"],
            max_popularity_views=500_000,
            exploration_rate=0.9,
        ),
        StationConfig(
            station_id="indie_groove_hooks",
            name="Indie Groove & Feel-Good",
            seed_tags=["indie pop", "folktronica", "indietronica"],
            seed_artists=["Oberhofer", "Royel Otis"],
            max_popularity_views=500_000,
            exploration_rate=1.0,
        ),
        StationConfig(
            station_id="us_hiphop_bouncy",
            name="US Hip-Hop & Bouncy Rap",
            seed_tags=["hip hop", "bouncy rap", "trap"],
            seed_artists=["bbno$", "Migos"],
            max_popularity_views=500_000,
            exploration_rate=0.95,
        ),
    ]


def classify_entry_to_station(
    artist: str,
    title: str,
    tags: list[str] | None = None,
) -> str:
    """Route a YouTube/YTMusic watch entry into one of the 5 Deep Music Theory DNA stations."""
    a_norm = normalize_artist(artist)
    combined = " ".join([artist or "", title or "", *(tags or [])])

    # 1. Japan Hip-Hop & Street (check explicit JP rap artists or JP script + hip-hop signal)
    if a_norm in _JP_HIPHOP_ARTISTS or any(k in a_norm for k in ("bad hop", "yzerr", "zorn", "awich", "creepy nuts", "舐達麻")):
        return "japan_hiphop_street"
    if _JP_SCRIPT_RE.search(combined) and _JP_HIPHOP_KW_RE.search(combined):
        return "japan_hiphop_street"

    # 2. Melodic Rap & Alt-Pop (check before general US hip-hop so Juice WRLD / Peep / Post Malone route accurately)
    if a_norm in _MELODIC_RAP_ALT_ARTISTS or _MELODIC_RAP_ALT_KW_RE.search(combined):
        return "melodic_rap_alt"

    # 3. EDM & Dance Party (remix/UKG/house/DJ signals)
    if a_norm in _EDM_ARTISTS or _EDM_KW_RE.search(combined):
        return "edm_dance_party"

    # 4. US Hip-Hop & Bouncy Rap
    if a_norm in _US_HIPHOP_ARTISTS or _US_HIPHOP_KW_RE.search(combined):
        return "us_hiphop_bouncy"

    # 5. Indie Groove & Feel-Good (explicit indie/funk/pop artists or default)
    if a_norm in _INDIE_GROOVE_ARTISTS:
        return "indie_groove_hooks"
    return "indie_groove_hooks"


def _extract_video_id(url: str) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if qs.get("v"):
            return str(qs["v"][0]).strip()
        if "youtu.be" in parsed.netloc and parsed.path.strip("/"):
            return parsed.path.strip("/").split("/")[0]
    except ValueError:
        return None
    return None


def _parse_iso_timestamp(raw_time: Any) -> datetime | None:
    if not raw_time:
        return None
    text = str(raw_time).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


def _clean_channel_artist(channel_name: str) -> str:
    cleaned = _CHANNEL_CLEAN_RE.sub("", channel_name or "").strip(" -_|")
    # Convert CamelCase VEVO names like "AviciiOfficialVEVO" -> "Avicii"
    cleaned = re.sub(r"Official$", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _clean_takeout_title(raw_title: str) -> str:
    title = _TAKEOUT_PREFIX_EN.sub("", raw_title or "")
    title = _TAKEOUT_SUFFIX_JP.sub("", title).strip()
    match = _JP_BRACKET_TITLE_RE.match(title)
    if match:
        title = match.group(1).strip()
    return title


def _is_google_ad_entry(entry: dict[str, Any]) -> bool:
    details = entry.get("details") or []
    for item in details:
        if isinstance(item, dict) and _AD_DETAIL_RE.search(str(item.get("name") or "")):
            return True
    return False


def _channel_matches_title_artist(channel_name: str, title_artist: str) -> bool:
    """Return True if channel name and 'Artist - Track' left-side artist match or contain each other."""
    if not channel_name or not title_artist:
        return False
    ch_norm = normalize_artist(_clean_channel_artist(channel_name)).replace(" ", "")
    ar_norm = normalize_artist(title_artist).replace(" ", "")
    if len(ch_norm) < 2 or len(ar_norm) < 2:
        return False
    return ch_norm == ar_norm or ar_norm in ch_norm or ch_norm in ar_norm


def parse_takeout_watch_history(
    takeout_path: Path | str,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract music video watch entries (with video_id and played_at timestamp) from Google Takeout JSON."""
    path = Path(takeout_path)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"Failed to parse Takeout JSON {path}: {exc}"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

    if not isinstance(raw, list):
        return []

    records: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # [M6] Exclude sponsored/Google Ads watch entries
        if _is_google_ad_entry(entry):
            continue

        header = str(entry.get("header") or "")
        url = str(entry.get("titleUrl") or "")
        raw_title = str(entry.get("title") or "")
        title = _clean_takeout_title(raw_title)
        if not title or title.lower().startswith("https://"):
            continue

        subtitles = entry.get("subtitles") or []
        channel_name = ""
        if subtitles and isinstance(subtitles[0], dict):
            channel_name = str(subtitles[0].get("name") or "").strip()

        is_yt_music = header.lower() == "youtube music" or "music.youtube.com" in url.lower()
        if not is_yt_music and (_NON_MUSIC_TITLE_RE.search(title) or _NON_MUSIC_TITLE_RE.search(channel_name)):
            continue

        dash_artist = ""
        dash_track = ""
        if " - " in title:
            left, right = title.split(" - ", 1)
            if 1 <= len(left.strip()) <= 45 and right.strip():
                dash_artist = left.strip()
                dash_track = _clean_takeout_title(right.strip())

        has_music_channel = bool(channel_name and _MUSIC_CHANNEL_RE.search(channel_name))
        has_music_title = bool(_MUSIC_SIGNAL_TITLE_RE.search(title))
        has_artist_channel_match = bool(dash_artist and _channel_matches_title_artist(channel_name, dash_artist))

        # [M6 / Bug #4] Require music channel, music title signal, or verified artist-channel match on "Artist - Track"
        if not (is_yt_music or has_music_channel or has_music_title or has_artist_channel_match):
            continue

        # Extract artist and clean track title
        artist = ""
        track_title = title
        if dash_artist and not is_yt_music:
            artist = dash_artist
            track_title = dash_track
        if not artist and channel_name:
            artist = _clean_channel_artist(channel_name)
        if not artist:
            continue

        records.append(
            {
                "artist": artist,
                "title": track_title,
                "video_id": _extract_video_id(url),
                "played_at": str(entry.get("time") or ""),
                "weight": 1.5 if is_yt_music else 1.0,
                "source": "takeout_ytmusic" if is_yt_music else "takeout_youtube",
            }
        )

    return records


def parse_ytmusic_history_and_likes(
    yt_client: YTMusic,
    liked_limit: int = 100,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Fetch history & liked songs via authenticated ytmusicapi client."""
    records: list[dict[str, Any]] = []
    now_iso = datetime.now(UTC).isoformat()

    try:
        history = yt_client.get_history() or []
        for item in history:
            artists = item.get("artists") or []
            artist = str(artists[0].get("name", "")).strip() if artists and isinstance(artists[0], dict) else ""
            title = str(item.get("title") or "").strip()
            if artist and title:
                records.append(
                    {
                        "artist": artist,
                        "title": title,
                        "video_id": item.get("videoId"),
                        "played_at": now_iso,
                        "weight": 1.5,
                        "source": "ytmusic_history",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        msg = f"ytmusicapi get_history() failed (auth may be expired): {exc}"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)

    try:
        liked = yt_client.get_liked_songs(limit=liked_limit) or {}
        for item in liked.get("tracks") or []:
            artists = item.get("artists") or []
            artist = str(artists[0].get("name", "")).strip() if artists and isinstance(artists[0], dict) else ""
            title = str(item.get("title") or "").strip()
            if artist and title:
                records.append(
                    {
                        "artist": artist,
                        "title": title,
                        "video_id": item.get("videoId"),
                        "played_at": now_iso,
                        "weight": 3.0,  # Liked songs carry higher preference weight
                        "source": "ytmusic_liked",
                    }
                )
    except Exception as exc:  # noqa: BLE001
        msg = f"ytmusicapi get_liked_songs() failed (auth may be expired): {exc}"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)

    return records


def build_weighted_theory_stations(
    records: list[dict[str, Any]],
    *,
    profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH,
    reference_time: datetime | None = None,
    max_weighted_seeds_per_station: int = 12,
    max_known_items: int = 1000,
) -> tuple[list[StationConfig], dict[str, list[str]]]:
    """Classify weighted YouTube history into the 5 DNA stations and build known_video_ids/track_keys."""
    now = reference_time or datetime.now(UTC)
    stations = default_theory_stations(profile_path=profile_path)
    station_by_id = {s.station_id: s for s in stations}

    # Aggregate artist play counts, most recent play age (days_ago), and top track per artist per station
    artist_counts: dict[str, Counter[str]] = defaultdict(Counter)
    artist_min_days: dict[str, dict[str, float]] = defaultdict(dict)
    artist_display_name: dict[str, str] = {}
    artist_top_track: dict[str, Counter[str]] = defaultdict(Counter)
    artist_liked_bonus: Counter[str] = Counter()

    known_video_ids: list[str] = []
    seen_vids: set[str] = set()
    known_track_keys: list[str] = []
    seen_keys: set[str] = set()

    for rec in records:
        artist = str(rec.get("artist") or "").strip()
        title = str(rec.get("title") or "").strip()
        if not artist:
            continue
        a_norm = normalize_artist(artist)
        if not a_norm:
            continue
        artist_display_name.setdefault(a_norm, artist)

        vid = str(rec.get("video_id") or "").strip()
        if vid and vid not in seen_vids:
            seen_vids.add(vid)
            known_video_ids.append(vid)
        if title:
            t_key = normalize_track_key(artist, title)
            if t_key not in seen_keys:
                seen_keys.add(t_key)
                known_track_keys.append(t_key)
            artist_top_track[a_norm][title] += 1

        dt = _parse_iso_timestamp(rec.get("played_at") or rec.get("time"))
        days_ago = max(0.0, (now - dt).total_seconds() / 86400.0) if dt is not None else 30.0

        st_id = classify_entry_to_station(artist, title)
        if st_id not in station_by_id:
            st_id = stations[0].station_id

        # Liked tracks count as stronger play signal while preserving frequency x recency formula
        increment = 3 if rec.get("source") == "ytmusic_liked" else 1
        artist_counts[st_id][a_norm] += increment
        if rec.get("source") == "ytmusic_liked":
            artist_liked_bonus[a_norm] += 1

        prev_days = artist_min_days[st_id].get(a_norm)
        if prev_days is None or days_ago < prev_days:
            artist_min_days[st_id][a_norm] = days_ago

    # Compute raw weights via log1p(play_count) * exp(-days_ago / 120.0) and normalize 0.0-1.0 per station
    for st in stations:
        counts = artist_counts.get(st.station_id)
        if not counts:
            continue

        scored_entries: list[dict[str, Any]] = []
        for a_norm, play_count in counts.items():
            days_ago = artist_min_days[st.station_id].get(a_norm, 30.0)
            raw_w = compute_recency_frequency_weight(play_count, days_ago)
            top_title = artist_top_track[a_norm].most_common(1)[0][0] if artist_top_track[a_norm] else ""
            scored_entries.append(
                {
                    "artist": artist_display_name[a_norm],
                    "top_title": top_title,
                    "play_count": int(play_count),
                    "days_ago": round(float(days_ago), 1),
                    "raw_weight": round(raw_w, 4),
                }
            )

        scored_entries.sort(key=lambda item: item["raw_weight"], reverse=True)
        top_entries = scored_entries[:max_weighted_seeds_per_station]
        max_raw = max((e["raw_weight"] for e in top_entries), default=1.0) or 1.0

        normalized_seeds: list[dict[str, Any]] = []
        for entry in top_entries:
            norm_w = round(float(entry["raw_weight"] / max_raw), 4)
            normalized_seeds.append(
                {
                    "artist": entry["artist"],
                    "top_title": entry["top_title"],
                    "play_count": entry["play_count"],
                    "days_ago": entry["days_ago"],
                    "weight": norm_w,
                }
            )

        st.weighted_seeds = normalized_seeds
        # Prepend high-weight history artists to seed_artists (deduplicated) so digging prioritizes them
        history_artists = [item["artist"] for item in normalized_seeds[:6]]
        merged_artists = list(dict.fromkeys([*history_artists, *st.seed_artists]))
        st.seed_artists = merged_artists[:14]

    history_index = {
        "known_video_ids": known_video_ids[:max_known_items],
        "known_track_keys": known_track_keys[:max_known_items],
    }
    return stations, history_index


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "hidden_gems"


def _heuristic_tags_for_artist(artist: str) -> list[str]:
    """Offline fallback genre guesser when MusicBrainz lookups are disabled/unreachable."""
    low = artist.lower()
    if any(k in low for k in ("avicii", "kygo", "calvin", "zedd", "martin garrix", "alesso")):
        return ["melodic progressive house", "electronic", "dance"]
    if any(k in low for k in ("four tet", "bonobo", "caribou", "floating points", "bicep")):
        return ["folktronica", "indietronica", "downtempo"]
    if any(k in low for k in ("oberhofer", "shins", "phoebe", "vampire", "tame")):
        return ["indietronica", "indie pop", "dream pop"]
    return ["indietronica", "indie", "alternative"]


def build_stations_from_history(
    records: list[dict[str, Any]],
    num_stations: int = 3,
    tag_resolver: Callable[[str], list[str]] | None = None,
    max_artists_to_resolve: int = 18,
) -> list[StationConfig]:
    """Cluster top artists and frequent tags from history into 2-4 coherent radio stations (legacy helper)."""
    target_stations = max(2, min(4, int(num_stations)))
    artist_scores: Counter[str] = Counter()
    for rec in records:
        artist = str(rec.get("artist") or "").strip()
        weight = float(rec.get("weight", 1.0))
        if artist:
            artist_scores[artist] += weight

    if not artist_scores:
        return default_fallback_stations()[:target_stations]

    top_artists = [artist for artist, _ in artist_scores.most_common(max_artists_to_resolve)]
    resolver = tag_resolver or _heuristic_tags_for_artist

    artist_tags: dict[str, list[str]] = {}
    tag_scores: Counter[str] = Counter()
    tag_to_artists: dict[str, list[str]] = defaultdict(list)

    for artist in top_artists:
        raw_tags = resolver(artist) or []
        clean_tags = [
            t.strip().lower()
            for t in raw_tags
            if t and t.strip().lower() not in _GENERIC_TAGS
        ]
        if not clean_tags:
            clean_tags = _heuristic_tags_for_artist(artist)
        artist_tags[artist] = clean_tags[:5]
        for rank_idx, tag in enumerate(artist_tags[artist]):
            tag_scores[tag] += artist_scores[artist] / (1.0 + 0.25 * rank_idx)
            if artist not in tag_to_artists[tag]:
                tag_to_artists[tag].append(artist)

    stations: list[StationConfig] = []
    used_anchors: set[str] = set()
    used_artist_sets: list[set[str]] = []
    ranked_tags = [tag for tag, _ in tag_scores.most_common(30)]

    for anchor_tag in ranked_tags:
        if len(stations) >= target_stations:
            break
        if anchor_tag in used_anchors:
            continue
        candidate_artists = sorted(
            tag_to_artists[anchor_tag],
            key=lambda a: artist_scores[a],
            reverse=True,
        )[:4]
        if not candidate_artists:
            continue

        artist_set = set(candidate_artists)
        if any(artist_set == prev for prev in used_artist_sets):
            continue

        co_tags: Counter[str] = Counter({anchor_tag: 10.0})
        for artist in candidate_artists:
            for t in artist_tags.get(artist, []):
                co_tags[t] += 1.0
        seed_tags = [t for t, _ in co_tags.most_common(3)]

        used_anchors.add(anchor_tag)
        used_artist_sets.append(artist_set)
        station_slug = f"gem_{_slugify(anchor_tag)}"
        if any(s.station_id == station_slug for s in stations):
            station_slug = f"{station_slug}_{len(stations) + 1}"

        stations.append(
            StationConfig(
                station_id=station_slug,
                name=f"{anchor_tag.title()} Hidden Gems",
                seed_tags=seed_tags,
                seed_artists=candidate_artists,
                max_popularity_views=500_000,
                exploration_rate=round(0.9 + 0.15 * len(stations), 2),
            )
        )

    if len(stations) < 2:
        for fallback in default_fallback_stations():
            if len(stations) >= max(2, target_stations):
                break
            if not any(s.station_id == fallback.station_id for s in stations):
                stations.append(fallback)

    return stations[:4]


def _select_stations_for_count(all_five: list[StationConfig], num_stations: int) -> list[StationConfig]:
    """Return all 5 canonical stations when num_stations>=5, or a balanced subset including EDM & Indie."""
    n = max(2, min(5, int(num_stations)))
    if n >= len(all_five):
        return list(all_five)
    # Prioritize edm_dance_party & indie_groove_hooks when slicing to <5 so fallback seeds stay covered
    priority_order = [
        "edm_dance_party",
        "indie_groove_hooks",
        "us_hiphop_bouncy",
        "melodic_rap_alt",
        "japan_hiphop_street",
    ]
    by_id = {s.station_id: s for s in all_five}
    ordered = [by_id[sid] for sid in priority_order if sid in by_id]
    for st in all_five:
        if st not in ordered:
            ordered.append(st)
    return ordered[:n]


def generate_stations_yaml(
    output_path: Path | str = Path("config/stations.yaml"),
    browser_auth_path: Path | str = Path("config/browser.json"),
    takeout_path: Path | str = Path("data/watch-history.json"),
    num_stations: int = 5,
    offline: bool = False,
    profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Read available history sources, build weighted 5-station DNA config, and write stations.yaml."""
    records: list[dict[str, Any]] = []
    sources_used: list[str] = []
    warnings: list[str] = []

    browser_path = Path(browser_auth_path)
    # Also auto-check data/browser.json and data/oauth.json if default config/browser.json is absent
    effective_auth: Path | None = browser_path if browser_path.exists() else None
    if effective_auth is None and str(browser_auth_path) == "config/browser.json":
        effective_auth = resolve_youtube_auth_file()

    if effective_auth is not None and effective_auth.exists():
        try:
            yt = YTMusic(str(effective_auth))
            yt_records = parse_ytmusic_history_and_likes(yt, warnings=warnings)
            if yt_records:
                records.extend(yt_records)
                sources_used.append(f"ytmusicapi:{effective_auth}")
            elif not warnings:
                warnings.append(
                    f"ytmusicapi auth {effective_auth} loaded but returned 0 history/liked tracks."
                )
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load ytmusicapi browser auth {effective_auth} (expired or invalid): {exc}"
            logger.warning(msg)
            warnings.append(msg)

    takeout_file = Path(takeout_path)
    if takeout_file.exists():
        takeout_records = parse_takeout_watch_history(takeout_file, warnings=warnings)
        if takeout_records:
            records.extend(takeout_records)
            sources_used.append(f"takeout:{takeout_file}")

    fallback_used = not bool(records)
    all_stations, history_index = build_weighted_theory_stations(
        records,
        profile_path=profile_path,
        reference_time=reference_time,
    )

    # Optionally enrich unknown artists' tags via MusicBrainz when online and history is present
    if records and not offline:
        lb_client = ListenBrainzClient(timeout=6.0)
        try:
            for st in all_stations:
                for seed_item in st.weighted_seeds[:2]:
                    _, mb_tags = lb_client.lookup_artist(str(seed_item["artist"]))
                    for t in mb_tags[:2]:
                        if t not in _GENERIC_TAGS and t not in st.seed_tags:
                            st.seed_tags.append(t)
        finally:
            lb_client.close()

    stations = _select_stations_for_count(all_stations, num_stations)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources_used": sources_used if sources_used else ["default_seed_fallback"],
        "records_imported": len(records),
        "fallback_used": fallback_used,
        "theory_profile": str(profile_path),
    }
    out = Path(output_path)
    save_stations_yaml(
        out,
        stations,
        meta=meta,
        history_index=history_index if (history_index["known_video_ids"] or history_index["known_track_keys"]) else None,
    )
    return {
        "output_path": str(out),
        "station_count": len(stations),
        "station_ids": [s.station_id for s in stations],
        "records_imported": len(records),
        "known_video_ids_count": len(history_index["known_video_ids"]),
        "known_track_keys_count": len(history_index["known_track_keys"]),
        "sources_used": meta["sources_used"],
        "fallback_used": fallback_used,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-generate config/stations.yaml from YouTube / YouTube Music history (5-station Theory DNA)."
    )
    parser.add_argument(
        "--output",
        default="config/stations.yaml",
        help="Destination YAML path (default: config/stations.yaml)",
    )
    parser.add_argument(
        "--browser-auth",
        default="config/browser.json",
        help="Path to ytmusicapi browser/oauth auth JSON (default: config/browser.json; also checks data/browser.json & data/oauth.json)",
    )
    parser.add_argument(
        "--takeout",
        default="data/watch-history.json",
        help="Path to Google Takeout watch-history.json (default: data/watch-history.json)",
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=5,
        choices=[2, 3, 4, 5],
        help="Number of stations to generate (2-5, default: 5)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip MusicBrainz tag lookups and use fast local heuristics",
    )
    args = parser.parse_args(argv)

    summary = generate_stations_yaml(
        output_path=args.output,
        browser_auth_path=args.browser_auth,
        takeout_path=args.takeout,
        num_stations=args.stations,
        offline=args.offline,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
