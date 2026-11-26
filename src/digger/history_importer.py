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
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ytmusicapi import YTMusic

from src.adapters.youtube import resolve_youtube_auth_file
from src.digger.listenbrainz import ListenBrainzClient
from src.models import (
    StationConfig,
    is_japanese_artist_or_text,
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
_NON_MUSIC_CHANNEL_RE = re.compile(
    r"(?:\b(?:tech|tips|gaming|gamer|podcast|news|channel|tv|clips|studio|academy|"
    r"coding|programming|cooking|kitchen|vlog|review)\b|"
    r"(?:実況|解説|ニュース|切り抜き|チャンネル|教室|料理|講座|速報))",
    re.IGNORECASE,
)
_EPISODE_OR_NUMBERED_RE = re.compile(
    r"(?:#\d+|\b(?:ep\.?|part|vol\.?)\s*\d+\b|第\d+回)",
    re.IGNORECASE,
)
_COLLAB_SPLIT_RE = re.compile(
    r"\s*(?:feat\.?|ft\.?|&|,|\s+x\s+|×|\bwith\b)\s*",
    re.IGNORECASE,
)
_MUSIC_SIGNAL_TITLE_RE = re.compile(
    r"(?:\b(?:official\s*(?:(?:hd|4k|music|lyric|audio|visuali[sz]er)\s*)*(?:video|audio|mv|m/v|lyric|visuali[sz]er)|"
    r"music\s*video|lyric\s*video|feat\.?|ft\.?|remix|acoustic|mv|full\s*song|topic|a\s+colors\s+show|boiler\s*room)\b|"
    r"\bM/V\b|"
    r"[\(\[\{]\s*(?:official\s*)?(?:audio|lyrics?|original\s+mix|remix|extended\s+mix|live\s+(?:on|at)\s+[^\)\]\}]*)\s*[\)\]\}])",
    re.IGNORECASE,
)
_TOPIC_CHANNEL_RE = re.compile(r"\s*-\s*(?:topic|トピック)$", re.IGNORECASE)
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

_WESTERN_ONLY_STATION_IDS = frozenset(
    {
        "us_hiphop_bouncy",
        "edm_dance_party",
        "melodic_rap_alt",
        "indie_groove_cult",
    }
)

# 6-station classification lexicons (strictly separating Western stations from JP Hip-Hop & J-Pop/City Pop)
_JP_HIPHOP_ARTISTS = {
    "bad hop", "yzerr", "t pablow", "ak 69", "zorn", "creepy nuts", "awich",
    "jp the wavy", "lex", "ralph", "jjj", "kid fresino", "bim", "kzm", "tohji",
    "watson", "lana", "chanmina", "ちゃんみな", "梅田サイファー", "舐達麻", "namedaruma", "般若",
    "kohh", "千葉雄喜", "yuki chiba", "yellow bucks", "eyden", "bonbero", "c o s a", "5lack",
    "punpee", "stuts", "仙人掌", "kandytown", "keiju", "io", "ozworld", "red eye",
    "jin dogg", "guca owl", "rykeydaddydirty", "anarchy", "norikiyo",
    "salu", "kohjiya", "dahlia", "mari", "百足", "韻マン", "mcニガリ", "r指定",
    "hentai shinshi club", "変態紳士クラブ", "wilywnka", "vigorman", "basi",
}
_JP_HIPHOP_KW_RE = re.compile(
    r"(?:ラップ|ヒップホップ|フリースタイル|サイファー|ストリート|ビーフ|ラップスタア|"
    r"\b(?:j-?rap|japanese\s*hip\s*hop|cypher|freestyle|hood|drill|trap|mc\s*battle|03-?\s*performance)\b)",
    re.IGNORECASE,
)
_JPOP_CITYPOP_ARTISTS = {
    "southern all stars", "サザンオールスターズ", "keisuke kuwata", "桑田佳祐",
    "tatsuro yamashita", "山下達郎", "mariya takeuchi", "竹内まりや",
    "taeko onuki", "大貫妙子", "toshiki kadomatsu", "角松敏生",
    "miki matsubara", "松原みき", "anri", "杏里", "junko ohashi", "junko yagami",
    "momoko kikuchi", "kiyotaka sugiyama", "omega tribe", "1986 omega tribe",
    "yumi matsutoya", "松任谷由実", "eiichi ohtaki", "大滝詠一", "haruomi hosono", "細野晴臣",
    "ryuichi sakamoto", "yukihiro takahashi", "yellow magic orchestra", "ymo",
    "hiroshi sato", "minako yoshida", "makoto matsushita", "masayoshi takanaka",
    "piper", "casiopea", "t square", "ryusenkei", "hitomitoi", "kirinji", "キリンジ",
    "original love", "fishmans", "shintaro sakamoto", "ichiko aoba",
    "imase", "fujii kaze", "藤井風", "vaundy", "official hige dandism", "official髭男dism",
    "gen hoshino", "星野源", "hikaru utada", "宇多田ヒカル", "kenshi yonezu", "米津玄師",
    "shiina ringo", "椎名林檎", "tokyo incidents", "東京事変", "spitz", "スピッツ",
    "sakanaction", "aimyon", "zutomayo", "yorushika", "king gnu", "millennium parade",
    "yoasobi", "mrs green apple", "back number", "mr children", "bump of chicken",
    "radwimps", "one ok rock", "perfume", "capsule", "ado", "yama",
    "suchmos", "nulbarich", "kroi", "lucky kilimanjaro", "sirup",
    "kan sano", "tendre", "iri", "chelmico", "tofubeats", "cero",
    "yogee new waves", "never young beach", "toconoma", "ovall", "lamp",
    "pizzicato five", "cornelius", "sugar babe", "tin pan alley",
}
_US_HIPHOP_ARTISTS = {
    "bbno", "migos", "j cole", "lil durk", "kyle", "macklemore", "lupe fiasco",
    "drake", "eminem", "kendrick lamar", "travis scott", "future", "21 savage",
    "metro boomin", "tyler the creator", "jid", "denzel curry", "a ap rocky",
    "asap rocky", "playboi carti", "gunna", "lil baby", "baby keem", "amine",
    "yung gravy", "snoop dogg", "kanye west", "pusha t", "joey bada", "jack harlow",
    "da baby", "dababy", "kodak black", "roddy ricch", "lil wayne", "50 cent",
    "outkast", "anderson paak", "mac miller", "wiz khalifa", "logic", "big sean",
    "lil yachty", "social house", "dj shadow", "run the jewels", "cardi b",
    "iggy azalea", "paul russell", "blanco brown", "kent jones", "tha supreme",
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
    "barry can t swim", "sammy virji", "interplanetary criminal", "gryffin", "showtek",
    "bakermat", "la roux", "gamper dadoni", "snakehips", "dj snake", "daddy yankee",
    "flo rida", "ellie goulding", "dua lipa", "justin bieber", "mike perry",
    "sam smith", "kim petras", "chris brown", "rihanna", "elite force", "data",
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
    "alec benjamin", "tate mcrae", "jvke", "g eazy", "lil nas x", "elyotto",
    "twenty one pilots", "shaed", "sub urban", "halsey", "imagine dragons",
    "ari hicks", "margo", "margø", "papithbk",
}
_MELODIC_RAP_ALT_KW_RE = re.compile(
    r"\b(?:emo\s*rap|melodic\s*rap|alt-?pop|pop\s*punk|sad\s*trap|emo|heartbreak|soundcloud\s*rap)\b",
    re.IGNORECASE,
)
_INDIE_GROOVE_ARTISTS = {
    "royel otis", "oberhofer", "foster the people", "fitz and the tantrums", "hozier",
    "earth wind fire", "surfaces", "dominic fike", "dope lemon", "phoenix",
    "bastille", "walk the moon", "tame impala", "parcels", "vulfpeck", "khruangbin",
    "two door cinema club", "vampire weekend", "mgmt", "jungle", "l imperatrice",
    "bruno mars", "silk sonic",
    "mac demarco", "men i trust", "clairo", "rex orange county", "still woozy",
    "glass animals", "metronomy", "the 1975", "arctic monkeys", "cage the elephant",
    "the strokes", "the shins", "phoebe bridgers", "dice", "rancid eddie",
    "the rare occasions", "borns", "børns", "architecture in helsinki", "the royal concept",
    "milky chance", "engelwood", "nicky youre", "eagles of death metal", "tally hall",
    "george ezra", "tai verdes", "mkto", "toploader", "the weeknd", "robin thicke",
    "miley cyrus", "charlie puth", "ashley park", "charlie faron", "ali danel",
    "dopamoon", "black lodge collective", "yael naim", "toto", "fall out boy",
    "benson boone", "journey", "the temper trap", "passion pit", "crystal castles",
    "fever ray", "keane", "azymuth", "balkan beat box", "black blood", "nina zilli",
    "babasonicos", "babasónicos", "kid francescoli", "french 79", "bon entendeur",
    "vendredi sur mer", "therapie taxi",
}


def compute_recency_frequency_weight(play_count: int, days_ago: float) -> float:
    """Compute YouTube history seed weight: log1p(play_count) * exp(-days_ago / 120.0)."""
    count = max(0, int(play_count))
    age_days = max(0.0, float(days_ago))
    return float(math.log1p(count) * math.exp(-age_days / 120.0))


def default_theory_stations(
    profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH,
) -> list[StationConfig]:
    """Load the 6 MIR + Collaborative Graph stations from config/music_theory_profile.yaml."""
    profile = load_music_theory_profile(profile_path)
    raw_stations = profile.get("stations") or []
    stations = [StationConfig.from_dict(item) for item in raw_stations if isinstance(item, dict)]
    if stations:
        return stations
    return default_fallback_stations()


def default_fallback_stations() -> list[StationConfig]:
    """Fallback stations prioritizing the 6 canonical stations."""
    profile_stations = load_music_theory_profile(DEFAULT_THEORY_PROFILE_PATH).get("stations") or []
    if profile_stations:
        return [StationConfig.from_dict(item) for item in profile_stations if isinstance(item, dict)]
    return [
        StationConfig(
            station_id="edm_dance_party",
            name="EDM, UKG & Dance Party",
            seed_tags=["melodic progressive house", "uk garage", "dance pop"],
            seed_artists=["Avicii", "Calvin Harris"],
            max_popularity_views=500_000_000,
            exploration_rate=0.9,
        ),
        StationConfig(
            station_id="indie_groove_cult",
            name="Indie Groove, PES Cult & Parisian Chic",
            seed_tags=["indie pop", "nu-disco", "french pop"],
            seed_artists=["Oberhofer", "Royel Otis", "L'Imperatrice"],
            max_popularity_views=500_000_000,
            exploration_rate=1.0,
        ),
        StationConfig(
            station_id="us_hiphop_bouncy",
            name="US Hip-Hop & Bouncy Rap",
            seed_tags=["hip hop", "bouncy rap", "trap"],
            seed_artists=["bbno$", "Migos"],
            max_popularity_views=500_000_000,
            exploration_rate=0.95,
        ),
    ]


def _artist_lookup_variants(name: str) -> set[str]:
    """Generate normalized lookup variants including ¥->y and $->s transliteration."""
    raw = str(name or "").strip()
    if not raw:
        return set()
    variants: set[str] = set()
    base = normalize_artist(raw)
    if base:
        variants.add(base)
    translit = raw.replace("¥", "y").replace("￥", "y").replace("$", "s")
    t_norm = normalize_artist(translit)
    if t_norm:
        variants.add(t_norm)
    return variants


def _split_artist_tokens(artist: str) -> list[str]:
    """Split collaborative artist strings into full string + individual artist tokens."""
    raw = str(artist or "").strip()
    if not raw:
        return []
    tokens = [raw]
    for part in _COLLAB_SPLIT_RE.split(raw):
        cleaned = part.strip(" ()[]")
        if cleaned and cleaned not in tokens:
            tokens.append(cleaned)
    return tokens


def classify_entry_to_station(
    artist: str,
    title: str,
    tags: list[str] | None = None,
) -> str:
    """Route a YouTube/YTMusic watch entry into one of the 6 canonical stations without JP/Western mixing."""
    candidate_norms: set[str] = set()
    for token in _split_artist_tokens(artist):
        candidate_norms.update(_artist_lookup_variants(token))
    a_norm = normalize_artist(artist)
    combined = " ".join([artist or "", title or "", *(tags or [])])
    is_jp = is_japanese_artist_or_text(artist, title)

    # Step 1: Japanese Hip-Hop vs Japanese Pop/City Pop/Groove (strictly prevented from entering Western stations)
    jp_hiphop_substrings = (
        "bad hop", "yzerr", "zorn", "awich", "creepy nuts", "舐達麻", "namedaruma", "yellow bucks", "yuki chiba",
    )
    if (candidate_norms & _JP_HIPHOP_ARTISTS) or any(
        k in norm for norm in candidate_norms for k in jp_hiphop_substrings
    ):
        return "japan_hiphop_street"
    if is_jp and _JP_HIPHOP_KW_RE.search(combined):
        return "japan_hiphop_street"
    if (candidate_norms & _JPOP_CITYPOP_ARTISTS) or is_jp:
        return "jpop_citypop_groove"

    # Step 2: Western explicit artist lexicons
    if candidate_norms & _MELODIC_RAP_ALT_ARTISTS:
        return "melodic_rap_alt"
    if candidate_norms & _US_HIPHOP_ARTISTS:
        return "us_hiphop_bouncy"
    if candidate_norms & _EDM_ARTISTS:
        return "edm_dance_party"
    if candidate_norms & _INDIE_GROOVE_ARTISTS:
        return "indie_groove_cult"

    # Step 3: Western keyword fallback
    if _MELODIC_RAP_ALT_KW_RE.search(combined):
        return "melodic_rap_alt"
    if _EDM_KW_RE.search(combined):
        return "edm_dance_party"
    if _US_HIPHOP_KW_RE.search(combined):
        return "us_hiphop_bouncy"
    if a_norm in _INDIE_GROOVE_ARTISTS:
        return "indie_groove_cult"
    return "indie_groove_cult"


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


def _iter_json_array_objects(path: Path, chunk_size: int = 131_072) -> Iterator[Any]:
    """Stream items from a top-level JSON array in fixed chunks using index-based raw_decode."""
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    eof = False
    started = False

    def _skip_ws() -> None:
        nonlocal pos
        n = len(buf)
        while pos < n and buf[pos] in " \t\r\n":
            pos += 1

    def _read_more(fh: Any) -> bool:
        nonlocal buf, pos, eof
        if pos > 0:
            buf = buf[pos:]
            pos = 0
        chunk = fh.read(chunk_size)
        if not chunk:
            eof = True
            return False
        buf += chunk
        return True

    with path.open("r", encoding="utf-8-sig") as fh:
        while True:
            while not started:
                _skip_ws()
                if pos < len(buf):
                    if buf[pos] != "[":
                        raise ValueError("Expected top-level JSON array")
                    pos += 1
                    started = True
                    break
                if not _read_more(fh):
                    raise ValueError("Empty JSON file")

            _skip_ws()
            while pos >= len(buf) and not eof:
                _read_more(fh)
                _skip_ws()

            if pos >= len(buf):
                raise ValueError("Unterminated JSON array")

            ch = buf[pos]
            if ch == "]":
                return
            if ch == ",":
                pos += 1
                continue

            while True:
                try:
                    obj, end_idx = decoder.raw_decode(buf, pos)
                    pos = end_idx
                    if pos >= chunk_size:
                        buf = buf[pos:]
                        pos = 0
                    yield obj
                    break
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Malformed JSON stream: {exc}") from exc
                    _read_more(fh)


def parse_takeout_watch_history(
    takeout_path: Path | str,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract music video watch entries (with video_id and played_at timestamp) from Google Takeout JSON."""
    path = Path(takeout_path)
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    try:
        for entry in _iter_json_array_objects(path):
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
            has_artist_channel_match = bool(
                dash_artist
                and _channel_matches_title_artist(channel_name, dash_artist)
                and not _NON_MUSIC_CHANNEL_RE.search(channel_name)
                and len(title) <= 60
                and not _EPISODE_OR_NUMBERED_RE.search(title)
                and not _NON_MUSIC_TITLE_RE.search(title)
            )

            # [M6 / Bug #4 / Audit #2] Require music channel, music title signal, or hardened artist-channel match
            if not (is_yt_music or has_music_channel or has_music_title or has_artist_channel_match):
                continue

            # Extract artist and clean track title (on - Topic channels, keep bilingual 'Title - Romaji' intact)
            artist = ""
            track_title = title
            is_topic_channel = bool(channel_name and _TOPIC_CHANNEL_RE.search(channel_name))
            if (
                dash_artist
                and not is_yt_music
                and (not is_topic_channel or _channel_matches_title_artist(channel_name, dash_artist))
            ):
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
    except (OSError, ValueError) as exc:
        msg = f"Failed to parse Takeout JSON {path}: {exc}"
        logger.warning(msg)
        if warnings is not None:
            warnings.append(msg)
        return []

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
    max_known_items: int = 5000,
) -> tuple[list[StationConfig], dict[str, list[str]]]:
    """Classify weighted YouTube history into the 6 DNA stations and build known_video_ids/track_keys/familiar_artists."""
    now = reference_time or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    stations = default_theory_stations(profile_path=profile_path)
    station_by_id = {s.station_id: s for s in stations}

    # Aggregate artist play counts, most recent play age (days_ago), and top track per artist per station
    artist_counts: dict[str, Counter[str]] = defaultdict(Counter)
    artist_min_days: dict[str, dict[str, float]] = defaultdict(dict)
    artist_display_name: dict[str, str] = {}
    artist_top_track: dict[str, Counter[str]] = defaultdict(Counter)
    artist_liked_bonus: Counter[str] = Counter()

    # Global per-artist counters for Known-Artist Suppression (>=5 plays or >=3 distinct tracks)
    global_artist_plays: Counter[str] = Counter()
    global_artist_tracks: dict[str, set[str]] = defaultdict(set)
    global_artist_min_days: dict[str, float] = {}

    # Track frequency x recency weights for known_video_ids and known_track_keys
    vid_counts: Counter[str] = Counter()
    vid_min_days: dict[str, float] = {}
    key_counts: Counter[str] = Counter()
    key_min_days: dict[str, float] = {}

    for rec in records:
        artist = str(rec.get("artist") or "").strip()
        title = str(rec.get("title") or "").strip()
        if not artist:
            continue
        a_norm = normalize_artist(artist)
        if not a_norm:
            continue
        artist_display_name.setdefault(a_norm, artist)

        dt = _parse_iso_timestamp(rec.get("played_at") or rec.get("time"))
        days_ago = max(0.0, (now - dt).total_seconds() / 86400.0) if dt is not None else 30.0
        increment = 3 if rec.get("source") == "ytmusic_liked" else 1

        global_artist_plays[a_norm] += increment
        prev_g_days = global_artist_min_days.get(a_norm)
        if prev_g_days is None or days_ago < prev_g_days:
            global_artist_min_days[a_norm] = days_ago

        vid = str(rec.get("video_id") or "").strip()
        if vid:
            vid_counts[vid] += increment
            prev_vid_days = vid_min_days.get(vid)
            if prev_vid_days is None or days_ago < prev_vid_days:
                vid_min_days[vid] = days_ago

        if title:
            t_key = normalize_track_key(artist, title)
            key_counts[t_key] += increment
            prev_key_days = key_min_days.get(t_key)
            if prev_key_days is None or days_ago < prev_key_days:
                key_min_days[t_key] = days_ago
            artist_top_track[a_norm][title] += 1
            global_artist_tracks[a_norm].add(t_key)

        st_id = classify_entry_to_station(artist, title)
        if st_id not in station_by_id:
            st_id = stations[0].station_id

        # Liked tracks count as stronger play signal while preserving frequency x recency formula
        artist_counts[st_id][a_norm] += increment
        if rec.get("source") == "ytmusic_liked":
            artist_liked_bonus[a_norm] += 1

        prev_days = artist_min_days[st_id].get(a_norm)
        if prev_days is None or days_ago < prev_days:
            artist_min_days[st_id][a_norm] = days_ago

    # Compute raw weights via log1p(play_count) * exp(-days_ago / 120.0) and normalize 0.0-1.0 per station
    for st in stations:
        if st.station_id in _WESTERN_ONLY_STATION_IDS:
            st.seed_artists = [a for a in st.seed_artists if not is_japanese_artist_or_text(a)]
            st.spotify_anchor_seeds = [
                s
                for s in st.spotify_anchor_seeds
                if not is_japanese_artist_or_text(*(s.split(" - ", 1) if " - " in s else (s, "")))
            ]

        counts = artist_counts.get(st.station_id)
        if not counts:
            continue

        scored_entries: list[dict[str, Any]] = []
        for a_norm, play_count in counts.items():
            disp_artist = artist_display_name[a_norm]
            top_title = artist_top_track[a_norm].most_common(1)[0][0] if artist_top_track[a_norm] else ""
            if st.station_id in _WESTERN_ONLY_STATION_IDS and is_japanese_artist_or_text(disp_artist, top_title):
                continue
            days_ago = artist_min_days[st.station_id].get(a_norm, 30.0)
            raw_w = compute_recency_frequency_weight(play_count, days_ago)
            scored_entries.append(
                {
                    "artist": disp_artist,
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

    sorted_known_vids = sorted(
        vid_counts.keys(),
        key=lambda v: compute_recency_frequency_weight(vid_counts[v], vid_min_days.get(v, 30.0)),
        reverse=True,
    )
    sorted_known_keys = sorted(
        key_counts.keys(),
        key=lambda k: compute_recency_frequency_weight(key_counts[k], key_min_days.get(k, 30.0)),
        reverse=True,
    )
    familiar_candidates = [
        a_norm
        for a_norm, plays in global_artist_plays.items()
        if plays >= 5 or len(global_artist_tracks.get(a_norm, set())) >= 3
    ]
    sorted_familiar_artists = sorted(
        familiar_candidates,
        key=lambda a: compute_recency_frequency_weight(
            global_artist_plays[a], global_artist_min_days.get(a, 30.0)
        ),
        reverse=True,
    )
    history_index = {
        "known_video_ids": sorted_known_vids[:max_known_items],
        "known_track_keys": sorted_known_keys[:max_known_items],
        "known_familiar_artists": sorted_familiar_artists[:max_known_items],
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
    """Return all 6 canonical stations when num_stations>=6, or a balanced subset when slicing."""
    n = max(2, min(6, int(num_stations)))
    if n >= len(all_five):
        return list(all_five)
    # Prioritize edm_dance_party & indie_groove_cult when slicing to <6 so fallback seeds stay covered
    priority_order = [
        "edm_dance_party",
        "indie_groove_cult",
        "us_hiphop_bouncy",
        "melodic_rap_alt",
        "japan_hiphop_street",
        "jpop_citypop_groove",
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
    num_stations: int = 6,
    offline: bool = False,
    profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH,
    reference_time: datetime | None = None,
    calibrate_db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Read available history sources, build weighted 6-station config, and optionally calibrate SQLite MIR profiles."""
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

    stations = _select_stations_for_count(all_five=all_stations, num_stations=num_stations)

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources_used": sources_used if sources_used else ["default_seed_fallback"],
        "records_imported": len(records),
        "fallback_used": fallback_used,
        "theory_profile": str(profile_path),
    }
    out = Path(output_path)
    has_history_idx = bool(
        history_index["known_video_ids"]
        or history_index["known_track_keys"]
        or history_index.get("known_familiar_artists")
    )
    save_stations_yaml(
        out,
        stations,
        meta=meta,
        history_index=history_index if has_history_idx else None,
    )

    calibrated_stations = 0
    if calibrate_db_path is not None:
        from src.db import RadioDatabase
        from src.recommender.calibrate_profile import calibrate_all_stations

        cal_db = RadioDatabase(calibrate_db_path)
        cal_res = calibrate_all_stations(
            cal_db,
            stations=stations,
            takeout_records=records,
            allow_network=not offline,
            reference_time=reference_time,
        )
        calibrated_stations = len(cal_res.get("stations") or {})

    return {
        "output_path": str(out),
        "station_count": len(stations),
        "station_ids": [s.station_id for s in stations],
        "records_imported": len(records),
        "calibrated_stations": calibrated_stations,
        "known_video_ids_count": len(history_index["known_video_ids"]),
        "known_track_keys_count": len(history_index["known_track_keys"]),
        "known_familiar_artists_count": len(history_index.get("known_familiar_artists", [])),
        "sources_used": meta["sources_used"],
        "fallback_used": fallback_used,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-generate config/stations.yaml from YouTube / YouTube Music history (6-station Theory DNA)."
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
        default=6,
        choices=[2, 3, 4, 5, 6],
        help="Number of stations to generate (2-6, default: 6)",
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
