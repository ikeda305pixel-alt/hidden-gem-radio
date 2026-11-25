"""Auto-generate config/stations.yaml from YouTube / YouTube Music watch history & likes.

Supported sources:
1. ytmusicapi browser auth (`config/browser.json`) -> get_history() & get_liked_songs()
2. Google Takeout `watch-history.json` (`data/watch-history.json`) -> music video extractor

When neither source exists or yields music records, cleanly falls back to the
default specification seed stations so MVP testing works immediately.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ytmusicapi import YTMusic

from src.digger.listenbrainz import ListenBrainzClient
from src.models import StationConfig, save_stations_yaml

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
    r"\b(?:official\s*(?:music\s*)?(?:video|audio|lyric|visualizer)|music\s*video|"
    r"lyric\s*video|feat\.?|ft\.?|remix|acoustic|mv|full\s*song|topic)\b",
    re.IGNORECASE,
)
_MUSIC_CHANNEL_RE = re.compile(
    r"(?:\s*-\s*(?:topic|トピック)$|vevo$|records?$|music$|official\s*(?:music|artist)|sounds?$|recordings?$)",
    re.IGNORECASE,
)
_CHANNEL_CLEAN_RE = re.compile(
    r"(?:\s*-\s*(?:topic|トピック)$|\s*official\s*vevo$|\s*vevo$|\s*official\s*(?:music|artist|channel)$|\s*official$)",
    re.IGNORECASE,
)
_JP_BRACKET_TITLE_RE = re.compile(r"^[「『](.+?)[」』]$")
_AD_DETAIL_RE = re.compile(r"(?:google\s*ads|広告)", re.IGNORECASE)


def default_fallback_stations() -> list[StationConfig]:
    """Specification example fallback stations when no personal history files are present."""
    return [
        StationConfig(
            station_id="melodic_folktronica_gems",
            name="Melodic House & Folktronica Gems",
            seed_tags=["melodic progressive house", "folktronica", "indietronica"],
            seed_artists=["Avicii", "Oberhofer"],
            max_popularity_views=500_000,
            exploration_rate=1.0,
        ),
        StationConfig(
            station_id="indietronica_dream_cuts",
            name="Indietronica & Dream Pop Deep Cuts",
            seed_tags=["indietronica", "dream pop", "downtempo"],
            seed_artists=["Oberhofer", "Four Tet"],
            max_popularity_views=350_000,
            exploration_rate=1.2,
        ),
        StationConfig(
            station_id="progressive_organic_house",
            name="Melodic Progressive & Organic Electronic",
            seed_tags=["melodic progressive house", "organic house", "deep house"],
            seed_artists=["Avicii", "Bonobo"],
            max_popularity_views=500_000,
            exploration_rate=0.9,
        ),
    ]


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


def parse_takeout_watch_history(
    takeout_path: Path | str,
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract music video watch entries from Google Takeout `watch-history.json`."""
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
        if not is_yt_music and _NON_MUSIC_TITLE_RE.search(title):
            continue

        has_music_channel = bool(channel_name and _MUSIC_CHANNEL_RE.search(channel_name))
        has_music_title = bool(_MUSIC_SIGNAL_TITLE_RE.search(title))

        # [M6] Do NOT treat bare " - " as music without a music channel or music title signal
        if not (is_yt_music or has_music_channel or has_music_title):
            continue

        # Extract artist and clean track title
        artist = ""
        track_title = title
        if " - " in title and not is_yt_music:
            left, right = title.split(" - ", 1)
            if 1 <= len(left.strip()) <= 45:
                artist = left.strip()
                track_title = _clean_takeout_title(right.strip())
        if not artist and channel_name:
            artist = _clean_channel_artist(channel_name)
        if not artist:
            continue

        records.append(
            {
                "artist": artist,
                "title": track_title,
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
    """Cluster top artists and frequent tags from history into 2-4 coherent radio stations."""
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
            # Prefer specific subgenres over rank-0 broad tags slightly
            tag_scores[tag] += artist_scores[artist] / (1.0 + 0.25 * rank_idx)
            if artist not in tag_to_artists[tag]:
                tag_to_artists[tag].append(artist)

    # Greedily select diverse anchor tags to build 2..4 distinct stations
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

        # Avoid creating duplicate stations with identical artist membership
        artist_set = set(candidate_artists)
        if any(artist_set == prev for prev in used_artist_sets):
            continue

        # Gather co-occurring tags among the station's seed artists
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

    # Guarantee at least 2 stations (per specification: 2-4 stations)
    if len(stations) < 2:
        for fallback in default_fallback_stations():
            if len(stations) >= max(2, target_stations):
                break
            if not any(s.station_id == fallback.station_id for s in stations):
                stations.append(fallback)

    return stations[:4]


def generate_stations_yaml(
    output_path: Path | str = Path("config/stations.yaml"),
    browser_auth_path: Path | str = Path("config/browser.json"),
    takeout_path: Path | str = Path("data/watch-history.json"),
    num_stations: int = 3,
    offline: bool = False,
) -> dict[str, Any]:
    """Read available history sources, generate 2-4 stations, and write config/stations.yaml."""
    records: list[dict[str, Any]] = []
    sources_used: list[str] = []
    warnings: list[str] = []

    browser_path = Path(browser_auth_path)
    if browser_path.exists():
        try:
            yt = YTMusic(str(browser_path))
            yt_records = parse_ytmusic_history_and_likes(yt, warnings=warnings)
            if yt_records:
                records.extend(yt_records)
                sources_used.append(f"ytmusicapi:{browser_path}")
            elif not warnings:
                warnings.append(
                    f"ytmusicapi auth {browser_path} loaded but returned 0 history/liked tracks."
                )
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed to load ytmusicapi browser auth {browser_path} (expired or invalid): {exc}"
            logger.warning(msg)
            warnings.append(msg)

    takeout_file = Path(takeout_path)
    if takeout_file.exists():
        takeout_records = parse_takeout_watch_history(takeout_file, warnings=warnings)
        if takeout_records:
            records.extend(takeout_records)
            sources_used.append(f"takeout:{takeout_file}")

    fallback_used = False
    if not records:
        fallback_used = True
        stations = default_fallback_stations()[: max(2, min(4, int(num_stations)))]
    else:
        lb_client: ListenBrainzClient | None = None
        if not offline:
            lb_client = ListenBrainzClient(timeout=8.0)

        def _resolve_tags(artist: str) -> list[str]:
            if lb_client is not None:
                _, mb_tags = lb_client.lookup_artist(artist)
                if mb_tags:
                    return mb_tags
            return _heuristic_tags_for_artist(artist)

        try:
            stations = build_stations_from_history(
                records,
                num_stations=num_stations,
                tag_resolver=_resolve_tags,
            )
        finally:
            if lb_client is not None:
                lb_client.close()

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sources_used": sources_used if sources_used else ["default_seed_fallback"],
        "records_imported": len(records),
        "fallback_used": fallback_used,
    }
    out = Path(output_path)
    save_stations_yaml(out, stations, meta=meta)
    return {
        "output_path": str(out),
        "station_count": len(stations),
        "station_ids": [s.station_id for s in stations],
        "records_imported": len(records),
        "sources_used": meta["sources_used"],
        "fallback_used": fallback_used,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Auto-generate config/stations.yaml from YouTube / YouTube Music history."
    )
    parser.add_argument(
        "--output",
        default="config/stations.yaml",
        help="Destination YAML path (default: config/stations.yaml)",
    )
    parser.add_argument(
        "--browser-auth",
        default="config/browser.json",
        help="Path to ytmusicapi browser auth JSON (default: config/browser.json)",
    )
    parser.add_argument(
        "--takeout",
        default="data/watch-history.json",
        help="Path to Google Takeout watch-history.json (default: data/watch-history.json)",
    )
    parser.add_argument(
        "--stations",
        type=int,
        default=3,
        choices=[2, 3, 4],
        help="Number of stations to generate (2-4, default: 3)",
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
