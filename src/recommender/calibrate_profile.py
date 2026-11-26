"""Automated 3-source measured baseline calibration for each station (YouTube 45% + Spotify 35% + PES/Emily 20%).

Computes per-station:
  - centroid[station] (16-dim): weighted acoustic/MIR + scene mean across real songs
  - axis_weights[station] (16-dim): consistency weights (lower variance -> higher weight)
  - __cult_chic_anchor__ (12-dim A1..C4): measured centroid of PES 2011 + Emily in Paris anchors

CLI:
    python -m src.recommender.calibrate_profile
    python -m src.recommender.calibrate_profile --takeout data/watch-history.json --offline
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.adapters.deezer import DeezerClient
from src.db import RadioDatabase
from src.models import StationConfig, TrackEntity, load_stations_yaml
from src.recommender.audio_mir import (
    DEFAULT_CULT_CHIC_A1_C4,
    MIR_DIM,
    AudioMIRExtractor,
)
from src.recommender.theory_features import DEFAULT_THEORY_PROFILE_PATH

logger = logging.getLogger(__name__)

# Source blend ratios required by specification
WEIGHT_YOUTUBE = 0.45
WEIGHT_SPOTIFY = 0.35
WEIGHT_CULT_ANCHORS = 0.20

# Deterministic acoustic anchor offsets for offline fallback calibration when Deezer/ffmpeg are unavailable
_STATION_OFFLINE_PRIORS: dict[str, list[float]] = {
    "us_hiphop_bouncy": [
        0.52, 0.78, 0.76, 0.78, 0.44, 0.48, 0.46, 0.46, 0.44, 0.28, 0.78, 0.62, 0.65, 0.80, 0.90, 0.55,
    ],
    "japan_hiphop_street": [
        0.48, 0.72, 0.66, 0.72, 0.38, 0.54, 0.50, 0.52, 0.40, 0.26, 0.74, 0.66, 0.65, 0.76, 0.88, 0.60,
    ],
    "edm_dance_party": [
        0.56, 0.88, 0.72, 0.74, 0.58, 0.56, 0.54, 0.54, 0.58, 0.36, 0.86, 0.58, 0.68, 0.84, 0.92, 0.64,
    ],
    "melodic_rap_alt": [
        0.64, 0.66, 0.54, 0.56, 0.54, 0.58, 0.58, 0.64, 0.48, 0.24, 0.72, 0.70, 0.65, 0.80, 0.90, 0.62,
    ],
    "indie_groove_cult": [
        0.50, 0.70, 0.62, 0.48, 0.60, 0.68, 0.58, 0.68, 0.46, 0.22, 0.66, 0.64, 0.70, 0.82, 0.88, 0.86,
    ],
}


def default_station_profile_vectors(station_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sensible 16-dim default (centroid, axis_weights) before first calibration."""
    prior = _STATION_OFFLINE_PRIORS.get(station_id)
    if prior is None:
        centroid = np.full(MIR_DIM, 0.55, dtype=np.float64)
    else:
        centroid = np.array(prior, dtype=np.float64)
    axis_weights = np.ones(MIR_DIM, dtype=np.float64)
    return centroid, axis_weights


def _parse_seed_pair(seed_str: str) -> tuple[str, str]:
    text = str(seed_str or "").strip()
    if " - " in text:
        left, right = text.split(" - ", 1)
        return left.strip(), right.strip()
    return text, ""


def _compute_weighted_mean_and_consistency(
    vectors: list[np.ndarray],
    weights: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean centroid and inverse-variance consistency axis_weights (mean ~ 1.0)."""
    mat = np.vstack(vectors).astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-12:
        w = np.ones(len(vectors), dtype=np.float64) / max(1, len(vectors))
    else:
        w = w / w_sum

    centroid = np.sum(mat * w[:, None], axis=0)
    var = np.sum(((mat - centroid[None, :]) ** 2) * w[:, None], axis=0)
    std = np.sqrt(np.maximum(0.0, var))

    # Tighter variance -> stronger consistency weight; normalize mean to 1.0 and clip to [0.45, 2.0]
    raw_axis_w = 1.0 / (std + 0.15)
    norm_axis_w = raw_axis_w / max(1e-9, float(np.mean(raw_axis_w)))
    axis_weights = np.clip(norm_axis_w, 0.45, 2.0)
    return centroid, axis_weights


def _collect_youtube_station_top_tracks(
    takeout_records: list[dict[str, Any]],
    *,
    reference_time: datetime | None = None,
    max_per_station: int = 30,
) -> dict[str, list[tuple[str, str, float]]]:
    """Group watch history records by station and return top 30 (artist, title, weight) per station."""
    from src.digger.history_importer import (
        classify_entry_to_station,
        compute_recency_frequency_weight,
    )

    now = reference_time or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    per_station: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for rec in takeout_records:
        artist = str(rec.get("artist") or "").strip()
        title = str(rec.get("title") or "").strip()
        if not artist or not title:
            continue
        st_id = classify_entry_to_station(artist, title)
        key = (artist.lower(), title.lower())
        bucket = per_station.setdefault(st_id, {})

        played_raw = str(rec.get("played_at") or rec.get("time") or "").strip()
        days_ago = 30.0
        if played_raw:
            try:
                ts = played_raw[:-1] + "+00:00" if played_raw.endswith("Z") else played_raw
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                days_ago = max(0.0, (now - dt).total_seconds() / 86400.0)
            except ValueError:
                days_ago = 30.0

        inc = 3 if rec.get("source") == "ytmusic_liked" else 1
        if key not in bucket:
            bucket[key] = {"artist": artist, "title": title, "count": inc, "min_days": days_ago}
        else:
            bucket[key]["count"] += inc
            bucket[key]["min_days"] = min(bucket[key]["min_days"], days_ago)

    result: dict[str, list[tuple[str, str, float]]] = {}
    for st_id, entries in per_station.items():
        scored = [
            (
                str(item["artist"]),
                str(item["title"]),
                compute_recency_frequency_weight(int(item["count"]), float(item["min_days"])),
            )
            for item in entries.values()
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        result[st_id] = scored[:max_per_station]
    return result


def calibrate_all_stations(
    db: RadioDatabase,
    *,
    stations: list[StationConfig] | None = None,
    stations_path: Path | str | None = None,
    profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH,
    takeout_path: Path | str | None = Path("data/watch-history.json"),
    takeout_records: list[dict[str, Any]] | None = None,
    mir_extractor: AudioMIRExtractor | None = None,
    allow_network: bool = True,
    max_youtube_per_station: int = 30,
    max_spotify_per_station: int = 40,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Calibrate 16-dim centroids & axis_weights from YouTube (45%) + Spotify (35%) + PES/Emily (20%)."""
    from src.digger.history_importer import (
        default_theory_stations,
        parse_takeout_watch_history,
    )
    if stations is None:
        profile_stations = default_theory_stations(profile_path=profile_path)
        profile_by_id = {s.station_id: s for s in profile_stations}
        if stations_path and Path(stations_path).exists():
            loaded = load_stations_yaml(Path(stations_path))
            # Enrich legacy stations.yaml entries with full Spotify/PES/Emily anchors from music_theory_profile.yaml
            for st in loaded:
                ref_st = profile_by_id.get(st.station_id)
                if ref_st is not None:
                    if not st.pes_anchors and ref_st.pes_anchors:
                        st.pes_anchors = list(ref_st.pes_anchors)
                    if not st.emily_anchors and ref_st.emily_anchors:
                        st.emily_anchors = list(ref_st.emily_anchors)
                    if len(st.spotify_anchor_seeds) < len(ref_st.spotify_anchor_seeds):
                        st.spotify_anchor_seeds = list(
                            dict.fromkeys([*st.spotify_anchor_seeds, *ref_st.spotify_anchor_seeds])
                        )
            stations = loaded
        else:
            stations = profile_stations
    if not stations:
        stations = default_theory_stations(profile_path=profile_path)

    deezer_client: DeezerClient | None = None
    owns_deezer = False
    if mir_extractor is None:
        deezer_client = DeezerClient(db=db) if allow_network else None
        owns_deezer = deezer_client is not None
        extractor = AudioMIRExtractor(db=db, deezer_client=deezer_client)
    else:
        extractor = mir_extractor

    try:
        # 1. Calibrate shared PES 2011 / Emily in Paris 12-dim acoustic centroid (__cult_chic_anchor__)
        all_cult_seeds: list[str] = []
        for st in stations:
            all_cult_seeds.extend(st.pes_anchors)
            all_cult_seeds.extend(st.emily_anchors)
        all_cult_seeds = list(dict.fromkeys(all_cult_seeds))

        cult_a1_c4_vecs: list[np.ndarray] = []
        for seed in all_cult_seeds:
            artist, title = _parse_seed_pair(seed)
            if not artist:
                continue
            vec12, _bpm, _meta, src = extractor.extract_a1_c4_for_track(
                artist,
                title,
                allow_network=allow_network,
            )
            if src in {"mir_pcm", "deezer_metadata", "cache"} or not allow_network:
                cult_a1_c4_vecs.append(vec12)

        if cult_a1_c4_vecs:
            cult_anchor_12 = np.mean(np.vstack(cult_a1_c4_vecs), axis=0)
        else:
            cult_anchor_12 = DEFAULT_CULT_CHIC_A1_C4.copy()

        extractor.set_cult_chic_anchor(cult_anchor_12)
        db.upsert_station_profile(
            station_id="__cult_chic_anchor__",
            centroid=[round(float(x), 5) for x in np.pad(cult_anchor_12, (0, 4), constant_values=0.8)],
            axis_weights=[1.0] * MIR_DIM,
            slow_weights=[round(float(x), 5) for x in np.pad(cult_anchor_12, (0, 4), constant_values=0.8)],
            sample_counts={"cult_chic_tracks": len(cult_a1_c4_vecs)},
        )

        # 2. Load YouTube watch history records (up to 30 top-weighted tracks per station)
        records = list(takeout_records) if takeout_records is not None else []
        if not records and takeout_path and Path(takeout_path).exists():
            records = parse_takeout_watch_history(takeout_path)
        yt_by_station = _collect_youtube_station_top_tracks(
            records,
            reference_time=reference_time,
            max_per_station=max_youtube_per_station,
        )

        def _extract_group(
            items: list[tuple[str, str, float]],
            source_blend: float,
            bucket_key: str,
            counts: dict[str, int],
            source_vecs: list[np.ndarray],
            source_weights: list[float],
        ) -> None:
            group_vecs: list[np.ndarray] = []
            group_raw_w: list[float] = []
            for artist, title, item_w in items:
                if not artist:
                    continue
                dummy = TrackEntity.build(
                    artist=artist,
                    title=title or artist,
                    cooccurrence_count=2,
                    view_count=5_000_000,
                )
                vec16, _bpm, _src = extractor.extract_16d_vector(
                    dummy,
                    allow_network=allow_network,
                )
                group_vecs.append(vec16)
                group_raw_w.append(max(1e-4, float(item_w)))

            if not group_vecs:
                return
            counts[bucket_key] = len(group_vecs)
            w_total = sum(group_raw_w)
            for v, rw in zip(group_vecs, group_raw_w, strict=True):
                source_vecs.append(v)
                source_weights.append(source_blend * (rw / w_total))

        # 3. Calibrate each station independently
        summary_stations: dict[str, Any] = {}
        for st in stations:
            source_vecs: list[np.ndarray] = []
            source_weights: list[float] = []
            counts = {"youtube": 0, "spotify": 0, "cult_anchors": 0}

            # Source 1: YouTube history top 30 (45%)
            yt_items = yt_by_station.get(st.station_id, [])
            if yt_items:
                _extract_group(yt_items, WEIGHT_YOUTUBE, "youtube", counts, source_vecs, source_weights)

            # Source 2: Spotify curated favorites (35%)
            sp_items = [
                (*_parse_seed_pair(seed), 1.0)
                for seed in st.spotify_anchor_seeds[:max_spotify_per_station]
            ]
            if sp_items:
                _extract_group(sp_items, WEIGHT_SPOTIFY, "spotify", counts, source_vecs, source_weights)

            # Source 3: PES 2011 / Emily in Paris anchors (20%)
            cult_list = [*st.pes_anchors, *st.emily_anchors]
            cult_items = [(*_parse_seed_pair(seed), 1.0) for seed in cult_list]
            if cult_items:
                _extract_group(cult_items, WEIGHT_CULT_ANCHORS, "cult_anchors", counts, source_vecs, source_weights)

            if source_vecs:
                centroid, axis_weights = _compute_weighted_mean_and_consistency(source_vecs, source_weights)
            else:
                centroid, axis_weights = default_station_profile_vectors(st.station_id)

            # Preserve existing slow_weights if already learned, otherwise initialize to calibrated anchor
            existing = db.get_station_profile(st.station_id)
            slow_weights = (
                existing["slow_weights"]
                if existing and len(existing.get("slow_weights", [])) == MIR_DIM
                else [round(float(x), 5) for x in centroid]
            )

            db.upsert_station_profile(
                station_id=st.station_id,
                centroid=[round(float(x), 5) for x in centroid],
                axis_weights=[round(float(x), 5) for x in axis_weights],
                slow_weights=slow_weights,
                sample_counts=counts,
            )
            summary_stations[st.station_id] = {
                "sample_counts": counts,
                "centroid": [round(float(x), 4) for x in centroid],
                "axis_weights": [round(float(x), 4) for x in axis_weights],
            }

        return {
            "status": "ok",
            "cult_chic_tracks": len(cult_a1_c4_vecs),
            "stations": summary_stations,
        }
    finally:
        if owns_deezer and deezer_client is not None:
            deezer_client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate 16-dim MIR station centroids & consistency weights from YouTube + Spotify + PES/Emily."
    )
    parser.add_argument("--db", default="data/radio.db", help="SQLite DB path")
    parser.add_argument("--stations", default="config/stations.yaml", help="stations.yaml path (optional)")
    parser.add_argument("--takeout", default="data/watch-history.json", help="Google Takeout watch-history.json path")
    parser.add_argument("--offline", action="store_true", help="Skip live Deezer/preview network calls")
    args = parser.parse_args(argv)

    db = RadioDatabase(args.db)
    result = calibrate_all_stations(
        db,
        stations_path=args.stations,
        takeout_path=args.takeout,
        allow_network=not args.offline,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
