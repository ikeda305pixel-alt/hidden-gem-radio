"""Deep Music Theory DNA 10-dimensional feature engine (Pure Python + NumPy, Termux-ready).

Generates the 10-dimensional context vector for Contextual Bandit learning:
  0: t1_tonal_suspension       (Absent / emergent tonic, IV-V-vi, dorian/mixolydian loops)
  1: t2_tresillo_bounce        (3-3-2 tresillo, UKG 2-step, off-beat accent, bouncy 808/funk bass)
  2: t3_harmonic_propulsion    (Harmonic minor i-V7, V/vi secondary dominant, slash bass, maj7/add9)
  3: t4_valence_contrast       (Bright major chords x melancholic vocal, stratified->unified chorus)
  4: t5_pulse_hook             (105-130 BPM direct or 135-170 BPM double/half-time, <15s hook)
  5: yt_history_affinity       (Proximity to weighted YouTube watch-history seeds)
  6: hidden_gem_score          (Inverse popularity score inside station view cap)
  7: cross_genre_bridge_score  (Cross-genre structural fusion bonus)
  8: tod_sin                   (sin(2*pi*hour/24))
  9: tod_cos                   (cos(2*pi*hour/24))
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.models import StationConfig, TrackEntity, normalize_artist

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_THEORY_PROFILE_PATH = PROJECT_ROOT / "config" / "music_theory_profile.yaml"

THEORY_AXIS_KEYS: tuple[str, ...] = ("T1", "T2", "T3", "T4", "T5")
THEORY_FEATURE_NAMES: tuple[str, ...] = (
    "t1_tonal_suspension",
    "t2_tresillo_bounce",
    "t3_harmonic_propulsion",
    "t4_valence_contrast",
    "t5_pulse_hook",
    "yt_history_affinity",
    "hidden_gem_score",
    "cross_genre_bridge_score",
    "tod_sin",
    "tod_cos",
)

_DEFAULT_AXIS_META: dict[str, dict[str, str]] = {
    "T1": {"key": "tonal_suspension", "label_ja": "不在トニック循環", "label_en": "Tonal Suspension"},
    "T2": {"key": "tresillo_bounce", "label_ja": "トレシーロ跳ね", "label_en": "Tresillo & Bounce"},
    "T3": {"key": "harmonic_propulsion", "label_ja": "和声推進力", "label_en": "Harmonic Propulsion"},
    "T4": {"key": "valence_contrast_lift", "label_ja": "明暗コントラスト爆発", "label_en": "Valence Contrast Lift"},
    "T5": {"key": "pulse_hook_immediacy", "label_ja": "即効パルス＆フック", "label_en": "Pulse & Hook Immediacy"},
    "T6": {"key": "affinity_x_gem", "label_ja": "履歴親和×隠れ度", "label_en": "History Affinity x Hidden Gem"},
}

# Structural keyword detectors (deterministic, regex-compiled once)
_SIGNAL_PATTERNS: dict[str, tuple[re.Pattern[str], dict[str, float]]] = {
    "ukg_2step": (
        re.compile(r"\b(?:ukg|uk\s*garage|2-?step|speed\s*garage|notion|boiler\s*room)\b", re.IGNORECASE),
        {"T1": 0.14, "T2": 0.24, "T3": 0.08, "T4": 0.06, "T5": 0.16},
    ),
    "house_edm": (
        re.compile(
            r"\b(?:progressive\s*house|melodic\s*house|deep\s*house|future\s*house|edm|"
            r"dance\s*pop|sidechain|avicii|calvin|tiesto|guetta|club\s*mix|remix|edit)\b",
            re.IGNORECASE,
        ),
        {"T1": 0.20, "T2": 0.16, "T3": 0.08, "T4": 0.12, "T5": 0.20},
    ),
    "funk_groove": (
        re.compile(
            r"\b(?:funk|nu-?disco|disco|groove|dorian|bassline|boogie|soul|city\s*pop|"
            r"parcels|vulfpeck|jungle|surfaces|earth\s*wind|imase)\b",
            re.IGNORECASE,
        ),
        {"T1": 0.18, "T2": 0.22, "T3": 0.16, "T4": 0.08, "T5": 0.14},
    ),
    "bouncy_rap": (
        re.compile(
            r"\b(?:bouncy|808|tresillo|hip\s*hop|rap|trap|drill|bbno|migos|cole|durk|drake|freestyle|cypher)\b",
            re.IGNORECASE,
        ),
        {"T1": 0.04, "T2": 0.20, "T3": 0.14, "T4": 0.02, "T5": 0.16},
    ),
    "japanese_hiphop": (
        re.compile(
            r"(?:\b(?:j-?rap|japanese\s*hip\s*hop|bad\s*hop|yzerr|zorn|awich|jjj|fresino|kandytown)\b|"
            r"(?:ラップ|ヒップホップ|ストリート|フリースタイル|サイファー))",
            re.IGNORECASE,
        ),
        {"T1": 0.08, "T2": 0.14, "T3": 0.18, "T4": 0.12, "T5": 0.14},
    ),
    "emo_melodic_guitar": (
        re.compile(
            r"\b(?:emo|melodic\s*rap|alt-?pop|pop\s*punk|guitar|acoustic|sad|heartbreak|"
            r"peep|juice\s*wrld|laroi|blackbear|iann\s*dior|mgk|zucker|dream\s*pop)\b",
            re.IGNORECASE,
        ),
        {"T1": 0.12, "T2": 0.04, "T3": 0.10, "T4": 0.26, "T5": 0.12},
    ),
    "indie_melodic": (
        re.compile(
            r"\b(?:indie|folktronica|indietronica|shoegaze|synth\s*pop|royal\s*road|"
            r"royel\s*otis|oberhofer|foster|phoenix|bastille|dope\s*lemon|hozier)\b",
            re.IGNORECASE,
        ),
        {"T1": 0.18, "T2": 0.14, "T3": 0.16, "T4": 0.16, "T5": 0.12},
    ),
    "harmonic_minor_jazz": (
        re.compile(r"\b(?:minor|jazz|neo-?soul|r&b|piano|chord|maj7|add9|melancholic|nocturne)\b", re.IGNORECASE),
        {"T1": 0.12, "T2": 0.06, "T3": 0.22, "T4": 0.10, "T5": 0.04},
    ),
    "japanese_script": (
        re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]"),
        {"T1": 0.06, "T2": 0.04, "T3": 0.10, "T4": 0.08, "T5": 0.04},
    ),
}

_CROSS_GENRE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:remix|flip|bootleg|edit|mashup|version)\b", re.IGNORECASE),
    re.compile(r"\b(?:feat\.?|ft\.?|x\s+)\b", re.IGNORECASE),
    re.compile(r"\b(?:indietronica|folktronica|emo\s*rap|melodic\s*rap|nu-?disco|funk\s*pop|uk\s*garage|jazz\s*rap)\b", re.IGNORECASE),
    re.compile(r"\b(?:acoustic|guitar|live\s+(?:on|at)|colors|boiler\s*room)\b", re.IGNORECASE),
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(slots=True)
class AudioTheorySignal:
    """Future extension hook for real audio DSP analysis (BPM, chroma, onset autocorrelation)."""

    bpm: float | None = None
    key_ambiguity: float | None = None          # Maps to T1 (0.0 - 1.0)
    tresillo_autocorr: float | None = None      # Maps to T2 (0.0 - 1.0)
    harmonic_tension: float | None = None       # Maps to T3 (0.0 - 1.0)
    valence_contrast: float | None = None       # Maps to T4 (0.0 - 1.0)
    hook_onset_sec: float | None = None         # Maps to T5 (<=15s -> high score)
    confidence: float = 0.0                     # Blend weight in [0.0, 1.0]


def merge_audio_theory_signal(
    base_t1_t5: list[float],
    audio_signal: AudioTheorySignal | None,
) -> list[float]:
    """Blend real DSP AudioTheorySignal into metadata-derived T1..T5 scores by confidence."""
    if audio_signal is None or audio_signal.confidence <= 0.0:
        return [_clip01(v) for v in base_t1_t5[:5]]

    alpha = _clip01(audio_signal.confidence)
    merged = [_clip01(v) for v in base_t1_t5[:5]]

    if audio_signal.key_ambiguity is not None:
        merged[0] = _clip01((1.0 - alpha) * merged[0] + alpha * _clip01(audio_signal.key_ambiguity))
    if audio_signal.tresillo_autocorr is not None:
        merged[1] = _clip01((1.0 - alpha) * merged[1] + alpha * _clip01(audio_signal.tresillo_autocorr))
    if audio_signal.harmonic_tension is not None:
        merged[2] = _clip01((1.0 - alpha) * merged[2] + alpha * _clip01(audio_signal.harmonic_tension))
    if audio_signal.valence_contrast is not None:
        merged[3] = _clip01((1.0 - alpha) * merged[3] + alpha * _clip01(audio_signal.valence_contrast))

    # T5 combines BPM sweet spots (105-130 or 135-170) and <=15s hook onset immediacy
    t5_parts: list[float] = []
    if audio_signal.bpm is not None and audio_signal.bpm > 0:
        bpm = float(audio_signal.bpm)
        if 105.0 <= bpm <= 130.0 or 135.0 <= bpm <= 170.0:
            t5_parts.append(0.95)
        elif 90.0 <= bpm <= 176.0:
            t5_parts.append(0.72)
        else:
            t5_parts.append(0.45)
    if audio_signal.hook_onset_sec is not None and audio_signal.hook_onset_sec >= 0:
        onset = float(audio_signal.hook_onset_sec)
        if onset <= 15.0:
            t5_parts.append(0.95)
        elif onset <= 30.0:
            t5_parts.append(0.70)
        else:
            t5_parts.append(0.40)
    if t5_parts:
        t5_audio = sum(t5_parts) / len(t5_parts)
        merged[4] = _clip01((1.0 - alpha) * merged[4] + alpha * t5_audio)

    return merged


def load_music_theory_profile(path: Path | str = DEFAULT_THEORY_PROFILE_PATH) -> dict[str, Any]:
    """Load config/music_theory_profile.yaml cleanly with safe empty fallback."""
    profile_path = Path(path)
    if not profile_path.exists():
        return {"axes": _DEFAULT_AXIS_META, "archetypes": {}, "stations": []}
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {"axes": _DEFAULT_AXIS_META, "archetypes": {}, "stations": []}


class TheoryFeatureExtractor:
    """Deterministic 10-dimensional Deep Music Theory DNA feature extractor."""

    def __init__(self, profile_path: Path | str = DEFAULT_THEORY_PROFILE_PATH) -> None:
        self.profile = load_music_theory_profile(profile_path)
        self.axes_meta: dict[str, dict[str, Any]] = dict(_DEFAULT_AXIS_META)
        for ax_id, meta in (self.profile.get("axes") or {}).items():
            if isinstance(meta, dict):
                self.axes_meta[str(ax_id).upper()] = meta
        self.archetypes: dict[str, dict[str, Any]] = {
            str(k): v for k, v in (self.profile.get("archetypes") or {}).items() if isinstance(v, dict)
        }

    def get_station_prior_vector(self, station: StationConfig | None) -> dict[str, float]:
        """Return station's target T1..T5 vector (defaulting to balanced 0.65 prior)."""
        prior = {ax: 0.65 for ax in THEORY_AXIS_KEYS}
        if station is not None and station.target_theory_vector:
            for ax in THEORY_AXIS_KEYS:
                if ax in station.target_theory_vector:
                    prior[ax] = _clip01(float(station.target_theory_vector[ax]))
        return prior

    def _compute_hidden_gem_score(self, view_count: int | None, max_popularity_views: int) -> float:
        if view_count is None or view_count < 0:
            return 0.58
        cap = max(50_000, int(max_popularity_views))
        if view_count <= 5_000:
            return 0.96
        if view_count <= 25_000:
            return 0.90
        if view_count <= 100_000:
            return 0.78
        if view_count <= cap:
            ratio = (view_count - 100_000) / max(1, cap - 100_000)
            return _clip01(0.75 - 0.35 * ratio)
        return 0.15

    def _compute_history_affinity(self, track: TrackEntity, station: StationConfig | None) -> float:
        if station is None:
            return 0.50

        weighted_seeds = station.weighted_seeds or []
        track_artist_norm = normalize_artist(track.artist)
        seed_artist_norm = normalize_artist(track.seed_artist or "")

        best_affinity = 0.0
        for item in weighted_seeds:
            w = _clip01(float(item.get("weight", 0.5)))
            s_artist = normalize_artist(str(item.get("artist") or ""))
            if not s_artist:
                continue
            if seed_artist_norm and (seed_artist_norm == s_artist or s_artist in seed_artist_norm):
                best_affinity = max(best_affinity, 0.55 + 0.45 * w)
            elif track_artist_norm and (track_artist_norm == s_artist or s_artist in track_artist_norm):
                best_affinity = max(best_affinity, 0.45 + 0.40 * w)

        if best_affinity > 0.0:
            return _clip01(best_affinity)

        # Fallback: proximity to station's anchor artists / seed tags
        anchor_norms = {normalize_artist(a) for a in station.seed_artists if a}
        if seed_artist_norm and seed_artist_norm in anchor_norms:
            return 0.78
        if track_artist_norm and track_artist_norm in anchor_norms:
            return 0.70

        if track.tags and station.seed_tags:
            overlap = len(set(track.tags) & set(station.seed_tags))
            if overlap > 0:
                return _clip01(0.45 + 0.15 * min(3, overlap))
        return 0.38 if track.source in {"watch_playlist", "ytmusic_watch_playlist"} else 0.32

    def _compute_cross_genre_bridge(self, track: TrackEntity, text_blob: str, station: StationConfig | None) -> float:
        hits = sum(1 for pat in _CROSS_GENRE_PATTERNS if pat.search(text_blob))
        tag_diversity = 0.0
        if track.tags:
            if station and station.seed_tags:
                non_seed_tags = set(track.tags) - set(station.seed_tags)
                tag_diversity = min(0.35, 0.12 * len(non_seed_tags))
            else:
                tag_diversity = min(0.30, 0.10 * len(set(track.tags)))
        return _clip01(0.25 + 0.22 * min(3, hits) + tag_diversity)

    def extract_vector(
        self,
        track: TrackEntity,
        *,
        station: StationConfig | None = None,
        max_popularity_views: int = 500_000,
        hour: int | None = None,
        audio_signal: AudioTheorySignal | None = None,
    ) -> np.ndarray:
        """Compute the exact 10-dimensional float64 context vector for MABWiser."""
        # 1. Start from neutral baseline blended gently with lineage/station archetype prior
        t_scores = {ax: 0.45 for ax in THEORY_AXIS_KEYS}

        # Apply lineage archetype if track was dug from a specific theory archetype
        if track.seed_archetype and track.seed_archetype in self.archetypes:
            arch_weights = self.archetypes[track.seed_archetype].get("weights") or {}
            for ax in THEORY_AXIS_KEYS:
                if ax in arch_weights:
                    t_scores[ax] = 0.35 * t_scores[ax] + 0.65 * float(arch_weights[ax])
        elif station is not None and station.target_theory_vector:
            # Soft station lineage prior (0.35 weight) so individual track signals still differentiate
            for ax in THEORY_AXIS_KEYS:
                if ax in station.target_theory_vector:
                    t_scores[ax] = 0.65 * t_scores[ax] + 0.35 * float(station.target_theory_vector[ax])

        # 2. Structural metadata signals from title + artist + tags + source
        text_blob = " ".join([track.artist, track.title, *track.tags, track.source or ""])
        for pattern, deltas in _SIGNAL_PATTERNS.values():
            if pattern.search(text_blob):
                for ax, delta in deltas.items():
                    t_scores[ax] += delta

        # 3. Continuous duration/year/view_count micro-biases so tagless legacy DB rows differentiate smoothly
        if track.duration_seconds is not None:
            dur = float(max(0, int(track.duration_seconds)))
            # Smooth Gaussian hook-density sweet spot centered at 195s (~3:15)
            t_scores["T5"] += 0.18 * math.exp(-((dur - 195.0) / 85.0) ** 2) - 0.06
            t_scores["T2"] += 0.04 * math.tanh((210.0 - dur) / 90.0)
            t_scores["T4"] += 0.04 * math.tanh((dur - 180.0) / 90.0)

        if track.release_year is not None:
            yr_norm = math.tanh((float(track.release_year) - 2015.0) / 10.0)
            t_scores["T1"] += 0.045 * yr_norm
            t_scores["T3"] += 0.045 * yr_norm
            t_scores["T4"] -= 0.035 * yr_norm

        if track.view_count is not None:
            log_v = math.log10(max(0, int(track.view_count)) + 1.0)
            v_norm = math.tanh((log_v - 4.5) / 1.5)
            t_scores["T1"] -= 0.03 * v_norm
            t_scores["T2"] += 0.03 * v_norm
            t_scores["T5"] += 0.035 * v_norm

        # 4. Clip and optionally merge real DSP audio signals
        t1_t5 = merge_audio_theory_signal([t_scores[ax] for ax in THEORY_AXIS_KEYS], audio_signal)

        # 5. Dimensions 5..9: yt_history_affinity, hidden_gem_score, cross_genre_bridge, tod_sin, tod_cos
        yt_affinity = self._compute_history_affinity(track, station)
        hidden_gem = self._compute_hidden_gem_score(track.view_count, max_popularity_views)
        cross_bridge = self._compute_cross_genre_bridge(track, text_blob, station)

        h = datetime.now().hour if hour is None else int(hour) % 24
        angle = 2.0 * math.pi * (h / 24.0)
        tod_sin = math.sin(angle)
        tod_cos = math.cos(angle)

        return np.array(
            [
                t1_t5[0],
                t1_t5[1],
                t1_t5[2],
                t1_t5[3],
                t1_t5[4],
                yt_affinity,
                hidden_gem,
                cross_bridge,
                tod_sin,
                tod_cos,
            ],
            dtype=np.float64,
        )

    def build_explanation(
        self,
        vector: np.ndarray,
        *,
        station: StationConfig | None = None,
        track: TrackEntity | None = None,
    ) -> dict[str, Any]:
        """Generate concise Japanese theory explanation + top 2 matched T1..T5 axes."""
        target = self.get_station_prior_vector(station)
        axis_entries: list[dict[str, Any]] = []
        axis_map: dict[str, float] = {}

        for idx, ax in enumerate(THEORY_AXIS_KEYS):
            score = round(float(vector[idx]), 3)
            axis_map[ax] = score
            meta = self.axes_meta.get(ax, _DEFAULT_AXIS_META[ax])
            # Match affinity weighs both track strength and station target relevance
            match_score = score * (0.5 + 0.5 * target[ax])
            axis_entries.append(
                {
                    "axis": ax,
                    "key": meta.get("key", ax.lower()),
                    "label": meta.get("label_ja", ax),
                    "score": score,
                    "_match": match_score,
                }
            )

        # T6 = affinity x hidden gem composite
        t6_score = round(float(math.sqrt(max(0.0, vector[5] * vector[6]))), 3)
        axis_map["T6"] = t6_score

        top_two = sorted(axis_entries, key=lambda item: item["_match"], reverse=True)[:2]
        clean_top_two = [
            {
                "axis": item["axis"],
                "key": item["key"],
                "label": item["label"],
                "score": item["score"],
            }
            for item in top_two
        ]

        gem_pct = round(float(vector[6]) * 100)
        arch_hint = ""
        if track and track.seed_archetype and track.seed_archetype in self.archetypes:
            arch_label = self.archetypes[track.seed_archetype].get("label_ja")
            if arch_label:
                arch_hint = f" / {arch_label}"

        summary = (
            f"{clean_top_two[0]['label']}({int(clean_top_two[0]['score'] * 100)}%) × "
            f"{clean_top_two[1]['label']}({int(clean_top_two[1]['score'] * 100)}%)"
            f"{arch_hint}｜隠れ度{gem_pct}%"
        )
        return {
            "summary": summary,
            "top_axes": clean_top_two,
            "axes": axis_map,
            "yt_history_affinity": round(float(vector[5]), 3),
            "hidden_gem_score": round(float(vector[6]), 3),
            "cross_genre_bridge_score": round(float(vector[7]), 3),
        }
