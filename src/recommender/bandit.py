"""Stage 2 Precision Scoring & Stage 3 Re-Ranking with Per-Station 3-Layer Stable MIR Learning.

Per-station completely isolated 3-layer effective preference synthesis:
    W_effective[s] = 0.75 * W_calibrated_anchor[s] + 0.20 * W_slow_learned[s] + 0.05 * W_session_mood[s]

Bounded slow learning rule (prevents single-click drift or cross-station bleed):
  - GOOD: +1.0, BAD: -1.0, COMPLETE (>=70%): +0.35, EARLY_SKIP (<15s): -0.40
  - lr = 0.03, per-step change clipped to [-0.02, +0.02], L2 pull toward calibrated anchor.

Stage 3 Re-Ranking:
  - Enforces 5-track recent artist cooldown (no duplicate artist in last 5 tracks).
  - Balances 80% Exploitation (high-score semi-classics / hidden gems) + 20% Exploration (co-occurring cult/chic gems).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import shutil
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from mabwiser.mab import MAB, LearningPolicy

from src.db import RadioDatabase
from src.models import (
    LEGACY_STATION_ID_MAP,
    RankedCandidate,
    StationConfig,
    TrackEntity,
    normalize_artist,
)
from src.recommender.audio_mir import (
    MIR_DIM,
    AudioMIRExtractor,
    summarize_4_layers,
)
from src.recommender.calibrate_profile import default_station_profile_vectors
from src.recommender.theory_features import (
    THEORY_AXIS_KEYS,
    AudioTheorySignal,
    TheoryFeatureExtractor,
)

logger = logging.getLogger(__name__)

_OOV_TAG_BUCKETS = 8
_DECADE_BUCKETS = 6  # <1990, 1990s, 2000s, 2010s, 2020s+, unknown
_VIEW_BUCKETS = 5    # <10k, 10k-50k, 50k-200k, 200k-500k, >=500k/unknown
_TIME_BUCKETS = 4    # morning, afternoon, evening, late_night
_WATCH_PLAYLIST_SOURCES = {"watch_playlist", "ytmusic_watch_playlist", "related_artist_deep_cut"}
THEORY_CONTEXT_DIM = 10

# 3-layer stable synthesis coefficients
ALPHA_ANCHOR = 0.75
ALPHA_SLOW = 0.20
ALPHA_SESSION = 0.05

# Bounded slow-learning hyperparameters
SLOW_LR = 0.03
SLOW_STEP_CLIP = 0.02
SLOW_L2_ANCHOR = 0.02


def _hash_bucket(text: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
    return int(digest, 16) % num_buckets


def _hidden_level(view_count: int | None, max_popularity_views: int) -> str:
    if view_count is None:
        return "発掘"
    if 300_000 <= view_count <= 50_000_000:
        return "S (名曲スイートスポット)"
    ratio = view_count / max(1, max_popularity_views)
    if view_count < 100_000:
        return "C (極小・要検証)"
    if view_count < 300_000 or ratio <= 0.25:
        return "A (隠れ名曲)"
    if view_count <= 500_000_000:
        return "B (準定番)"
    return "C (超定番)"


def _sigmoid(x: float) -> float:
    clipped = max(-10.0, min(10.0, float(x)))
    return 1.0 / (1.0 + math.exp(-clipped))


class ContextualBanditRecommender:
    """Per-station 3-layer stable MIR + MABWiser LinUCB scorer and Stage 3 re-ranker."""

    def __init__(
        self,
        station_id: str,
        seed_tags: list[str],
        exploration_rate: float = 1.0,
        model_path: Path | str | None = None,
        seed: int = 42,
        station_config: StationConfig | None = None,
        theory_extractor: TheoryFeatureExtractor | None = None,
        mir_extractor: AudioMIRExtractor | None = None,
        db: RadioDatabase | None = None,
    ) -> None:
        self.station_id = station_id
        self.seed_tags = [t.strip().lower() for t in seed_tags if t and t.strip()]
        self.tag_vocab = list(dict.fromkeys(self.seed_tags))
        self.exploration_rate = float(max(0.05, exploration_rate))
        self.model_path = Path(model_path) if model_path else Path(f"data/bandit_{station_id}.pkl")
        self.seed = int(seed)
        self.station_config = station_config
        self.theory_extractor = theory_extractor or TheoryFeatureExtractor()
        self.db = db
        self.mir_extractor = mir_extractor or AudioMIRExtractor(db=db)
        self._lock = threading.Lock()

        # Per-station completely isolated 16-dim 3-layer profile state
        default_centroid, default_axis_w = default_station_profile_vectors(station_id)
        self._w_anchor = default_centroid.copy()
        self._axis_weights = default_axis_w.copy()
        self._w_slow = default_centroid.copy()
        self._w_session = default_centroid.copy()
        self._calibrated_from_db = False
        self._profile_updated_at = 0.0
        self._sample_counts: dict[str, int] = {}
        self._load_calibrated_profile_locked()

        # Primary tag/metadata LinUCB model (preserves backward compatibility)
        self._mab = MAB(
            arms=["play"],
            learning_policy=LearningPolicy.LinUCB(alpha=self.exploration_rate),
            seed=self.seed,
        )
        # 10-dimensional Theory DNA LinUCB model
        self._theory_mab = MAB(
            arms=["play"],
            learning_policy=LearningPolicy.LinUCB(alpha=self.exploration_rate),
            seed=self.seed,
        )
        self._n_updates = 0
        self._theory_n_updates = 0
        self._load_if_exists()

    def _load_calibrated_profile_locked(self, *, preserve_slow_if_loaded: bool = False) -> None:
        if self.db is None:
            return
        prof = self.db.get_station_profile(self.station_id)
        if not prof:
            return
        centroid = prof.get("centroid") or []
        axis_w = prof.get("axis_weights") or []
        slow_w = prof.get("slow_weights") or []
        updated_at = float(prof.get("updated_at") or 0.0)
        if len(centroid) == MIR_DIM and len(axis_w) == MIR_DIM:
            self._w_anchor = np.asarray(centroid, dtype=np.float64)
            self._axis_weights = np.asarray(axis_w, dtype=np.float64)
            if not preserve_slow_if_loaded:
                self._w_slow = (
                    np.asarray(slow_w, dtype=np.float64)
                    if len(slow_w) == MIR_DIM
                    else self._w_anchor.copy()
                )
                self._w_session = self._w_anchor.copy()
            self._calibrated_from_db = True
            self._profile_updated_at = updated_at
            self._sample_counts = dict(prof.get("sample_counts") or {})

    def _sync_external_profile_locked(self) -> None:
        """Detect external CLI calibrate_profile updates via updated_at and reload anchor/axis_weights."""
        if self.db is None:
            return
        prof = self.db.get_station_profile(self.station_id)
        if not prof:
            return
        updated_at = float(prof.get("updated_at") or 0.0)
        if updated_at > self._profile_updated_at + 1e-6:
            # Reload calibrated anchor & consistency weights while preserving online-learned slow_weights
            self._load_calibrated_profile_locked(preserve_slow_if_loaded=self._calibrated_from_db)

    def reload_calibrated_profile(self) -> None:
        """Reload calibrated anchor & consistency axis weights from SQLite without losing slow state."""
        with self._lock:
            self._load_calibrated_profile_locked()

    def get_effective_weights(self) -> np.ndarray:
        """Return W_effective[s] = 0.75 * W_anchor + 0.20 * W_slow + 0.05 * W_session."""
        with self._lock:
            return (
                ALPHA_ANCHOR * self._w_anchor
                + ALPHA_SLOW * self._w_slow
                + ALPHA_SESSION * self._w_session
            ).copy()

    def get_slow_weights(self) -> np.ndarray:
        with self._lock:
            return self._w_slow.copy()

    def get_calibrated_anchor(self) -> np.ndarray:
        with self._lock:
            return self._w_anchor.copy()

    def get_axis_weights(self) -> np.ndarray:
        with self._lock:
            return self._axis_weights.copy()

    @property
    def n_updates(self) -> int:
        with self._lock:
            return self._n_updates

    @property
    def theory_n_updates(self) -> int:
        with self._lock:
            return self._theory_n_updates

    @property
    def feature_dim(self) -> int:
        return (
            len(self.tag_vocab)
            + _OOV_TAG_BUCKETS
            + _DECADE_BUCKETS
            + _VIEW_BUCKETS
            + _TIME_BUCKETS
            + 1  # is_new_artist
            + 1  # is_watch_playlist_source
            + 1  # intercept bias
        )

    @staticmethod
    def compute_reward(
        event_type: str,
        listen_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> float:
        """Map feedback event & listening progress to scalar bandit reward in [0.0, 1.0]."""
        ev = (event_type or "").strip().upper()
        if ev == "GOOD":
            return 1.0
        if ev == "COMPLETE":
            return 0.7
        if ev == "MID_SKIP":
            return 0.3
        if ev in {"BAD", "EARLY_SKIP"}:
            return 0.0

        # Derive from listening duration for generic SKIP / ENDED events
        if listen_seconds is not None and listen_seconds < 15.0:
            return 0.0
        if (
            listen_seconds is not None
            and duration_seconds is not None
            and duration_seconds > 0
            and (listen_seconds / duration_seconds) >= 0.7
        ):
            return 0.7
        if ev == "ENDED":
            return 0.7
        return 0.3

    @staticmethod
    def compute_theory_reward(
        event_type: str,
        listen_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> float:
        """Signed reward for Theory DNA / MIR online learning:
        GOOD=+1.0, BAD=-1.0, COMPLETE (>=70%)=+0.35, EARLY_SKIP (<15s)=-0.40.
        """
        ev = (event_type or "").strip().upper()
        if ev == "GOOD":
            return 1.0
        if ev == "BAD":
            return -1.0
        if ev == "EARLY_SKIP":
            return -0.40
        if ev == "COMPLETE":
            return 0.35
        if ev == "MID_SKIP":
            return 0.0

        if listen_seconds is not None and listen_seconds < 15.0:
            return -0.40
        if (
            listen_seconds is not None
            and duration_seconds is not None
            and duration_seconds > 0
            and (listen_seconds / duration_seconds) >= 0.70
        ):
            return 0.35
        if ev == "ENDED":
            return 0.35
        return 0.0

    def featurize(
        self,
        track: TrackEntity,
        *,
        is_new_artist: bool,
        hour: int | None = None,
    ) -> np.ndarray:
        """Build legacy tag/metadata context vector for backward-compatible LinUCB."""
        vec = np.zeros(self.feature_dim, dtype=np.float64)
        offset = 0

        vocab_index = {tag: i for i, tag in enumerate(self.tag_vocab)}
        for tag in track.tags:
            norm_tag = tag.strip().lower()
            if not norm_tag:
                continue
            if norm_tag in vocab_index:
                vec[offset + vocab_index[norm_tag]] = 1.0
            else:
                bucket = _hash_bucket(norm_tag, _OOV_TAG_BUCKETS)
                vec[offset + len(self.tag_vocab) + bucket] = 1.0
        offset += len(self.tag_vocab) + _OOV_TAG_BUCKETS

        year = track.release_year
        if year is None:
            dec_idx = 5
        elif year < 1990:
            dec_idx = 0
        elif year < 2000:
            dec_idx = 1
        elif year < 2010:
            dec_idx = 2
        elif year < 2020:
            dec_idx = 3
        else:
            dec_idx = 4
        vec[offset + dec_idx] = 1.0
        offset += _DECADE_BUCKETS

        views = track.view_count
        if views is None or views < 0:
            view_idx = 4
        else:
            log_v = math.log10(views + 1.0)
            if log_v < 4.0:
                view_idx = 0
            elif log_v < 4.7:
                view_idx = 1
            elif log_v < 5.3:
                view_idx = 2
            elif log_v < 5.7:
                view_idx = 3
            else:
                view_idx = 4
        vec[offset + view_idx] = 1.0
        offset += _VIEW_BUCKETS

        h = datetime.now().hour if hour is None else int(hour) % 24
        if 5 <= h < 11:
            tod_idx = 0
        elif 11 <= h < 17:
            tod_idx = 1
        elif 17 <= h < 23:
            tod_idx = 2
        else:
            tod_idx = 3
        vec[offset + tod_idx] = 1.0
        offset += _TIME_BUCKETS

        vec[offset] = 1.0 if is_new_artist else 0.0
        vec[offset + 1] = 1.0 if track.source in _WATCH_PLAYLIST_SOURCES else 0.0
        vec[offset + 2] = 1.0
        return vec

    def featurize_theory(
        self,
        track: TrackEntity,
        *,
        max_popularity_views: int = 500_000_000,
        hour: int | None = None,
        audio_signal: AudioTheorySignal | None = None,
    ) -> np.ndarray:
        """Build 10-dimensional Theory DNA context vector."""
        cap = (
            self.station_config.max_popularity_views
            if self.station_config is not None
            else max_popularity_views
        )
        return self.theory_extractor.extract_vector(
            track,
            station=self.station_config,
            max_popularity_views=cap,
            hour=hour,
            audio_signal=audio_signal,
        )

    def featurize_mir(
        self,
        track: TrackEntity,
        *,
        allow_network: bool = False,
    ) -> tuple[np.ndarray, float | None]:
        """Extract 16-dimensional real MIR + co-occurrence + sweetspot + cult-chic vector."""
        vec16, est_bpm, _src = self.mir_extractor.extract_16d_vector(track, allow_network=allow_network)
        return vec16, est_bpm

    def update(
        self,
        track: TrackEntity,
        *,
        event_type: str,
        listen_seconds: float | None = None,
        duration_seconds: float | None = None,
        is_new_artist: bool = True,
        hour: int | None = None,
        audio_signal: AudioTheorySignal | None = None,
    ) -> float:
        """Thread-safely update per-station bounded 3-layer MIR weights and LinUCB with atomic save & rollback."""
        reward = self.compute_reward(event_type, listen_seconds, duration_seconds)
        signed_reward = self.compute_theory_reward(event_type, listen_seconds, duration_seconds)
        context = self.featurize(track, is_new_artist=is_new_artist, hour=hour).reshape(1, -1)
        theory_context = self.featurize_theory(track, hour=hour, audio_signal=audio_signal).reshape(1, -1)
        mir_vec, _bpm = self.featurize_mir(track, allow_network=False)

        with self._lock:
            # Reload externally calibrated centroid/axis_weights if CLI calibrate_profile updated SQLite
            self._sync_external_profile_locked()

            prev_n_updates = self._n_updates
            prev_theory_n_updates = self._theory_n_updates
            prev_mab_snapshot = pickle.dumps(self._mab)
            prev_theory_snapshot = pickle.dumps(self._theory_mab)
            prev_slow = self._w_slow.copy()
            prev_session = self._w_session.copy()
            prev_disk_bytes = self.model_path.read_bytes() if self.model_path.exists() else None
            saved_to_disk = False
            try:
                if self._n_updates == 0:
                    self._mab.fit(decisions=["play"], rewards=[reward], contexts=context)
                else:
                    self._mab.partial_fit(decisions=["play"], rewards=[reward], contexts=context)

                if self._theory_n_updates == 0:
                    self._theory_mab.fit(decisions=["play"], rewards=[signed_reward], contexts=theory_context)
                else:
                    self._theory_mab.partial_fit(decisions=["play"], rewards=[signed_reward], contexts=theory_context)

                # Bounded per-station 3-layer MIR slow update (lr=0.03, clip ±0.02, L2 pull to calibrated anchor)
                if abs(signed_reward) > 1e-9:
                    raw_step = (
                        SLOW_LR * signed_reward * (mir_vec - self._w_slow)
                        - SLOW_L2_ANCHOR * (self._w_slow - self._w_anchor)
                    )
                    clipped_step = np.clip(raw_step, -SLOW_STEP_CLIP, SLOW_STEP_CLIP)
                    self._w_slow = np.clip(self._w_slow + clipped_step, 0.0, 1.0)

                    session_step = np.clip(
                        0.06 * signed_reward * (mir_vec - self._w_session),
                        -SLOW_STEP_CLIP,
                        SLOW_STEP_CLIP,
                    )
                    self._w_session = np.clip(self._w_session + session_step, 0.0, 1.0)

                self._n_updates += 1
                self._theory_n_updates += 1
                self._save_locked()
                saved_to_disk = True

                # Always persist slow_weights to SQLite (even before calibration) using a partial update
                # so externally calibrated centroid & axis_weights are NEVER overwritten!
                if self.db is not None:
                    self.db.update_station_slow_weights(
                        station_id=self.station_id,
                        slow_weights=[round(float(x), 5) for x in self._w_slow],
                        default_centroid=[round(float(x), 5) for x in self._w_anchor],
                        default_axis_weights=[round(float(x), 5) for x in self._axis_weights],
                    )
            except Exception:
                self._n_updates = prev_n_updates
                self._theory_n_updates = prev_theory_n_updates
                self._mab = pickle.loads(prev_mab_snapshot)
                self._theory_mab = pickle.loads(prev_theory_snapshot)
                self._w_slow = prev_slow
                self._w_session = prev_session
                if saved_to_disk:
                    try:
                        if prev_disk_bytes is not None:
                            self.model_path.write_bytes(prev_disk_bytes)
                        elif self.model_path.exists():
                            self.model_path.unlink()
                    except OSError:
                        pass
                raise
        return reward

    def get_learned_theory_profile(self) -> dict[str, Any]:
        """Return 5-axis summary + 4-layer real MIR calibrated preference profile for this station."""
        prior = self.theory_extractor.get_station_prior_vector(self.station_config)
        with self._lock:
            t_updates = self._theory_n_updates
            w_eff = (
                ALPHA_ANCHOR * self._w_anchor
                + ALPHA_SLOW * self._w_slow
                + ALPHA_SESSION * self._w_session
            ).copy()
            calibrated = self._calibrated_from_db
            sample_counts = dict(self._sample_counts)
            beta: np.ndarray | None = None
            if t_updates > 0:
                model = self._theory_mab._imp.arm_to_model.get("play")
                if model is not None and getattr(model, "beta", None) is not None and len(model.beta) >= 5:
                    beta = np.array(model.beta[:5], dtype=np.float64)

        learn_blend = min(0.85, t_updates / (t_updates + 6.0)) if t_updates > 0 else 0.0
        axes_out: list[dict[str, Any]] = []
        scores_map: dict[str, float] = {}

        for idx, ax in enumerate(THEORY_AXIS_KEYS):
            meta = self.theory_extractor.axes_meta.get(ax, {})
            prior_val = float(prior[ax])
            if beta is not None:
                raw_w = float(beta[idx])
                learned_val = _sigmoid(raw_w * 2.2)
                blended = (1.0 - learn_blend) * prior_val + learn_blend * learned_val
            else:
                raw_w = 0.0
                blended = prior_val

            score = round(float(max(0.0, min(1.0, blended))), 3)
            scores_map[ax] = score
            axes_out.append(
                {
                    "axis": ax,
                    "key": meta.get("key", ax.lower()),
                    "label_ja": meta.get("label_ja", ax),
                    "label_en": meta.get("label_en", ax),
                    "score": score,
                    "prior": round(prior_val, 3),
                    "learned_weight": round(raw_w, 4),
                }
            )

        layers_dict = summarize_4_layers(w_eff)
        return {
            "station_id": self.station_id,
            "updates": t_updates,
            "calibrated": calibrated,
            "sample_counts": sample_counts,
            "scores": scores_map,
            "axes": axes_out,
            "layers": list(layers_dict.values()),
            "effective_mir_vector": [round(float(x), 4) for x in w_eff],
        }

    def _score_mir_precision(
        self,
        mir_matrix: np.ndarray,
        w_effective: np.ndarray,
        axis_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stage 2 Precision Scoring: weighted 16-dim acoustic similarity + scene/sweetspot quality."""
        aw = np.maximum(0.1, axis_weights)
        aw_norm = aw / float(np.sum(aw))

        # 1. Weighted closeness across all 16 MIR dims (especially acoustic A1..C4)
        l1_dist = np.sum(np.abs(mir_matrix - w_effective[None, :]) * aw_norm[None, :], axis=1)
        closeness = np.clip(1.0 - l1_dist * 1.45, 0.0, 1.0)

        # 2. Weighted cosine similarity on 12-dim acoustic core (A1..C4)
        sqrt_a = np.sqrt(aw[:12])
        a_cand = mir_matrix[:, :12] * sqrt_a[None, :]
        a_tgt = w_effective[:12] * sqrt_a
        denom = np.maximum(1e-9, np.linalg.norm(a_cand, axis=1) * float(np.linalg.norm(a_tgt)))
        acoustic_cos = np.clip(np.dot(a_cand, a_tgt) / denom, 0.0, 1.0)

        # 3. Stage 1 co-occurrence + Deezer rank + YouTube 300k-50M sweetspot + PES/Emily cult-chic similarity
        d1_cooc = mir_matrix[:, 12]
        d2_rank = mir_matrix[:, 13]
        d3_views = mir_matrix[:, 14]
        d4_cult = mir_matrix[:, 15]

        exploit = (
            0.35 * closeness
            + 0.25 * acoustic_cos
            + 0.16 * d1_cooc
            + 0.10 * d2_rank
            + 0.14 * d3_views
        )
        # Exploration crossover affinity prioritizes co-occurring PES/Emily cult-chic gems
        crossover_explore = 0.45 * d1_cooc + 0.35 * d4_cult + 0.20 * d3_views
        return exploit, crossover_explore

    def rank_candidates(
        self,
        candidates: list[TrackEntity],
        *,
        seen_artists: set[str],
        max_popularity_views: int = 500_000_000,
        hour: int | None = None,
        limit: int = 3,
        recent_artists: list[str] | None = None,
    ) -> list[RankedCandidate]:
        """Stage 2 Precision Scoring + Stage 3 Re-Ranking (5-track artist cooldown + 80/20 exploit/explore)."""
        if not candidates:
            return []

        X = np.vstack(
            [
                self.featurize(
                    track,
                    is_new_artist=(normalize_artist(track.artist) not in seen_artists),
                    hour=hour,
                )
                for track in candidates
            ]
        )
        X_theory = np.vstack(
            [
                self.featurize_theory(
                    track,
                    max_popularity_views=max_popularity_views,
                    hour=hour,
                )
                for track in candidates
            ]
        )
        mir_results = [self.featurize_mir(track, allow_network=False) for track in candidates]
        X_mir = np.vstack([pair[0] for pair in mir_results])
        measured_bpms = [pair[1] for pair in mir_results]

        has_theory_station = bool(self.station_config and self.station_config.target_theory_vector)
        target_prior = self.theory_extractor.get_station_prior_vector(self.station_config)
        target_vec5 = np.array([target_prior[ax] for ax in THEORY_AXIS_KEYS], dtype=np.float64)

        with self._lock:
            self._sync_external_profile_locked()
            n_updates = self._n_updates
            theory_updates = self._theory_n_updates
            w_eff = (
                ALPHA_ANCHOR * self._w_anchor
                + ALPHA_SLOW * self._w_slow
                + ALPHA_SESSION * self._w_session
            ).copy()
            axis_w = self._axis_weights.copy()
            has_calibrated = self._calibrated_from_db

            if n_updates == 0:
                base_explore = self.exploration_rate * np.linalg.norm(X, axis=1)
                tag_overlap = np.array(
                    [len(set(t.tags) & set(self.seed_tags)) * 0.05 for t in candidates],
                    dtype=np.float64,
                )
                base_exploit = tag_overlap
                base_total = base_exploit + base_explore
            else:
                raw_exp = self._mab.predict_expectations(contexts=X)
                exp_list = [raw_exp] if isinstance(raw_exp, dict) else list(raw_exp)
                base_total = np.array([float(item["play"]) for item in exp_list], dtype=np.float64)
                lin_model = self._mab._imp.arm_to_model["play"]
                base_exploit = np.dot(X, lin_model.beta)
                base_explore = np.maximum(0.0, base_total - base_exploit)

            if theory_updates == 0:
                t_norms = np.linalg.norm(X_theory[:, :5], axis=1) * (np.linalg.norm(target_vec5) + 1e-9)
                cosine = np.dot(X_theory[:, :5], target_vec5) / np.maximum(1e-9, t_norms)
                t6_bonus = 0.25 * X_theory[:, 5] + 0.25 * X_theory[:, 6] + 0.10 * X_theory[:, 7]
                theory_exploit = 0.60 * cosine + t6_bonus
                theory_explore = self.exploration_rate * 0.35 * np.linalg.norm(X_theory, axis=1)
                theory_total = theory_exploit + theory_explore
            else:
                raw_t = self._theory_mab.predict_expectations(contexts=X_theory)
                t_list = [raw_t] if isinstance(raw_t, dict) else list(raw_t)
                theory_total = np.array([float(item["play"]) for item in t_list], dtype=np.float64)
                t_model = self._theory_mab._imp.arm_to_model["play"]
                theory_exploit = np.dot(X_theory, t_model.beta)
                theory_explore = np.maximum(0.0, theory_total - theory_exploit)

        mir_exploit, mir_crossover = self._score_mir_precision(X_mir, w_eff, axis_w)
        mir_total = mir_exploit + 0.15 * self.exploration_rate * mir_crossover

        # Blend weights: calibrated/MIR stations lean heavily on real 16-dim MIR + theory;
        # legacy tag-only stations preserve tag LinUCB dominance for backward compatibility.
        if has_theory_station or has_calibrated:
            w_mir = 0.50
            w_theory = 0.35
            w_base = 0.15
        else:
            w_mir = 0.10
            w_theory = 0.10
            w_base = 0.80

        exploit_scores = w_base * base_exploit + w_theory * theory_exploit + w_mir * mir_exploit
        explore_scores = w_base * base_explore + w_theory * theory_explore + w_mir * mir_crossover
        total_scores = w_base * base_total + w_theory * theory_total + w_mir * mir_total

        scored: list[tuple[RankedCandidate, float]] = []
        seed_tag_set = set(self.seed_tags)
        for idx, track in enumerate(candidates):
            exploit = float(exploit_scores[idx])
            explore = float(explore_scores[idx])
            total = float(total_scores[idx])
            mode = "活用" if (n_updates > 0 and exploit >= explore) else "探索"
            matched = [t for t in track.tags if t in seed_tag_set]
            if not matched and track.tags:
                matched = track.tags[:3]
            t_vec = X_theory[idx]
            m_vec = X_mir[idx]
            explanation = self.theory_extractor.build_explanation(
                t_vec,
                station=self.station_config,
                track=track,
            )
            layers = summarize_4_layers(m_vec, centroid=w_eff)
            sweetspot = float(0.65 * m_vec[14] + 0.35 * m_vec[13])
            rc = RankedCandidate(
                track=track,
                score=total,
                exploit_score=exploit,
                explore_bonus=explore,
                mode=mode,
                hidden_level=_hidden_level(track.view_count, max_popularity_views),
                matched_tags=matched,
                theory_vector=t_vec.tolist(),
                theory_explanation=explanation,
                mir_vector=m_vec.tolist(),
                mir_layers=layers,
                measured_bpm=measured_bpms[idx],
                sweetspot_score=sweetspot,
            )
            scored.append((rc, float(mir_crossover[idx])))

        # Stage 3 Re-Ranking:
        # 1. Recent 6-track artist cooldown + strict unique-artist-per-batch spacing (never >2 even in tiny pools)
        recent_set = {normalize_artist(a) for a in (recent_artists or [])[:6] if a}

        exploit_pool = sorted(scored, key=lambda pair: pair[0].score, reverse=True)
        # Exploration crossover pool: co-occurring / cult-chic crossover gems
        explore_pool = sorted(
            scored,
            key=lambda pair: (pair[0].track.cooccurrence_count >= 2, pair[1], pair[0].score),
            reverse=True,
        )

        selected: list[RankedCandidate] = []
        used_track_ids: set[str] = set()
        batch_artist_counts: dict[str, int] = {}

        def _pick_from(
            pool: list[tuple[RankedCandidate, float]],
            *,
            allow_recent_artist: bool,
            max_in_batch: int = 1,
        ) -> RankedCandidate | None:
            for cand, _cross in pool:
                if cand.track.track_id in used_track_ids:
                    continue
                a_norm = normalize_artist(cand.track.artist)
                if batch_artist_counts.get(a_norm, 0) >= max_in_batch:
                    continue
                if not allow_recent_artist and a_norm in recent_set:
                    continue
                used_track_ids.add(cand.track.track_id)
                batch_artist_counts[a_norm] = batch_artist_counts.get(a_norm, 0) + 1
                return cand
            return None

        # 80% Exploit + 20% Explore slot pattern (slot indices 4, 9, ... are exploration crossover slots)
        for slot in range(limit):
            is_explore_slot = (slot % 5) == 4
            primary_pool = explore_pool if is_explore_slot else exploit_pool
            secondary_pool = exploit_pool if is_explore_slot else explore_pool

            chosen = _pick_from(primary_pool, allow_recent_artist=False, max_in_batch=1)
            if chosen is None:
                chosen = _pick_from(secondary_pool, allow_recent_artist=False, max_in_batch=1)
            if chosen is None:
                # Relax recent-history cooldown while still forbidding duplicate artist within the same batch
                chosen = _pick_from(primary_pool, allow_recent_artist=True, max_in_batch=1)
            if chosen is None:
                # Final emergency fallback for tiny single-artist pools (still hard-capped at <=2)
                chosen = _pick_from(primary_pool, allow_recent_artist=True, max_in_batch=2)
            if chosen is None:
                break
            if is_explore_slot:
                chosen.mode = "探索"
            selected.append(chosen)

        return selected[:limit]

    def _apply_exploration_rate_locked(self) -> None:
        """Apply current YAML exploration_rate (alpha) to both loaded MABWiser LinUCB internals."""
        for mab in (self._mab, self._theory_mab):
            mab._imp.alpha = self.exploration_rate
            for model in mab._imp.arm_to_model.values():
                if hasattr(model, "alpha"):
                    model.alpha = self.exploration_rate

    def _save_locked(self) -> None:
        """Atomically persist bandit & 3-layer MIR state via unique NamedTemporaryFile + os.replace."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "station_id": self.station_id,
            "tag_vocab": self.tag_vocab,
            "feature_dim": self.feature_dim,
            "theory_dim": THEORY_CONTEXT_DIM,
            "mir_dim": MIR_DIM,
            "alpha": self.exploration_rate,
            "exploration_rate": self.exploration_rate,
            "n_updates": self._n_updates,
            "theory_n_updates": self._theory_n_updates,
            "mab": self._mab,
            "theory_mab": self._theory_mab,
            "w_slow": self._w_slow.tolist(),
        }
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.model_path.parent,
                prefix=f".{self.model_path.stem}_",
                suffix=".tmp",
                delete=False,
            ) as fh:
                tmp_name = fh.name
                pickle.dump(state, fh)
                fh.flush()
            os.replace(tmp_name, self.model_path)
            tmp_name = None
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def _load_if_exists(self) -> None:
        if not self.model_path.exists():
            # Auto-migrate legacy pickle filename (e.g. bandit_indie_groove_hooks.pkl -> bandit_indie_groove_cult.pkl)
            for old_id, new_id in LEGACY_STATION_ID_MAP.items():
                if new_id == self.station_id:
                    legacy_path = self.model_path.parent / f"bandit_{old_id}.pkl"
                    if legacy_path.exists():
                        try:
                            shutil.copy2(legacy_path, self.model_path)
                            logger.info("Migrated legacy bandit pickle %s -> %s", legacy_path, self.model_path)
                            break
                        except OSError as exc:
                            logger.warning("Failed to migrate legacy pickle %s: %s", legacy_path, exc)
            if not self.model_path.exists():
                return
        with self._lock:
            try:
                with self.model_path.open("rb") as fh:
                    state = pickle.load(fh)
                saved_mab = state.get("mab") if isinstance(state, dict) else None
                saved_dim = state.get("feature_dim") if isinstance(state, dict) else None
                if saved_dim is None:
                    saved_dim = getattr(getattr(saved_mab, "_imp", None), "num_features", None)
                dim_matches = saved_dim is not None and int(saved_dim) == self.feature_dim
                if (
                    isinstance(state, dict)
                    and state.get("tag_vocab") == self.tag_vocab
                    and dim_matches
                    and isinstance(saved_mab, MAB)
                ):
                    self._mab = saved_mab
                    self._n_updates = int(state.get("n_updates", 0))

                    saved_theory_mab = state.get("theory_mab")
                    saved_theory_dim = state.get("theory_dim")
                    if saved_theory_dim is None and saved_theory_mab is not None:
                        saved_theory_dim = getattr(getattr(saved_theory_mab, "_imp", None), "num_features", None)
                    if (
                        isinstance(saved_theory_mab, MAB)
                        and saved_theory_dim is not None
                        and int(saved_theory_dim) == THEORY_CONTEXT_DIM
                    ):
                        self._theory_mab = saved_theory_mab
                        self._theory_n_updates = int(state.get("theory_n_updates", 0))

                    saved_slow = state.get("w_slow")
                    if isinstance(saved_slow, list) and len(saved_slow) == MIR_DIM and not self._calibrated_from_db:
                        self._w_slow = np.asarray(saved_slow, dtype=np.float64)

                    self._apply_exploration_rate_locked()
                else:
                    backup_path = self.model_path.with_suffix(".pkl.bak")
                    shutil.copy2(self.model_path, backup_path)
                    logger.warning(
                        "Bandit tag_vocab/feature_dim changed for station %s; backed up old weights to %s and starting fresh",
                        self.station_id,
                        backup_path,
                    )
            except (OSError, pickle.UnpicklingError, ValueError, TypeError, AttributeError, EOFError) as exc:
                logger.warning("Failed to load bandit weights from %s (%s); starting fresh", self.model_path, exc)
