"""Stage 2 Contextual Bandit Re-ranker using MABWiser LinUCB + 10-dim Deep Music Theory DNA."""

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

from src.models import RankedCandidate, StationConfig, TrackEntity, normalize_artist
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
_WATCH_PLAYLIST_SOURCES = {"watch_playlist", "ytmusic_watch_playlist"}
THEORY_CONTEXT_DIM = 10


def _hash_bucket(text: str, num_buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).hexdigest()
    return int(digest, 16) % num_buckets


def _hidden_level(view_count: int | None, max_popularity_views: int) -> str:
    if view_count is None:
        return "発掘"
    ratio = view_count / max(1, max_popularity_views)
    if view_count <= 20_000 or ratio <= 0.05:
        return "S (超穴場)"
    if view_count <= 100_000 or ratio <= 0.25:
        return "A (隠れ名曲)"
    if view_count <= 300_000 or ratio <= 0.65:
        return "B (準穴場)"
    return "C (注目圏)"


def _sigmoid(x: float) -> float:
    clipped = max(-10.0, min(10.0, float(x)))
    return 1.0 / (1.0 + math.exp(-clipped))


class ContextualBanditRecommender:
    """Per-station Contextual Bandit (MABWiser LinUCB) integrating 10-dim Deep Music Theory DNA."""

    def __init__(
        self,
        station_id: str,
        seed_tags: list[str],
        exploration_rate: float = 1.0,
        model_path: Path | str | None = None,
        seed: int = 42,
        station_config: StationConfig | None = None,
        theory_extractor: TheoryFeatureExtractor | None = None,
    ) -> None:
        self.station_id = station_id
        self.seed_tags = [t.strip().lower() for t in seed_tags if t and t.strip()]
        self.tag_vocab = list(dict.fromkeys(self.seed_tags))
        self.exploration_rate = float(max(0.05, exploration_rate))
        self.model_path = Path(model_path) if model_path else Path(f"data/bandit_{station_id}.pkl")
        self.seed = int(seed)
        self.station_config = station_config
        self.theory_extractor = theory_extractor or TheoryFeatureExtractor()
        self._lock = threading.Lock()

        # Primary tag/metadata LinUCB model (preserves full backward compatibility)
        self._mab = MAB(
            arms=["play"],
            learning_policy=LearningPolicy.LinUCB(alpha=self.exploration_rate),
            seed=self.seed,
        )
        # 10-dimensional Deep Music Theory DNA LinUCB model (GOOD=+1.0, BAD=-1.0)
        self._theory_mab = MAB(
            arms=["play"],
            learning_policy=LearningPolicy.LinUCB(alpha=self.exploration_rate),
            seed=self.seed,
        )
        self._n_updates = 0
        self._theory_n_updates = 0
        self._load_if_exists()

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
            + 1  # is_watch_playlist_source ([C2] tagless Watch Playlist indicator)
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
            and (listen_seconds / duration_seconds) >= 0.8
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
        """Signed reward for 10-dim Theory DNA online learning: GOOD=+1.0, BAD=-1.0."""
        ev = (event_type or "").strip().upper()
        if ev == "GOOD":
            return 1.0
        if ev == "BAD":
            return -1.0
        if ev == "COMPLETE":
            return 0.7
        if ev == "EARLY_SKIP":
            return -0.8
        if ev == "MID_SKIP":
            return 0.1

        if listen_seconds is not None and listen_seconds < 15.0:
            return -0.8
        if (
            listen_seconds is not None
            and duration_seconds is not None
            and duration_seconds > 0
            and (listen_seconds / duration_seconds) >= 0.8
        ):
            return 0.7
        if ev == "ENDED":
            return 0.7
        return 0.1

    def featurize(
        self,
        track: TrackEntity,
        *,
        is_new_artist: bool,
        hour: int | None = None,
    ) -> np.ndarray:
        """Build `[tag_one_hot, release_decade, log_view_bucket, time_of_day_bucket, is_new_artist, is_watch_playlist, bias]`."""
        vec = np.zeros(self.feature_dim, dtype=np.float64)
        offset = 0

        # 1. tag_one_hot (exact station seed tags + hashed OOV buckets)
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

        # 2. release_decade one-hot
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

        # 3. log_view_bucket one-hot
        views = track.view_count
        if views is None or views < 0:
            view_idx = 4
        else:
            log_v = math.log10(views + 1.0)
            if log_v < 4.0:        # < 10k
                view_idx = 0
            elif log_v < 4.7:      # 10k - ~50k
                view_idx = 1
            elif log_v < 5.3:      # 50k - ~200k
                view_idx = 2
            elif log_v < 5.7:      # 200k - ~500k
                view_idx = 3
            else:
                view_idx = 4
        vec[offset + view_idx] = 1.0
        offset += _VIEW_BUCKETS

        # 4. time_of_day_bucket one-hot
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

        # 5. is_new_artist, is_watch_playlist_source, intercept bias
        vec[offset] = 1.0 if is_new_artist else 0.0
        vec[offset + 1] = 1.0 if track.source in _WATCH_PLAYLIST_SOURCES else 0.0
        vec[offset + 2] = 1.0
        return vec

    def featurize_theory(
        self,
        track: TrackEntity,
        *,
        max_popularity_views: int = 500_000,
        hour: int | None = None,
        audio_signal: AudioTheorySignal | None = None,
    ) -> np.ndarray:
        """Build the 10-dimensional Deep Music Theory DNA context vector."""
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
        """Thread-safely update both tag LinUCB and 10-dim Theory DNA LinUCB, with atomic save & rollback."""
        reward = self.compute_reward(event_type, listen_seconds, duration_seconds)
        theory_reward = self.compute_theory_reward(event_type, listen_seconds, duration_seconds)
        context = self.featurize(track, is_new_artist=is_new_artist, hour=hour).reshape(1, -1)
        theory_context = self.featurize_theory(track, hour=hour, audio_signal=audio_signal).reshape(1, -1)

        with self._lock:
            prev_n_updates = self._n_updates
            prev_theory_n_updates = self._theory_n_updates
            prev_mab_snapshot = pickle.dumps(self._mab)
            prev_theory_snapshot = pickle.dumps(self._theory_mab)
            try:
                if self._n_updates == 0:
                    self._mab.fit(decisions=["play"], rewards=[reward], contexts=context)
                else:
                    self._mab.partial_fit(decisions=["play"], rewards=[reward], contexts=context)

                if self._theory_n_updates == 0:
                    self._theory_mab.fit(decisions=["play"], rewards=[theory_reward], contexts=theory_context)
                else:
                    self._theory_mab.partial_fit(decisions=["play"], rewards=[theory_reward], contexts=theory_context)

                self._n_updates += 1
                self._theory_n_updates += 1
                self._save_locked()
            except Exception:
                self._n_updates = prev_n_updates
                self._theory_n_updates = prev_theory_n_updates
                self._mab = pickle.loads(prev_mab_snapshot)
                self._theory_mab = pickle.loads(prev_theory_snapshot)
                raise
        return reward

    def get_learned_theory_profile(self) -> dict[str, Any]:
        """Return the current 5-axis (T1..T5) learned user preference profile in [0.0, 1.0]."""
        prior = self.theory_extractor.get_station_prior_vector(self.station_config)
        with self._lock:
            t_updates = self._theory_n_updates
            beta: np.ndarray | None = None
            if t_updates > 0:
                model = self._theory_mab._imp.arm_to_model.get("play")
                if model is not None and getattr(model, "beta", None) is not None and len(model.beta) >= 5:
                    beta = np.array(model.beta[:5], dtype=np.float64)

        # Blend station prior with online-learned LinUCB beta weights smoothly as feedback accumulates
        learn_blend = min(0.85, t_updates / (t_updates + 6.0)) if t_updates > 0 else 0.0
        axes_out: list[dict[str, Any]] = []
        scores_map: dict[str, float] = {}

        for idx, ax in enumerate(THEORY_AXIS_KEYS):
            meta = self.theory_extractor.axes_meta.get(ax, {})
            prior_val = float(prior[ax])
            if beta is not None:
                raw_w = float(beta[idx])
                # Map signed ridge weight (~[-1.5, +1.5]) to [0.0, 1.0] preference score
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

        return {
            "station_id": self.station_id,
            "updates": t_updates,
            "scores": scores_map,
            "axes": axes_out,
        }

    def rank_candidates(
        self,
        candidates: list[TrackEntity],
        *,
        seen_artists: set[str],
        max_popularity_views: int = 500_000,
        hour: int | None = None,
        limit: int = 3,
    ) -> list[RankedCandidate]:
        """Thread-safely score and rank unheard candidates via MABWiser LinUCB + 10-dim Theory DNA."""
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

        has_theory_station = bool(self.station_config and self.station_config.target_theory_vector)
        target_prior = self.theory_extractor.get_station_prior_vector(self.station_config)
        target_vec5 = np.array([target_prior[ax] for ax in THEORY_AXIS_KEYS], dtype=np.float64)

        with self._lock:
            n_updates = self._n_updates
            theory_updates = self._theory_n_updates

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
                # Cold-start theory alignment: cosine proximity to station target T1..T5 + T6 (affinity * gem)
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

        # Blend weights: full Theory DNA stations weigh 10-dim theory bandit heavily (55%);
        # generic tag-only stations preserve primary tag LinUCB dominance while still computing theory explanations.
        w_theory = 0.55 if has_theory_station else 0.12
        w_base = 1.0 - w_theory

        exploit_scores = w_base * base_exploit + w_theory * theory_exploit
        explore_scores = w_base * base_explore + w_theory * theory_explore
        total_scores = w_base * base_total + w_theory * theory_total

        # Build ranked entries using ONLY genuine track tags ([C2] no fabricated matches)
        scored: list[RankedCandidate] = []
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
            explanation = self.theory_extractor.build_explanation(
                t_vec,
                station=self.station_config,
                track=track,
            )
            scored.append(
                RankedCandidate(
                    track=track,
                    score=total,
                    exploit_score=exploit,
                    explore_bonus=explore,
                    mode=mode,
                    hidden_level=_hidden_level(track.view_count, max_popularity_views),
                    matched_tags=matched,
                    theory_vector=t_vec.tolist(),
                    theory_explanation=explanation,
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)

        # Apply light artist diversity across the top prefetch batch so back-to-back tracks vary
        selected: list[RankedCandidate] = []
        overflow: list[RankedCandidate] = []
        batch_artists: set[str] = set()
        for item in scored:
            a_norm = normalize_artist(item.track.artist)
            if a_norm not in batch_artists:
                selected.append(item)
                batch_artists.add(a_norm)
            else:
                overflow.append(item)
            if len(selected) >= limit:
                break

        if len(selected) < limit:
            selected.extend(overflow[: limit - len(selected)])
        return selected[:limit]

    def _apply_exploration_rate_locked(self) -> None:
        """Apply current YAML exploration_rate (alpha) to both loaded MABWiser LinUCB internals."""
        for mab in (self._mab, self._theory_mab):
            mab._imp.alpha = self.exploration_rate
            for model in mab._imp.arm_to_model.values():
                if hasattr(model, "alpha"):
                    model.alpha = self.exploration_rate

    def _save_locked(self) -> None:
        """Atomically persist bandit state via unique NamedTemporaryFile + os.replace."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "station_id": self.station_id,
            "tag_vocab": self.tag_vocab,
            "feature_dim": self.feature_dim,
            "theory_dim": THEORY_CONTEXT_DIM,
            "alpha": self.exploration_rate,
            "exploration_rate": self.exploration_rate,
            "n_updates": self._n_updates,
            "theory_n_updates": self._theory_n_updates,
            "mab": self._mab,
            "theory_mab": self._theory_mab,
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

                    # [M8] Apply latest YAML exploration_rate even if persisted alpha differed
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
