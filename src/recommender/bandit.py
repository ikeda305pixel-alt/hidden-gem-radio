"""Stage 2 Contextual Bandit Re-ranker using MABWiser LinUCB."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from mabwiser.mab import LearningPolicy, MAB

from src.models import RankedCandidate, TrackEntity, normalize_artist

logger = logging.getLogger(__name__)

_OOV_TAG_BUCKETS = 8
_DECADE_BUCKETS = 6  # <1990, 1990s, 2000s, 2010s, 2020s+, unknown
_VIEW_BUCKETS = 5    # <10k, 10k-50k, 50k-200k, 200k-500k, >=500k/unknown
_TIME_BUCKETS = 4    # morning, afternoon, evening, late_night


def _hash_bucket(text: str, num_buckets: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


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


class ContextualBanditRecommender:
    """Per-station Contextual Bandit (MABWiser LinUCB) with local pickle persistence."""

    def __init__(
        self,
        station_id: str,
        seed_tags: list[str],
        exploration_rate: float = 1.0,
        model_path: Path | str | None = None,
        seed: int = 42,
    ) -> None:
        self.station_id = station_id
        self.seed_tags = [t.strip().lower() for t in seed_tags if t and t.strip()]
        self.tag_vocab = list(dict.fromkeys(self.seed_tags))
        self.exploration_rate = float(max(0.05, exploration_rate))
        self.model_path = Path(model_path) if model_path else Path(f"data/bandit_{station_id}.pkl")
        self.seed = int(seed)

        self._mab = MAB(
            arms=["play"],
            learning_policy=LearningPolicy.LinUCB(alpha=self.exploration_rate),
            seed=self.seed,
        )
        self._n_updates = 0
        self._load_if_exists()

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def feature_dim(self) -> int:
        return (
            len(self.tag_vocab)
            + _OOV_TAG_BUCKETS
            + _DECADE_BUCKETS
            + _VIEW_BUCKETS
            + _TIME_BUCKETS
            + 1  # is_new_artist
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

    def featurize(
        self,
        track: TrackEntity,
        *,
        is_new_artist: bool,
        hour: int | None = None,
    ) -> np.ndarray:
        """Build `[tag_one_hot, release_decade, log_view_bucket, time_of_day_bucket, is_new_artist, bias]`."""
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

        # 5. is_new_artist binary indicator + intercept
        vec[offset] = 1.0 if is_new_artist else 0.0
        vec[offset + 1] = 1.0
        return vec

    def update(
        self,
        track: TrackEntity,
        *,
        event_type: str,
        listen_seconds: float | None = None,
        duration_seconds: float | None = None,
        is_new_artist: bool = True,
        hour: int | None = None,
    ) -> float:
        """Immediately update LinUCB model on feedback and persist weights locally."""
        reward = self.compute_reward(event_type, listen_seconds, duration_seconds)
        context = self.featurize(track, is_new_artist=is_new_artist, hour=hour).reshape(1, -1)

        if self._n_updates == 0:
            self._mab.fit(decisions=["play"], rewards=[reward], contexts=context)
        else:
            self._mab.partial_fit(decisions=["play"], rewards=[reward], contexts=context)

        self._n_updates += 1
        self._save()
        return reward

    def rank_candidates(
        self,
        candidates: list[TrackEntity],
        *,
        seen_artists: set[str],
        max_popularity_views: int = 500_000,
        hour: int | None = None,
        limit: int = 3,
    ) -> list[RankedCandidate]:
        """Score and rank unheard candidate tracks via MABWiser LinUCB with artist diversity."""
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

        if self._n_updates == 0:
            # Cold-start: pure LinUCB exploration norm + subtle preference for seed-tag match
            explore_scores = self.exploration_rate * np.linalg.norm(X, axis=1)
            tag_overlap = np.array(
                [len(set(t.tags) & set(self.seed_tags)) * 0.05 for t in candidates],
                dtype=np.float64,
            )
            exploit_scores = tag_overlap
            total_scores = exploit_scores + explore_scores
        else:
            raw_exp = self._mab.predict_expectations(contexts=X)
            exp_list = [raw_exp] if isinstance(raw_exp, dict) else list(raw_exp)
            total_scores = np.array([float(item["play"]) for item in exp_list], dtype=np.float64)

            lin_model = self._mab._imp.arm_to_model["play"]
            exploit_scores = np.dot(X, lin_model.beta)
            explore_scores = np.maximum(0.0, total_scores - exploit_scores)

        # Build ranked entries
        scored: list[RankedCandidate] = []
        seed_tag_set = set(self.seed_tags)
        for idx, track in enumerate(candidates):
            exploit = float(exploit_scores[idx])
            explore = float(explore_scores[idx])
            total = float(total_scores[idx])
            mode = "活用" if (self._n_updates > 0 and exploit >= explore) else "探索"
            matched = [t for t in track.tags if t in seed_tag_set]
            if not matched:
                matched = track.tags[:3]
            scored.append(
                RankedCandidate(
                    track=track,
                    score=total,
                    exploit_score=exploit,
                    explore_bonus=explore,
                    mode=mode,
                    hidden_level=_hidden_level(track.view_count, max_popularity_views),
                    matched_tags=matched,
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

    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.model_path.with_suffix(".pkl.tmp")
        state: dict[str, Any] = {
            "station_id": self.station_id,
            "tag_vocab": self.tag_vocab,
            "exploration_rate": self.exploration_rate,
            "n_updates": self._n_updates,
            "mab": self._mab,
        }
        with tmp_path.open("wb") as fh:
            pickle.dump(state, fh)
        os.replace(tmp_path, self.model_path)

    def _load_if_exists(self) -> None:
        if not self.model_path.exists():
            return
        try:
            with self.model_path.open("rb") as fh:
                state = pickle.load(fh)
            if (
                isinstance(state, dict)
                and state.get("tag_vocab") == self.tag_vocab
                and isinstance(state.get("mab"), MAB)
            ):
                self._mab = state["mab"]
                self._n_updates = int(state.get("n_updates", 0))
            else:
                logger.info(
                    "Tag vocabulary changed for station %s; starting fresh bandit model",
                    self.station_id,
                )
        except Exception as exc:
            logger.warning("Failed to load bandit weights from %s (%s); starting fresh", self.model_path, exc)
