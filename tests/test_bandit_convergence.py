"""Contextual Bandit convergence tests (evidence_environment: FIXTURE, counts_toward_l5_certification: false)."""

from __future__ import annotations

import threading
from pathlib import Path

from src.models import TrackEntity
from src.recommender.bandit import ContextualBanditRecommender


def _make_track(
    idx: int,
    tag: str,
    artist: str,
    views: int = 25_000,
    year: int = 2019,
    source: str = "listenbrainz",
) -> TrackEntity:
    return TrackEntity(
        track_id=f"trk_{tag}_{idx}",
        canonical_key=f"{artist.lower()}::{tag}_{idx}",
        artist=artist,
        title=f"Hidden Gem {tag} #{idx}",
        tags=[tag] if tag else [],
        release_year=year,
        view_count=views,
        video_id=f"vid_{tag}_{idx:03d}",
        source=source,
    )


def test_bandit_converges_to_preferred_tag_after_20_feedback_pairs(tmp_path: Path) -> None:
    """Initial exploration bonus must be high, decay after 20 updates, and exploit preferred tag."""
    model_path = tmp_path / "bandit_test.pkl"
    bandit = ContextualBanditRecommender(
        station_id="convergence_station",
        seed_tags=["folktronica", "hardstyle", "indietronica"],
        exploration_rate=0.5,
        model_path=model_path,
        seed=42,
    )

    probe_folk = _make_track(900, tag="folktronica", artist="ProbeFolk")
    probe_hard = _make_track(901, tag="hardstyle", artist="ProbeHard")

    # Before learning: pure exploration mode with high exploration bonus
    initial_ranked = bandit.rank_candidates(
        [probe_folk, probe_hard],
        seen_artists=set(),
        max_popularity_views=500_000,
        hour=21,
        limit=2,
    )
    assert all(item.mode == "探索" for item in initial_ranked)
    initial_folk_explore = next(i.explore_bonus for i in initial_ranked if i.track.track_id == probe_folk.track_id)

    for i in range(20):
        good_track = _make_track(i, tag="folktronica", artist=f"FolkArtist{i}")
        bad_track = _make_track(i, tag="hardstyle", artist=f"HardArtist{i}")
        bandit.update(good_track, event_type="GOOD", is_new_artist=True, hour=21)
        bandit.update(bad_track, event_type="BAD", is_new_artist=True, hour=21)

    assert model_path.exists(), "Bandit model weights must be persisted locally on update"

    # Reload from persisted pickle to verify persistence
    reloaded = ContextualBanditRecommender(
        station_id="convergence_station",
        seed_tags=["folktronica", "hardstyle", "indietronica"],
        exploration_rate=0.5,
        model_path=model_path,
        seed=42,
    )

    candidates = [
        _make_track(100 + i, tag="folktronica", artist=f"NewFolk{i}") for i in range(5)
    ] + [
        _make_track(200 + i, tag="hardstyle", artist=f"NewHard{i}") for i in range(5)
    ]

    ranked = reloaded.rank_candidates(
        candidates,
        seen_artists=set(),
        max_popularity_views=500_000,
        hour=21,
        limit=10,
    )

    top_5_tags = [item.track.tags[0] for item in ranked[:5]]
    assert top_5_tags.count("folktronica") >= 4, f"Expected folktronica to dominate top 5, got {top_5_tags}"

    folk_items = [item for item in ranked if "folktronica" in item.track.tags]
    hard_items = [item for item in ranked if "hardstyle" in item.track.tags]
    assert sum(i.score for i in folk_items) / len(folk_items) > (
        sum(i.score for i in hard_items) / len(hard_items) + 0.35
    )

    # Verify exploration bonus decayed and preferred tag shifted to exploitation mode ("活用")
    late_folk_explore = folk_items[0].explore_bonus
    assert late_folk_explore < initial_folk_explore * 0.5, (
        f"Expected exploration bonus to decay: initial={initial_folk_explore:.3f}, late={late_folk_explore:.3f}"
    )
    assert folk_items[0].mode == "活用"


def test_bandit_concurrent_8_thread_updates_have_zero_errors(tmp_path: Path) -> None:
    """[C1] 8 threads updating the same recommender concurrently must raise zero FileNotFoundError."""
    model_path = tmp_path / "bandit_concurrent.pkl"
    bandit = ContextualBanditRecommender(
        station_id="concurrent_station",
        seed_tags=["folktronica", "indietronica"],
        exploration_rate=1.0,
        model_path=model_path,
        seed=42,
    )
    errors: list[Exception] = []

    def worker(worker_id: int) -> None:
        for i in range(25):
            try:
                tag = "folktronica" if i % 2 == 0 else "indietronica"
                ev = "GOOD" if i % 2 == 0 else "BAD"
                bandit.update(
                    _make_track(worker_id * 100 + i, tag=tag, artist=f"A_{worker_id}_{i}"),
                    event_type=ev,
                    is_new_artist=True,
                    hour=12,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Unexpected concurrent update errors: {errors[:3]}"
    assert bandit.n_updates == 200

    reloaded = ContextualBanditRecommender(
        station_id="concurrent_station",
        seed_tags=["folktronica", "indietronica"],
        exploration_rate=1.0,
        model_path=model_path,
        seed=42,
    )
    assert reloaded.n_updates == 200


def test_bandit_applies_updated_alpha_and_backs_up_on_vocab_change(tmp_path: Path) -> None:
    """[M8] YAML exploration_rate changes must apply on reload; vocab change backs up to .pkl.bak."""
    model_path = tmp_path / "bandit_m8.pkl"
    bandit = ContextualBanditRecommender(
        station_id="m8_station",
        seed_tags=["folktronica", "indietronica"],
        exploration_rate=1.0,
        model_path=model_path,
    )
    bandit.update(_make_track(1, tag="folktronica", artist="Artist1"), event_type="GOOD", hour=12)

    # Reload with higher exploration_rate (3.0) -> alpha must update in live LinUCB model
    reloaded_alpha = ContextualBanditRecommender(
        station_id="m8_station",
        seed_tags=["folktronica", "indietronica"],
        exploration_rate=3.0,
        model_path=model_path,
    )
    assert reloaded_alpha.n_updates == 1
    assert reloaded_alpha._mab.learning_policy.alpha == 3.0
    assert reloaded_alpha._mab._imp.arm_to_model["play"].alpha == 3.0

    # Reload with changed tag vocabulary -> old pkl backed up to .pkl.bak and fresh model started
    reloaded_vocab = ContextualBanditRecommender(
        station_id="m8_station",
        seed_tags=["folktronica", "indietronica", "dream pop"],
        exploration_rate=1.0,
        model_path=model_path,
    )
    assert reloaded_vocab.n_updates == 0
    assert model_path.with_suffix(".pkl.bak").exists()


def test_reward_mapping_matches_specification() -> None:
    """Verify exact reward mapping: GOOD=1.0, COMPLETE=0.7, MID_SKIP=0.3, BAD/EARLY_SKIP=0.0."""
    assert ContextualBanditRecommender.compute_reward("GOOD") == 1.0
    assert ContextualBanditRecommender.compute_reward("COMPLETE") == 0.7
    assert ContextualBanditRecommender.compute_reward("MID_SKIP") == 0.3
    assert ContextualBanditRecommender.compute_reward("BAD") == 0.0
    assert ContextualBanditRecommender.compute_reward("EARLY_SKIP") == 0.0
    # Derived from listen ratio / seconds when event is SKIP or ENDED
    assert ContextualBanditRecommender.compute_reward("ENDED", listen_seconds=180, duration_seconds=200) == 0.7
    assert ContextualBanditRecommender.compute_reward("SKIP", listen_seconds=10, duration_seconds=200) == 0.0
    assert ContextualBanditRecommender.compute_reward("SKIP", listen_seconds=60, duration_seconds=200) == 0.3
