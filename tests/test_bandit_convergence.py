"""Contextual Bandit convergence tests (evidence_environment: FIXTURE, counts_toward_l5_certification: false)."""

from __future__ import annotations

from pathlib import Path

from src.models import TrackEntity
from src.recommender.bandit import ContextualBanditRecommender


def _make_track(idx: int, tag: str, artist: str, views: int = 25_000, year: int = 2019) -> TrackEntity:
    return TrackEntity(
        track_id=f"trk_{tag}_{idx}",
        canonical_key=f"{artist.lower()}::{tag}_{idx}",
        artist=artist,
        title=f"Hidden Gem {tag} #{idx}",
        tags=[tag],
        release_year=year,
        view_count=views,
        video_id=f"vid_{tag}_{idx:03d}",
    )


def test_bandit_converges_to_preferred_tag_after_20_feedback_pairs(tmp_path: Path) -> None:
    """Feeding 20 GOOD (1.0) on tag A and 20 BAD (0.0) on tag B should converge toward tag A."""
    model_path = tmp_path / "bandit_test.pkl"
    bandit = ContextualBanditRecommender(
        station_id="convergence_station",
        seed_tags=["folktronica", "hardstyle", "indietronica"],
        exploration_rate=0.5,
        model_path=model_path,
        seed=42,
    )

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

    folk_scores = [item.score for item in ranked if "folktronica" in item.track.tags]
    hard_scores = [item.score for item in ranked if "hardstyle" in item.track.tags]
    assert sum(folk_scores) / len(folk_scores) > sum(hard_scores) / len(hard_scores) + 0.35


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
