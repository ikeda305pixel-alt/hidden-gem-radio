"""Stage 2 Contextual Bandit re-ranking powered by MABWiser + 10-dim Deep Music Theory DNA."""

from src.recommender.bandit import ContextualBanditRecommender
from src.recommender.theory_features import (
    THEORY_AXIS_KEYS,
    THEORY_FEATURE_NAMES,
    AudioTheorySignal,
    TheoryFeatureExtractor,
    merge_audio_theory_signal,
)

__all__ = [
    "THEORY_AXIS_KEYS",
    "THEORY_FEATURE_NAMES",
    "AudioTheorySignal",
    "ContextualBanditRecommender",
    "TheoryFeatureExtractor",
    "merge_audio_theory_signal",
]
