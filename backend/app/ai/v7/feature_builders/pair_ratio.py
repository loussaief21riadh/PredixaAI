from __future__ import annotations

from collections.abc import Sequence

from app.ai.v7.feature_builders.pair_statistics import (
    PairStatistics,
)
from app.models.draw import Draw


class PairRatioBuilder:
    """
    Build a pair ratio feature.

    Combines pair frequency and pair recency into
    a single normalized score.
    """

    FEATURE_NAME = "pair_ratio"

    EPSILON = 1e-6

    @classmethod
    def build(
        cls,
        history: Sequence[Draw],
        candidate_number: int,
    ) -> dict[str, float]:

        statistics = PairStatistics(
            history=history,
        )

        frequency = (
            statistics.candidate_frequency(
                candidate_number
            )
        )

        recency_ratio = (
            statistics.candidate_recency_ratio(
                candidate_number
            )
        )

        score = frequency / (
            recency_ratio + cls.EPSILON
        )

        return {
            cls.FEATURE_NAME: round(
                float(score),
                6,
            ),
        }