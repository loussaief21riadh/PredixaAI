from __future__ import annotations

from collections.abc import Sequence

from app.ai.v7.feature_builders.pair_statistics import (
    PairStatistics,
)
from app.models.draw import Draw


class PairRecencyBuilder:
    """
    Build normalized pair-recency features for one candidate.

    The builder delegates all computations to PairStatistics.
    """

    FEATURE_NAME = "pair_recency"

    @classmethod
    def build(
        cls,
        history: Sequence[Draw],
        candidate_number: int,
    ) -> dict[str, float]:

        statistics = PairStatistics(
            history=history,
        )

        return {
            cls.FEATURE_NAME: (
                statistics.candidate_recency_ratio(
                    candidate_number
                )
            ),
        }