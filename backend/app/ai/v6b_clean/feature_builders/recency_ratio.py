from __future__ import annotations

from app.ai.v6b_clean.feature_builders.base import (
    BaseFeatureBuilder,
)
from app.ai.v6b_clean.feature_builders.recency import (
    RecencyBuilder,
)


class RecencyRatioBuilder(BaseFeatureBuilder):
    """
    V6B implementation of the V5 recency_ratio feature.

    Returns:
        {
            "recency_ratio": float
        }
    """

    def build(
        self,
        history,
        candidate_number,
    ) -> dict[str, float]:

        self.validate_history(history)
        self.validate_candidate(candidate_number)

        recency = (
            RecencyBuilder()
            .build(
                history=history,
                candidate_number=candidate_number,
            )["recency"]
        )

        return {
            "recency_ratio": round(
                recency / len(history),
                6,
            )
        }