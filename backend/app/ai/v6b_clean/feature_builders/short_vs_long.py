from __future__ import annotations

from app.ai.v6b_clean.feature_builders.base import (
    BaseFeatureBuilder,
)
from app.ai.v6b_clean.feature_builders.frequency import (
    FrequencyBuilder,
)


class ShortVsLongBuilder(BaseFeatureBuilder):
    """
    V6B implementation of the V5 short_vs_long feature.

    Formula:
        rate_10 - rate_100

    Returns:
        {
            "short_vs_long": float
        }
    """

    def build(
        self,
        history,
        candidate_number,
    ) -> dict[str, float]:

        self.validate_history(history)
        self.validate_candidate(candidate_number)

        rates = FrequencyBuilder().build(
            history=history,
            candidate_number=candidate_number,
        )

        value = round(
            rates["rate_10"] - rates["rate_100"],
            6,
        )

        return {
            "short_vs_long": value,
        }