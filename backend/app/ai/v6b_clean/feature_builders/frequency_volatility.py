from __future__ import annotations

from statistics import pstdev

from app.ai.v6b_clean.feature_builders.base import (
    BaseFeatureBuilder,
)
from app.ai.v6b_clean.feature_builders.frequency import (
    FrequencyBuilder,
)


class FrequencyVolatilityBuilder(BaseFeatureBuilder):
    """
    V6B implementation of the V5 frequency_volatility feature.

    Formula:
        pstdev(rate_10, rate_20, rate_50, rate_100)

    Returns:
        {
            "frequency_volatility": float
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
            pstdev(
                [
                    rates["rate_10"],
                    rates["rate_20"],
                    rates["rate_50"],
                    rates["rate_100"],
                ]
            ),
            6,
        )

        return {
            "frequency_volatility": value,
        }