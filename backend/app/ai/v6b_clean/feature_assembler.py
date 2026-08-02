from __future__ import annotations

from app.ai.v6b_clean.feature_builders.frequency import (
    FrequencyBuilder,
)
from app.ai.v6b_clean.feature_builders.frequency_volatility import (
    FrequencyVolatilityBuilder,
)
from app.ai.v6b_clean.feature_builders.global_statistics import (
    GlobalStatisticsBuilder,
)
from app.ai.v6b_clean.feature_builders.recency import (
    RecencyBuilder,
)
from app.ai.v6b_clean.feature_builders.recency_ratio import (
    RecencyRatioBuilder,
)
from app.ai.v6b_clean.feature_builders.short_vs_long import (
    ShortVsLongBuilder,
)


class FeatureAssembler:
    """
    Assemble all V6B feature builders into one
    V5-compatible feature dictionary.
    """

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    def __init__(self) -> None:

        self.frequency = FrequencyBuilder()
        self.recency = RecencyBuilder()
        self.recency_ratio = RecencyRatioBuilder()
        self.short_vs_long = ShortVsLongBuilder()
        self.frequency_volatility = (
            FrequencyVolatilityBuilder()
        )
        self.global_statistics = (
            GlobalStatisticsBuilder()
        )

    def build(
        self,
        history,
    ) -> dict:

        features = {}

        # -----------------------------
        # Global features
        # -----------------------------

        features.update(
            self.global_statistics.build(
                history=history,
            )
        )

        # -----------------------------
        # Candidate features
        # -----------------------------

        for number in range(
            self.NUMBER_MIN,
            self.NUMBER_MAX + 1,
        ):

            frequency = self.frequency.build(
                history=history,
                candidate_number=number,
            )

            recency = self.recency.build(
                history=history,
                candidate_number=number,
            )

            recency_ratio = (
                self.recency_ratio.build(
                    history=history,
                    candidate_number=number,
                )
            )

            short_vs_long = (
                self.short_vs_long.build(
                    history=history,
                    candidate_number=number,
                )
            )

            volatility = (
                self.frequency_volatility.build(
                    history=history,
                    candidate_number=number,
                )
            )

            for name, value in frequency.items():
                features[
                    f"{name}_{number}"
                ] = value

            features[
                f"recency_{number}"
            ] = recency["recency"]

            features[
                f"recency_ratio_{number}"
            ] = recency_ratio[
                "recency_ratio"
            ]

            features[
                f"short_vs_long_{number}"
            ] = short_vs_long[
                "short_vs_long"
            ]

            features[
                f"frequency_volatility_{number}"
            ] = volatility[
                "frequency_volatility"
            ]

        return features