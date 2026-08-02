from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

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
from app.ai.v7.constants import (
    NUMBER_MAX,
    NUMBER_MIN,
)
from app.ai.v7.feature_builders.pair_ratio import (
    PairRatioBuilder,
)
from app.ai.v7.feature_builders.pair_statistics import (
    PairStatistics,
)
from app.models.draw import Draw


class V7FeatureAssembler:
    """
    Predixa AI V7 feature assembler with pair features.

    Feature structure:

        4 global features

        49 candidates × 11 candidate features:
            - rate_10
            - rate_20
            - rate_50
            - rate_100
            - recency
            - recency_ratio
            - short_vs_long
            - frequency_volatility
            - pair_frequency
            - pair_recency
            - pair_ratio

    Total:
        4 + (49 × 11) = 543 features
    """

    VERSION = "V7-FEATURE-ASSEMBLER-PAIR-FEATURES"

    GLOBAL_FEATURES = (
        "history_size",
        "average_sum",
        "average_even_count",
        "average_consecutive_pairs",
    )

    BASELINE_CANDIDATE_FEATURES = (
        "rate_10",
        "rate_20",
        "rate_50",
        "rate_100",
        "recency",
        "recency_ratio",
        "short_vs_long",
        "frequency_volatility",
    )

    PAIR_CANDIDATE_FEATURES = (
        "pair_frequency",
        "pair_recency",
        "pair_ratio",
    )

    CANDIDATE_FEATURES = (
        *BASELINE_CANDIDATE_FEATURES,
        *PAIR_CANDIDATE_FEATURES,
    )

    EXPECTED_FEATURE_COUNT = (
        len(GLOBAL_FEATURES)
        + (
            NUMBER_MAX
            - NUMBER_MIN
            + 1
        )
        * len(CANDIDATE_FEATURES)
    )

    def __init__(self) -> None:
        self.frequency_builder = FrequencyBuilder()
        self.recency_builder = RecencyBuilder()
        self.recency_ratio_builder = RecencyRatioBuilder()
        self.short_vs_long_builder = ShortVsLongBuilder()

        self.frequency_volatility_builder = (
            FrequencyVolatilityBuilder()
        )

        self.global_statistics_builder = (
            GlobalStatisticsBuilder()
        )

    @staticmethod
    def _validate_history(
        history: Sequence[Draw],
    ) -> None:
        if not isinstance(
            history,
            Sequence,
        ):
            raise ValueError(
                "history must be a sequence of Draw objects."
            )

        if len(history) == 0:
            raise ValueError(
                "history cannot be empty."
            )

        for index, draw in enumerate(
            history,
        ):
            if not isinstance(
                draw,
                Draw,
            ):
                raise ValueError(
                    "history contains an invalid item at "
                    f"index {index}. Expected Draw, received "
                    f"{type(draw).__name__}."
                )

    @classmethod
    def _validate_features(
        cls,
        features: dict[str, int | float],
    ) -> None:
        if not isinstance(
            features,
            dict,
        ):
            raise ValueError(
                "features must be a dictionary."
            )

        if (
            len(features)
            != cls.EXPECTED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected V7 feature count. "
                f"Expected {cls.EXPECTED_FEATURE_COUNT}, "
                f"received {len(features)}."
            )

        missing_global_features = [
            feature_name
            for feature_name in cls.GLOBAL_FEATURES
            if feature_name not in features
        ]

        if missing_global_features:
            raise ValueError(
                "Missing V7 global features: "
                f"{missing_global_features}"
            )

        missing_candidate_features: list[str] = []

        for candidate_number in range(
            NUMBER_MIN,
            NUMBER_MAX + 1,
        ):
            for feature_family in (
                cls.CANDIDATE_FEATURES
            ):
                feature_name = (
                    f"{feature_family}_"
                    f"{candidate_number}"
                )

                if feature_name not in features:
                    missing_candidate_features.append(
                        feature_name
                    )

        if missing_candidate_features:
            raise ValueError(
                "Missing V7 candidate features: "
                f"{missing_candidate_features[:20]}"
            )

        invalid_type_features: list[str] = []
        non_finite_features: list[str] = []

        for feature_name, value in (
            features.items()
        ):
            if isinstance(
                value,
                bool,
            ) or not isinstance(
                value,
                (int, float),
            ):
                invalid_type_features.append(
                    feature_name
                )
                continue

            if not isfinite(
                float(value)
            ):
                non_finite_features.append(
                    feature_name
                )

        if invalid_type_features:
            raise ValueError(
                "V7 features contain non-numeric values: "
                f"{invalid_type_features[:20]}"
            )

        if non_finite_features:
            raise ValueError(
                "V7 features contain non-finite values: "
                f"{non_finite_features[:20]}"
            )

    def build(
        self,
        history: Sequence[Draw],
    ) -> dict[str, int | float]:
        """
        Build the complete V7 feature dictionary.
        """

        self._validate_history(
            history
        )

        features: dict[
            str,
            int | float
        ] = {}

        global_features = (
            self.global_statistics_builder.build(
                history=history,
            )
        )

        features.update(
            global_features
        )

        pair_statistics = PairStatistics(
            history=history,
        )

        for candidate_number in range(
            NUMBER_MIN,
            NUMBER_MAX + 1,
        ):
            frequency_features = (
                self.frequency_builder.build(
                    history=history,
                    candidate_number=(
                        candidate_number
                    ),
                )
            )

            recency_features = (
                self.recency_builder.build(
                    history=history,
                    candidate_number=(
                        candidate_number
                    ),
                )
            )

            recency_ratio_features = (
                self.recency_ratio_builder.build(
                    history=history,
                    candidate_number=(
                        candidate_number
                    ),
                )
            )

            short_vs_long_features = (
                self.short_vs_long_builder.build(
                    history=history,
                    candidate_number=(
                        candidate_number
                    ),
                )
            )

            volatility_features = (
                self.frequency_volatility_builder.build(
                    history=history,
                    candidate_number=(
                        candidate_number
                    ),
                )
            )

            for feature_name, value in (
                frequency_features.items()
            ):
                features[
                    f"{feature_name}_"
                    f"{candidate_number}"
                ] = value

            features[
                f"recency_{candidate_number}"
            ] = recency_features[
                "recency"
            ]

            features[
                f"recency_ratio_{candidate_number}"
            ] = recency_ratio_features[
                "recency_ratio"
            ]

            features[
                f"short_vs_long_{candidate_number}"
            ] = short_vs_long_features[
                "short_vs_long"
            ]

            features[
                f"frequency_volatility_"
                f"{candidate_number}"
            ] = volatility_features[
                "frequency_volatility"
            ]

            pair_frequency = (
                pair_statistics.candidate_frequency(
                    candidate_number
                )
            )

            pair_recency = (
                pair_statistics
                .candidate_recency_ratio(
                    candidate_number
                )
            )

            pair_ratio = (
                pair_frequency
                / (
                    pair_recency
                    + PairRatioBuilder.EPSILON
                )
            )

            features[
                f"pair_frequency_{candidate_number}"
            ] = pair_frequency

            features[
                f"pair_recency_{candidate_number}"
            ] = pair_recency

            features[
                f"pair_ratio_{candidate_number}"
            ] = round(
                float(
                    pair_ratio
                ),
                6,
            )

        self._validate_features(
            features
        )

        return features


FeatureAssembler = V7FeatureAssembler