from __future__ import annotations

import pandas as pd

from app.ai.v7.feature_assembler import (
    V7FeatureAssembler,
)
from app.models.draw import Draw


class V7RankingDataset:
    """
    Predixa AI V7 Ranking Dataset.

    Converts each historical target draw into 49 candidate rows.

    Temporal protocol:
        - target draw = T
        - feature history ends at T-2
        - T-1 excluded from feature computation
        - T used only for labels
        - strict chronological processing

    Each candidate row contains:
        - candidate_number
        - 4 global historical features
        - 8 candidate-specific features
        - target = 1 if candidate appears in T, else 0
    """

    VERSION = "PREDIXA-V7-RANKING-DATASET"

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    WINNING_NUMBERS_PER_DRAW = 5

    LAG_DRAWS = 1

    EXPECTED_FULL_FEATURE_COUNT = (
        V7FeatureAssembler.EXPECTED_FEATURE_COUNT
    )

    GLOBAL_FEATURES = (
        "history_size",
        "average_sum",
        "average_even_count",
        "average_consecutive_pairs",
    )

    CANDIDATE_FEATURES = (
        "rate_10",
        "rate_20",
        "rate_50",
        "rate_100",
        "recency",
        "recency_ratio",
        "short_vs_long",
        "frequency_volatility",
        "pair_frequency",
        "pair_recency",
        "pair_ratio",
    )

    MODEL_FEATURES = (
        *GLOBAL_FEATURES,
        *CANDIDATE_FEATURES,
    )

    def __init__(self) -> None:
        self.feature_assembler = V7FeatureAssembler()

    @staticmethod
    def _main_numbers(
        draw: Draw,
    ) -> list[int]:
        """
        Return the five main numbers of one draw.
        """

        return [
            int(draw.n1),
            int(draw.n2),
            int(draw.n3),
            int(draw.n4),
            int(draw.n5),
        ]

    @classmethod
    def _validate_target_numbers(
        cls,
        target_numbers: set[int],
    ) -> None:
        """
        Validate the target draw numbers.
        """

        if (
            len(target_numbers)
            != cls.WINNING_NUMBERS_PER_DRAW
        ):
            raise ValueError(
                "Each target draw must contain exactly "
                f"{cls.WINNING_NUMBERS_PER_DRAW} unique numbers."
            )

        invalid_numbers = sorted(
            number
            for number in target_numbers
            if not (
                cls.NUMBER_MIN
                <= number
                <= cls.NUMBER_MAX
            )
        )

        if invalid_numbers:
            raise ValueError(
                "Target draw contains numbers outside "
                f"{cls.NUMBER_MIN}-{cls.NUMBER_MAX}: "
                f"{invalid_numbers}"
            )

    @classmethod
    def _validate_full_features(
        cls,
        features: dict[str, int | float],
    ) -> None:
        """
        Validate the complete 396-feature dictionary.
        """

        if not isinstance(
            features,
            dict,
        ):
            raise ValueError(
                "FeatureAssembler must return a dictionary."
            )

        if (
            len(features)
            != cls.EXPECTED_FULL_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected V7 full feature count. "
                f"Expected "
                f"{cls.EXPECTED_FULL_FEATURE_COUNT}, "
                f"received {len(features)}."
            )

        missing_global = [
            feature_name
            for feature_name in cls.GLOBAL_FEATURES
            if feature_name not in features
        ]

        if missing_global:
            raise ValueError(
                "Missing global features: "
                f"{missing_global}"
            )

        for candidate_number in range(
            cls.NUMBER_MIN,
            cls.NUMBER_MAX + 1,
        ):
            for family in cls.CANDIDATE_FEATURES:
                full_feature_name = (
                    f"{family}_{candidate_number}"
                )

                if full_feature_name not in features:
                    raise ValueError(
                        "Missing candidate feature: "
                        f"{full_feature_name}"
                    )

    @classmethod
    def _candidate_row(
        cls,
        features: dict[str, int | float],
        candidate_number: int,
        target_numbers: set[int],
    ) -> dict[str, int | float]:
        """
        Build one candidate-level row.
        """

        if not (
            cls.NUMBER_MIN
            <= candidate_number
            <= cls.NUMBER_MAX
        ):
            raise ValueError(
                "Candidate number must be between "
                f"{cls.NUMBER_MIN} and "
                f"{cls.NUMBER_MAX}."
            )

        row: dict[str, int | float] = {
            "candidate_number": (
                candidate_number
            ),
        }

        for feature_name in cls.GLOBAL_FEATURES:
            row[feature_name] = features[
                feature_name
            ]

        for family in cls.CANDIDATE_FEATURES:
            full_feature_name = (
                f"{family}_{candidate_number}"
            )

            row[family] = features[
                full_feature_name
            ]

        row["target"] = int(
            candidate_number
            in target_numbers
        )

        return row

    def build_from_draws(
        self,
        draws: list[Draw],
        window_size: int = 100,
        max_training_targets: int = 1500,
    ) -> tuple[
        pd.DataFrame,
        list[dict[str, object]],
    ]:
        """
        Build the candidate-level ranking dataset.

        For every target draw T:

            feature history:
                exactly window_size draws
                ending at T-2

            T-1:
                excluded from features

            T:
                used only for candidate labels

        Parameters
        ----------
        draws
            Chronologically ordered draw sequence.

        window_size
            Number of historical draws used for features.

        max_training_targets
            Maximum number of target dates included.
            Use 0 to disable the limit.
        """

        if not isinstance(
            draws,
            list,
        ):
            raise ValueError(
                "draws must be provided as a list."
            )

        if window_size < 100:
            raise ValueError(
                "V7 requires "
                "window_size >= 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        minimum_required = (
            window_size
            + self.LAG_DRAWS
            + 1
        )

        if len(draws) < minimum_required:
            raise ValueError(
                "Not enough draws to build the "
                "V6B-CLEAN ranking dataset. "
                f"Required at least {minimum_required}, "
                f"received {len(draws)}."
            )

        first_target_index = (
            window_size
            + self.LAG_DRAWS
        )

        if max_training_targets > 0:
            first_target_index = max(
                first_target_index,
                len(draws)
                - max_training_targets,
            )

        rows: list[
            dict[str, int | float | str]
        ] = []

        target_metadata: list[
            dict[str, object]
        ] = []

        for target_index in range(
            first_target_index,
            len(draws),
        ):
            feature_end_index = (
                target_index
                - self.LAG_DRAWS
            )

            feature_start_index = (
                feature_end_index
                - window_size
            )

            history = draws[
                feature_start_index:
                feature_end_index
            ]

            if len(history) != window_size:
                raise ValueError(
                    "Invalid V7 feature-history size. "
                    f"Expected {window_size}, "
                    f"received {len(history)}."
                )

            target_draw = draws[
                target_index
            ]

            excluded_previous_draw = draws[
                target_index - 1
            ]

            if (
                history[-1].draw_date
                >= excluded_previous_draw.draw_date
            ):
                raise ValueError(
                    "Temporal leakage detected: "
                    "feature history does not end before T-1."
                )

            if (
                excluded_previous_draw.draw_date
                >= target_draw.draw_date
            ):
                raise ValueError(
                    "Draw chronology is invalid around "
                    "the target date."
                )

            features = (
                self.feature_assembler.build(
                    history=history,
                )
            )

            self._validate_full_features(
                features
            )

            target_numbers = set(
                self._main_numbers(
                    target_draw
                )
            )

            self._validate_target_numbers(
                target_numbers
            )

            target_rows: list[
                dict[str, int | float | str]
            ] = []

            positive_count = 0

            for candidate_number in range(
                self.NUMBER_MIN,
                self.NUMBER_MAX + 1,
            ):
                row = self._candidate_row(
                    features=features,
                    candidate_number=(
                        candidate_number
                    ),
                    target_numbers=(
                        target_numbers
                    ),
                )

                positive_count += int(
                    row["target"]
                )

                row[
                    "target_draw_index"
                ] = target_index

                row[
                    "target_draw_date"
                ] = str(
                    target_draw.draw_date
                )

                target_rows.append(
                    row
                )

            if len(target_rows) != 49:
                raise ValueError(
                    "Each target draw must produce "
                    "exactly 49 candidate rows."
                )

            if positive_count != 5:
                raise ValueError(
                    "Each target draw must produce "
                    "exactly 5 positive labels. "
                    f"Received {positive_count}."
                )

            rows.extend(
                target_rows
            )

            target_metadata.append(
                {
                    "target_index": (
                        target_index
                    ),
                    "target_date": str(
                        target_draw.draw_date
                    ),
                    "target_numbers": sorted(
                        target_numbers
                    ),
                    "feature_history_first_date": str(
                        history[0].draw_date
                    ),
                    "feature_history_last_date": str(
                        history[-1].draw_date
                    ),
                    "excluded_previous_draw_date": str(
                        excluded_previous_draw.draw_date
                    ),
                    "feature_history_size": (
                        len(history)
                    ),
                }
            )

        dataset = pd.DataFrame(
            rows
        )

        if dataset.empty:
            raise ValueError(
                "V7 ranking dataset is empty."
            )

        expected_rows = (
            len(target_metadata)
            * 49
        )

        if len(dataset) != expected_rows:
            raise ValueError(
                "Unexpected V7 dataset size. "
                f"Expected {expected_rows}, "
                f"received {len(dataset)}."
            )

        if dataset.isnull().any().any():
            null_columns = (
                dataset.columns[
                    dataset.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "V7 ranking dataset contains "
                "missing values in columns: "
                f"{null_columns}"
            )

        total_positives = int(
            dataset[
                "target"
            ].sum()
        )

        expected_positives = (
            len(target_metadata)
            * 5
        )

        if (
            total_positives
            != expected_positives
        ):
            raise ValueError(
                "Unexpected number of positive V7 labels. "
                f"Expected {expected_positives}, "
                f"received {total_positives}."
            )

        rows_per_target = (
            dataset
            .groupby(
                "target_draw_index"
            )
            .size()
        )

        if not (
            rows_per_target == 49
        ).all():
            raise ValueError(
                "At least one target draw does not "
                "contain exactly 49 candidate rows."
            )

        positives_per_target = (
            dataset
            .groupby(
                "target_draw_index"
            )[
                "target"
            ]
            .sum()
        )

        if not (
            positives_per_target == 5
        ).all():
            raise ValueError(
                "At least one target draw does not "
                "contain exactly 5 positive labels."
            )

        return (
            dataset,
            target_metadata,
        )

    @classmethod
    def feature_columns(
        cls,
    ) -> list[str]:
        """
        Return the ordered model feature columns.

        candidate_number is deliberately excluded so the model
        does not learn arbitrary number identity.
        """

        return list(
            cls.MODEL_FEATURES
        )

RankingDataset = V7RankingDataset
