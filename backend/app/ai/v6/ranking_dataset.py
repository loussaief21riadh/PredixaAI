import pandas as pd

from app.ai.v5.feature_engineering_v5b import (
    V5BFeatureEngineering,
)
from app.models.draw import Draw


class V6RankingDataset:
    """
    Predixa AI V6 Ranking Dataset.

    Converts each training target draw into 49 candidate rows.

    Temporal protocol:
        target = T
        feature history ends at T-2
        T-1 excluded
        target T excluded from features

    Each candidate row contains:
        - candidate_number
        - global historical features
        - candidate-specific historical features
        - target = 1 if candidate appears in T else 0
    """

    VERSION = "V6-RANKING-DATASET"

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    LAG_DRAWS = 1

    GLOBAL_FEATURES = (
        "history_size",
        "average_sum",
        "average_even_count",
        "average_consecutive_pairs",
    )

    CANDIDATE_FAMILIES = (
        "rate_10",
        "rate_20",
        "rate_50",
        "rate_100",
        "recency",
        "recency_ratio",
        "short_vs_long",
        "frequency_volatility",
    )

    @staticmethod
    def _main_numbers(
        draw: Draw,
    ) -> list[int]:
        return [
            draw.n1,
            draw.n2,
            draw.n3,
            draw.n4,
            draw.n5,
        ]

    @classmethod
    def _candidate_row(
        cls,
        features: dict,
        candidate_number: int,
        target_numbers: set[int],
    ) -> dict:
        row = {
            "candidate_number": (
                candidate_number
            ),
        }

        # --------------------------------------------------
        # Global features
        # --------------------------------------------------

        for name in cls.GLOBAL_FEATURES:
            if name not in features:
                raise ValueError(
                    f"Missing global feature: {name}"
                )

            row[name] = features[name]

        # --------------------------------------------------
        # Candidate-specific features
        # --------------------------------------------------

        for family in cls.CANDIDATE_FAMILIES:
            feature_name = (
                f"{family}_{candidate_number}"
            )

            if feature_name not in features:
                raise ValueError(
                    f"Missing candidate feature: "
                    f"{feature_name}"
                )

            row[family] = (
                features[
                    feature_name
                ]
            )

        row["target"] = (
            1
            if candidate_number
            in target_numbers
            else 0
        )

        return row

    @classmethod
    def build_from_draws(
        cls,
        draws: list[Draw],
        window_size: int = 100,
        max_training_targets: int = 1500,
    ):
        """
        Build candidate-level ranking dataset.

        For each target draw T:

            history:
                exactly window_size draws
                ending at T-2

            T-1:
                excluded from feature history

            T:
                used only for the binary candidate target
        """

        if window_size < 100:
            raise ValueError(
                "V6 requires window_size >= 100."
            )

        minimum_required = (
            window_size
            + cls.LAG_DRAWS
            + 1
        )

        if len(draws) < minimum_required:
            raise ValueError(
                "Not enough draws to build "
                "the V6 ranking dataset."
            )

        first_target_index = (
            window_size
            + cls.LAG_DRAWS
        )

        if max_training_targets > 0:
            first_target_index = max(
                first_target_index,
                len(draws)
                - max_training_targets,
            )

        rows = []

        target_metadata = []

        for target_index in range(
            first_target_index,
            len(draws),
        ):
            feature_end_index = (
                target_index
                - cls.LAG_DRAWS
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
                continue

            target_draw = draws[
                target_index
            ]

            features = (
                V5BFeatureEngineering
                .build_from_history(
                    history,
                    window_size=window_size,
                    variant="full",
                )
            )

            if len(features) != 396:
                raise ValueError(
                    "Unexpected V5 full feature count. "
                    f"Expected 396, received "
                    f"{len(features)}."
                )

            target_numbers = set(
                cls._main_numbers(
                    target_draw
                )
            )

            positive_count = 0

            for candidate_number in range(
                cls.NUMBER_MIN,
                cls.NUMBER_MAX + 1,
            ):
                row = cls._candidate_row(
                    features=features,
                    candidate_number=(
                        candidate_number
                    ),
                    target_numbers=(
                        target_numbers
                    ),
                )

                positive_count += (
                    row["target"]
                )

                row["target_draw_index"] = (
                    target_index
                )

                row["target_draw_date"] = str(
                    target_draw.draw_date
                )

                rows.append(
                    row
                )

            if positive_count != 5:
                raise ValueError(
                    "Each target draw must produce "
                    "exactly 5 positive candidates."
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
                "V6 ranking dataset is empty."
            )

        expected_rows = (
            len(target_metadata)
            * 49
        )

        if len(dataset) != expected_rows:
            raise ValueError(
                "Unexpected V6 dataset size."
            )

        if dataset.isnull().any().any():
            raise ValueError(
                "V6 ranking dataset contains "
                "missing values."
            )

        total_positives = int(
            dataset["target"].sum()
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
                "Unexpected number of positive "
                "V6 targets."
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
        Features intended for the ranking model.

        candidate_number is intentionally excluded
        so the model does not learn arbitrary number identity.
        """

        return [
            *cls.GLOBAL_FEATURES,
            *cls.CANDIDATE_FAMILIES,
        ]