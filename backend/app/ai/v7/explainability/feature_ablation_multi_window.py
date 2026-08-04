from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
)
from app.ai.v7.ranking_dataset import (
    V7RankingDataset,
)
from app.core.settings import (
    MAX_DEPTH,
    N_ESTIMATORS,
    RANDOM_STATE,
)
from app.database import SessionLocal
from app.models.draw import Draw


class V7FeatureAblationMultiWindowReport:
    """
    Run feature-family ablation over multiple chronological windows.

    Protocol
    --------
    - Build the complete V7 candidate-level dataset.
    - Select several non-overlapping validation windows.
    - Use only targets preceding each validation window for training.
    - Purge the target immediately preceding each validation window.
    - Optionally limit the number of training targets.
    - Train one Random Forest for every feature-family ablation.
    - Aggregate results across all windows.

    Experiments
    -----------
    - baseline
    - without_global
    - without_frequency
    - without_recency
    - without_trend
    - without_volatility
    """

    VERSION = (
        "V7-FEATURE-FAMILY-ABLATION-"
        "MULTI-WINDOW-PURGED-V1"
    )

    MODERN_LOTO_START_DATE = "2008-10-06"

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_MAX_TRAINING_TARGETS = 1500
    DEFAULT_VALIDATION_TARGETS = 50
    DEFAULT_WINDOWS = 5
    DEFAULT_TOP_K = 5
    DEFAULT_PURGE_TARGETS = 1

    MINIMUM_TRAINING_TARGETS = 100

    REQUIRED_DATASET_COLUMNS = (
        "candidate_number",
        "target",
        "target_draw_index",
        "target_draw_date",
    )

    BACKEND_DIR = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    DEFAULT_OUTPUT_DIR = (
        BACKEND_DIR
        / "reports"
        / "v7"
        / "feature_ablation_multi_window"
    )

    # ==========================================================
    # PARAMETER VALIDATION
    # ==========================================================

    @classmethod
    def _validate_parameters(
        cls,
        window_size: int,
        max_training_targets: int,
        validation_targets: int,
        windows: int,
        top_k: int,
        purge_targets: int,
    ) -> None:
        if window_size < 100:
            raise ValueError(
                "window_size must be at least 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        if (
            max_training_targets > 0
            and max_training_targets
            < cls.MINIMUM_TRAINING_TARGETS
        ):
            raise ValueError(
                "max_training_targets must be at least "
                f"{cls.MINIMUM_TRAINING_TARGETS}, "
                "or 0 to disable the limit."
            )

        if validation_targets < 5:
            raise ValueError(
                "validation_targets must be at least 5."
            )

        if windows < 2:
            raise ValueError(
                "windows must be at least 2."
            )

        if not (
            1
            <= top_k
            <= 49
        ):
            raise ValueError(
                "top_k must be between 1 and 49."
            )

        if purge_targets < 0:
            raise ValueError(
                "purge_targets cannot be negative."
            )

    # ==========================================================
    # DATABASE
    # ==========================================================

    @classmethod
    def _load_draws(
        cls,
        db: Any,
    ) -> list[Draw]:
        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= cls.MODERN_LOTO_START_DATE
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        if not draws:
            raise ValueError(
                "No modern draws were found."
            )

        return list(
            draws
        )

    # ==========================================================
    # DATASET VALIDATION
    # ==========================================================

    @classmethod
    def _validate_dataset(
        cls,
        dataset: pd.DataFrame,
    ) -> None:
        if not isinstance(
            dataset,
            pd.DataFrame,
        ):
            raise ValueError(
                "dataset must be a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                "dataset cannot be empty."
            )

        missing_required_columns = [
            column
            for column in (
                cls.REQUIRED_DATASET_COLUMNS
            )
            if column not in dataset.columns
        ]

        if missing_required_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing_required_columns}"
            )

        model_features = (
            V7RankingDataset
            .feature_columns()
        )

        missing_model_features = [
            feature_name
            for feature_name in model_features
            if feature_name
            not in dataset.columns
        ]

        if missing_model_features:
            raise ValueError(
                "Dataset is missing model features: "
                f"{missing_model_features}"
            )

        if dataset.isnull().any().any():
            null_columns = (
                dataset.columns[
                    dataset.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Dataset contains missing values: "
                f"{null_columns}"
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
                "Every target must contain exactly "
                "49 candidate rows."
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
                "Every target must contain exactly "
                "5 positive labels."
            )

    # ==========================================================
    # FEATURE CONFIGURATION
    # ==========================================================

    @staticmethod
    def _validate_feature_configuration(
    ) -> None:
        model_features = set(
            V7RankingDataset
            .feature_columns()
        )

        configured_features = {
            feature_name
            for family_name in (
                FEATURE_FAMILY_ORDER
            )
            for feature_name in (
                FEATURE_FAMILIES[
                    family_name
                ]
            )
        }

        missing_configuration = sorted(
            model_features
            - configured_features
        )

        unknown_configuration = sorted(
            configured_features
            - model_features
        )

        if missing_configuration:
            raise ValueError(
                "Model features missing from family "
                f"configuration: {missing_configuration}"
            )

        if unknown_configuration:
            raise ValueError(
                "Configured features not used by "
                f"the model: {unknown_configuration}"
            )

    @staticmethod
    def _feature_sets() -> dict[
        str,
        list[str],
    ]:
        baseline_features = (
            V7RankingDataset
            .feature_columns()
        )

        feature_sets: dict[
            str,
            list[str],
        ] = {
            "baseline": list(
                baseline_features
            ),
        }

        for family_name in (
            FEATURE_FAMILY_ORDER
        ):
            removed_features = set(
                FEATURE_FAMILIES[
                    family_name
                ]
            )

            retained_features = [
                feature_name
                for feature_name in (
                    baseline_features
                )
                if feature_name
                not in removed_features
            ]

            if not retained_features:
                raise ValueError(
                    "Ablation removed every model "
                    f"feature for family: {family_name}"
                )

            feature_sets[
                f"without_{family_name}"
            ] = retained_features

        return feature_sets

    # ==========================================================
    # WINDOW CONSTRUCTION
    # ==========================================================

    @classmethod
    def _build_windows(
        cls,
        dataset: pd.DataFrame,
        windows: int,
        validation_targets: int,
        max_training_targets: int,
        purge_targets: int,
    ) -> list[dict[str, Any]]:
        target_indices = sorted(
            dataset[
                "target_draw_index"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        required_validation_targets = (
            windows
            * validation_targets
        )

        minimum_required_targets = (
            cls.MINIMUM_TRAINING_TARGETS
            + purge_targets
            + required_validation_targets
        )

        if (
            len(target_indices)
            < minimum_required_targets
        ):
            raise ValueError(
                "Not enough dataset targets for "
                "multi-window ablation. "
                f"Required at least "
                f"{minimum_required_targets}, "
                f"received {len(target_indices)}."
            )

        validation_start_position = (
            len(target_indices)
            - required_validation_targets
        )

        window_definitions: list[
            dict[str, Any]
        ] = []

        for window_offset in range(
            windows
        ):
            validation_start = (
                validation_start_position
                + (
                    window_offset
                    * validation_targets
                )
            )

            validation_end = (
                validation_start
                + validation_targets
            )

            validation_target_indices = (
                target_indices[
                    validation_start:
                    validation_end
                ]
            )

            training_end = (
                validation_start
                - purge_targets
            )

            if training_end <= 0:
                raise ValueError(
                    "No targets remain before the "
                    "validation window after purging."
                )

            available_training_indices = (
                target_indices[
                    :training_end
                ]
            )

            if max_training_targets > 0:
                training_target_indices = (
                    available_training_indices[
                        -max_training_targets:
                    ]
                )
            else:
                training_target_indices = (
                    available_training_indices
                )

            if (
                len(training_target_indices)
                < cls.MINIMUM_TRAINING_TARGETS
            ):
                raise ValueError(
                    "A validation window contains fewer "
                    "than the required number of "
                    "training targets. "
                    f"Window={window_offset + 1}, "
                    f"required="
                    f"{cls.MINIMUM_TRAINING_TARGETS}, "
                    f"received="
                    f"{len(training_target_indices)}."
                )

            training_dataset = (
                dataset[
                    dataset[
                        "target_draw_index"
                    ].isin(
                        training_target_indices
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            validation_dataset = (
                dataset[
                    dataset[
                        "target_draw_index"
                    ].isin(
                        validation_target_indices
                    )
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            if training_dataset.empty:
                raise ValueError(
                    "A multi-window training dataset "
                    "is empty."
                )

            if validation_dataset.empty:
                raise ValueError(
                    "A multi-window validation dataset "
                    "is empty."
                )

            training_last_target = int(
                training_target_indices[
                    -1
                ]
            )

            validation_first_target = int(
                validation_target_indices[
                    0
                ]
            )

            if not (
                training_last_target
                < validation_first_target
            ):
                raise ValueError(
                    "Temporal leakage detected between "
                    "training and validation targets."
                )

            training_first_date = str(
                training_dataset[
                    "target_draw_date"
                ].iloc[0]
            )

            training_last_date = str(
                training_dataset[
                    "target_draw_date"
                ].iloc[-1]
            )

            validation_first_date = str(
                validation_dataset[
                    "target_draw_date"
                ].iloc[0]
            )

            validation_last_date = str(
                validation_dataset[
                    "target_draw_date"
                ].iloc[-1]
            )

            window_definitions.append(
                {
                    "window_number": (
                        window_offset + 1
                    ),
                    "training_target_indices": (
                        training_target_indices
                    ),
                    "validation_target_indices": (
                        validation_target_indices
                    ),
                    "training_dataset": (
                        training_dataset
                    ),
                    "validation_dataset": (
                        validation_dataset
                    ),
                    "training_targets": len(
                        training_target_indices
                    ),
                    "validation_targets": len(
                        validation_target_indices
                    ),
                    "training_rows": len(
                        training_dataset
                    ),
                    "validation_rows": len(
                        validation_dataset
                    ),
                    "training_first_date": (
                        training_first_date
                    ),
                    "training_last_date": (
                        training_last_date
                    ),
                    "validation_first_date": (
                        validation_first_date
                    ),
                    "validation_last_date": (
                        validation_last_date
                    ),
                }
            )

        return window_definitions

    # ==========================================================
    # MODEL
    # ==========================================================

    @staticmethod
    def _build_model() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

    # ==========================================================
    # EXPERIMENT EVALUATION
    # ==========================================================

    @classmethod
    def _evaluate_experiment(
        cls,
        experiment_name: str,
        feature_columns: list[str],
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        top_k: int,
    ) -> dict[str, Any]:
        if not feature_columns:
            raise ValueError(
                f"{experiment_name}: feature "
                "column list is empty."
            )

        training_X = (
            training_dataset[
                feature_columns
            ]
            .astype(float)
        )

        training_y = (
            training_dataset[
                "target"
            ]
            .astype(int)
        )

        unique_targets = set(
            training_y
            .unique()
            .tolist()
        )

        if unique_targets != {
            0,
            1,
        }:
            raise ValueError(
                f"{experiment_name}: training data "
                "must contain both binary classes."
            )

        model = cls._build_model()

        training_started = (
            perf_counter()
        )

        model.fit(
            training_X,
            training_y,
        )

        training_seconds = (
            perf_counter()
            - training_started
        )

        model_classes = list(
            model.classes_
        )

        if 1 not in model_classes:
            raise ValueError(
                f"{experiment_name}: positive class "
                "is unavailable."
            )

        positive_class_index = (
            model_classes.index(1)
        )

        evaluation_started = (
            perf_counter()
        )

        maximum_hits = min(
            top_k,
            5,
        )

        hit_distribution = {
            hit_count: 0
            for hit_count in range(
                maximum_hits + 1
            )
        }

        total_hits = 0
        evaluated_targets = 0
        at_least_one_hit = 0
        at_least_two_hits = 0

        details: list[
            dict[str, Any]
        ] = []

        grouped_validation = (
            validation_dataset
            .groupby(
                "target_draw_index",
                sort=True,
            )
        )

        for (
            target_draw_index,
            target_rows,
        ) in grouped_validation:
            ordered_rows = (
                target_rows
                .sort_values(
                    by="candidate_number",
                    ascending=True,
                )
                .reset_index(
                    drop=True
                )
            )

            if len(
                ordered_rows
            ) != 49:
                raise ValueError(
                    f"{experiment_name}: target "
                    f"{target_draw_index} does not "
                    "contain 49 candidates."
                )

            candidate_numbers = (
                ordered_rows[
                    "candidate_number"
                ]
                .astype(int)
                .to_numpy()
            )

            actual_numbers = set(
                ordered_rows.loc[
                    ordered_rows[
                        "target"
                    ].astype(int)
                    == 1,
                    "candidate_number",
                ]
                .astype(int)
                .tolist()
            )

            if len(
                actual_numbers
            ) != 5:
                raise ValueError(
                    f"{experiment_name}: target "
                    f"{target_draw_index} does not "
                    "contain exactly 5 winners."
                )

            probabilities = (
                model.predict_proba(
                    ordered_rows[
                        feature_columns
                    ].astype(float)
                )[
                    :,
                    positive_class_index
                ]
            )

            if not np.all(
                np.isfinite(
                    probabilities
                )
            ):
                raise ValueError(
                    f"{experiment_name}: prediction "
                    "contains non-finite values."
                )

            ranking_indices = sorted(
                range(
                    len(
                        candidate_numbers
                    )
                ),
                key=lambda index: (
                    -float(
                        probabilities[
                            index
                        ]
                    ),
                    int(
                        candidate_numbers[
                            index
                        ]
                    ),
                ),
            )

            predicted_numbers = [
                int(
                    candidate_numbers[
                        index
                    ]
                )
                for index in (
                    ranking_indices[
                        :top_k
                    ]
                )
            ]

            hits = len(
                set(
                    predicted_numbers
                )
                & actual_numbers
            )

            total_hits += hits
            evaluated_targets += 1

            hit_distribution[
                hits
            ] += 1

            if hits >= 1:
                at_least_one_hit += 1

            if hits >= 2:
                at_least_two_hits += 1

            details.append(
                {
                    "target_draw_index": int(
                        target_draw_index
                    ),
                    "target_draw_date": str(
                        ordered_rows[
                            "target_draw_date"
                        ].iloc[0]
                    ),
                    "predicted_top_k": (
                        predicted_numbers
                    ),
                    "actual_numbers": sorted(
                        actual_numbers
                    ),
                    "hits": hits,
                }
            )

        evaluation_seconds = (
            perf_counter()
            - evaluation_started
        )

        if evaluated_targets == 0:
            raise ValueError(
                f"{experiment_name}: no validation "
                "targets were evaluated."
            )

        average_hits = (
            total_hits
            / evaluated_targets
        )

        return {
            "experiment": (
                experiment_name
            ),
            "removed_family": (
                None
                if experiment_name
                == "baseline"
                else experiment_name.replace(
                    "without_",
                    "",
                    1,
                )
            ),
            "feature_count": len(
                feature_columns
            ),
            "features": list(
                feature_columns
            ),
            "training_rows": len(
                training_dataset
            ),
            "validation_rows": len(
                validation_dataset
            ),
            "evaluated_targets": (
                evaluated_targets
            ),
            "total_hits": (
                total_hits
            ),
            "average_hits_at_5": round(
                average_hits,
                6,
            ),
            "precision_at_k": round(
                total_hits
                / (
                    evaluated_targets
                    * top_k
                ),
                6,
            ),
            "at_least_1_hit_rate": round(
                at_least_one_hit
                / evaluated_targets,
                6,
            ),
            "at_least_2_hit_rate": round(
                at_least_two_hits
                / evaluated_targets,
                6,
            ),
            "hit_distribution": (
                hit_distribution
            ),
            "training_seconds": round(
                training_seconds,
                6,
            ),
            "evaluation_seconds": round(
                evaluation_seconds,
                6,
            ),
            "details": details,
        }

    # ==========================================================
    # BASELINE COMPARISON
    # ==========================================================

    @staticmethod
    def _add_window_comparison(
        experiments: list[
            dict[str, Any]
        ],
    ) -> None:
        baseline = next(
            (
                experiment
                for experiment in experiments
                if experiment[
                    "experiment"
                ]
                == "baseline"
            ),
            None,
        )

        if baseline is None:
            raise ValueError(
                "Window baseline experiment "
                "is missing."
            )

        baseline_hits = float(
            baseline[
                "average_hits_at_5"
            ]
        )

        for experiment in experiments:
            experiment_hits = float(
                experiment[
                    "average_hits_at_5"
                ]
            )

            delta = (
                experiment_hits
                - baseline_hits
            )

            experiment[
                "delta_vs_baseline"
            ] = round(
                delta,
                6,
            )

            if (
                experiment[
                    "experiment"
                ]
                == "baseline"
            ):
                conclusion = "reference"
            elif delta > 0:
                conclusion = (
                    "improved_without_family"
                )
            elif delta < 0:
                conclusion = (
                    "degraded_without_family"
                )
            else:
                conclusion = (
                    "unchanged_without_family"
                )

            experiment[
                "conclusion"
            ] = conclusion

    # ==========================================================
    # AGGREGATION
    # ==========================================================

    @classmethod
    def _aggregate_experiments(
        cls,
        windows_results: list[
            dict[str, Any]
        ],
        experiment_names: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        aggregates: list[
            dict[str, Any]
        ] = []

        random_expectation = (
            top_k
            * 5
            / 49
        )

        for experiment_name in (
            experiment_names
        ):
            window_experiments = [
                experiment
                for window_result in (
                    windows_results
                )
                for experiment in (
                    window_result[
                        "experiments"
                    ]
                )
                if experiment[
                    "experiment"
                ]
                == experiment_name
            ]

            if (
                len(window_experiments)
                != len(windows_results)
            ):
                raise ValueError(
                    "Unexpected number of window "
                    f"results for {experiment_name}."
                )

            hit_values = np.array(
                [
                    float(
                        experiment[
                            "average_hits_at_5"
                        ]
                    )
                    for experiment in (
                        window_experiments
                    )
                ],
                dtype=float,
            )

            delta_values = np.array(
                [
                    float(
                        experiment[
                            "delta_vs_baseline"
                        ]
                    )
                    for experiment in (
                        window_experiments
                    )
                ],
                dtype=float,
            )

            improved_windows = int(
                np.sum(
                    delta_values > 0
                )
            )

            degraded_windows = int(
                np.sum(
                    delta_values < 0
                )
            )

            unchanged_windows = int(
                np.sum(
                    delta_values == 0
                )
            )

            mean_hits = float(
                np.mean(
                    hit_values
                )
            )

            mean_delta = float(
                np.mean(
                    delta_values
                )
            )

            if (
                experiment_name
                == "baseline"
            ):
                conclusion = "reference"
            elif (
                mean_delta > 0
                and improved_windows
                > degraded_windows
            ):
                conclusion = (
                    "family_may_add_noise"
                )
            elif (
                mean_delta < 0
                and degraded_windows
                > improved_windows
            ):
                conclusion = (
                    "family_likely_useful"
                )
            else:
                conclusion = "inconclusive"

            aggregates.append(
                {
                    "experiment": (
                        experiment_name
                    ),
                    "removed_family": (
                        None
                        if experiment_name
                        == "baseline"
                        else experiment_name.replace(
                            "without_",
                            "",
                            1,
                        )
                    ),
                    "feature_count": int(
                        window_experiments[
                            0
                        ][
                            "feature_count"
                        ]
                    ),
                    "windows": len(
                        window_experiments
                    ),
                    "total_evaluated_targets": sum(
                        int(
                            experiment[
                                "evaluated_targets"
                            ]
                        )
                        for experiment in (
                            window_experiments
                        )
                    ),
                    "total_hits": sum(
                        int(
                            experiment[
                                "total_hits"
                            ]
                        )
                        for experiment in (
                            window_experiments
                        )
                    ),
                    "window_mean_hits_at_5": round(
                        mean_hits,
                        6,
                    ),
                    "window_std_hits_at_5": round(
                        float(
                            np.std(
                                hit_values,
                                ddof=0,
                            )
                        ),
                        6,
                    ),
                    "minimum_hits_at_5": round(
                        float(
                            np.min(
                                hit_values
                            )
                        ),
                        6,
                    ),
                    "maximum_hits_at_5": round(
                        float(
                            np.max(
                                hit_values
                            )
                        ),
                        6,
                    ),
                    "mean_delta_vs_baseline": round(
                        mean_delta,
                        6,
                    ),
                    "minimum_delta_vs_baseline": round(
                        float(
                            np.min(
                                delta_values
                            )
                        ),
                        6,
                    ),
                    "maximum_delta_vs_baseline": round(
                        float(
                            np.max(
                                delta_values
                            )
                        ),
                        6,
                    ),
                    "improved_windows": (
                        improved_windows
                    ),
                    "degraded_windows": (
                        degraded_windows
                    ),
                    "unchanged_windows": (
                        unchanged_windows
                    ),
                    "average_training_seconds": round(
                        float(
                            np.mean(
                                [
                                    float(
                                        experiment[
                                            "training_seconds"
                                        ]
                                    )
                                    for experiment in (
                                        window_experiments
                                    )
                                ]
                            )
                        ),
                        6,
                    ),
                    "average_evaluation_seconds": round(
                        float(
                            np.mean(
                                [
                                    float(
                                        experiment[
                                            "evaluation_seconds"
                                        ]
                                    )
                                    for experiment in (
                                        window_experiments
                                    )
                                ]
                            )
                        ),
                        6,
                    ),
                    "random_expectation": round(
                        random_expectation,
                        6,
                    ),
                    "mean_lift_vs_random": round(
                        mean_hits
                        - random_expectation,
                        6,
                    ),
                    "conclusion": conclusion,
                }
            )

        return aggregates

    # ==========================================================
    # DATAFRAMES
    # ==========================================================

    @staticmethod
    def _aggregate_dataframe(
        aggregates: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        columns = [
            "experiment",
            "removed_family",
            "feature_count",
            "windows",
            "total_evaluated_targets",
            "total_hits",
            "window_mean_hits_at_5",
            "window_std_hits_at_5",
            "minimum_hits_at_5",
            "maximum_hits_at_5",
            "mean_delta_vs_baseline",
            "minimum_delta_vs_baseline",
            "maximum_delta_vs_baseline",
            "improved_windows",
            "degraded_windows",
            "unchanged_windows",
            "average_training_seconds",
            "average_evaluation_seconds",
            "random_expectation",
            "mean_lift_vs_random",
            "conclusion",
        ]

        return pd.DataFrame(
            aggregates,
            columns=columns,
        )

    @staticmethod
    def _window_dataframe(
        windows_results: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        rows: list[
            dict[str, Any]
        ] = []

        for window_result in (
            windows_results
        ):
            for experiment in (
                window_result[
                    "experiments"
                ]
            ):
                rows.append(
                    {
                        "window_number": (
                            window_result[
                                "window_number"
                            ]
                        ),
                        "training_first_date": (
                            window_result[
                                "training_first_date"
                            ]
                        ),
                        "training_last_date": (
                            window_result[
                                "training_last_date"
                            ]
                        ),
                        "validation_first_date": (
                            window_result[
                                "validation_first_date"
                            ]
                        ),
                        "validation_last_date": (
                            window_result[
                                "validation_last_date"
                            ]
                        ),
                        "training_targets": (
                            window_result[
                                "training_targets"
                            ]
                        ),
                        "validation_targets": (
                            window_result[
                                "validation_targets"
                            ]
                        ),
                        "experiment": (
                            experiment[
                                "experiment"
                            ]
                        ),
                        "removed_family": (
                            experiment[
                                "removed_family"
                            ]
                        ),
                        "feature_count": (
                            experiment[
                                "feature_count"
                            ]
                        ),
                        "total_hits": (
                            experiment[
                                "total_hits"
                            ]
                        ),
                        "average_hits_at_5": (
                            experiment[
                                "average_hits_at_5"
                            ]
                        ),
                        "delta_vs_baseline": (
                            experiment[
                                "delta_vs_baseline"
                            ]
                        ),
                        "at_least_1_hit_rate": (
                            experiment[
                                "at_least_1_hit_rate"
                            ]
                        ),
                        "at_least_2_hit_rate": (
                            experiment[
                                "at_least_2_hit_rate"
                            ]
                        ),
                        "training_seconds": (
                            experiment[
                                "training_seconds"
                            ]
                        ),
                        "evaluation_seconds": (
                            experiment[
                                "evaluation_seconds"
                            ]
                        ),
                        "conclusion": (
                            experiment[
                                "conclusion"
                            ]
                        ),
                    }
                )

        return pd.DataFrame(
            rows
        )

    # ==========================================================
    # EXPORTS
    # ==========================================================

    @staticmethod
    def _write_csv(
        dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        dataframe.to_csv(
            output_path,
            index=False,
        )

        if not output_path.exists():
            raise ValueError(
                "CSV export failed."
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                "Generated CSV file is empty."
            )

        return output_path.resolve()

    @staticmethod
    def _write_json(
        result: dict[str, Any],
        output_path: Path,
    ) -> Path:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=4,
            )

        if not output_path.exists():
            raise ValueError(
                "JSON export failed."
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                "Generated JSON file is empty."
            )

        return output_path.resolve()

    @staticmethod
    def _write_text(
        result: dict[str, Any],
        output_path: Path,
    ) -> Path:
        lines = [
            "=" * 120,
            (
                "PREDIXA AI V7 FEATURE FAMILY "
                "ABLATION - MULTI-WINDOW"
            ),
            "=" * 120,
            "",
            (
                "Version                    : "
                f"{result['version']}"
            ),
            (
                "Windows                    : "
                f"{result['windows']}"
            ),
            (
                "Validation targets/window  : "
                f"{result['validation_targets_per_window']}"
            ),
            (
                "Maximum training targets   : "
                f"{result['max_training_targets']}"
            ),
            (
                "Purged targets             : "
                f"{result['purge_targets']}"
            ),
            (
                "Window size                : "
                f"{result['window_size']}"
            ),
            (
                "Top-K                      : "
                f"{result['top_k']}"
            ),
            "",
            "AGGREGATE RESULTS",
            "-" * 120,
            (
                f"{'Experiment':<24}"
                f"{'Features':>10}"
                f"{'Mean':>12}"
                f"{'Std':>12}"
                f"{'Min':>12}"
                f"{'Max':>12}"
                f"{'Mean Δ':>12}"
                f"{'Better':>10}"
                f"{'Worse':>10}"
                f"{'Same':>8}"
            ),
            "-" * 120,
        ]

        for aggregate in (
            result[
                "aggregates"
            ]
        ):
            lines.append(
                f"{aggregate['experiment']:<24}"
                f"{int(aggregate['feature_count']):>10}"
                f"{float(aggregate['window_mean_hits_at_5']):>12.6f}"
                f"{float(aggregate['window_std_hits_at_5']):>12.6f}"
                f"{float(aggregate['minimum_hits_at_5']):>12.6f}"
                f"{float(aggregate['maximum_hits_at_5']):>12.6f}"
                f"{float(aggregate['mean_delta_vs_baseline']):>12.6f}"
                f"{int(aggregate['improved_windows']):>10}"
                f"{int(aggregate['degraded_windows']):>10}"
                f"{int(aggregate['unchanged_windows']):>8}"
            )

        lines.extend(
            [
                "",
                "WINDOW RESULTS",
                "-" * 120,
            ]
        )

        for window_result in (
            result[
                "window_results"
            ]
        ):
            lines.extend(
                [
                    "",
                    (
                        f"Window "
                        f"{window_result['window_number']} | "
                        f"{window_result['validation_first_date']} "
                        f"-> "
                        f"{window_result['validation_last_date']}"
                    ),
                    (
                        f"Training targets: "
                        f"{window_result['training_targets']} | "
                        f"Validation targets: "
                        f"{window_result['validation_targets']}"
                    ),
                    "-" * 120,
                ]
            )

            for experiment in (
                window_result[
                    "experiments"
                ]
            ):
                lines.append(
                    f"{experiment['experiment']:<24}"
                    f"Hits@5="
                    f"{float(experiment['average_hits_at_5']):.6f} | "
                    f"Delta="
                    f"{float(experiment['delta_vs_baseline']):+.6f} | "
                    f"Total hits="
                    f"{int(experiment['total_hits'])}"
                )

        lines.extend(
            [
                "",
                "=" * 120,
                "INTERPRETATION",
                "=" * 120,
                "",
                (
                    "A negative mean delta means that "
                    "removing the family reduced Hits@5."
                ),
                (
                    "A positive mean delta means that "
                    "removing the family improved Hits@5."
                ),
                (
                    "A family should only be removed when "
                    "the improvement is stable across "
                    "multiple chronological windows."
                ),
                "",
            ]
        )

        output_path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        if not output_path.exists():
            raise ValueError(
                "Text export failed."
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                "Generated text file is empty."
            )

        return output_path.resolve()

    # ==========================================================
    # MAIN RUNNER
    # ==========================================================

    @classmethod
    def run(
        cls,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_training_targets: int = (
            DEFAULT_MAX_TRAINING_TARGETS
        ),
        validation_targets: int = (
            DEFAULT_VALIDATION_TARGETS
        ),
        windows: int = DEFAULT_WINDOWS,
        top_k: int = DEFAULT_TOP_K,
        purge_targets: int = (
            DEFAULT_PURGE_TARGETS
        ),
    ) -> dict[str, Any]:
        cls._validate_parameters(
            window_size=window_size,
            max_training_targets=(
                max_training_targets
            ),
            validation_targets=(
                validation_targets
            ),
            windows=windows,
            top_k=top_k,
            purge_targets=(
                purge_targets
            ),
        )

        cls._validate_feature_configuration()

        destination = Path(
            output_dir
        ).expanduser().resolve()

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        db = SessionLocal()

        try:
            draws = cls._load_draws(
                db
            )

            dataset, metadata = (
                V7RankingDataset()
                .build_from_draws(
                    draws=draws,
                    window_size=window_size,
                    max_training_targets=0,
                )
            )

        finally:
            db.close()

        cls._validate_dataset(
            dataset
        )

        window_definitions = (
            cls._build_windows(
                dataset=dataset,
                windows=windows,
                validation_targets=(
                    validation_targets
                ),
                max_training_targets=(
                    max_training_targets
                ),
                purge_targets=(
                    purge_targets
                ),
            )
        )

        feature_sets = (
            cls._feature_sets()
        )

        window_results: list[
            dict[str, Any]
        ] = []

        for window_definition in (
            window_definitions
        ):
            experiments: list[
                dict[str, Any]
            ] = []

            for (
                experiment_name,
                feature_columns,
            ) in feature_sets.items():
                experiment_result = (
                    cls._evaluate_experiment(
                        experiment_name=(
                            experiment_name
                        ),
                        feature_columns=(
                            feature_columns
                        ),
                        training_dataset=(
                            window_definition[
                                "training_dataset"
                            ]
                        ),
                        validation_dataset=(
                            window_definition[
                                "validation_dataset"
                            ]
                        ),
                        top_k=top_k,
                    )
                )

                experiments.append(
                    experiment_result
                )

            cls._add_window_comparison(
                experiments
            )

            window_results.append(
                {
                    "window_number": (
                        window_definition[
                            "window_number"
                        ]
                    ),
                    "training_targets": (
                        window_definition[
                            "training_targets"
                        ]
                    ),
                    "validation_targets": (
                        window_definition[
                            "validation_targets"
                        ]
                    ),
                    "training_rows": (
                        window_definition[
                            "training_rows"
                        ]
                    ),
                    "validation_rows": (
                        window_definition[
                            "validation_rows"
                        ]
                    ),
                    "training_first_date": (
                        window_definition[
                            "training_first_date"
                        ]
                    ),
                    "training_last_date": (
                        window_definition[
                            "training_last_date"
                        ]
                    ),
                    "validation_first_date": (
                        window_definition[
                            "validation_first_date"
                        ]
                    ),
                    "validation_last_date": (
                        window_definition[
                            "validation_last_date"
                        ]
                    ),
                    "experiments": (
                        experiments
                    ),
                }
            )

        experiment_names = list(
            feature_sets
        )

        aggregates = (
            cls._aggregate_experiments(
                windows_results=(
                    window_results
                ),
                experiment_names=(
                    experiment_names
                ),
                top_k=top_k,
            )
        )

        result: dict[str, Any] = {
            "status": "success",
            "version": cls.VERSION,
            "model_type": (
                "RandomForestClassifier"
            ),
            "draw_count": len(
                draws
            ),
            "dataset_rows": len(
                dataset
            ),
            "dataset_targets": len(
                metadata
            ),
            "window_size": (
                window_size
            ),
            "max_training_targets": (
                max_training_targets
            ),
            "validation_targets_per_window": (
                validation_targets
            ),
            "windows": windows,
            "top_k": top_k,
            "purge_targets": (
                purge_targets
            ),
            "aggregates": aggregates,
            "window_results": (
                window_results
            ),
        }

        aggregate_dataframe = (
            cls._aggregate_dataframe(
                aggregates
            )
        )

        window_dataframe = (
            cls._window_dataframe(
                window_results
            )
        )

        aggregate_csv_path = cls._write_csv(
            dataframe=aggregate_dataframe,
            output_path=(
                destination
                / (
                    "feature_ablation_"
                    "multi_window_summary.csv"
                )
            ),
        )

        windows_csv_path = cls._write_csv(
            dataframe=window_dataframe,
            output_path=(
                destination
                / (
                    "feature_ablation_"
                    "multi_window_windows.csv"
                )
            ),
        )

        text_path = cls._write_text(
            result=result,
            output_path=(
                destination
                / (
                    "feature_ablation_"
                    "multi_window.txt"
                )
            ),
        )

        result[
            "files"
        ] = {
            "summary_csv": str(
                aggregate_csv_path
            ),
            "windows_csv": str(
                windows_csv_path
            ),
            "json": str(
                destination
                / (
                    "feature_ablation_"
                    "multi_window.json"
                )
            ),
            "text": str(
                text_path
            ),
        }

        json_path = cls._write_json(
            result=result,
            output_path=(
                destination
                / (
                    "feature_ablation_"
                    "multi_window.json"
                )
            ),
        )

        result[
            "files"
        ][
            "json"
        ] = str(
            json_path
        )

        return result

    # ==========================================================
    # CONSOLE OUTPUT
    # ==========================================================

    @staticmethod
    def print_result(
        result: dict[str, Any],
    ) -> None:
        print("=" * 120)
        print(
            "PREDIXA AI V7 FEATURE FAMILY "
            "ABLATION - MULTI-WINDOW"
        )
        print("=" * 120)

        print(
            f"Status                    : "
            f"{result['status']}"
        )

        print(
            f"Windows                   : "
            f"{result['windows']}"
        )

        print(
            f"Validation targets/window : "
            f"{result['validation_targets_per_window']}"
        )

        print(
            f"Maximum training targets  : "
            f"{result['max_training_targets']}"
        )

        print(
            f"Purged targets            : "
            f"{result['purge_targets']}"
        )

        print()

        print(
            f"{'Experiment':<24}"
            f"{'Features':>10}"
            f"{'Mean':>12}"
            f"{'Std':>12}"
            f"{'Min':>12}"
            f"{'Max':>12}"
            f"{'Mean Δ':>12}"
            f"{'Better':>10}"
            f"{'Worse':>10}"
            f"{'Same':>8}"
        )

        print("-" * 120)

        for aggregate in (
            result[
                "aggregates"
            ]
        ):
            print(
                f"{aggregate['experiment']:<24}"
                f"{int(aggregate['feature_count']):>10}"
                f"{float(aggregate['window_mean_hits_at_5']):>12.6f}"
                f"{float(aggregate['window_std_hits_at_5']):>12.6f}"
                f"{float(aggregate['minimum_hits_at_5']):>12.6f}"
                f"{float(aggregate['maximum_hits_at_5']):>12.6f}"
                f"{float(aggregate['mean_delta_vs_baseline']):>12.6f}"
                f"{int(aggregate['improved_windows']):>10}"
                f"{int(aggregate['degraded_windows']):>10}"
                f"{int(aggregate['unchanged_windows']):>8}"
            )

        print()
        print("WINDOWS")
        print("-" * 120)

        for window_result in (
            result[
                "window_results"
            ]
        ):
            baseline = next(
                experiment
                for experiment in (
                    window_result[
                        "experiments"
                    ]
                )
                if experiment[
                    "experiment"
                ]
                == "baseline"
            )

            print(
                f"window_{window_result['window_number']} | "
                f"{window_result['validation_first_date']} "
                f"-> "
                f"{window_result['validation_last_date']} | "
                f"Baseline Hits@5 = "
                f"{float(baseline['average_hits_at_5']):.6f}"
            )

        print()
        print("GENERATED FILES")
        print("-" * 120)

        for file_type, file_path in (
            result[
                "files"
            ].items()
        ):
            print(
                f"{file_type.upper():<12}: "
                f"{file_path}"
            )

        print()
        print("=" * 120)
        print("SUCCESS")
        print("=" * 120)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run PredixaAI V7 feature-family "
            "ablation over multiple temporal windows."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_WINDOW_SIZE
        ),
    )

    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_MAX_TRAINING_TARGETS
        ),
    )

    parser.add_argument(
        "--validation-targets",
        type=int,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_VALIDATION_TARGETS
        ),
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_WINDOWS
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_TOP_K
        ),
    )

    parser.add_argument(
        "--purge-targets",
        type=int,
        default=(
            V7FeatureAblationMultiWindowReport
            .DEFAULT_PURGE_TARGETS
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()

    arguments = parser.parse_args()

    result = (
        V7FeatureAblationMultiWindowReport.run(
            output_dir=arguments.output_dir,
            window_size=arguments.window_size,
            max_training_targets=(
                arguments.max_training_targets
            ),
            validation_targets=(
                arguments.validation_targets
            ),
            windows=arguments.windows,
            top_k=arguments.top_k,
            purge_targets=(
                arguments.purge_targets
            ),
        )
    )

    V7FeatureAblationMultiWindowReport.print_result(
        result
    )


if __name__ == "__main__":
    main()