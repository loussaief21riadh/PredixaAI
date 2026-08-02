from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.v6b_clean.walk_forward_backtester import (
    V6BCleanWalkForwardBacktester,
)
from app.models.draw import Draw


class _TruncatedQuery:
    """
    Minimal SQLAlchemy-query-compatible wrapper.
    """

    def __init__(
        self,
        draws: list[Draw],
    ) -> None:
        self._draws = draws

    def filter(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "_TruncatedQuery":
        return self

    def order_by(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> "_TruncatedQuery":
        return self

    def all(
        self,
    ) -> list[Draw]:
        return list(
            self._draws
        )


class _TruncatedDB:
    """
    Minimal database wrapper exposing only a truncated draw history.
    """

    def __init__(
        self,
        draws: list[Draw],
    ) -> None:
        self._draws = draws

    def query(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> _TruncatedQuery:
        return _TruncatedQuery(
            self._draws
        )


class V6BCleanMultiWindowValidator:
    """
    Predixa AI V6B-CLEAN multi-window temporal validator.

    Runs strict V6B-CLEAN walk-forward evaluation across
    chronological, non-overlapping windows.
    """

    VERSION = "V6B-CLEAN-MULTI-WINDOW-PURGED-T2"

    MODERN_LOTO_START_DATE = "2008-10-06"

    RANDOM_EXPECTATION = 25 / 49

    @staticmethod
    def _metric_summary(
        values: list[float],
    ) -> dict[str, float]:
        if not values:
            raise ValueError(
                "Cannot summarize an empty metric list."
            )

        array = np.asarray(
            values,
            dtype=float,
        )

        if not np.all(
            np.isfinite(array)
        ):
            raise ValueError(
                "Metric list contains non-finite values."
            )

        return {
            "mean": round(
                float(
                    np.mean(array)
                ),
                6,
            ),
            "std": round(
                float(
                    np.std(
                        array,
                        ddof=0,
                    )
                ),
                6,
            ),
            "min": round(
                float(
                    np.min(array)
                ),
                6,
            ),
            "max": round(
                float(
                    np.max(array)
                ),
                6,
            ),
        }

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
                "No modern Loto draws were found."
            )

        return list(
            draws
        )

    @classmethod
    def _validate_window_result(
        cls,
        result: dict[str, Any],
        expected_draws: list[Draw],
        window_number: int,
        test_draws_per_window: int,
    ) -> None:
        if result.get(
            "status"
        ) != "success":
            raise ValueError(
                f"Window {window_number}: "
                "walk-forward execution did not succeed."
            )

        if result.get(
            "evaluated_draws"
        ) != test_draws_per_window:
            raise ValueError(
                f"Window {window_number}: "
                "unexpected evaluated draw count."
            )

        details = result.get(
            "details",
            [],
        )

        if len(
            details
        ) != test_draws_per_window:
            raise ValueError(
                f"Window {window_number}: "
                "unexpected detail-row count."
            )

        expected_start = str(
            expected_draws[
                0
            ].draw_date
        )

        expected_end = str(
            expected_draws[
                -1
            ].draw_date
        )

        actual_start = str(
            details[
                0
            ][
                "draw_date"
            ]
        )

        actual_end = str(
            details[
                -1
            ][
                "draw_date"
            ]
        )

        if (
            actual_start
            != expected_start
            or actual_end
            != expected_end
        ):
            raise ValueError(
                f"Window {window_number}: "
                "evaluated dates do not match the "
                "requested chronological window."
            )

        for detail_position, detail in enumerate(
            details,
            start=1,
        ):
            probabilities = detail.get(
                "probabilities",
                {},
            )

            ranking = detail.get(
                "ranking",
                [],
            )

            predicted_top_5 = detail.get(
                "predicted_top_5",
                [],
            )

            if len(
                probabilities
            ) != 49:
                raise ValueError(
                    f"Window {window_number}, "
                    f"detail {detail_position}: "
                    "probability vector must contain 49 values."
                )

            if len(
                ranking
            ) != 49:
                raise ValueError(
                    f"Window {window_number}, "
                    f"detail {detail_position}: "
                    "ranking must contain 49 candidates."
                )

            if (
                len(predicted_top_5)
                != 5
                or len(
                    set(
                        predicted_top_5
                    )
                )
                != 5
            ):
                raise ValueError(
                    f"Window {window_number}, "
                    f"detail {detail_position}: "
                    "Top-5 must contain five unique numbers."
                )

            last_allowed = str(
                detail[
                    "last_allowed_feature_draw"
                ]
            )

            excluded_previous = str(
                detail[
                    "excluded_previous_draw"
                ]
            )

            target_date = str(
                detail[
                    "draw_date"
                ]
            )

            last_training_target = str(
                detail[
                    "last_training_target_available"
                ]
            )

            if not (
                last_allowed
                < excluded_previous
                < target_date
            ):
                raise ValueError(
                    f"Window {window_number}, "
                    f"detail {detail_position}: "
                    "invalid T-2/T-1/T chronology."
                )

            if not (
                last_training_target
                < excluded_previous
            ):
                raise ValueError(
                    f"Window {window_number}, "
                    f"detail {detail_position}: "
                    "T-1 was not purged from training targets."
                )

    @classmethod
    def run(
        cls,
        db: Any,
        test_draws_per_window: int = 20,
        number_of_windows: int = 3,
        window_size: int = 100,
        max_training_targets: int = 1500,
        monte_carlo_simulations: int = 1000,
    ) -> dict[str, Any]:
        if test_draws_per_window < 5:
            raise ValueError(
                "test_draws_per_window must be at least 5."
            )

        if number_of_windows < 1:
            raise ValueError(
                "number_of_windows must be positive."
            )

        if window_size < 100:
            raise ValueError(
                "window_size must be at least 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        if monte_carlo_simulations < 100:
            raise ValueError(
                "monte_carlo_simulations must be at least 100."
            )

        all_draws = cls._load_draws(
            db
        )

        required_test_draws = (
            test_draws_per_window
            * number_of_windows
        )

        minimum_required = (
            window_size
            + V6BCleanWalkForwardBacktester.LAG_DRAWS
            + V6BCleanWalkForwardBacktester.PURGE_DRAWS
            + required_test_draws
            + 1
        )

        if len(
            all_draws
        ) < minimum_required:
            raise ValueError(
                "Not enough draws for V6B-CLEAN "
                "multi-window validation. "
                f"Required at least {minimum_required}, "
                f"received {len(all_draws)}."
            )

        first_window_start = (
            len(all_draws)
            - required_test_draws
        )

        window_results: list[
            dict[str, Any]
        ] = []

        all_details: list[
            dict[str, Any]
        ] = []

        for window_number in range(
            1,
            number_of_windows + 1,
        ):
            window_start_index = (
                first_window_start
                + (
                    window_number - 1
                )
                * test_draws_per_window
            )

            window_end_index = (
                window_start_index
                + test_draws_per_window
            )

            expected_window_draws = all_draws[
                window_start_index:
                window_end_index
            ]

            if (
                len(expected_window_draws)
                != test_draws_per_window
            ):
                raise ValueError(
                    f"Window {window_number} has "
                    "an invalid size."
                )

            truncated_draws = all_draws[
                :window_end_index
            ]

            truncated_db = _TruncatedDB(
                truncated_draws
            )

            result = (
                V6BCleanWalkForwardBacktester
                .run(
                    db=truncated_db,
                    test_draws=(
                        test_draws_per_window
                    ),
                    window_size=(
                        window_size
                    ),
                    max_training_targets=(
                        max_training_targets
                    ),
                    monte_carlo_simulations=(
                        monte_carlo_simulations
                    ),
                )
            )

            cls._validate_window_result(
                result=result,
                expected_draws=(
                    expected_window_draws
                ),
                window_number=(
                    window_number
                ),
                test_draws_per_window=(
                    test_draws_per_window
                ),
            )

            details = result[
                "details"
            ]

            window_label = (
                f"window_{window_number}"
            )

            window_start_date = str(
                expected_window_draws[
                    0
                ].draw_date
            )

            window_end_date = str(
                expected_window_draws[
                    -1
                ].draw_date
            )

            for detail in details:
                detail[
                    "multi_window_label"
                ] = window_label

                detail[
                    "multi_window_number"
                ] = window_number

            result[
                "multi_window_label"
            ] = window_label

            result[
                "multi_window_number"
            ] = window_number

            result[
                "multi_window_start_date"
            ] = window_start_date

            result[
                "multi_window_end_date"
            ] = window_end_date

            window_results.append(
                {
                    "window": (
                        window_label
                    ),
                    "window_number": (
                        window_number
                    ),
                    "start_date": (
                        window_start_date
                    ),
                    "end_date": (
                        window_end_date
                    ),
                    "evaluated_draws": (
                        result[
                            "evaluated_draws"
                        ]
                    ),
                    "average_hits_at_5": (
                        result[
                            "model"
                        ][
                            "average_hits_at_5"
                        ]
                    ),
                    "total_hits": (
                        result[
                            "model"
                        ][
                            "total_hits"
                        ]
                    ),
                    "at_least_1_hit_rate": (
                        result[
                            "model"
                        ][
                            "at_least_1_hit_rate"
                        ]
                    ),
                    "at_least_2_hit_rate": (
                        result[
                            "model"
                        ][
                            "at_least_2_hit_rate"
                        ]
                    ),
                    "frequency_average_hits_at_5": (
                        result[
                            "frequency_baseline"
                        ][
                            "average_hits_at_5"
                        ]
                    ),
                    "frequency_total_hits": (
                        result[
                            "frequency_baseline"
                        ][
                            "total_hits"
                        ]
                    ),
                    "previous_draw_average_hits_at_5": (
                        result[
                            "previous_draw_baseline"
                        ][
                            "average_hits_at_5"
                        ]
                    ),
                    "previous_draw_total_hits": (
                        result[
                            "previous_draw_baseline"
                        ][
                            "total_hits"
                        ]
                    ),
                    "monte_carlo_p_value": (
                        result[
                            "random_baseline"
                        ][
                            "monte_carlo"
                        ][
                            "empirical_p_value"
                        ]
                    ),
                    "result": (
                        result
                    ),
                }
            )

            all_details.extend(
                details
            )

        model_window_hits = [
            float(
                window[
                    "average_hits_at_5"
                ]
            )
            for window in window_results
        ]

        frequency_window_hits = [
            float(
                window[
                    "frequency_average_hits_at_5"
                ]
            )
            for window in window_results
        ]

        previous_window_hits = [
            float(
                window[
                    "previous_draw_average_hits_at_5"
                ]
            )
            for window in window_results
        ]

        model_summary = (
            cls._metric_summary(
                model_window_hits
            )
        )

        frequency_summary = (
            cls._metric_summary(
                frequency_window_hits
            )
        )

        previous_summary = (
            cls._metric_summary(
                previous_window_hits
            )
        )

        total_evaluated_draws = sum(
            int(
                window[
                    "evaluated_draws"
                ]
            )
            for window in window_results
        )

        total_hits = sum(
            int(
                window[
                    "total_hits"
                ]
            )
            for window in window_results
        )

        total_frequency_hits = sum(
            int(
                window[
                    "frequency_total_hits"
                ]
            )
            for window in window_results
        )

        total_previous_hits = sum(
            int(
                window[
                    "previous_draw_total_hits"
                ]
            )
            for window in window_results
        )

        weighted_average_hits = (
            total_hits
            / total_evaluated_draws
        )

        weighted_frequency_hits = (
            total_frequency_hits
            / total_evaluated_draws
        )

        weighted_previous_hits = (
            total_previous_hits
            / total_evaluated_draws
        )

        if len(
            all_details
        ) != total_evaluated_draws:
            raise ValueError(
                "Unexpected combined detail-row count."
            )

        return {
            "status": "success",
            "version": cls.VERSION,
            "evaluation_type": (
                "multi_window_strict_walk_forward_purged"
            ),
            "architecture": (
                "single_global_candidate_ranking_model"
            ),
            "number_of_windows": (
                number_of_windows
            ),
            "test_draws_per_window": (
                test_draws_per_window
            ),
            "total_evaluated_draws": (
                total_evaluated_draws
            ),
            "window_size": (
                window_size
            ),
            "max_training_targets": (
                max_training_targets
            ),
            "monte_carlo_simulations": (
                monte_carlo_simulations
            ),
            "feature_count": 12,
            "candidate_count": 49,
            "top_k": 5,
            "probability_vectors_included": True,
            "probability_vector_size": 49,
            "random_expectation": round(
                cls.RANDOM_EXPECTATION,
                8,
            ),
            "model": {
                "average_hits_at_5": round(
                    weighted_average_hits,
                    6,
                ),
                "window_mean_hits_at_5": (
                    model_summary[
                        "mean"
                    ]
                ),
                "window_std_hits_at_5": (
                    model_summary[
                        "std"
                    ]
                ),
                "minimum_window_hits_at_5": (
                    model_summary[
                        "min"
                    ]
                ),
                "maximum_window_hits_at_5": (
                    model_summary[
                        "max"
                    ]
                ),
                "total_hits": (
                    total_hits
                ),
            },
            "frequency_baseline": {
                "average_hits_at_5": round(
                    weighted_frequency_hits,
                    6,
                ),
                "window_mean_hits_at_5": (
                    frequency_summary[
                        "mean"
                    ]
                ),
                "window_std_hits_at_5": (
                    frequency_summary[
                        "std"
                    ]
                ),
                "minimum_window_hits_at_5": (
                    frequency_summary[
                        "min"
                    ]
                ),
                "maximum_window_hits_at_5": (
                    frequency_summary[
                        "max"
                    ]
                ),
                "total_hits": (
                    total_frequency_hits
                ),
            },
            "previous_draw_baseline": {
                "average_hits_at_5": round(
                    weighted_previous_hits,
                    6,
                ),
                "window_mean_hits_at_5": (
                    previous_summary[
                        "mean"
                    ]
                ),
                "window_std_hits_at_5": (
                    previous_summary[
                        "std"
                    ]
                ),
                "minimum_window_hits_at_5": (
                    previous_summary[
                        "min"
                    ]
                ),
                "maximum_window_hits_at_5": (
                    previous_summary[
                        "max"
                    ]
                ),
                "total_hits": (
                    total_previous_hits
                ),
            },
            "comparison": {
                "absolute_lift_vs_random": round(
                    weighted_average_hits
                    - cls.RANDOM_EXPECTATION,
                    6,
                ),
                "absolute_lift_vs_frequency": round(
                    weighted_average_hits
                    - weighted_frequency_hits,
                    6,
                ),
                "absolute_lift_vs_previous_draw": round(
                    weighted_average_hits
                    - weighted_previous_hits,
                    6,
                ),
                "beats_random_expectation": (
                    weighted_average_hits
                    > cls.RANDOM_EXPECTATION
                ),
                "beats_frequency": (
                    weighted_average_hits
                    > weighted_frequency_hits
                ),
                "beats_previous_draw": (
                    weighted_average_hits
                    > weighted_previous_hits
                ),
            },
            "window_results": (
                window_results
            ),
            "details": (
                all_details
            ),
        }
