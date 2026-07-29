from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from sqlalchemy.orm import Session

from app.ai.v5.walk_forward_backtester_v5b import (
    V5BWalkForwardBacktester,
)
from app.models.draw import Draw


@dataclass
class WindowDefinition:
    """
    One chronological evaluation window.
    """

    label: str
    start_index: int
    end_index: int


class V5CMultiWindowValidator:
    """
    Predixa AI V5-C Multi-Window Stability Validator.

    Goal:
        Compare selected V5-B variants across multiple
        non-overlapping historical evaluation windows.

    Candidates:
        - full
        - no_recency_ratio
        - rates_only

    Temporal protocol:
        - strict chronological walk-forward
        - T-2 prediction features
        - T-1 excluded from prediction features
        - T-1 target purged from training
    """

    VERSION = "V5-C-MULTI-WINDOW"

    DEFAULT_VARIANTS = (
        "full",
        "no_recency_ratio",
        "rates_only",
    )

    # Expected random Hits@5:
    #
    # 5 predicted numbers
    # × probability that one number appears in a 5/49 draw
    #
    # = 5 × (5 / 49)
    # = 25 / 49
    # ≈ 0.510204
    RANDOM_EXPECTATION = 25 / 49

    MODERN_LOTO_START_DATE = "2008-10-06"

    @staticmethod
    def _load_draws(
        db: Session,
    ) -> list[Draw]:

        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= V5CMultiWindowValidator.MODERN_LOTO_START_DATE
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

        return draws

    @staticmethod
    def _build_windows(
        total_draws: int,
        test_draws_per_window: int,
        number_of_windows: int,
    ) -> list[WindowDefinition]:

        if test_draws_per_window < 5:
            raise ValueError(
                "test_draws_per_window must be at least 5."
            )

        if number_of_windows < 2:
            raise ValueError(
                "number_of_windows must be at least 2."
            )

        required_evaluation_draws = (
            test_draws_per_window
            * number_of_windows
        )

        if total_draws <= required_evaluation_draws:
            raise ValueError(
                "Not enough draws for the requested "
                "multi-window validation."
            )

        windows: list[WindowDefinition] = []

        final_end = total_draws

        for reverse_index in range(
            number_of_windows
        ):

            end_index = (
                final_end
                - (
                    reverse_index
                    * test_draws_per_window
                )
            )

            start_index = (
                end_index
                - test_draws_per_window
            )

            if start_index < 0:
                raise ValueError(
                    "Historical window start became negative."
                )

            windows.append(
                WindowDefinition(
                    label=(
                        f"window_"
                        f"{number_of_windows - reverse_index}"
                    ),
                    start_index=start_index,
                    end_index=end_index,
                )
            )

        windows.reverse()

        return windows

    @staticmethod
    def _metric_summary(
        values: list[float],
    ) -> dict[str, float]:

        if not values:
            raise ValueError(
                "Cannot summarize an empty metric list."
            )

        return {
            "mean": round(
                mean(values),
                6,
            ),
            "std": round(
                pstdev(values),
                6,
            ),
            "min": round(
                min(values),
                6,
            ),
            "max": round(
                max(values),
                6,
            ),
        }

    @staticmethod
    def _variant_summary(
        window_results: list[dict[str, Any]],
    ) -> dict[str, Any]:

        if not window_results:
            raise ValueError(
                "No window results available."
            )

        hits = [
            result[
                "model"
            ][
                "average_hits_at_5"
            ]
            for result in window_results
        ]

        at_least_1 = [
            result[
                "model"
            ][
                "at_least_1_hit_rate"
            ]
            for result in window_results
        ]

        at_least_2 = [
            result[
                "model"
            ][
                "at_least_2_hit_rate"
            ]
            for result in window_results
        ]

        frequency = [
            result[
                "frequency_baseline"
            ][
                "average_hits_at_5"
            ]
            for result in window_results
        ]

        previous = [
            result[
                "previous_draw_baseline"
            ][
                "average_hits_at_5"
            ]
            for result in window_results
        ]

        total_hits = sum(
            result[
                "model"
            ][
                "total_hits"
            ]
            for result in window_results
        )

        total_evaluated = sum(
            result[
                "evaluated_draws"
            ]
            for result in window_results
        )

        weighted_average_hits = (
            total_hits
            / total_evaluated
        )

        mean_window_hits = mean(
            hits
        )

        return {
            "windows": len(
                window_results
            ),

            "total_evaluated_draws": (
                total_evaluated
            ),

            "total_hits": total_hits,

            "weighted_average_hits_at_5": round(
                weighted_average_hits,
                6,
            ),

            "average_hits_at_5": (
                V5CMultiWindowValidator
                ._metric_summary(
                    hits
                )
            ),

            "at_least_1_hit_rate": (
                V5CMultiWindowValidator
                ._metric_summary(
                    at_least_1
                )
            ),

            "at_least_2_hit_rate": (
                V5CMultiWindowValidator
                ._metric_summary(
                    at_least_2
                )
            ),

            "frequency_baseline": (
                V5CMultiWindowValidator
                ._metric_summary(
                    frequency
                )
            ),

            "previous_draw_baseline": (
                V5CMultiWindowValidator
                ._metric_summary(
                    previous
                )
            ),

            "random_expectation": round(
                V5CMultiWindowValidator
                .RANDOM_EXPECTATION,
                6,
            ),

            "mean_lift_vs_random": round(
                mean_window_hits
                - V5CMultiWindowValidator.RANDOM_EXPECTATION,
                6,
            ),

            "weighted_lift_vs_random": round(
                weighted_average_hits
                - V5CMultiWindowValidator.RANDOM_EXPECTATION,
                6,
            ),

            "mean_lift_vs_frequency": round(
                mean_window_hits
                - mean(
                    frequency
                ),
                6,
            ),

            "mean_lift_vs_previous_draw": round(
                mean_window_hits
                - mean(
                    previous
                ),
                6,
            ),
        }

    @staticmethod
    def _rank_variants(
        summaries: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:

        ranking: list[dict[str, Any]] = []

        for variant, summary in summaries.items():

            average_hits = (
                summary[
                    "average_hits_at_5"
                ][
                    "mean"
                ]
            )

            std_hits = (
                summary[
                    "average_hits_at_5"
                ][
                    "std"
                ]
            )

            minimum_hits = (
                summary[
                    "average_hits_at_5"
                ][
                    "min"
                ]
            )

            ranking.append(
                {
                    "variant": variant,

                    "average_hits_at_5": (
                        average_hits
                    ),

                    "weighted_average_hits_at_5": (
                        summary[
                            "weighted_average_hits_at_5"
                        ]
                    ),

                    "std_hits_at_5": (
                        std_hits
                    ),

                    "minimum_window_hits_at_5": (
                        minimum_hits
                    ),

                    "mean_lift_vs_random": (
                        summary[
                            "mean_lift_vs_random"
                        ]
                    ),
                }
            )

        ranking.sort(
            key=lambda item: (
                item[
                    "average_hits_at_5"
                ],
                -item[
                    "std_hits_at_5"
                ],
                item[
                    "minimum_window_hits_at_5"
                ],
            ),
            reverse=True,
        )

        for position, item in enumerate(
            ranking,
            start=1,
        ):
            item[
                "rank"
            ] = position

        return ranking

    @staticmethod
    def run(
        db: Session,
        test_draws_per_window: int = 100,
        number_of_windows: int = 3,
        window_size: int = 100,
        max_training_samples: int = 1500,
        monte_carlo_simulations: int = 10000,
        variants: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """
        Run V5-C multi-window stability validation.

        Windows are non-overlapping.

        Each historical window is evaluated using the
        existing V5-B purged walk-forward engine.
        """

        if variants is None:
            variants = (
                V5CMultiWindowValidator
                .DEFAULT_VARIANTS
            )

        if not variants:
            raise ValueError(
                "At least one variant is required."
            )

        valid_variants = (
            V5BWalkForwardBacktester
            .EXPECTED_FEATURE_COUNTS
        )

        for variant in variants:
            if variant not in valid_variants:
                raise ValueError(
                    f"Unknown V5-C variant: "
                    f"{variant}"
                )

        draws = (
            V5CMultiWindowValidator
            ._load_draws(
                db
            )
        )

        windows = (
            V5CMultiWindowValidator
            ._build_windows(
                total_draws=len(draws),
                test_draws_per_window=(
                    test_draws_per_window
                ),
                number_of_windows=(
                    number_of_windows
                ),
            )
        )

        variant_window_results: dict[
            str,
            list[dict[str, Any]],
        ] = {
            variant: []
            for variant in variants
        }

        window_metadata: list[
            dict[str, Any]
        ] = []

        for window in windows:

            truncated_draws = draws[
                :window.end_index
            ]

            evaluation_draws = draws[
                window.start_index:
                window.end_index
            ]

            if len(
                evaluation_draws
            ) != test_draws_per_window:

                raise ValueError(
                    "Unexpected evaluation "
                    "window size."
                )

            window_metadata.append(
                {
                    "label": window.label,

                    "start_date": str(
                        evaluation_draws[
                            0
                        ].draw_date
                    ),

                    "end_date": str(
                        evaluation_draws[
                            -1
                        ].draw_date
                    ),

                    "draw_count": len(
                        evaluation_draws
                    ),

                    "historical_draws_available": (
                        len(
                            truncated_draws
                        )
                    ),
                }
            )

            for variant in variants:

                original_query = db.query

                class DrawQueryProxy:
                    def __init__(
                        self,
                        rows: list[Draw],
                    ):
                        self.rows = rows

                    def filter(
                        self,
                        *args,
                        **kwargs,
                    ):
                        return self

                    def order_by(
                        self,
                        *args,
                        **kwargs,
                    ):
                        return self

                    def all(
                        self,
                    ):
                        return self.rows

                def query_override(
                    model,
                ):
                    if model is Draw:
                        return DrawQueryProxy(
                            truncated_draws
                        )

                    return original_query(
                        model
                    )

                db.query = query_override

                try:
                    result = (
                        V5BWalkForwardBacktester
                        .run(
                            db=db,

                            variant=variant,

                            test_draws=(
                                test_draws_per_window
                            ),

                            window_size=(
                                window_size
                            ),

                            max_training_samples=(
                                max_training_samples
                            ),

                            monte_carlo_simulations=(
                                monte_carlo_simulations
                            ),
                        )
                    )

                finally:
                    db.query = original_query

                result[
                    "v5c_window"
                ] = window.label

                result[
                    "v5c_window_start_date"
                ] = str(
                    evaluation_draws[
                        0
                    ].draw_date
                )

                result[
                    "v5c_window_end_date"
                ] = str(
                    evaluation_draws[
                        -1
                    ].draw_date
                )

                variant_window_results[
                    variant
                ].append(
                    result
                )

        summaries = {
            variant: (
                V5CMultiWindowValidator
                ._variant_summary(
                    results
                )
            )
            for variant, results
            in variant_window_results.items()
        }

        ranking = (
            V5CMultiWindowValidator
            ._rank_variants(
                summaries
            )
        )

        return {
            "status": "success",

            "version": (
                V5CMultiWindowValidator.VERSION
            ),

            "evaluation_type": (
                "multi_window_strict_walk_forward_purged"
            ),

            "variants": list(
                variants
            ),

            "number_of_windows": (
                number_of_windows
            ),

            "test_draws_per_window": (
                test_draws_per_window
            ),

            "total_evaluated_draws_per_variant": (
                number_of_windows
                * test_draws_per_window
            ),

            "window_size": (
                window_size
            ),

            "max_training_samples": (
                max_training_samples
            ),

            "monte_carlo_simulations": (
                monte_carlo_simulations
            ),

            "random_expectation": round(
                V5CMultiWindowValidator
                .RANDOM_EXPECTATION,
                6,
            ),

            "windows": (
                window_metadata
            ),

            "summaries": (
                summaries
            ),

            "ranking": (
                ranking
            ),

            "details": (
                variant_window_results
            ),
        }