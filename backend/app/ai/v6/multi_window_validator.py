from typing import Any

import numpy as np

from app.ai.v6.walk_forward_backtester import (
    V6WalkForwardBacktester,
)
from app.models.draw import Draw


class V6MultiWindowValidator:
    """
    Predixa AI V6 - Multi-Window Temporal Validator.

    Runs V6 on several chronological, non-overlapping windows.

    For every window:
        - only draws available up to the end of that window are exposed
          to the V6 walk-forward backtester;
        - V6 keeps its strict T-2 prediction protocol;
        - T-1 remains purged from training;
        - windows do not overlap.

    This class does not modify V5.
    """

    VERSION = "V6-MULTI-WINDOW-PURGED-T2"

    MODERN_LOTO_START_DATE = "2008-10-06"

    @classmethod
    def run(
        cls,
        db,
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

        all_draws = (
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

        required_test_draws = (
            test_draws_per_window
            * number_of_windows
        )

        minimum_required = (
            window_size
            + required_test_draws
            + 2
        )

        if len(all_draws) < minimum_required:
            raise ValueError(
                "Not enough draws for V6 multi-window validation."
            )

        first_window_start = (
            len(all_draws)
            - required_test_draws
        )

        window_results = []

        all_details = []

        # ======================================================
        # WINDOWS
        # ======================================================

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

            # --------------------------------------------------
            # Temporal truncation
            #
            # The existing V6 backtester always evaluates the
            # last N draws visible to it.
            #
            # Therefore, expose only history through this
            # window's final target draw.
            # --------------------------------------------------

            truncated_draws = all_draws[
                :window_end_index
            ]

            # --------------------------------------------------
            # Temporarily provide a query-like DB wrapper
            # containing only the truncated chronological data.
            # --------------------------------------------------

            class _TruncatedQuery:
                def __init__(
                    self,
                    draws,
                ):
                    self._draws = draws

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
                    return list(
                        self._draws
                    )

            class _TruncatedDB:
                def __init__(
                    self,
                    draws,
                ):
                    self._draws = draws

                def query(
                    self,
                    *args,
                    **kwargs,
                ):
                    return _TruncatedQuery(
                        self._draws
                    )

            truncated_db = _TruncatedDB(
                truncated_draws
            )

            result = (
                V6WalkForwardBacktester.run(
                    db=truncated_db,
                    test_draws=(
                        test_draws_per_window
                    ),
                    window_size=window_size,
                    max_training_targets=(
                        max_training_targets
                    ),
                    monte_carlo_simulations=(
                        monte_carlo_simulations
                    ),
                )
            )

            details = result.get(
                "details",
                [],
            )

            if (
                len(details)
                != test_draws_per_window
            ):
                raise ValueError(
                    f"Window {window_number}: "
                    "unexpected evaluated draw count."
                )

            expected_start = str(
                expected_window_draws[
                    0
                ].draw_date
            )

            expected_end = str(
                expected_window_draws[
                    -1
                ].draw_date
            )

            actual_start = (
                details[
                    0
                ][
                    "draw_date"
                ]
            )

            actual_end = (
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
                    "temporal truncation mismatch. "
                    f"Expected "
                    f"{expected_start} -> {expected_end}, "
                    f"received "
                    f"{actual_start} -> {actual_end}."
                )

            total_hits = int(
                result[
                    "model"
                ][
                    "total_hits"
                ]
            )

            hits_at_5 = (
                total_hits
                / test_draws_per_window
            )

            frequency_hits = int(
                result[
                    "frequency_baseline"
                ][
                    "total_hits"
                ]
            )

            previous_hits = int(
                result[
                    "previous_draw_baseline"
                ][
                    "total_hits"
                ]
            )

            window_name = (
                f"window_{window_number}"
            )

            for detail in details:
                detail_copy = dict(
                    detail
                )

                detail_copy[
                    "v6_window"
                ] = window_name

                all_details.append(
                    detail_copy
                )

            window_results.append(
                {
                    "window": window_name,

                    "start_date": (
                        actual_start
                    ),

                    "end_date": (
                        actual_end
                    ),

                    "evaluated_draws": (
                        test_draws_per_window
                    ),

                    "total_hits": (
                        total_hits
                    ),

                    "hits_at_5": round(
                        hits_at_5,
                        8,
                    ),

                    "frequency_total_hits": (
                        frequency_hits
                    ),

                    "frequency_hits_at_5": round(
                        frequency_hits
                        / test_draws_per_window,
                        8,
                    ),

                    "previous_total_hits": (
                        previous_hits
                    ),

                    "previous_hits_at_5": round(
                        previous_hits
                        / test_draws_per_window,
                        8,
                    ),

                    "details": (
                        details
                    ),
                }
            )

        # ======================================================
        # AGGREGATION
        # ======================================================

        total_evaluated = (
            test_draws_per_window
            * number_of_windows
        )

        total_hits = sum(
            row["total_hits"]
            for row in window_results
        )

        total_frequency_hits = sum(
            row["frequency_total_hits"]
            for row in window_results
        )

        total_previous_hits = sum(
            row["previous_total_hits"]
            for row in window_results
        )

        window_hit_rates = np.asarray(
            [
                row["hits_at_5"]
                for row in window_results
            ],
            dtype=float,
        )

        weighted_hits_at_5 = (
            total_hits
            / total_evaluated
        )

        random_expectation = (
            V6WalkForwardBacktester
            .RANDOM_EXPECTATION
        )

        return {
            "status": "success",

            "version": cls.VERSION,

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
                total_evaluated
            ),

            "window_size": (
                window_size
            ),

            "max_training_targets": (
                max_training_targets
            ),

            "summary": {
                "total_hits": (
                    total_hits
                ),

                "weighted_hits_at_5": round(
                    weighted_hits_at_5,
                    8,
                ),

                "mean_window_hits_at_5": round(
                    float(
                        np.mean(
                            window_hit_rates
                        )
                    ),
                    8,
                ),

                "std_window_hits_at_5": round(
                    float(
                        np.std(
                            window_hit_rates,
                            ddof=0,
                        )
                    ),
                    8,
                ),

                "minimum_window_hits_at_5": round(
                    float(
                        np.min(
                            window_hit_rates
                        )
                    ),
                    8,
                ),

                "maximum_window_hits_at_5": round(
                    float(
                        np.max(
                            window_hit_rates
                        )
                    ),
                    8,
                ),

                "random_expectation": round(
                    random_expectation,
                    8,
                ),

                "lift_vs_random": round(
                    weighted_hits_at_5
                    - random_expectation,
                    8,
                ),

                "frequency_total_hits": (
                    total_frequency_hits
                ),

                "frequency_hits_at_5": round(
                    total_frequency_hits
                    / total_evaluated,
                    8,
                ),

                "previous_total_hits": (
                    total_previous_hits
                ),

                "previous_hits_at_5": round(
                    total_previous_hits
                    / total_evaluated,
                    8,
                ),
            },

            "windows": (
                window_results
            ),

            "details": (
                all_details
            ),
        }
