from math import comb
from typing import Any


class V5DRankDiagnostics:
    """
    Predixa AI V5-D Rank Diagnostics.

    Evaluates whether the ordering produced by the model
    contains useful information independently from probability
    calibration.

    Metrics:
        - hit rate at each exact rank
        - cumulative total hits for Top-K
        - probability of >=1 hit for Top-K
        - theoretical random baselines
        - lift versus random
        - concentration of observed hits by rank

    No model training is performed here.
    """

    VERSION = "V5-D-RANK-DIAGNOSTICS"

    NUMBER_COUNT = 49
    WINNING_NUMBERS = 5
    MAX_K = 5

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def _validate(
        cls,
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:

        if not isinstance(result, dict):
            raise ValueError(
                "Result must be a dictionary."
            )

        details = result.get("details")

        if not isinstance(details, list):
            raise ValueError(
                "Result does not contain a valid details list."
            )

        if not details:
            raise ValueError(
                "Result details are empty."
            )

        for index, detail in enumerate(
            details,
            start=1,
        ):
            predicted = detail.get(
                "predicted_top_5"
            )

            actual = detail.get(
                "actual_numbers"
            )

            if not isinstance(predicted, list):
                raise ValueError(
                    f"Detail {index} has no predicted_top_5."
                )

            if len(predicted) != cls.MAX_K:
                raise ValueError(
                    f"Detail {index} must contain "
                    f"{cls.MAX_K} predicted numbers."
                )

            if len(set(predicted)) != cls.MAX_K:
                raise ValueError(
                    f"Detail {index} contains duplicate predictions."
                )

            if not isinstance(actual, list):
                raise ValueError(
                    f"Detail {index} has no actual_numbers."
                )

            if len(actual) != cls.WINNING_NUMBERS:
                raise ValueError(
                    f"Detail {index} must contain "
                    f"{cls.WINNING_NUMBERS} actual numbers."
                )

        return details

    # ==========================================================
    # RANDOM BASELINES
    # ==========================================================

    @classmethod
    def _random_expected_hits(
        cls,
        k: int,
    ) -> float:
        """
        Expected number of matches when selecting k numbers
        uniformly from 49 against 5 winning numbers.
        """

        return (
            k
            * cls.WINNING_NUMBERS
            / cls.NUMBER_COUNT
        )

    @classmethod
    def _random_probability_at_least_one(
        cls,
        k: int,
    ) -> float:
        """
        Hypergeometric probability:

            P(at least one hit)
            = 1 - P(zero hits)
        """

        zero_hits = (
            comb(
                cls.NUMBER_COUNT
                - cls.WINNING_NUMBERS,
                k,
            )
            /
            comb(
                cls.NUMBER_COUNT,
                k,
            )
        )

        return 1.0 - zero_hits

    # ==========================================================
    # ANALYSIS
    # ==========================================================

    @classmethod
    def analyze(
        cls,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        details = cls._validate(
            result
        )

        evaluated = len(details)

        exact_rank_hits = {
            rank: 0
            for rank in range(
                1,
                cls.MAX_K + 1,
            )
        }

        cumulative_total_hits = {
            k: 0
            for k in range(
                1,
                cls.MAX_K + 1,
            )
        }

        cumulative_draws_with_hit = {
            k: 0
            for k in range(
                1,
                cls.MAX_K + 1,
            )
        }

        for detail in details:
            predicted = detail[
                "predicted_top_5"
            ]

            actual = set(
                detail["actual_numbers"]
            )

            for rank, number in enumerate(
                predicted,
                start=1,
            ):
                if number in actual:
                    exact_rank_hits[
                        rank
                    ] += 1

            for k in range(
                1,
                cls.MAX_K + 1,
            ):
                hits = sum(
                    1
                    for number in predicted[:k]
                    if number in actual
                )

                cumulative_total_hits[
                    k
                ] += hits

                if hits >= 1:
                    cumulative_draws_with_hit[
                        k
                    ] += 1

        total_top_5_hits = (
            cumulative_total_hits[
                cls.MAX_K
            ]
        )

        exact_ranks = {}

        random_single_rank_rate = (
            cls.WINNING_NUMBERS
            / cls.NUMBER_COUNT
        )

        for rank in range(
            1,
            cls.MAX_K + 1,
        ):
            hits = exact_rank_hits[
                rank
            ]

            observed_rate = (
                hits
                / evaluated
            )

            exact_ranks[
                f"rank_{rank}"
            ] = {
                "hits": hits,

                "observed_rate": round(
                    observed_rate,
                    8,
                ),

                "random_rate": round(
                    random_single_rank_rate,
                    8,
                ),

                "rate_lift_vs_random": round(
                    observed_rate
                    - random_single_rank_rate,
                    8,
                ),

                "share_of_all_model_hits": round(
                    (
                        hits
                        / total_top_5_hits
                    )
                    if total_top_5_hits
                    else 0.0,
                    8,
                ),
            }

        cumulative = {}

        for k in range(
            1,
            cls.MAX_K + 1,
        ):
            observed_total_hits = (
                cumulative_total_hits[k]
            )

            observed_average_hits = (
                observed_total_hits
                / evaluated
            )

            expected_random_hits = (
                cls._random_expected_hits(
                    k
                )
            )

            observed_at_least_one = (
                cumulative_draws_with_hit[
                    k
                ]
                / evaluated
            )

            random_at_least_one = (
                cls._random_probability_at_least_one(
                    k
                )
            )

            cumulative[
                f"top_{k}"
            ] = {
                "total_hits": (
                    observed_total_hits
                ),

                "average_hits": round(
                    observed_average_hits,
                    8,
                ),

                "random_expected_average_hits": round(
                    expected_random_hits,
                    8,
                ),

                "average_hits_lift_vs_random": round(
                    observed_average_hits
                    - expected_random_hits,
                    8,
                ),

                "draws_with_at_least_1_hit": (
                    cumulative_draws_with_hit[
                        k
                    ]
                ),

                "observed_at_least_1_rate": round(
                    observed_at_least_one,
                    8,
                ),

                "random_at_least_1_rate": round(
                    random_at_least_one,
                    8,
                ),

                "at_least_1_lift_vs_random": round(
                    observed_at_least_one
                    - random_at_least_one,
                    8,
                ),
            }

        first_three_hits = sum(
            exact_rank_hits[rank]
            for rank in (
                1,
                2,
                3,
            )
        )

        last_two_hits = sum(
            exact_rank_hits[rank]
            for rank in (
                4,
                5,
            )
        )

        return {
            "status": "success",

            "version": cls.VERSION,

            "source_version": result.get(
                "version"
            ),

            "evaluated_draws": (
                evaluated
            ),

            "total_top_5_hits": (
                total_top_5_hits
            ),

            "exact_rank_results": (
                exact_ranks
            ),

            "cumulative_results": (
                cumulative
            ),

            "rank_group_summary": {
                "ranks_1_to_3_hits": (
                    first_three_hits
                ),

                "ranks_4_to_5_hits": (
                    last_two_hits
                ),

                "ranks_1_to_3_share": round(
                    (
                        first_three_hits
                        / total_top_5_hits
                    )
                    if total_top_5_hits
                    else 0.0,
                    8,
                ),

                "ranks_4_to_5_share": round(
                    (
                        last_two_hits
                        / total_top_5_hits
                    )
                    if total_top_5_hits
                    else 0.0,
                    8,
                ),
            },
        }
