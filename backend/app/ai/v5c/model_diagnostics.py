from collections import Counter
from statistics import mean
from typing import Any


class V5CModelDiagnostics:
    """
    Predixa AI V5-C Model Diagnostics.

    Analyse an existing walk-forward result without changing
    the model architecture or feature engineering.

    Diagnostics:
        - hit rate by prediction rank
        - cumulative Hits@1 ... Hits@5
        - predicted-number frequency
        - actual-number frequency
        - over-predicted / under-predicted numbers
        - prediction concentration
        - Top-5 diversity
        - previous-draw overlap diagnostics
        - optional probability calibration / Brier score
    """

    VERSION = "V5-C-DIAGNOSTICS"

    NUMBER_MIN = 1
    NUMBER_MAX = 49
    TOP_K = 5

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @staticmethod
    def _validate_result(
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:

        if not isinstance(result, dict):
            raise ValueError(
                "Walk-forward result must be a dictionary."
            )

        details = result.get("details")

        if not isinstance(details, list):
            raise ValueError(
                "Walk-forward result does not contain "
                "a valid details list."
            )

        if not details:
            raise ValueError(
                "Walk-forward details are empty."
            )

        for index, detail in enumerate(
            details,
            start=1,
        ):
            if not isinstance(detail, dict):
                raise ValueError(
                    f"Detail {index} is not a dictionary."
                )

            if "predicted_top_5" not in detail:
                raise ValueError(
                    f"Detail {index} is missing predicted_top_5."
                )

            if "actual_numbers" not in detail:
                raise ValueError(
                    f"Detail {index} is missing actual_numbers."
                )

            predicted = detail["predicted_top_5"]
            actual = detail["actual_numbers"]

            if len(predicted) != 5:
                raise ValueError(
                    f"Detail {index} does not contain "
                    "exactly 5 predicted numbers."
                )

            if len(actual) != 5:
                raise ValueError(
                    f"Detail {index} does not contain "
                    "exactly 5 actual numbers."
                )

        return details

    # ==========================================================
    # RANK DIAGNOSTICS
    # ==========================================================

    @staticmethod
    def _rank_diagnostics(
        details: list[dict[str, Any]],
    ) -> dict[str, Any]:

        total_draws = len(details)

        rank_hits = {
            rank: 0
            for rank in range(
                1,
                V5CModelDiagnostics.TOP_K + 1,
            )
        }

        cumulative_draw_hits = {
            k: 0
            for k in range(
                1,
                V5CModelDiagnostics.TOP_K + 1,
            )
        }

        cumulative_total_hits = {
            k: 0
            for k in range(
                1,
                V5CModelDiagnostics.TOP_K + 1,
            )
        }

        for detail in details:
            predicted = detail["predicted_top_5"]
            actual = set(detail["actual_numbers"])

            for rank, number in enumerate(
                predicted,
                start=1,
            ):
                if number in actual:
                    rank_hits[rank] += 1

            for k in range(
                1,
                V5CModelDiagnostics.TOP_K + 1,
            ):
                top_k = predicted[:k]

                hits = sum(
                    1
                    for number in top_k
                    if number in actual
                )

                cumulative_total_hits[k] += hits

                if hits >= 1:
                    cumulative_draw_hits[k] += 1

        by_rank = {}

        for rank in range(
            1,
            V5CModelDiagnostics.TOP_K + 1,
        ):
            hits = rank_hits[rank]

            by_rank[f"rank_{rank}"] = {
                "hits": hits,
                "hit_rate": round(
                    hits / total_draws,
                    6,
                ),
            }

        cumulative = {}

        for k in range(
            1,
            V5CModelDiagnostics.TOP_K + 1,
        ):
            cumulative[f"top_{k}"] = {
                "total_hits": (
                    cumulative_total_hits[k]
                ),
                "average_hits": round(
                    cumulative_total_hits[k]
                    / total_draws,
                    6,
                ),
                "draws_with_at_least_1_hit": (
                    cumulative_draw_hits[k]
                ),
                "at_least_1_hit_rate": round(
                    cumulative_draw_hits[k]
                    / total_draws,
                    6,
                ),
            }

        return {
            "by_rank": by_rank,
            "cumulative": cumulative,
        }

    # ==========================================================
    # NUMBER FREQUENCY
    # ==========================================================

    @staticmethod
    def _number_frequency_diagnostics(
        details: list[dict[str, Any]],
    ) -> dict[str, Any]:

        total_draws = len(details)

        predicted_counter = Counter()
        actual_counter = Counter()

        for detail in details:
            predicted_counter.update(
                detail["predicted_top_5"]
            )

            actual_counter.update(
                detail["actual_numbers"]
            )

        per_number = {}

        for number in range(
            V5CModelDiagnostics.NUMBER_MIN,
            V5CModelDiagnostics.NUMBER_MAX + 1,
        ):
            predicted_count = (
                predicted_counter.get(
                    number,
                    0,
                )
            )

            actual_count = (
                actual_counter.get(
                    number,
                    0,
                )
            )

            difference = (
                predicted_count
                - actual_count
            )

            per_number[number] = {
                "predicted_count": predicted_count,
                "actual_count": actual_count,
                "difference": difference,
                "predicted_rate": round(
                    predicted_count
                    / total_draws,
                    6,
                ),
                "actual_rate": round(
                    actual_count
                    / total_draws,
                    6,
                ),
            }

        most_predicted = sorted(
            per_number.items(),
            key=lambda item: (
                item[1]["predicted_count"],
                -item[0],
            ),
            reverse=True,
        )[:10]

        most_actual = sorted(
            per_number.items(),
            key=lambda item: (
                item[1]["actual_count"],
                -item[0],
            ),
            reverse=True,
        )[:10]

        most_over_predicted = sorted(
            per_number.items(),
            key=lambda item: (
                item[1]["difference"],
                item[1]["predicted_count"],
            ),
            reverse=True,
        )[:10]

        most_under_predicted = sorted(
            per_number.items(),
            key=lambda item: (
                item[1]["difference"],
                -item[1]["actual_count"],
            ),
        )[:10]

        def serialize(
            rows: list[
                tuple[
                    int,
                    dict[str, Any],
                ]
            ],
        ) -> list[dict[str, Any]]:

            return [
                {
                    "number": number,
                    **metrics,
                }
                for number, metrics
                in rows
            ]

        return {
            "per_number": per_number,

            "most_predicted": serialize(
                most_predicted
            ),

            "most_actual": serialize(
                most_actual
            ),

            "most_over_predicted": serialize(
                most_over_predicted
            ),

            "most_under_predicted": serialize(
                most_under_predicted
            ),
        }

    # ==========================================================
    # CONCENTRATION / DIVERSITY
    # ==========================================================

    @staticmethod
    def _concentration_diagnostics(
        details: list[dict[str, Any]],
    ) -> dict[str, Any]:

        predicted_counter = Counter()

        unique_top_5_sets = set()
        all_prediction_sets = []

        for detail in details:
            predicted = detail["predicted_top_5"]

            predicted_counter.update(
                predicted
            )

            normalized = tuple(
                sorted(predicted)
            )

            unique_top_5_sets.add(
                normalized
            )

            all_prediction_sets.append(
                set(predicted)
            )

        total_prediction_slots = (
            len(details)
            * V5CModelDiagnostics.TOP_K
        )

        top_5_numbers = (
            predicted_counter.most_common(5)
        )

        top_10_numbers = (
            predicted_counter.most_common(10)
        )

        top_5_share = (
            sum(
                count
                for _, count
                in top_5_numbers
            )
            / total_prediction_slots
        )

        top_10_share = (
            sum(
                count
                for _, count
                in top_10_numbers
            )
            / total_prediction_slots
        )

        consecutive_overlaps = []

        for index in range(
            1,
            len(all_prediction_sets),
        ):
            overlap = len(
                all_prediction_sets[index - 1]
                &
                all_prediction_sets[index]
            )

            consecutive_overlaps.append(
                overlap
            )

        average_consecutive_overlap = (
            mean(consecutive_overlaps)
            if consecutive_overlaps
            else 0.0
        )

        return {
            "unique_numbers_predicted": len(
                predicted_counter
            ),

            "unique_top_5_combinations": len(
                unique_top_5_sets
            ),

            "combination_uniqueness_rate": round(
                len(unique_top_5_sets)
                / len(details),
                6,
            ),

            "top_5_number_concentration": round(
                top_5_share,
                6,
            ),

            "top_10_number_concentration": round(
                top_10_share,
                6,
            ),

            "average_overlap_between_consecutive_predictions": round(
                average_consecutive_overlap,
                6,
            ),

            "most_common_predicted_numbers": [
                {
                    "number": number,
                    "count": count,
                    "share_of_prediction_slots": round(
                        count
                        / total_prediction_slots,
                        6,
                    ),
                }
                for number, count
                in predicted_counter.most_common(
                    10
                )
            ],
        }

    # ==========================================================
    # PREVIOUS DRAW
    # ==========================================================

    @staticmethod
    def _previous_draw_diagnostics(
        details: list[dict[str, Any]],
    ) -> dict[str, Any]:

        overlaps = []
        exact_copy_count = 0
        available = 0

        for detail in details:
            previous = detail.get(
                "previous_draw_top_5"
            )

            predicted = detail.get(
                "predicted_top_5"
            )

            if previous is None:
                continue

            available += 1

            overlap = len(
                set(previous)
                &
                set(predicted)
            )

            overlaps.append(overlap)

            if set(previous) == set(predicted):
                exact_copy_count += 1

        if available == 0:
            return {
                "available": False,
            }

        return {
            "available": True,

            "evaluated_draws": available,

            "average_overlap": round(
                mean(overlaps),
                6,
            ),

            "minimum_overlap": min(
                overlaps
            ),

            "maximum_overlap": max(
                overlaps
            ),

            "exact_copy_count": (
                exact_copy_count
            ),

            "exact_copy_rate": round(
                exact_copy_count
                / available,
                6,
            ),
        }

    # ==========================================================
    # PROBABILITY CALIBRATION
    # ==========================================================

    @staticmethod
    def _probability_diagnostics(
        details: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Requires complete probabilities for numbers 1..49
        on every evaluated draw.
        """

        probability_rows = []

        for detail in details:
            probabilities = detail.get(
                "probabilities"
            )

            if not isinstance(
                probabilities,
                dict,
            ):
                continue

            if len(probabilities) != 49:
                continue

            probability_rows.append(
                detail
            )

        if len(probability_rows) != len(details):
            return {
                "available": False,
                "reason": (
                    "Complete 49-number probability vectors "
                    "are not available for every evaluated draw."
                ),
                "brier_score": None,
                "calibration": None,
            }

        squared_errors = []
        calibration_pairs = []

        for detail in probability_rows:
            actual = set(
                detail["actual_numbers"]
            )

            probabilities = detail[
                "probabilities"
            ]

            for number in range(
                1,
                50,
            ):
                probability = probabilities.get(
                    number
                )

                if probability is None:
                    probability = probabilities.get(
                        str(number)
                    )

                if probability is None:
                    raise ValueError(
                        f"Missing probability for "
                        f"number {number}."
                    )

                probability = float(
                    probability
                )

                target = (
                    1.0
                    if number in actual
                    else 0.0
                )

                squared_errors.append(
                    (
                        probability
                        - target
                    )
                    ** 2
                )

                calibration_pairs.append(
                    (
                        probability,
                        target,
                    )
                )

        brier_score = mean(
            squared_errors
        )

        bins = [
            (0.00, 0.05),
            (0.05, 0.10),
            (0.10, 0.15),
            (0.15, 0.20),
            (0.20, 0.30),
            (0.30, 0.50),
            (0.50, 1.000001),
        ]

        calibration = []

        for lower, upper in bins:
            rows = [
                (
                    probability,
                    target,
                )
                for probability, target
                in calibration_pairs
                if lower
                <= probability
                < upper
            ]

            if not rows:
                continue

            calibration.append(
                {
                    "range": (
                        f"{lower:.2f}-"
                        f"{min(upper, 1.0):.2f}"
                    ),

                    "count": len(rows),

                    "mean_predicted_probability": round(
                        mean(
                            probability
                            for probability, _
                            in rows
                        ),
                        6,
                    ),

                    "observed_frequency": round(
                        mean(
                            target
                            for _, target
                            in rows
                        ),
                        6,
                    ),
                }
            )

        return {
            "available": True,

            "brier_score": round(
                brier_score,
                8,
            ),

            "calibration": calibration,
        }

    # ==========================================================
    # MAIN
    # ==========================================================

    @staticmethod
    def analyze(
        result: dict[str, Any],
    ) -> dict[str, Any]:

        details = (
            V5CModelDiagnostics
            ._validate_result(
                result
            )
        )

        total_hits = sum(
            len(
                set(
                    detail["predicted_top_5"]
                )
                &
                set(
                    detail["actual_numbers"]
                )
            )
            for detail in details
        )

        total_draws = len(details)

        return {
            "status": "success",

            "version": (
                V5CModelDiagnostics.VERSION
            ),

            "source_version": (
                result.get("version")
            ),

            "evaluated_draws": (
                total_draws
            ),

            "total_hits": (
                total_hits
            ),

            "average_hits_at_5": round(
                total_hits
                / total_draws,
                6,
            ),

            "rank_diagnostics": (
                V5CModelDiagnostics
                ._rank_diagnostics(
                    details
                )
            ),

            "number_diagnostics": (
                V5CModelDiagnostics
                ._number_frequency_diagnostics(
                    details
                )
            ),

            "concentration_diagnostics": (
                V5CModelDiagnostics
                ._concentration_diagnostics(
                    details
                )
            ),

            "previous_draw_diagnostics": (
                V5CModelDiagnostics
                ._previous_draw_diagnostics(
                    details
                )
            ),

            "probability_diagnostics": (
                V5CModelDiagnostics
                ._probability_diagnostics(
                    details
                )
            ),
        }
