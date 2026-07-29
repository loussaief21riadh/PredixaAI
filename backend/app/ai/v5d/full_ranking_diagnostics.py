from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


class V5DFullRankingDiagnostics:
    """
    Predixa AI V5-D - Full 49-number ranking diagnostics.

    Evaluates whether the complete 49-number score ranking
    contains information about the 5 winning numbers.

    No model training is performed.

    Metrics:
        - ROC-AUC per draw
        - Average Precision per draw
        - Mean / median / std
        - Fraction of draws above random ROC-AUC = 0.50
        - Average Precision compared with prevalence = 5/49
        - Top-K hit metrics
    """

    VERSION = "V5-D-FULL-RANKING-DIAGNOSTICS"

    NUMBER_COUNT = 49
    WINNING_NUMBERS = 5

    RANDOM_ROC_AUC = 0.5
    RANDOM_AP = WINNING_NUMBERS / NUMBER_COUNT

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

        if not isinstance(details, list) or not details:
            raise ValueError(
                "Result must contain a non-empty details list."
            )

        for index, detail in enumerate(
            details,
            start=1,
        ):
            probabilities = detail.get(
                "probabilities"
            )

            actual = detail.get(
                "actual_numbers"
            )

            if not isinstance(probabilities, dict):
                raise ValueError(
                    f"Detail {index} has no probabilities."
                )

            if len(probabilities) != cls.NUMBER_COUNT:
                raise ValueError(
                    f"Detail {index}: expected 49 scores, "
                    f"received {len(probabilities)}."
                )

            if not isinstance(actual, list):
                raise ValueError(
                    f"Detail {index} has no actual_numbers."
                )

            if len(actual) != cls.WINNING_NUMBERS:
                raise ValueError(
                    f"Detail {index}: expected 5 actual numbers."
                )

        return details

    @staticmethod
    def _score(
        probabilities: dict,
        number: int,
    ) -> float:

        value = probabilities.get(number)

        if value is None:
            value = probabilities.get(
                str(number)
            )

        if value is None:
            raise ValueError(
                f"Missing score for number {number}."
            )

        return float(value)

    @classmethod
    def analyze(
        cls,
        result: dict[str, Any],
    ) -> dict[str, Any]:

        details = cls._validate(
            result
        )

        auc_values = []
        ap_values = []

        top_k_hits = {
            k: 0
            for k in range(1, 6)
        }

        per_draw = []

        for detail in details:
            actual = set(
                detail["actual_numbers"]
            )

            probabilities = detail[
                "probabilities"
            ]

            y_true = np.array(
                [
                    1 if number in actual else 0
                    for number in range(
                        1,
                        cls.NUMBER_COUNT + 1,
                    )
                ],
                dtype=int,
            )

            y_score = np.array(
                [
                    cls._score(
                        probabilities,
                        number,
                    )
                    for number in range(
                        1,
                        cls.NUMBER_COUNT + 1,
                    )
                ],
                dtype=float,
            )

            auc = float(
                roc_auc_score(
                    y_true,
                    y_score,
                )
            )

            ap = float(
                average_precision_score(
                    y_true,
                    y_score,
                )
            )

            auc_values.append(auc)
            ap_values.append(ap)

            ranking = sorted(
                range(
                    1,
                    cls.NUMBER_COUNT + 1,
                ),
                key=lambda number: (
                    -cls._score(
                        probabilities,
                        number,
                    ),
                    number,
                ),
            )

            draw_top_k = {}

            for k in range(1, 6):
                hits = len(
                    set(ranking[:k])
                    & actual
                )

                top_k_hits[k] += hits

                draw_top_k[
                    f"top_{k}_hits"
                ] = hits

            per_draw.append(
                {
                    "draw_date": detail.get(
                        "draw_date"
                    ),
                    "roc_auc": round(
                        auc,
                        8,
                    ),
                    "average_precision": round(
                        ap,
                        8,
                    ),
                    **draw_top_k,
                }
            )

        auc_array = np.asarray(
            auc_values,
            dtype=float,
        )

        ap_array = np.asarray(
            ap_values,
            dtype=float,
        )

        evaluated = len(details)

        cumulative = {}

        for k in range(1, 6):
            observed = (
                top_k_hits[k]
                / evaluated
            )

            random_expected = (
                k
                * cls.WINNING_NUMBERS
                / cls.NUMBER_COUNT
            )

            cumulative[
                f"top_{k}"
            ] = {
                "total_hits": (
                    top_k_hits[k]
                ),
                "average_hits": round(
                    observed,
                    8,
                ),
                "random_expected": round(
                    random_expected,
                    8,
                ),
                "lift": round(
                    observed
                    - random_expected,
                    8,
                ),
            }

        return {
            "status": "success",
            "version": cls.VERSION,
            "evaluated_draws": evaluated,

            "roc_auc": {
                "random_baseline": (
                    cls.RANDOM_ROC_AUC
                ),
                "mean": round(
                    float(np.mean(auc_array)),
                    8,
                ),
                "median": round(
                    float(np.median(auc_array)),
                    8,
                ),
                "std": round(
                    float(
                        np.std(
                            auc_array,
                            ddof=1,
                        )
                    ),
                    8,
                ),
                "minimum": round(
                    float(np.min(auc_array)),
                    8,
                ),
                "maximum": round(
                    float(np.max(auc_array)),
                    8,
                ),
                "draws_above_0_5": int(
                    np.sum(
                        auc_array > 0.5
                    )
                ),
                "draws_below_0_5": int(
                    np.sum(
                        auc_array < 0.5
                    )
                ),
                "fraction_above_0_5": round(
                    float(
                        np.mean(
                            auc_array > 0.5
                        )
                    ),
                    8,
                ),
            },

            "average_precision": {
                "random_prevalence_baseline": round(
                    cls.RANDOM_AP,
                    8,
                ),
                "mean": round(
                    float(np.mean(ap_array)),
                    8,
                ),
                "median": round(
                    float(np.median(ap_array)),
                    8,
                ),
                "std": round(
                    float(
                        np.std(
                            ap_array,
                            ddof=1,
                        )
                    ),
                    8,
                ),
                "minimum": round(
                    float(np.min(ap_array)),
                    8,
                ),
                "maximum": round(
                    float(np.max(ap_array)),
                    8,
                ),
                "mean_minus_prevalence": round(
                    float(
                        np.mean(ap_array)
                        - cls.RANDOM_AP
                    ),
                    8,
                ),
            },

            "top_k": cumulative,

            "per_draw": per_draw,
        }
