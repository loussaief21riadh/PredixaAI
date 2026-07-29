from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


class V5DPermutationTest:
    """
    Predixa AI V5-D Statistical Permutation Test.

    Tests whether the observed model ranking performs better
    than random rankings under the exact 5-of-49 structure.

    Metrics:
        - Mean Hits@5
        - Mean ROC-AUC
        - Mean Average Precision

    Null hypothesis:
        The model scores contain no information about which
        5 numbers are winners.

    The actual winning sets are preserved.
    For every simulation, the score/ranking assignment is
    randomly permuted within each draw.

    No model training is performed.
    """

    VERSION = "V5-D-PERMUTATION-TEST"

    NUMBER_COUNT = 49
    WINNING_NUMBERS = 5
    TOP_K = 5

    RANDOM_STATE = 42

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
                    f"Detail {index} has no probability vector."
                )

            if len(probabilities) != cls.NUMBER_COUNT:
                raise ValueError(
                    f"Detail {index}: expected 49 probabilities."
                )

            if not isinstance(actual, list):
                raise ValueError(
                    f"Detail {index} has no actual_numbers."
                )

            if len(actual) != cls.WINNING_NUMBERS:
                raise ValueError(
                    f"Detail {index}: expected 5 actual numbers."
                )

            if len(set(actual)) != cls.WINNING_NUMBERS:
                raise ValueError(
                    f"Detail {index}: duplicate actual numbers."
                )

        return details

    @staticmethod
    def _get_score(
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

        value = float(value)

        if not np.isfinite(value):
            raise ValueError(
                f"Invalid score for number {number}."
            )

        return value

    @classmethod
    def _prepare(
        cls,
        details: list[dict[str, Any]],
    ):
        true_rows = []
        score_rows = []

        for detail in details:
            actual = set(
                detail["actual_numbers"]
            )

            y_true = np.array(
                [
                    1 if number in actual else 0
                    for number in range(
                        1,
                        cls.NUMBER_COUNT + 1,
                    )
                ],
                dtype=np.int8,
            )

            probabilities = detail[
                "probabilities"
            ]

            y_score = np.array(
                [
                    cls._get_score(
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

            true_rows.append(
                y_true
            )

            score_rows.append(
                y_score
            )

        return (
            np.vstack(true_rows),
            np.vstack(score_rows),
        )

    @classmethod
    def _metrics(
        cls,
        y_true: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[float, float, float]:

        draw_count = y_true.shape[0]

        total_hits = 0
        auc_values = []
        ap_values = []

        for index in range(
            draw_count
        ):
            true_row = y_true[
                index
            ]

            score_row = y_score[
                index
            ]

            ranking = np.lexsort(
                (
                    np.arange(
                        cls.NUMBER_COUNT
                    ),
                    -score_row,
                )
            )

            top_indices = ranking[
                :cls.TOP_K
            ]

            total_hits += int(
                true_row[
                    top_indices
                ].sum()
            )

            auc_values.append(
                roc_auc_score(
                    true_row,
                    score_row,
                )
            )

            ap_values.append(
                average_precision_score(
                    true_row,
                    score_row,
                )
            )

        mean_hits = (
            total_hits
            / draw_count
        )

        mean_auc = float(
            np.mean(
                auc_values
            )
        )

        mean_ap = float(
            np.mean(
                ap_values
            )
        )

        return (
            float(mean_hits),
            mean_auc,
            mean_ap,
        )

    @classmethod
    def run(
        cls,
        result: dict[str, Any],
        simulations: int = 10000,
        random_state: int | None = None,
    ) -> dict[str, Any]:

        if simulations < 100:
            raise ValueError(
                "simulations must be at least 100."
            )

        details = cls._validate(
            result
        )

        y_true, y_score = cls._prepare(
            details
        )

        observed_hits, observed_auc, observed_ap = (
            cls._metrics(
                y_true,
                y_score,
            )
        )

        seed = (
            cls.RANDOM_STATE
            if random_state is None
            else random_state
        )

        rng = np.random.default_rng(
            seed
        )

        simulated_hits = np.empty(
            simulations,
            dtype=float,
        )

        simulated_auc = np.empty(
            simulations,
            dtype=float,
        )

        simulated_ap = np.empty(
            simulations,
            dtype=float,
        )

        draw_count = y_true.shape[0]

        for simulation in range(
            simulations
        ):
            permuted_scores = np.empty_like(
                y_score
            )

            for draw_index in range(
                draw_count
            ):
                permutation = rng.permutation(
                    cls.NUMBER_COUNT
                )

                permuted_scores[
                    draw_index
                ] = y_score[
                    draw_index,
                    permutation
                ]

            (
                simulated_hits[simulation],
                simulated_auc[simulation],
                simulated_ap[simulation],
            ) = cls._metrics(
                y_true,
                permuted_scores,
            )

        def summarize(
            observed: float,
            simulated: np.ndarray,
        ) -> dict[str, Any]:

            empirical_p = (
                (
                    np.sum(
                        simulated >= observed
                    )
                    + 1
                )
                /
                (
                    simulations
                    + 1
                )
            )

            percentile = np.mean(
                simulated <= observed
            )

            return {
                "observed": round(
                    observed,
                    8,
                ),

                "null_mean": round(
                    float(
                        np.mean(simulated)
                    ),
                    8,
                ),

                "null_std": round(
                    float(
                        np.std(
                            simulated,
                            ddof=1,
                        )
                    ),
                    8,
                ),

                "null_95_lower": round(
                    float(
                        np.percentile(
                            simulated,
                            2.5,
                        )
                    ),
                    8,
                ),

                "null_95_upper": round(
                    float(
                        np.percentile(
                            simulated,
                            97.5,
                        )
                    ),
                    8,
                ),

                "percentile": round(
                    float(percentile),
                    8,
                ),

                "one_sided_p_value": round(
                    float(empirical_p),
                    8,
                ),

                "significant_0_05": bool(
                    empirical_p < 0.05
                ),
            }

        return {
            "status": "success",

            "version": cls.VERSION,

            "evaluated_draws": len(
                details
            ),

            "simulations": simulations,

            "hits_at_5": summarize(
                observed_hits,
                simulated_hits,
            ),

            "roc_auc": summarize(
                observed_auc,
                simulated_auc,
            ),

            "average_precision": summarize(
                observed_ap,
                simulated_ap,
            ),
        }
