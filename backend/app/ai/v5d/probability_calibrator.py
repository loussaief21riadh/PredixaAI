import math
from typing import Any


class V5DProbabilityCalibrator:
    """
    Predixa AI V5-D Probability Calibration.

    Initial diagnostic calibrator.

    IMPORTANT:
    This module does NOT train the Random Forest.
    It operates on already-saved probability vectors.

    The first objective is to measure:
        - raw Brier score
        - constant 5/49 baseline Brier score
        - probability overconfidence
        - simple shrinkage toward the 5/49 base rate

    This stage is diagnostic only.

    A production calibration model must later be fitted
    strictly out-of-sample / temporally to avoid leakage.
    """

    VERSION = "V5-D-PROBABILITY-CALIBRATION"

    NUMBER_COUNT = 49
    NUMBERS_PER_DRAW = 5

    BASE_RATE = (
        NUMBERS_PER_DRAW
        / NUMBER_COUNT
    )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def _validate_details(
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
            probabilities = detail.get(
                "probabilities"
            )

            actual_numbers = detail.get(
                "actual_numbers"
            )

            if not isinstance(
                probabilities,
                dict,
            ):
                raise ValueError(
                    f"Detail {index} has no probability vector."
                )

            if len(probabilities) != cls.NUMBER_COUNT:
                raise ValueError(
                    f"Detail {index} does not contain "
                    f"{cls.NUMBER_COUNT} probabilities."
                )

            if not isinstance(
                actual_numbers,
                list,
            ):
                raise ValueError(
                    f"Detail {index} has no actual_numbers."
                )

            if len(actual_numbers) != cls.NUMBERS_PER_DRAW:
                raise ValueError(
                    f"Detail {index} does not contain "
                    f"{cls.NUMBERS_PER_DRAW} actual numbers."
                )

        return details

    # ==========================================================
    # SCORE EXTRACTION
    # ==========================================================

    @staticmethod
    def _get_probability(
        probabilities: dict,
        number: int,
    ) -> float:

        value = probabilities.get(
            number
        )

        if value is None:
            value = probabilities.get(
                str(number)
            )

        if value is None:
            raise ValueError(
                f"Missing probability for number {number}."
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite probability for number {number}."
            )

        if value < 0.0 or value > 1.0:
            raise ValueError(
                f"Probability outside [0, 1] "
                f"for number {number}: {value}"
            )

        return value

    # ==========================================================
    # BRIER SCORE
    # ==========================================================

    @classmethod
    def brier_score(
        cls,
        details: list[dict[str, Any]],
        shrinkage: float = 0.0,
    ) -> float:
        """
        shrinkage = 0:
            raw RF probabilities.

        shrinkage = 1:
            constant 5/49 base-rate probability.

        Intermediate values:
            p_calibrated =
                (1 - shrinkage) * p_raw
                + shrinkage * base_rate

        This transformation is monotonic for shrinkage < 1,
        so ranking is preserved.
        """

        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError(
                "shrinkage must be between 0 and 1."
            )

        squared_errors = []

        for detail in details:
            actual = set(
                detail["actual_numbers"]
            )

            probabilities = detail[
                "probabilities"
            ]

            for number in range(
                1,
                cls.NUMBER_COUNT + 1,
            ):
                raw_probability = (
                    cls._get_probability(
                        probabilities,
                        number,
                    )
                )

                calibrated_probability = (
                    (
                        1.0
                        - shrinkage
                    )
                    * raw_probability
                    +
                    shrinkage
                    * cls.BASE_RATE
                )

                target = (
                    1.0
                    if number in actual
                    else 0.0
                )

                squared_errors.append(
                    (
                        calibrated_probability
                        - target
                    )
                    ** 2
                )

        return (
            sum(squared_errors)
            / len(squared_errors)
        )

    # ==========================================================
    # SHRINKAGE GRID
    # ==========================================================

    @classmethod
    def evaluate_shrinkage_grid(
        cls,
        details: list[dict[str, Any]],
        step: float = 0.05,
    ) -> list[dict[str, float]]:

        if step <= 0.0 or step > 1.0:
            raise ValueError(
                "step must be in (0, 1]."
            )

        rows = []

        shrinkage = 0.0

        while shrinkage <= 1.0000001:
            shrinkage = min(
                shrinkage,
                1.0,
            )

            score = cls.brier_score(
                details,
                shrinkage=shrinkage,
            )

            rows.append(
                {
                    "shrinkage": round(
                        shrinkage,
                        6,
                    ),
                    "brier_score": round(
                        score,
                        8,
                    ),
                }
            )

            if shrinkage >= 1.0:
                break

            shrinkage += step

        return rows

    # ==========================================================
    # MAIN ANALYSIS
    # ==========================================================

    @classmethod
    def analyze(
        cls,
        result: dict[str, Any],
        grid_step: float = 0.05,
    ) -> dict[str, Any]:

        details = cls._validate_details(
            result
        )

        raw_brier = cls.brier_score(
            details,
            shrinkage=0.0,
        )

        baseline_brier = cls.brier_score(
            details,
            shrinkage=1.0,
        )

        grid = cls.evaluate_shrinkage_grid(
            details,
            step=grid_step,
        )

        best = min(
            grid,
            key=lambda row: (
                row["brier_score"],
                row["shrinkage"],
            ),
        )

        return {
            "status": "success",

            "version": cls.VERSION,

            "evaluated_draws": len(
                details
            ),

            "evaluated_binary_predictions": (
                len(details)
                * cls.NUMBER_COUNT
            ),

            "base_rate": round(
                cls.BASE_RATE,
                8,
            ),

            "raw_brier_score": round(
                raw_brier,
                8,
            ),

            "constant_baseline_brier_score": round(
                baseline_brier,
                8,
            ),

            "raw_minus_baseline": round(
                raw_brier
                - baseline_brier,
                8,
            ),

            "best_in_sample_shrinkage": (
                best["shrinkage"]
            ),

            "best_in_sample_brier_score": (
                best["brier_score"]
            ),

            "grid": grid,

            "warning": (
                "Best shrinkage is measured in-sample and "
                "must not be treated as production calibration. "
                "Temporal out-of-sample calibration is required."
            ),
        }
