from __future__ import annotations

from math import isclose
from typing import Any

from app.ai.v6.walk_forward_backtester import (
    V6WalkForwardBacktester,
)
from app.ai.v6b_clean.walk_forward_backtester import (
    V6BCleanWalkForwardBacktester,
)
from app.database import SessionLocal


TEST_DRAWS = 5
WINDOW_SIZE = 100
MAX_TRAINING_TARGETS = 1500
MONTE_CARLO_SIMULATIONS = 1000

FLOAT_ABS_TOLERANCE = 1e-15

EXPECTED_DIFFERENT_FIELDS = {
    "version",
}


def compare_values(
    v6_value: Any,
    v6b_value: Any,
    path: str = "result",
) -> list[str]:
    """
    Recursively compare two nested walk-forward results.

    The version identifier is intentionally allowed to differ.
    Every other field must match.
    """

    field_name = path.rsplit(".", maxsplit=1)[-1]

    if field_name in EXPECTED_DIFFERENT_FIELDS:
        return []

    differences: list[str] = []

    if isinstance(v6_value, dict):
        if not isinstance(v6b_value, dict):
            return [
                f"{path}: type mismatch "
                f"(V6={type(v6_value).__name__}, "
                f"V6B={type(v6b_value).__name__})"
            ]

        v6_keys = set(v6_value)
        v6b_keys = set(v6b_value)

        missing_in_v6b = sorted(
            v6_keys - v6b_keys
        )
        extra_in_v6b = sorted(
            v6b_keys - v6_keys
        )

        if missing_in_v6b:
            differences.append(
                f"{path}: keys missing in V6B: "
                f"{missing_in_v6b}"
            )

        if extra_in_v6b:
            differences.append(
                f"{path}: extra keys in V6B: "
                f"{extra_in_v6b}"
            )

        for key in sorted(
            v6_keys & v6b_keys,
            key=str,
        ):
            differences.extend(
                compare_values(
                    v6_value[key],
                    v6b_value[key],
                    f"{path}.{key}",
                )
            )

        return differences

    if isinstance(v6_value, (list, tuple)):
        if not isinstance(
            v6b_value,
            (list, tuple),
        ):
            return [
                f"{path}: type mismatch "
                f"(V6={type(v6_value).__name__}, "
                f"V6B={type(v6b_value).__name__})"
            ]

        if len(v6_value) != len(v6b_value):
            differences.append(
                f"{path}: length mismatch "
                f"(V6={len(v6_value)}, "
                f"V6B={len(v6b_value)})"
            )
            return differences

        for index, (
            v6_item,
            v6b_item,
        ) in enumerate(
            zip(
                v6_value,
                v6b_value,
            )
        ):
            differences.extend(
                compare_values(
                    v6_item,
                    v6b_item,
                    f"{path}[{index}]",
                )
            )

        return differences

    if isinstance(v6_value, bool):
        if not isinstance(v6b_value, bool):
            return [
                f"{path}: type mismatch "
                f"(V6=bool, "
                f"V6B={type(v6b_value).__name__})"
            ]

        if v6_value != v6b_value:
            differences.append(
                f"{path}: value mismatch "
                f"(V6={v6_value!r}, "
                f"V6B={v6b_value!r})"
            )

        return differences

    if isinstance(v6_value, (int, float)):
        if not isinstance(
            v6b_value,
            (int, float),
        ) or isinstance(
            v6b_value,
            bool,
        ):
            return [
                f"{path}: type mismatch "
                f"(V6={type(v6_value).__name__}, "
                f"V6B={type(v6b_value).__name__})"
            ]

        if not isclose(
            float(v6_value),
            float(v6b_value),
            rel_tol=0.0,
            abs_tol=FLOAT_ABS_TOLERANCE,
        ):
            differences.append(
                f"{path}: numeric mismatch "
                f"(V6={v6_value!r}, "
                f"V6B={v6b_value!r})"
            )

        return differences

    if v6_value != v6b_value:
        differences.append(
            f"{path}: value mismatch "
            f"(V6={v6_value!r}, "
            f"V6B={v6b_value!r})"
        )

    return differences


def validate_temporal_details(
    details: list[dict[str, Any]],
    label: str,
) -> None:
    """
    Validate T-2 prediction and T-1 purge metadata.
    """

    if len(details) != TEST_DRAWS:
        raise AssertionError(
            f"{label}: expected {TEST_DRAWS} detail rows, "
            f"received {len(details)}."
        )

    for position, detail in enumerate(
        details,
        start=1,
    ):
        probabilities = detail[
            "probabilities"
        ]

        ranking = detail[
            "ranking"
        ]

        predicted_top_5 = detail[
            "predicted_top_5"
        ]

        if len(probabilities) != 49:
            raise AssertionError(
                f"{label} draw {position}: "
                "probability vector must contain 49 values."
            )

        if len(ranking) != 49:
            raise AssertionError(
                f"{label} draw {position}: "
                "ranking must contain 49 candidates."
            )

        if len(predicted_top_5) != 5:
            raise AssertionError(
                f"{label} draw {position}: "
                "Top-5 must contain 5 numbers."
            )

        if len(set(predicted_top_5)) != 5:
            raise AssertionError(
                f"{label} draw {position}: "
                "Top-5 contains duplicate numbers."
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
            raise AssertionError(
                f"{label} draw {position}: "
                "invalid prediction chronology. "
                "Expected T-2 < T-1 < T."
            )

        if not (
            last_training_target
            < excluded_previous
        ):
            raise AssertionError(
                f"{label} draw {position}: "
                "T-1 was not purged from training targets."
            )


def print_summary(
    v6_result: dict[str, Any],
    v6b_result: dict[str, Any],
) -> None:
    """
    Print a compact parity summary.
    """

    print()
    print("SUMMARY")
    print("-" * 80)

    print(
        "Evaluated draws :",
        v6_result[
            "evaluated_draws"
        ],
    )

    print(
        "V6 Hits@5      :",
        v6_result[
            "model"
        ][
            "average_hits_at_5"
        ],
    )

    print(
        "V6B Hits@5     :",
        v6b_result[
            "model"
        ][
            "average_hits_at_5"
        ],
    )

    print(
        "V6 total hits  :",
        v6_result[
            "model"
        ][
            "total_hits"
        ],
    )

    print(
        "V6B total hits :",
        v6b_result[
            "model"
        ][
            "total_hits"
        ],
    )

    print(
        "Frequency      :",
        v6_result[
            "frequency_baseline"
        ][
            "average_hits_at_5"
        ],
    )

    print(
        "Previous draw  :",
        v6_result[
            "previous_draw_baseline"
        ][
            "average_hits_at_5"
        ],
    )

    print(
        "Monte Carlo p  :",
        v6_result[
            "random_baseline"
        ][
            "monte_carlo"
        ][
            "empirical_p_value"
        ],
    )

    print()
    print("DRAW-BY-DRAW")
    print("-" * 80)

    for position, (
        v6_detail,
        v6b_detail,
    ) in enumerate(
        zip(
            v6_result[
                "details"
            ],
            v6b_result[
                "details"
            ],
        ),
        start=1,
    ):
        print(
            position,
            "|",
            v6_detail[
                "draw_date"
            ],
            "| V6 =",
            v6_detail[
                "predicted_top_5"
            ],
            "| V6B =",
            v6b_detail[
                "predicted_top_5"
            ],
            "| hits =",
            v6_detail[
                "hits"
            ],
        )


def main() -> None:
    """
    Run strict V6 versus V6B-CLEAN walk-forward parity.
    """

    print("=" * 80)
    print("PREDIXA AI V6B-CLEAN - WALK-FORWARD PARITY TEST")
    print("=" * 80)

    db = SessionLocal()

    try:
        v6_result = (
            V6WalkForwardBacktester.run(
                db=db,
                test_draws=TEST_DRAWS,
                window_size=WINDOW_SIZE,
                max_training_targets=(
                    MAX_TRAINING_TARGETS
                ),
                monte_carlo_simulations=(
                    MONTE_CARLO_SIMULATIONS
                ),
            )
        )

        db.expire_all()

        v6b_result = (
            V6BCleanWalkForwardBacktester.run(
                db=db,
                test_draws=TEST_DRAWS,
                window_size=WINDOW_SIZE,
                max_training_targets=(
                    MAX_TRAINING_TARGETS
                ),
                monte_carlo_simulations=(
                    MONTE_CARLO_SIMULATIONS
                ),
            )
        )

    finally:
        db.close()

    validate_temporal_details(
        v6_result[
            "details"
        ],
        label="V6",
    )

    validate_temporal_details(
        v6b_result[
            "details"
        ],
        label="V6B",
    )

    differences = compare_values(
        v6_result,
        v6b_result,
    )

    print_summary(
        v6_result=v6_result,
        v6b_result=v6b_result,
    )

    print()
    print("PARITY")
    print("-" * 80)

    print(
        "Same aggregate metrics :",
        not differences,
    )

    print(
        "Same Top-5 sequences   :",
        [
            detail[
                "predicted_top_5"
            ]
            for detail in v6_result[
                "details"
            ]
        ]
        == [
            detail[
                "predicted_top_5"
            ]
            for detail in v6b_result[
                "details"
            ]
        ],
    )

    print(
        "Same probability sets  :",
        all(
            v6_detail[
                "probabilities"
            ]
            == v6b_detail[
                "probabilities"
            ]
            for (
                v6_detail,
                v6b_detail,
            ) in zip(
                v6_result[
                    "details"
                ],
                v6b_result[
                    "details"
                ],
            )
        ),
    )

    print(
        "Temporal checks valid  :",
        True,
    )

    if differences:
        print()
        print("FIRST DIFFERENCES")
        print("-" * 80)

        for difference in differences[:50]:
            print(difference)

        raise AssertionError(
            "V6B-CLEAN walk-forward output "
            "does not match V6."
        )

    print()
    print("=" * 80)
    print("WALK-FORWARD PARITY : TRUE")
    print("=" * 80)

    print(
        "V6B-CLEAN reproduces V6 draw-by-draw, "
        "including Top-5 predictions, 49 probabilities, "
        "rankings, hits, baselines, diagnostics, "
        "Monte Carlo results, and aggregate metrics."
    )


if __name__ == "__main__":
    main()