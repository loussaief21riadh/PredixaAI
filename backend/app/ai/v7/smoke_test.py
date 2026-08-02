from __future__ import annotations

from math import isclose
from typing import Any

from app.ai.v6b_clean.walk_forward_backtester import (
    V6BCleanWalkForwardBacktester,
)
from app.ai.v7.walk_forward_backtester import (
    V7WalkForwardBacktester,
)
from app.database import SessionLocal


TEST_DRAWS = 5
WINDOW_SIZE = 100
MAX_TRAINING_TARGETS = 1500
MONTE_CARLO_SIMULATIONS = 1000

FLOAT_TOLERANCE = 1e-15

IGNORED_PATHS = {
    "result.version",
}


def compare_nested(
    baseline: Any,
    candidate: Any,
    path: str = "result",
) -> list[str]:
    if path in IGNORED_PATHS:
        return []

    differences: list[str] = []

    if isinstance(baseline, dict):
        if not isinstance(candidate, dict):
            return [
                f"{path}: type mismatch "
                f"(baseline={type(baseline).__name__}, "
                f"candidate={type(candidate).__name__})"
            ]

        baseline_keys = set(baseline)
        candidate_keys = set(candidate)

        missing_keys = sorted(
            baseline_keys - candidate_keys
        )
        extra_keys = sorted(
            candidate_keys - baseline_keys
        )

        if missing_keys:
            differences.append(
                f"{path}: missing keys in V7: {missing_keys}"
            )

        if extra_keys:
            differences.append(
                f"{path}: extra keys in V7: {extra_keys}"
            )

        for key in sorted(
            baseline_keys & candidate_keys,
            key=str,
        ):
            differences.extend(
                compare_nested(
                    baseline[key],
                    candidate[key],
                    f"{path}.{key}",
                )
            )

        return differences

    if isinstance(baseline, (list, tuple)):
        if not isinstance(candidate, (list, tuple)):
            return [
                f"{path}: type mismatch "
                f"(baseline={type(baseline).__name__}, "
                f"candidate={type(candidate).__name__})"
            ]

        if len(baseline) != len(candidate):
            return [
                f"{path}: length mismatch "
                f"(baseline={len(baseline)}, "
                f"candidate={len(candidate)})"
            ]

        for index, (
            baseline_item,
            candidate_item,
        ) in enumerate(
            zip(
                baseline,
                candidate,
            )
        ):
            differences.extend(
                compare_nested(
                    baseline_item,
                    candidate_item,
                    f"{path}[{index}]",
                )
            )

        return differences

    if isinstance(baseline, bool):
        if candidate is not baseline:
            differences.append(
                f"{path}: value mismatch "
                f"(baseline={baseline!r}, "
                f"candidate={candidate!r})"
            )

        return differences

    if isinstance(baseline, (int, float)):
        if not isinstance(candidate, (int, float)) or isinstance(
            candidate,
            bool,
        ):
            return [
                f"{path}: type mismatch "
                f"(baseline={type(baseline).__name__}, "
                f"candidate={type(candidate).__name__})"
            ]

        if not isclose(
            float(baseline),
            float(candidate),
            rel_tol=0.0,
            abs_tol=FLOAT_TOLERANCE,
        ):
            differences.append(
                f"{path}: numeric mismatch "
                f"(baseline={baseline!r}, "
                f"candidate={candidate!r})"
            )

        return differences

    if baseline != candidate:
        differences.append(
            f"{path}: value mismatch "
            f"(baseline={baseline!r}, "
            f"candidate={candidate!r})"
        )

    return differences


def validate_result(
    result: dict[str, Any],
    label: str,
) -> None:
    if result.get("status") != "success":
        raise AssertionError(
            f"{label}: invalid status."
        )

    if result.get("evaluated_draws") != TEST_DRAWS:
        raise AssertionError(
            f"{label}: expected {TEST_DRAWS} evaluated draws."
        )

    details = result.get("details", [])

    if len(details) != TEST_DRAWS:
        raise AssertionError(
            f"{label}: invalid detail count."
        )

    for position, detail in enumerate(
        details,
        start=1,
    ):
        predicted_top_5 = detail["predicted_top_5"]
        probabilities = detail["probabilities"]
        ranking = detail["ranking"]

        if (
            len(predicted_top_5) != 5
            or len(set(predicted_top_5)) != 5
        ):
            raise AssertionError(
                f"{label} draw {position}: invalid Top-5."
            )

        if len(probabilities) != 49:
            raise AssertionError(
                f"{label} draw {position}: "
                "invalid probability vector."
            )

        if len(ranking) != 49:
            raise AssertionError(
                f"{label} draw {position}: invalid ranking."
            )

        last_allowed = str(
            detail["last_allowed_feature_draw"]
        )
        excluded_previous = str(
            detail["excluded_previous_draw"]
        )
        target_date = str(
            detail["draw_date"]
        )
        last_training_target = str(
            detail["last_training_target_available"]
        )

        if not (
            last_allowed
            < excluded_previous
            < target_date
        ):
            raise AssertionError(
                f"{label} draw {position}: "
                "invalid T-2/T-1/T chronology."
            )

        if not (
            last_training_target
            < excluded_previous
        ):
            raise AssertionError(
                f"{label} draw {position}: "
                "T-1 was not purged."
            )


def main() -> None:
    print("=" * 80)
    print("PREDIXA AI V7 BASELINE PARITY TEST")
    print("=" * 80)

    db = SessionLocal()

    try:
        baseline_result = (
            V6BCleanWalkForwardBacktester.run(
                db=db,
                test_draws=TEST_DRAWS,
                window_size=WINDOW_SIZE,
                max_training_targets=MAX_TRAINING_TARGETS,
                monte_carlo_simulations=MONTE_CARLO_SIMULATIONS,
            )
        )

        db.expire_all()

        v7_result = (
            V7WalkForwardBacktester.run(
                db=db,
                test_draws=TEST_DRAWS,
                window_size=WINDOW_SIZE,
                max_training_targets=MAX_TRAINING_TARGETS,
                monte_carlo_simulations=MONTE_CARLO_SIMULATIONS,
            )
        )

    finally:
        db.close()

    validate_result(
        baseline_result,
        label="V6B-CLEAN",
    )

    validate_result(
        v7_result,
        label="V7",
    )

    differences = compare_nested(
        baseline_result,
        v7_result,
    )

    print()
    print("RESULTS")
    print("-" * 80)

    print(
        "V6B-CLEAN Hits@5 :",
        baseline_result["model"]["average_hits_at_5"],
    )

    print(
        "V7 Hits@5        :",
        v7_result["model"]["average_hits_at_5"],
    )

    print()
    print("DRAW-BY-DRAW")
    print("-" * 80)

    for position, (
        baseline_detail,
        v7_detail,
    ) in enumerate(
        zip(
            baseline_result["details"],
            v7_result["details"],
        ),
        start=1,
    ):
        print(
            position,
            "|",
            baseline_detail["draw_date"],
            "| V6B =",
            baseline_detail["predicted_top_5"],
            "| V7 =",
            v7_detail["predicted_top_5"],
            "| hits =",
            baseline_detail["hits"],
        )

    print()
    print("PARITY")
    print("-" * 80)

    print(
        "Same Top-5 sequences :",
        [
            detail["predicted_top_5"]
            for detail in baseline_result["details"]
        ]
        == [
            detail["predicted_top_5"]
            for detail in v7_result["details"]
        ],
    )

    print(
        "Same metrics         :",
        not differences,
    )

    print(
        "Temporal checks      :",
        True,
    )

    if differences:
        print()
        print("FIRST DIFFERENCES")
        print("-" * 80)

        for difference in differences[:50]:
            print(difference)

        raise AssertionError(
            "V7 baseline does not reproduce V6B-CLEAN."
        )

    print()
    print("=" * 80)
    print("V7 BASELINE PARITY : TRUE")
    print("=" * 80)

    print(
        "V7 reproduces V6B-CLEAN draw-by-draw, "
        "including Top-5 predictions, rankings, "
        "probabilities, metrics, baselines, "
        "diagnostics, and temporal protocol."
    )


if __name__ == "__main__":
    main()
