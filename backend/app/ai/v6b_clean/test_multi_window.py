from __future__ import annotations

from app.ai.v6b_clean.multi_window_validator import (
    V6BCleanMultiWindowValidator,
)
from app.database import SessionLocal


def main() -> None:
    db = SessionLocal()

    try:
        result = V6BCleanMultiWindowValidator.run(
            db=db,
            test_draws_per_window=20,
            number_of_windows=3,
            window_size=100,
            max_training_targets=1500,
            monte_carlo_simulations=1000,
        )

        print("=" * 80)
        print("PREDIXA AI V6B-CLEAN MULTI-WINDOW VALIDATION")
        print("=" * 80)

        print("Status                :", result["status"])
        print("Version               :", result["version"])
        print("Windows               :", result["number_of_windows"])
        print("Total evaluated draws :", result["total_evaluated_draws"])

        print()
        print("MODEL")
        print("-" * 80)

        print(
            "Average Hits@5 :",
            result["model"]["average_hits_at_5"],
        )
        print(
            "Window Mean    :",
            result["model"]["window_mean_hits_at_5"],
        )
        print(
            "Window Std     :",
            result["model"]["window_std_hits_at_5"],
        )
        print(
            "Minimum        :",
            result["model"]["minimum_window_hits_at_5"],
        )
        print(
            "Maximum        :",
            result["model"]["maximum_window_hits_at_5"],
        )
        print(
            "Total hits     :",
            result["model"]["total_hits"],
        )

        print()
        print("BASELINES")
        print("-" * 80)

        print(
            "Random expectation :",
            result["random_expectation"],
        )
        print(
            "Frequency Hits@5   :",
            result["frequency_baseline"]["average_hits_at_5"],
        )
        print(
            "Previous Hits@5    :",
            result["previous_draw_baseline"]["average_hits_at_5"],
        )

        print()
        print("WINDOWS")
        print("-" * 80)

        for window in result["window_results"]:
            print(
                window["window"],
                "|",
                window["start_date"],
                "->",
                window["end_date"],
                "| Hits@5 =",
                window["average_hits_at_5"],
                "| Hits =",
                window["total_hits"],
            )

        print()
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)

    finally:
        db.close()


if __name__ == "__main__":
    main()
