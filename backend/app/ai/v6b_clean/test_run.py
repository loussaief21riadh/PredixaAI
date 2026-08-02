from app.ai.v6b_clean.walk_forward_backtester import (
    V6BCleanWalkForwardBacktester,
)
from app.database import SessionLocal


def main():
    db = SessionLocal()

    try:
        result = V6BCleanWalkForwardBacktester.run(
            db=db,
            test_draws=5,
            window_size=100,
            max_training_targets=1500,
            monte_carlo_simulations=1000,
        )

        print("=" * 80)
        print("V6B CLEAN WALK FORWARD")
        print("=" * 80)

        print("Status:", result["status"])
        print("Version:", result["version"])
        print("Evaluated draws:", result["evaluated_draws"])
        print(
            "Average hits:",
            result["model"]["average_hits_at_5"],
        )

        print()
        print("Top-5 of first evaluated draw:")
        print(
            result["details"][0]["predicted_top_5"]
        )

        print()
        print("SUCCESS")

    finally:
        db.close()


if __name__ == "__main__":
    main()