from __future__ import annotations

from sqlalchemy import select

from app.ai.v5.feature_engineering import (
    V5FeatureEngineering,
)
from app.ai.v6b_clean.feature_builders.frequency import (
    FrequencyBuilder,
)
from app.database import SessionLocal
from app.models.draw import Draw


WINDOW_SIZE = 100
CANDIDATE_NUMBER = 17


def load_recent_draws(
    limit: int,
) -> list[Draw]:
    """
    Load the most recent draws in chronological order.

    The database query retrieves the newest rows first,
    then the result is reversed so that the returned
    history is ordered from oldest to newest.
    """

    if limit <= 0:
        raise ValueError(
            "The draw limit must be greater than zero."
        )

    with SessionLocal() as db:
        statement = (
            select(Draw)
            .order_by(
                Draw.draw_date.desc(),
                Draw.id.desc(),
            )
            .limit(limit)
        )

        draws = list(
            db.scalars(statement).all()
        )

    draws.reverse()

    return draws


def compare_frequency_features(
    history: list[Draw],
    candidate_number: int,
) -> bool:
    """
    Compare V6B frequency features with the V5 reference.
    """

    v5_features = (
        V5FeatureEngineering.build_from_history(
            draws=history,
            window_size=WINDOW_SIZE,
        )
    )

    v6b_features = FrequencyBuilder().build(
        history=history,
        candidate_number=candidate_number,
    )

    success = True

    print()
    print(f"Candidate number: {candidate_number}")
    print("-" * 72)

    for window in FrequencyBuilder.WINDOWS:
        v5_name = (
            f"rate_{window}_{candidate_number}"
        )

        v6b_name = f"rate_{window}"

        v5_value = v5_features[v5_name]
        v6b_value = v6b_features[v6b_name]

        matches = v5_value == v6b_value

        status = (
            "OK"
            if matches
            else "FAIL"
        )

        print(
            f"{v6b_name:<12}"
            f"V5={v5_value:<12}"
            f"V6B={v6b_value:<12}"
            f"{status}"
        )

        if not matches:
            success = False

    return success


def main() -> None:
    """
    Run the V6B frequency compatibility smoke test.
    """

    print("=" * 72)
    print("PredixaAI V6B Frequency Smoke Test")
    print("=" * 72)

    history = load_recent_draws(
        limit=WINDOW_SIZE,
    )

    if len(history) != WINDOW_SIZE:
        raise RuntimeError(
            "Not enough draws were loaded. "
            f"Expected {WINDOW_SIZE}, "
            f"received {len(history)}."
        )

    print(
        f"History size: {len(history)}"
    )
    print(
        f"First draw:  {history[0].draw_date}"
    )
    print(
        f"Last draw:   {history[-1].draw_date}"
    )

    success = compare_frequency_features(
        history=history,
        candidate_number=CANDIDATE_NUMBER,
    )

    print()
    print("=" * 72)

    if not success:
        raise AssertionError(
            "FrequencyBuilder does not match "
            "V5FeatureEngineering."
        )

    print(
        "SUCCESS: FrequencyBuilder matches "
        "V5FeatureEngineering."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()