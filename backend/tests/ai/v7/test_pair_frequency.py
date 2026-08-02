from __future__ import annotations

from datetime import date

import pytest

from app.ai.v7.feature_builders.pair_frequency import (
    PairFrequencyBuilder,
)
from app.models.draw import Draw


def make_draw(
    numbers: list[int],
    draw_id: int = 1,
) -> Draw:
    return Draw(
        id=draw_id,
        draw_date=date(2026, 1, draw_id),
        n1=numbers[0],
        n2=numbers[1],
        n3=numbers[2],
        n4=numbers[3],
        n5=numbers[4],
    )


def test_builder_exists() -> None:
    builder = PairFrequencyBuilder()

    assert builder is not None
    assert hasattr(builder, "build")


def test_empty_history() -> None:
    with pytest.raises(ValueError):
        PairFrequencyBuilder.build(
            history=[],
            candidate_number=10,
        )


def test_invalid_candidate_low() -> None:
    history = [
        make_draw([1, 2, 3, 4, 5]),
    ]

    with pytest.raises(ValueError):
        PairFrequencyBuilder.build(
            history=history,
            candidate_number=0,
        )


def test_invalid_candidate_high() -> None:
    history = [
        make_draw([1, 2, 3, 4, 5]),
    ]

    with pytest.raises(ValueError):
        PairFrequencyBuilder.build(
            history=history,
            candidate_number=50,
        )


def test_feature_name() -> None:
    history = [
        make_draw([1, 2, 3, 4, 5]),
    ]

    result = PairFrequencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert set(result) == {
        "pair_frequency",
    }


def test_result_range() -> None:
    history = [
        make_draw(
            [1, 2, 3, 4, 5],
            draw_id=1,
        ),
        make_draw(
            [1, 6, 7, 8, 9],
            draw_id=2,
        ),
    ]

    result = PairFrequencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert (
        0.0
        <= result["pair_frequency"]
        <= 1.0
    )


def test_expected_frequency() -> None:
    history = [
        make_draw(
            [1, 2, 3, 4, 5],
            draw_id=1,
        ),
        make_draw(
            [1, 6, 7, 8, 9],
            draw_id=2,
        ),
    ]

    result = PairFrequencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert result["pair_frequency"] == 1.0


def test_absent_candidate_frequency() -> None:
    history = [
        make_draw(
            [1, 2, 3, 4, 5],
            draw_id=1,
        ),
        make_draw(
            [6, 7, 8, 9, 10],
            draw_id=2,
        ),
    ]

    result = PairFrequencyBuilder.build(
        history=history,
        candidate_number=49,
    )

    assert result["pair_frequency"] == 0.0


def test_deterministic_result() -> None:
    history = [
        make_draw(
            [1, 2, 3, 4, 5],
            draw_id=1,
        ),
        make_draw(
            [1, 6, 7, 8, 9],
            draw_id=2,
        ),
    ]

    first_result = PairFrequencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    second_result = PairFrequencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert first_result == second_result
