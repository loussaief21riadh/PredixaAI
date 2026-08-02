from __future__ import annotations

from datetime import date

import pytest

from app.ai.v7.feature_builders.pair_recency import (
    PairRecencyBuilder,
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


def test_builder_exists():
    builder = PairRecencyBuilder()

    assert builder is not None
    assert hasattr(builder, "build")


def test_empty_history():
    with pytest.raises(ValueError):
        PairRecencyBuilder.build(
            history=[],
            candidate_number=10,
        )


def test_invalid_candidate_low():
    history = [
        make_draw([1, 2, 3, 4, 5]),
    ]

    with pytest.raises(ValueError):
        PairRecencyBuilder.build(
            history=history,
            candidate_number=0,
        )


def test_invalid_candidate_high():
    history = [
        make_draw([1, 2, 3, 4, 5]),
    ]

    with pytest.raises(ValueError):
        PairRecencyBuilder.build(
            history=history,
            candidate_number=50,
        )


def test_feature_name():
    history = [
        make_draw([1, 2, 3, 4, 5]),
    ]

    result = PairRecencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert set(result) == {
        "pair_recency",
    }


def test_result_range():
    history = [
        make_draw([1, 2, 3, 4, 5], 1),
        make_draw([1, 6, 7, 8, 9], 2),
    ]

    result = PairRecencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert (
        0.0
        <= result["pair_recency"]
        <= 1.0
    )


def test_candidate_absent():
    history = [
        make_draw([1, 2, 3, 4, 5], 1),
        make_draw([6, 7, 8, 9, 10], 2),
    ]

    result = PairRecencyBuilder.build(
        history=history,
        candidate_number=49,
    )

    assert result["pair_recency"] == 1.0


def test_candidate_recent():
    history = [
        make_draw([1, 2, 3, 4, 5], 1),
        make_draw([6, 7, 8, 9, 10], 2),
        make_draw([11, 12, 13, 14, 15], 3),
    ]

    result = PairRecencyBuilder.build(
        history=history,
        candidate_number=15,
    )

    assert result["pair_recency"] == 0.0


def test_deterministic_result():
    history = [
        make_draw([1, 2, 3, 4, 5], 1),
        make_draw([1, 6, 7, 8, 9], 2),
    ]

    first = PairRecencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    second = PairRecencyBuilder.build(
        history=history,
        candidate_number=1,
    )

    assert first == second