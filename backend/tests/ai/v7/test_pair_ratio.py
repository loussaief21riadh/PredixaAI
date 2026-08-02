from __future__ import annotations

from datetime import date

import pytest

from app.ai.v7.feature_builders.pair_ratio import (
    PairRatioBuilder,
)
from app.models.draw import Draw


def make_draw(numbers: list[int], draw_id: int = 1) -> Draw:
    return Draw(
        id=draw_id,
        draw_date=date(2026, 1, min(draw_id, 28)),
        n1=numbers[0],
        n2=numbers[1],
        n3=numbers[2],
        n4=numbers[3],
        n5=numbers[4],
    )


def test_builder_exists():
    builder = PairRatioBuilder()
    assert builder is not None
    assert hasattr(builder, "build")


def test_empty_history():
    with pytest.raises(ValueError):
        PairRatioBuilder.build([], 10)


def test_invalid_candidate_low():
    history = [make_draw([1, 2, 3, 4, 5])]
    with pytest.raises(ValueError):
        PairRatioBuilder.build(history, 0)


def test_invalid_candidate_high():
    history = [make_draw([1, 2, 3, 4, 5])]
    with pytest.raises(ValueError):
        PairRatioBuilder.build(history, 50)


def test_feature_name():
    history = [make_draw([1, 2, 3, 4, 5])]
    result = PairRatioBuilder.build(history, 1)
    assert "pair_ratio" in result


def test_ratio_non_negative():
    history = [
        make_draw([1, 2, 3, 4, 5], 1),
        make_draw([1, 6, 7, 8, 9], 2),
    ]

    result = PairRatioBuilder.build(history, 1)

    assert result["pair_ratio"] >= 0.0


def test_deterministic():
    history = [
        make_draw([1, 2, 3, 4, 5], 1),
        make_draw([1, 6, 7, 8, 9], 2),
    ]

    first = PairRatioBuilder.build(history, 1)
    second = PairRatioBuilder.build(history, 1)

    assert first == second