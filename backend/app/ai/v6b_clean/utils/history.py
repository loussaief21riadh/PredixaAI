from __future__ import annotations

from collections.abc import Sequence

from app.models.draw import Draw


NUMBER_MIN = 1
NUMBER_MAX = 49


def main_numbers(draw: Draw) -> tuple[int, int, int, int, int]:
    """
    Return the five main lottery numbers of a draw.

    The returned order matches the values stored in the database.
    """

    return (
        draw.n1,
        draw.n2,
        draw.n3,
        draw.n4,
        draw.n5,
    )


def contains_number(
    draw: Draw,
    number: int,
) -> bool:
    """
    Return True if the candidate number appears in the draw.
    """

    return number in main_numbers(draw)


def validate_candidate_number(
    number: int,
) -> None:
    """
    Validate that a lottery number is between 1 and 49.
    """

    if not NUMBER_MIN <= number <= NUMBER_MAX:
        raise ValueError(
            f"Candidate number must be between "
            f"{NUMBER_MIN} and {NUMBER_MAX}. "
            f"Received {number}."
        )


def validate_history(
    history: Sequence[Draw],
) -> None:
    """
    Validate that the historical sequence is usable.
    """

    if len(history) == 0:
        raise ValueError(
            "History cannot be empty."
        )


def count_occurrences(
    history: Sequence[Draw],
    number: int,
) -> int:
    """
    Count the number of historical draws containing
    the candidate number.
    """

    validate_candidate_number(number)
    validate_history(history)

    return sum(
        1
        for draw in history
        if contains_number(draw, number)
    )


def frequency_rate(
    history: Sequence[Draw],
    number: int,
    window: int,
) -> float:
    """
    Compute the normalized occurrence rate over the
    last 'window' historical draws.

    This reproduces the V5 implementation exactly.
    """

    validate_candidate_number(number)
    validate_history(history)

    recent_history = history[-window:]

    if len(recent_history) == 0:
        return 0.0

    occurrences = count_occurrences(
        recent_history,
        number,
    )

    return occurrences / len(recent_history)