from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from app.models.draw import Draw


class PairFrequencyBuilder:
    """
    Build normalized pair-frequency features for one candidate.

    For each historical draw, all unordered pairs of the five
    main numbers are generated.

    For a given candidate number, the builder measures how often
    that candidate appears inside historical pairs.

    Normalization:
        candidate pair occurrences / (history_size * 4)

    Each candidate can participate in at most four pairs per draw,
    so the resulting feature remains inside [0, 1].
    """

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    NUMBERS_PER_DRAW = 5

    PAIRS_PER_CANDIDATE_PER_DRAW = (
        NUMBERS_PER_DRAW - 1
    )

    FEATURE_NAME = "pair_frequency"

    @staticmethod
    def _main_numbers(
        draw: Draw,
    ) -> list[int]:
        return [
            int(draw.n1),
            int(draw.n2),
            int(draw.n3),
            int(draw.n4),
            int(draw.n5),
        ]

    @classmethod
    def _validate_candidate_number(
        cls,
        candidate_number: int,
    ) -> None:
        if not isinstance(
            candidate_number,
            int,
        ):
            raise ValueError(
                "candidate_number must be an integer."
            )

        if not (
            cls.NUMBER_MIN
            <= candidate_number
            <= cls.NUMBER_MAX
        ):
            raise ValueError(
                "candidate_number must be between "
                f"{cls.NUMBER_MIN} and "
                f"{cls.NUMBER_MAX}."
            )

    @classmethod
    def _validate_history(
        cls,
        history: Sequence[Draw],
    ) -> None:
        if not isinstance(
            history,
            Sequence,
        ):
            raise ValueError(
                "history must be a sequence of Draw objects."
            )

        if len(history) == 0:
            raise ValueError(
                "history cannot be empty."
            )

        for index, draw in enumerate(
            history,
        ):
            if not isinstance(
                draw,
                Draw,
            ):
                raise ValueError(
                    "history contains an invalid item at "
                    f"index {index}. Expected Draw, received "
                    f"{type(draw).__name__}."
                )

            numbers = cls._main_numbers(
                draw
            )

            if len(
                set(numbers)
            ) != cls.NUMBERS_PER_DRAW:
                raise ValueError(
                    "Each historical draw must contain "
                    "five unique main numbers."
                )

            invalid_numbers = [
                number
                for number in numbers
                if not (
                    cls.NUMBER_MIN
                    <= number
                    <= cls.NUMBER_MAX
                )
            ]

            if invalid_numbers:
                raise ValueError(
                    "Historical draw contains numbers outside "
                    f"{cls.NUMBER_MIN}-{cls.NUMBER_MAX}: "
                    f"{sorted(invalid_numbers)}"
                )

    @classmethod
    def _pair_counter(
        cls,
        history: Sequence[Draw],
    ) -> Counter[tuple[int, int]]:
        pair_counter: Counter[
            tuple[int, int]
        ] = Counter()

        for draw in history:
            numbers = sorted(
                cls._main_numbers(
                    draw
                )
            )

            pair_counter.update(
                combinations(
                    numbers,
                    2,
                )
            )

        return pair_counter

    @classmethod
    def build(
        cls,
        history: Sequence[Draw],
        candidate_number: int,
    ) -> dict[str, float]:
        """
        Build normalized pair frequency for one candidate.
        """

        cls._validate_history(
            history
        )

        cls._validate_candidate_number(
            candidate_number
        )

        pair_counter = cls._pair_counter(
            history
        )

        candidate_pair_occurrences = sum(
            count
            for pair, count in pair_counter.items()
            if candidate_number in pair
        )

        maximum_possible_occurrences = (
            len(history)
            * cls.PAIRS_PER_CANDIDATE_PER_DRAW
        )

        if maximum_possible_occurrences <= 0:
            raise ValueError(
                "Unable to normalize pair frequency."
            )

        pair_frequency = (
            candidate_pair_occurrences
            / maximum_possible_occurrences
        )

        if not (
            0.0
            <= pair_frequency
            <= 1.0
        ):
            raise ValueError(
                "pair_frequency must remain inside [0, 1]."
            )

        return {
            cls.FEATURE_NAME: round(
                float(
                    pair_frequency
                ),
                6,
            ),
        }
