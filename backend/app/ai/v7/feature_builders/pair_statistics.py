from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from app.models.draw import Draw


class PairStatistics:
    """
    Compute reusable pair statistics from historical draws.

    The statistics are calculated once and can be reused by:

    - PairFrequencyBuilder
    - PairRecencyBuilder
    - PairRatioBuilder
    """

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    NUMBERS_PER_DRAW = 5
    PAIRS_PER_DRAW = 10
    PAIRS_PER_CANDIDATE_PER_DRAW = 4

    def __init__(
        self,
        history: Sequence[Draw],
    ) -> None:
        self._validate_history(
            history
        )

        self.history = list(
            history
        )

        self.history_size = len(
            self.history
        )

        self.pair_counter: Counter[
            tuple[int, int]
        ] = Counter()

        self.pair_last_seen: dict[
            tuple[int, int],
            int
        ] = {}

        self.candidate_pair_occurrences: Counter[
            int
        ] = Counter()

        self.candidate_last_seen: dict[
            int,
            int
        ] = {}

        self._calculate()

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
            history
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
                numbers
            ) != cls.NUMBERS_PER_DRAW:
                raise ValueError(
                    "Each draw must contain exactly "
                    "five main numbers."
                )

            if len(
                set(numbers)
            ) != cls.NUMBERS_PER_DRAW:
                raise ValueError(
                    "Each draw must contain five "
                    "unique main numbers."
                )

            invalid_numbers = sorted(
                number
                for number in numbers
                if not (
                    cls.NUMBER_MIN
                    <= number
                    <= cls.NUMBER_MAX
                )
            )

            if invalid_numbers:
                raise ValueError(
                    "Historical draw contains numbers "
                    f"outside {cls.NUMBER_MIN}-"
                    f"{cls.NUMBER_MAX}: "
                    f"{invalid_numbers}"
                )

    @classmethod
    def validate_candidate_number(
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

    @staticmethod
    def normalize_pair(
        first_number: int,
        second_number: int,
    ) -> tuple[int, int]:
        if first_number == second_number:
            raise ValueError(
                "A pair must contain two different numbers."
            )

        return tuple(
            sorted(
                (
                    int(first_number),
                    int(second_number),
                )
            )
        )

    def _calculate(
        self,
    ) -> None:
        for draw_index, draw in enumerate(
            self.history
        ):
            numbers = sorted(
                self._main_numbers(
                    draw
                )
            )

            draw_pairs = list(
                combinations(
                    numbers,
                    2,
                )
            )

            if (
                len(draw_pairs)
                != self.PAIRS_PER_DRAW
            ):
                raise ValueError(
                    "Each draw must generate exactly "
                    f"{self.PAIRS_PER_DRAW} pairs."
                )

            for pair in draw_pairs:
                normalized_pair = (
                    self.normalize_pair(
                        pair[0],
                        pair[1],
                    )
                )

                self.pair_counter[
                    normalized_pair
                ] += 1

                self.pair_last_seen[
                    normalized_pair
                ] = draw_index

                first_number, second_number = (
                    normalized_pair
                )

                self.candidate_pair_occurrences[
                    first_number
                ] += 1

                self.candidate_pair_occurrences[
                    second_number
                ] += 1

                self.candidate_last_seen[
                    first_number
                ] = draw_index

                self.candidate_last_seen[
                    second_number
                ] = draw_index

    def pair_count(
        self,
        first_number: int,
        second_number: int,
    ) -> int:
        pair = self.normalize_pair(
            first_number,
            second_number,
        )

        return int(
            self.pair_counter.get(
                pair,
                0,
            )
        )

    def pair_recency(
        self,
        first_number: int,
        second_number: int,
    ) -> int:
        pair = self.normalize_pair(
            first_number,
            second_number,
        )

        last_seen_index = (
            self.pair_last_seen.get(
                pair
            )
        )

        if last_seen_index is None:
            return self.history_size

        return (
            self.history_size
            - 1
            - last_seen_index
        )

    def candidate_occurrences(
        self,
        candidate_number: int,
    ) -> int:
        self.validate_candidate_number(
            candidate_number
        )

        return int(
            self.candidate_pair_occurrences.get(
                candidate_number,
                0,
            )
        )

    def candidate_frequency(
        self,
        candidate_number: int,
    ) -> float:
        occurrences = (
            self.candidate_occurrences(
                candidate_number
            )
        )

        maximum_occurrences = (
            self.history_size
            * self.PAIRS_PER_CANDIDATE_PER_DRAW
        )

        if maximum_occurrences <= 0:
            raise ValueError(
                "Unable to normalize candidate frequency."
            )

        frequency = (
            occurrences
            / maximum_occurrences
        )

        if not (
            0.0
            <= frequency
            <= 1.0
        ):
            raise ValueError(
                "Candidate pair frequency must remain "
                "inside [0, 1]."
            )

        return round(
            float(
                frequency
            ),
            6,
        )

    def candidate_recency(
        self,
        candidate_number: int,
    ) -> int:
        self.validate_candidate_number(
            candidate_number
        )

        last_seen_index = (
            self.candidate_last_seen.get(
                candidate_number
            )
        )

        if last_seen_index is None:
            return self.history_size

        return (
            self.history_size
            - 1
            - last_seen_index
        )

    def candidate_recency_ratio(
        self,
        candidate_number: int,
    ) -> float:
        recency = self.candidate_recency(
            candidate_number
        )

        return round(
            recency
            / self.history_size,
            6,
        )
