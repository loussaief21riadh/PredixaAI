from __future__ import annotations

from statistics import mean

from app.ai.v6b_clean.feature_builders.base import (
    BaseFeatureBuilder,
)
from app.ai.v6b_clean.utils.history import (
    main_numbers,
)


class GlobalStatisticsBuilder(BaseFeatureBuilder):
    """
    Builds the four global structural features
    used by V5.
    """

    def build(
        self,
        history,
        candidate_number=None,
    ) -> dict[str, int | float]:

        self.validate_history(history)

        sums = []
        even_counts = []
        consecutive_counts = []

        for draw in history:

            numbers = sorted(
                main_numbers(draw)
            )

            sums.append(
                sum(numbers)
            )

            even_counts.append(
                sum(
                    1
                    for n in numbers
                    if n % 2 == 0
                )
            )

            consecutive_counts.append(
                sum(
                    1
                    for i in range(
                        len(numbers) - 1
                    )
                    if (
                        numbers[i + 1]
                        == numbers[i] + 1
                    )
                )
            )

        return {
            "history_size": len(history),

            "average_sum": round(
                mean(sums),
                6,
            ),

            "average_even_count": round(
                mean(even_counts),
                6,
            ),

            "average_consecutive_pairs": round(
                mean(consecutive_counts),
                6,
            ),
        }