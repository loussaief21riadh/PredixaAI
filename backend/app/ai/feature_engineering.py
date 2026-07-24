from statistics import mean

from app.ai.features.frequency import FrequencyFeatures
from app.models.draw import Draw


class FeatureEngineering:
    """
    Predixa AI V4-E Feature Engineering.

    Experimental configuration:

    - T-2 lag is handled by the dataset/backtester.
    - Delay features removed.
    - Momentum features removed.
    - Trend features removed.
    - Raw frequency features retained.

    Expected feature count:

        4 global structural features
        196 frequency features

        Total = 200 features
    """

    @staticmethod
    def _get_main_numbers(
        draw: Draw,
    ) -> list[int]:

        return [
            draw.n1,
            draw.n2,
            draw.n3,
            draw.n4,
            draw.n5,
        ]

    @staticmethod
    def build_from_history(
        draws: list[Draw],
        window_size: int = 100,
    ) -> dict[str, int | float]:

        if not draws:
            raise ValueError(
                "At least one historical draw is required."
            )

        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero."
            )

        history = draws[
            -window_size:
        ]

        sums: list[int] = []
        even_counts: list[int] = []
        consecutive_counts: list[int] = []

        # --------------------------------------------------
        # Global structural statistics
        # --------------------------------------------------

        for draw in history:

            numbers = sorted(
                FeatureEngineering
                ._get_main_numbers(
                    draw
                )
            )

            sums.append(
                sum(numbers)
            )

            even_count = sum(
                1
                for number in numbers
                if number % 2 == 0
            )

            even_counts.append(
                even_count
            )

            consecutive_count = sum(
                1
                for index in range(
                    len(numbers) - 1
                )
                if (
                    numbers[index + 1]
                    == numbers[index] + 1
                )
            )

            consecutive_counts.append(
                consecutive_count
            )

        features: dict[
            str,
            int | float
        ] = {
            "history_size": len(
                history
            ),

            "average_sum": round(
                mean(sums),
                6,
            ),

            "average_even_count": round(
                mean(even_counts),
                6,
            ),

            "average_consecutive_pairs": round(
                mean(
                    consecutive_counts
                ),
                6,
            ),
        }

        # --------------------------------------------------
        # Raw frequency features only
        #
        # freq_10_1 ... freq_10_49
        # freq_20_1 ... freq_20_49
        # freq_50_1 ... freq_50_49
        # freq_100_1 ... freq_100_49
        #
        # Total: 196
        # --------------------------------------------------

        (
            frequency_features,
            _frequency_by_window,
        ) = FrequencyFeatures.build(
            history
        )

        features.update(
            frequency_features
        )

        # --------------------------------------------------
        # Intentionally removed in V4-E
        #
        # overdue_*
        # overdue_ratio_*
        # momentum_*
        # trend_*
        # --------------------------------------------------

        return features