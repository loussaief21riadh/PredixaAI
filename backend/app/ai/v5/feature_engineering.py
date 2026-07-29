from statistics import mean, pstdev

from app.models.draw import Draw


class V5FeatureEngineering:
    """
    Predixa AI V5-A Feature Engineering.

    Design principles:
    - Historical information only.
    - T-2 / purge policy is handled by the dataset/backtester.
    - Normalized frequency instead of raw counts.
    - Recency information.
    - Short-vs-long frequency signal.
    - Frequency volatility.
    - No pairwise co-occurrence yet.

    Per-number features:
        rate_10
        rate_20
        rate_50
        rate_100
        recency
        recency_ratio
        short_vs_long
        frequency_volatility

    49 * 8 = 392 number-level features

    Global features:
        history_size
        average_sum
        average_even_count
        average_consecutive_pairs

    Total expected features = 396
    """

    WINDOWS = (
        10,
        20,
        50,
        100,
    )

    @staticmethod
    def _main_numbers(
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
    def _frequency_rate(
        history: list[Draw],
        number: int,
        window: int,
    ) -> float:

        recent = history[
            -window:
        ]

        if not recent:
            return 0.0

        occurrences = sum(
            1
            for draw in recent
            if number
            in V5FeatureEngineering
            ._main_numbers(draw)
        )

        return (
            occurrences
            / len(recent)
        )

    @staticmethod
    def _recency(
        history: list[Draw],
        number: int,
    ) -> int:
        """
        Number of draws since the most recent appearance.

        0:
            number appeared in the latest allowed draw.

        1:
            appeared one draw before that.

        If absent from the complete feature history,
        return len(history).
        """

        for distance, draw in enumerate(
            reversed(history)
        ):

            if (
                number
                in V5FeatureEngineering
                ._main_numbers(draw)
            ):
                return distance

        return len(history)

    @staticmethod
    def build_from_history(
        draws: list[Draw],
        window_size: int = 100,
    ) -> dict[str, int | float]:

        if not draws:
            raise ValueError(
                "At least one historical draw is required."
            )

        if window_size < 100:
            raise ValueError(
                "Predixa V5-A requires window_size >= 100."
            )

        history = draws[
            -window_size:
        ]

        if len(history) < 100:
            raise ValueError(
                "Predixa V5-A requires at least "
                "100 historical draws."
            )

        # --------------------------------------------------
        # Global structural statistics
        # --------------------------------------------------

        sums = []
        even_counts = []
        consecutive_counts = []

        for draw in history:

            numbers = sorted(
                V5FeatureEngineering
                ._main_numbers(draw)
            )

            sums.append(
                sum(numbers)
            )

            even_counts.append(
                sum(
                    1
                    for number in numbers
                    if number % 2 == 0
                )
            )

            consecutive_counts.append(
                sum(
                    1
                    for index in range(
                        len(numbers) - 1
                    )
                    if (
                        numbers[index + 1]
                        == numbers[index] + 1
                    )
                )
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
        # Number-level V5 features
        # --------------------------------------------------

        for number in range(
            1,
            50,
        ):

            rates = {}

            for window in (
                V5FeatureEngineering.WINDOWS
            ):

                rates[
                    window
                ] = (
                    V5FeatureEngineering
                    ._frequency_rate(
                        history,
                        number,
                        window,
                    )
                )

                features[
                    f"rate_{window}_{number}"
                ] = round(
                    rates[window],
                    6,
                )

            # ----------------------------------------------
            # Recency
            # ----------------------------------------------

            recency = (
                V5FeatureEngineering
                ._recency(
                    history,
                    number,
                )
            )

            features[
                f"recency_{number}"
            ] = recency

            features[
                f"recency_ratio_{number}"
            ] = round(
                recency
                / len(history),
                6,
            )

            # ----------------------------------------------
            # Short-term versus long-term frequency
            #
            # Positive:
            # recent frequency > long-term frequency
            #
            # Negative:
            # recent frequency < long-term frequency
            # ----------------------------------------------

            features[
                f"short_vs_long_{number}"
            ] = round(
                rates[10]
                - rates[100],
                6,
            )

            # ----------------------------------------------
            # Frequency volatility
            #
            # Standard deviation across normalized
            # 10/20/50/100 rates.
            # ----------------------------------------------

            features[
                f"frequency_volatility_{number}"
            ] = round(
                pstdev(
                    [
                        rates[10],
                        rates[20],
                        rates[50],
                        rates[100],
                    ]
                ),
                6,
            )

        return features
