from collections import Counter

from app.models.draw import Draw


class FrequencyFeatures:
    """
    Predixa AI V4 frequency features.

    Supports two modes:

    Normal mode:
        Uses the complete supplied history.

        Example when predicting T:
            [..., T-3, T-2, T-1]

    Diagnostic lag mode:
        Excludes the most recent historical draw.

        Example when predicting T:
            [..., T-3, T-2]

    This diagnostic mode helps determine whether the
    Random Forest is learning to reproduce T-1.
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
        """
        Return the five main numbers from one draw.
        """

        return [
            draw.n1,
            draw.n2,
            draw.n3,
            draw.n4,
            draw.n5,
        ]

    @staticmethod
    def _frequency_for_window(
        draws: list[Draw],
        window_size: int,
    ) -> Counter:
        """
        Calculate raw occurrence counts over the
        requested historical window.
        """

        recent_draws = draws[
            -window_size:
        ]

        counter = Counter()

        for draw in recent_draws:
            counter.update(
                FrequencyFeatures
                ._main_numbers(
                    draw
                )
            )

        return counter

    @staticmethod
    def build(
        history: list[Draw],
        exclude_latest: bool = False,
    ) -> tuple[
        dict[str, int],
        dict[int, Counter],
    ]:
        """
        Build frequency features.

        Parameters
        ----------
        history:
            Chronological historical draws.

        exclude_latest:
            False:
                Use the complete history.

            True:
                Remove the most recent historical draw
                before calculating frequencies.

                This is intended for diagnostic ablation
                experiments.

        Returns
        -------
        tuple:
            features:
                196 frequency features.

            frequency_by_window:
                Counter objects reused by momentum
                and trend calculations.
        """

        if not history:
            raise ValueError(
                "At least one historical draw is required."
            )

        if exclude_latest:

            if len(history) < 2:
                raise ValueError(
                    "At least two historical draws are "
                    "required when exclude_latest=True."
                )

            working_history = history[
                :-1
            ]

        else:
            working_history = history

        features: dict[
            str,
            int
        ] = {}

        frequency_by_window: dict[
            int,
            Counter
        ] = {}

        for window in (
            FrequencyFeatures.WINDOWS
        ):

            actual_window = min(
                window,
                len(
                    working_history
                ),
            )

            counter = (
                FrequencyFeatures
                ._frequency_for_window(
                    working_history,
                    actual_window,
                )
            )

            frequency_by_window[
                window
            ] = counter

            for number in range(
                1,
                50,
            ):

                features[
                    f"freq_{window}_{number}"
                ] = counter.get(
                    number,
                    0,
                )

        return (
            features,
            frequency_by_window,
        )