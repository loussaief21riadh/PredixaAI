from statistics import mean

from app.models.draw import Draw


class DelayFeatures:
    """
    Predixa AI V4-B delay features.

    This ablation version intentionally removes
    the raw overdue_<number> features because they
    strongly identify the numbers from the most
    recent draw and may cause shortcut learning.

    Retained feature:
        overdue_ratio_<number>

    overdue_ratio compares the current delay with
    the historical average gap between appearances.
    """

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
    def _appearance_indices(
        history: list[Draw],
        number: int,
    ) -> list[int]:
        """
        Return chronological indices where
        the number appeared.
        """

        indices = []

        for index, draw in enumerate(
            history
        ):
            if number in (
                DelayFeatures
                ._main_numbers(draw)
            ):
                indices.append(
                    index
                )

        return indices

    @staticmethod
    def _current_overdue(
        history: list[Draw],
        number: int,
    ) -> int:
        """
        Number of draws since the latest appearance.
        """

        for index, draw in enumerate(
            reversed(history)
        ):
            if number in (
                DelayFeatures
                ._main_numbers(draw)
            ):
                return index

        return len(history)

    @staticmethod
    def _average_gap(
        history: list[Draw],
        number: int,
    ) -> float:
        """
        Calculate the historical average gap
        between consecutive appearances.
        """

        indices = (
            DelayFeatures
            ._appearance_indices(
                history,
                number,
            )
        )

        if len(indices) < 2:
            return float(
                len(history)
            )

        gaps = [
            indices[index]
            - indices[index - 1]
            for index in range(
                1,
                len(indices),
            )
        ]

        return float(
            mean(gaps)
        )

    @staticmethod
    def build(
        history: list[Draw],
    ) -> dict[str, float]:
        """
        Build 49 overdue-ratio features.

        Raw overdue features are intentionally
        excluded in this V4-B ablation experiment.
        """

        if not history:
            raise ValueError(
                "At least one historical draw is required."
            )

        features: dict[
            str,
            float
        ] = {}

        for number in range(
            1,
            50,
        ):

            overdue = (
                DelayFeatures
                ._current_overdue(
                    history,
                    number,
                )
            )

            average_gap = (
                DelayFeatures
                ._average_gap(
                    history,
                    number,
                )
            )

            safe_average_gap = max(
                average_gap,
                1.0,
            )

            overdue_ratio = (
                overdue
                / safe_average_gap
            )

            features[
                f"overdue_ratio_{number}"
            ] = round(
                overdue_ratio,
                6,
            )

        return features