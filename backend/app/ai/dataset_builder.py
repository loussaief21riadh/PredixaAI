import pandas as pd
from sqlalchemy.orm import Session

from app.ai.feature_engineering import FeatureEngineering
from app.models.draw import Draw


class DatasetBuilder:
    """
    Predixa AI V4-D lagged dataset builder.

    Rule:
        To predict target T, features are built only
        from draws up to T-2.

    Therefore T-1 is intentionally excluded from X.

    This experiment is designed to prevent the model
    from directly learning to reproduce the immediately
    preceding draw.
    """

    WINDOW_SIZE = 100

    MODERN_LOTO_START_DATE = "2008-10-06"

    LAG_DRAWS = 1

    @staticmethod
    def build(
        db: Session,
        window_size: int | None = None,
    ):
        if window_size is None:
            window_size = DatasetBuilder.WINDOW_SIZE

        if window_size < 100:
            raise ValueError(
                "Predixa AI V4-D requires "
                "window_size >= 100."
            )

        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= DatasetBuilder.MODERN_LOTO_START_DATE
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        minimum_required = (
            window_size
            + DatasetBuilder.LAG_DRAWS
            + 1
        )

        if len(draws) < minimum_required:
            raise ValueError(
                "Not enough modern draws. "
                f"At least {minimum_required} are required."
            )

        feature_rows = []
        target_rows = []

        # --------------------------------------------------
        # Example with window_size=100
        #
        # target T = draws[101]
        #
        # T-1 = draws[100] -> intentionally excluded
        #
        # features:
        # draws[0:100]
        #
        # target:
        # draws[101]
        # --------------------------------------------------

        first_target_index = (
            window_size
            + DatasetBuilder.LAG_DRAWS
        )

        for target_index in range(
            first_target_index,
            len(draws),
        ):

            feature_end_index = (
                target_index
                - DatasetBuilder.LAG_DRAWS
            )

            feature_start_index = (
                feature_end_index
                - window_size
            )

            history = draws[
                feature_start_index:
                feature_end_index
            ]

            target_draw = draws[
                target_index
            ]

            if len(history) != window_size:
                raise ValueError(
                    "Invalid lagged history size."
                )

            features = (
                FeatureEngineering
                .build_from_history(
                    history,
                    window_size=window_size,
                )
            )

            target_numbers = {
                target_draw.n1,
                target_draw.n2,
                target_draw.n3,
                target_draw.n4,
                target_draw.n5,
            }

            targets = {
                f"target_{number}": (
                    1
                    if number in target_numbers
                    else 0
                )
                for number in range(
                    1,
                    50,
                )
            }

            feature_rows.append(
                features
            )

            target_rows.append(
                targets
            )

        X = pd.DataFrame(
            feature_rows
        )

        y = pd.DataFrame(
            target_rows
        )

        if X.empty or y.empty:
            raise ValueError(
                "The V4-D dataset is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "Feature and target datasets "
                "have different sizes."
            )

        if X.isnull().any().any():
            raise ValueError(
                "Feature dataset contains missing values."
            )

        if y.isnull().any().any():
            raise ValueError(
                "Target dataset contains missing values."
            )

        return X, y