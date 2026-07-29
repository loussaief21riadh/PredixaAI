import pandas as pd
from sqlalchemy.orm import Session

from app.ai.v5.feature_engineering import V5FeatureEngineering
from app.models.draw import Draw


class V5DatasetBuilder:
    """
    Predixa AI V5-A time-aware dataset builder.

    Temporal rule
    -------------
    For target draw T:

        Features use exactly 100 historical draws
        ending at T-2.

        T-1 is excluded from the feature vector.

        Target y is T.

    Example
    -------
        history:
            draws[0:100]

        excluded:
            draws[100] = T-1

        target:
            draws[101] = T

    This preserves the lagged temporal protocol used
    by the validated V4-F benchmark.

    Important
    ---------
    This builder creates the dataset only.

    The additional V4-F training purge must still be
    applied by the V5 walk-forward backtester when
    evaluating future targets.
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
            window_size = (
                V5DatasetBuilder.WINDOW_SIZE
            )

        if window_size < 100:
            raise ValueError(
                "Predixa AI V5-A requires "
                "window_size >= 100."
            )

        # --------------------------------------------------
        # Load modern Loto draws chronologically
        # --------------------------------------------------

        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= V5DatasetBuilder
                .MODERN_LOTO_START_DATE
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        minimum_required = (
            window_size
            + V5DatasetBuilder.LAG_DRAWS
            + 1
        )

        if len(draws) < minimum_required:
            raise ValueError(
                "Not enough modern draws. "
                f"At least {minimum_required} "
                "draws are required."
            )

        feature_rows = []
        target_rows = []

        # --------------------------------------------------
        # First possible target:
        #
        # window_size = 100
        # lag = 1
        #
        # draws[0:100] = features
        # draws[100]   = excluded T-1
        # draws[101]   = target T
        # --------------------------------------------------

        first_target_index = (
            window_size
            + V5DatasetBuilder.LAG_DRAWS
        )

        for target_index in range(
            first_target_index,
            len(draws),
        ):

            # ----------------------------------------------
            # End is exclusive.
            #
            # target_index = T
            # feature_end_index = T-1
            #
            # Therefore the last included draw is T-2.
            # ----------------------------------------------

            feature_end_index = (
                target_index
                - V5DatasetBuilder.LAG_DRAWS
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
                    "Invalid V5-A lagged "
                    "history size."
                )

            # ----------------------------------------------
            # Build V5-A features
            # ----------------------------------------------

            features = (
                V5FeatureEngineering
                .build_from_history(
                    history,
                    window_size=window_size,
                )
            )

            # ----------------------------------------------
            # 49 binary targets
            # ----------------------------------------------

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

        # --------------------------------------------------
        # DataFrames
        # --------------------------------------------------

        X = pd.DataFrame(
            feature_rows
        )

        y = pd.DataFrame(
            target_rows
        )

        # --------------------------------------------------
        # Safety checks
        # --------------------------------------------------

        if X.empty or y.empty:
            raise ValueError(
                "The V5-A dataset is empty."
            )

        if len(X) != len(y):
            raise ValueError(
                "V5-A feature and target "
                "datasets have different sizes."
            )

        if X.isnull().any().any():
            raise ValueError(
                "V5-A feature dataset "
                "contains missing values."
            )

        if y.isnull().any().any():
            raise ValueError(
                "V5-A target dataset "
                "contains missing values."
            )

        expected_features = (
            4
            + 49 * 8
        )

        if X.shape[1] != expected_features:
            raise ValueError(
                "Unexpected V5-A feature count. "
                f"Expected {expected_features}, "
                f"received {X.shape[1]}."
            )

        if y.shape[1] != 49:
            raise ValueError(
                "Unexpected V5-A target count. "
                f"Expected 49, "
                f"received {y.shape[1]}."
            )

        return X, y
