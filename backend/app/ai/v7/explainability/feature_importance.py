from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ai.v7.ranking_model import V7RankingModel


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance from a fitted V7RankingModel.

    The analyzer exposes reusable methods to:

    - verify feature-importance availability;
    - return raw feature importance values;
    - return sorted feature importance values;
    - build a ranked pandas DataFrame;
    - export the ranked results to CSV.
    """

    DATAFRAME_COLUMNS = (
        "rank",
        "feature",
        "importance",
    )

    def __init__(
        self,
        model: V7RankingModel,
    ) -> None:
        if not isinstance(
            model,
            V7RankingModel,
        ):
            raise ValueError(
                "model must be a V7RankingModel."
            )

        self.model = model

    def is_available(
        self,
    ) -> bool:
        """
        Return True when the fitted underlying estimator exposes
        feature_importances_.
        """

        if not self.model.is_fitted:
            return False

        return hasattr(
            self.model.model,
            "feature_importances_",
        )

    def feature_importances(
        self,
    ) -> dict[str, float]:
        """
        Return feature importances mapped by feature name.
        """

        if not self.model.is_fitted:
            raise ValueError(
                "Model must be fitted before extracting "
                "feature importances."
            )

        if not self.is_available():
            raise ValueError(
                "Underlying model does not expose "
                "feature importances."
            )

        raw_importances = (
            self.model.model.feature_importances_
        )

        feature_names = list(
            self.model.feature_columns
        )

        if len(raw_importances) != len(feature_names):
            raise ValueError(
                "Feature importance length mismatch. "
                f"Expected {len(feature_names)}, "
                f"received {len(raw_importances)}."
            )

        importances = {
            feature_name: float(importance)
            for feature_name, importance in zip(
                feature_names,
                raw_importances,
            )
        }

        if len(importances) != len(feature_names):
            raise ValueError(
                "Feature importance mapping contains "
                "duplicate feature names."
            )

        return importances

    def sorted_feature_importances(
        self,
    ) -> list[tuple[str, float]]:
        """
        Return feature importances sorted by descending importance.

        Feature names are used as deterministic tie-breakers.
        """

        return sorted(
            self.feature_importances().items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

    def to_dataframe(
        self,
    ) -> pd.DataFrame:
        """
        Return a ranked feature-importance DataFrame.

        Columns:
            rank
            feature
            importance
        """

        rows = [
            {
                "rank": rank,
                "feature": feature_name,
                "importance": importance,
            }
            for rank, (
                feature_name,
                importance,
            ) in enumerate(
                self.sorted_feature_importances(),
                start=1,
            )
        ]

        dataframe = pd.DataFrame(
            rows,
            columns=list(
                self.DATAFRAME_COLUMNS
            ),
        )

        if dataframe.empty:
            raise ValueError(
                "Feature importance DataFrame is empty."
            )

        expected_rows = len(
            self.model.feature_columns
        )

        if len(dataframe) != expected_rows:
            raise ValueError(
                "Unexpected feature importance row count. "
                f"Expected {expected_rows}, "
                f"received {len(dataframe)}."
            )

        if dataframe[
            "feature"
        ].duplicated().any():
            raise ValueError(
                "Feature importance DataFrame contains "
                "duplicate feature names."
            )

        if dataframe[
            "importance"
        ].isnull().any():
            raise ValueError(
                "Feature importance DataFrame contains "
                "missing importance values."
            )

        return dataframe

    def to_csv(
        self,
        output_path: str | Path,
    ) -> Path:
        """
        Export ranked feature importances to a CSV file.

        Parameters
        ----------
        output_path
            Destination CSV path.

        Returns
        -------
        Path
            Resolved path of the generated CSV file.
        """

        destination = Path(
            output_path
        ).expanduser()

        if destination.suffix.lower() != ".csv":
            raise ValueError(
                "output_path must use the .csv extension."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = self.to_dataframe()

        dataframe.to_csv(
            destination,
            index=False,
        )

        if not destination.exists():
            raise ValueError(
                "CSV export failed."
            )

        if destination.stat().st_size == 0:
            raise ValueError(
                "Generated CSV file is empty."
            )

        return destination.resolve()