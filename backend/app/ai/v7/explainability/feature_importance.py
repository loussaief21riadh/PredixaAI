from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.ai.v7.ranking_model import V7RankingModel


class FeatureImportanceAnalyzer:
    """
    Analyze feature importance from a fitted V7RankingModel.

    Features:
        • availability check
        • raw importances
        • sorted importances
        • pandas dataframe
        • CSV export
        • JSON export
        • text report
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

        if not self.model.is_fitted:
            return False

        return hasattr(
            self.model.model,
            "feature_importances_",
        )

    def feature_importances(
        self,
    ) -> dict[str, float]:

        if not self.model.is_fitted:
            raise ValueError(
                "Model must be fitted."
            )

        if not self.is_available():
            raise ValueError(
                "Feature importances are unavailable."
            )

        importances = (
            self.model.model.feature_importances_
        )

        feature_names = (
            self.model.feature_columns
        )

        if len(importances) != len(feature_names):
            raise ValueError(
                "Feature importance length mismatch."
            )

        return {
            feature: float(score)
            for feature, score in zip(
                feature_names,
                importances,
            )
        }

    def sorted_feature_importances(
        self,
    ) -> list[
        tuple[str, float]
    ]:

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

        rows = []

        for rank, (
            feature,
            importance,
        ) in enumerate(
            self.sorted_feature_importances(),
            start=1,
        ):

            rows.append(
                {
                    "rank": rank,
                    "feature": feature,
                    "importance": importance,
                }
            )

        dataframe = pd.DataFrame(
            rows,
            columns=list(
                self.DATAFRAME_COLUMNS
            ),
        )

        if dataframe.empty:
            raise ValueError(
                "Feature importance dataframe is empty."
            )

        return dataframe

    def to_csv(
        self,
        output_path: str | Path,
    ) -> Path:

        output_path = Path(
            output_path
        ).expanduser()

        if output_path.suffix.lower() != ".csv":
            raise ValueError(
                "CSV filename must end with .csv"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.to_dataframe().to_csv(
            output_path,
            index=False,
        )

        if not output_path.exists():
            raise ValueError(
                "CSV export failed."
            )

        return output_path.resolve()

    def to_json(
        self,
        output_path: str | Path,
    ) -> Path:
        """
        Export feature importance to JSON.
        """

        output_path = Path(
            output_path
        ).expanduser()

        if output_path.suffix.lower() != ".json":
            raise ValueError(
                "JSON filename must end with .json"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = self.to_dataframe()

        records = dataframe.to_dict(
            orient="records"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                records,
                file,
                indent=4,
            )

        if not output_path.exists():
            raise ValueError(
                "JSON export failed."
            )

        return output_path.resolve()

    def to_text(
        self,
    ) -> str:
        """
        Return a formatted text report.
        """

        dataframe = self.to_dataframe()

        lines = []

        lines.append(
            "=" * 60
        )

        lines.append(
            "PredixaAI Feature Importance Report"
        )

        lines.append(
            "=" * 60
        )

        lines.append("")

        for _, row in dataframe.iterrows():

            lines.append(
                f"{int(row['rank']):>2}. "
                f"{row['feature']:<35}"
                f"{row['importance']:.6f}"
            )

        return "\n".join(
            lines
        )