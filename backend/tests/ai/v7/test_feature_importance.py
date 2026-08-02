from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from app.ai.v7.explainability.feature_importance import (
    FeatureImportanceAnalyzer,
)
from app.ai.v7.ranking_model import (
    V7RankingModel,
)


def build_trained_model() -> V7RankingModel:
    """
    Build and fit a real V7RankingModel using a synthetic dataset.
    """

    model = V7RankingModel()

    features, target = make_classification(
        n_samples=500,
        n_features=len(model.feature_columns),
        n_informative=max(
            2,
            len(model.feature_columns) // 2,
        ),
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )

    dataset = pd.DataFrame(
        features,
        columns=model.feature_columns,
    )

    dataset["target"] = target

    model.fit(dataset)

    return model


def test_analyzer_creation() -> None:

    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    assert analyzer is not None


def test_is_available() -> None:

    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    assert analyzer.is_available()


def test_feature_importances() -> None:

    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    result = analyzer.feature_importances()

    assert isinstance(
        result,
        dict,
    )

    assert len(result) == len(
        analyzer.model.feature_columns
    )


def test_sorted_feature_importances() -> None:

    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    values = analyzer.sorted_feature_importances()

    assert len(values) == len(
        analyzer.model.feature_columns
    )

    scores = [
        score
        for _, score in values
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )


def test_dataframe() -> None:

    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    dataframe = analyzer.to_dataframe()

    assert not dataframe.empty

    assert list(
        dataframe.columns
    ) == [
        "rank",
        "feature",
        "importance",
    ]

    assert len(
        dataframe
    ) == len(
        analyzer.model.feature_columns
    )

    assert dataframe["rank"].iloc[0] == 1


def test_csv_export(
    tmp_path: Path,
) -> None:

    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    output = (
        tmp_path
        / "feature_importance.csv"
    )

    csv_path = analyzer.to_csv(
        output
    )

    assert csv_path.exists()

    dataframe = pd.read_csv(
        csv_path
    )

    assert not dataframe.empty

    assert list(
        dataframe.columns
    ) == [
        "rank",
        "feature",
        "importance",
    ]


def test_invalid_model() -> None:

    with pytest.raises(
        ValueError,
    ):
        FeatureImportanceAnalyzer(
            object()
        )