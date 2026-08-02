from __future__ import annotations

import json
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
    Build and fit a real V7RankingModel using synthetic data.
    """

    model = V7RankingModel()

    features, target = make_classification(
        n_samples=500,
        n_features=len(
            model.feature_columns
        ),
        n_informative=max(
            2,
            len(
                model.feature_columns
            )
            // 2,
        ),
        n_redundant=0,
        n_repeated=0,
        random_state=42,
    )

    dataset = pd.DataFrame(
        features,
        columns=model.feature_columns,
    )

    dataset[
        "target"
    ] = target

    model.fit(
        dataset
    )

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

    assert analyzer.is_available() is True


def test_feature_importances() -> None:
    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    result = analyzer.feature_importances()

    assert isinstance(
        result,
        dict,
    )

    assert len(
        result
    ) == len(
        analyzer.model.feature_columns
    )

    assert set(
        result
    ) == set(
        analyzer.model.feature_columns
    )

    assert all(
        isinstance(
            value,
            float,
        )
        for value in result.values()
    )


def test_sorted_feature_importances() -> None:
    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    values = (
        analyzer.sorted_feature_importances()
    )

    assert len(
        values
    ) == len(
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

    assert dataframe[
        "rank"
    ].tolist() == list(
        range(
            1,
            len(
                dataframe
            )
            + 1,
        )
    )

    assert not dataframe[
        "feature"
    ].duplicated().any()

    assert dataframe[
        "importance"
    ].notnull().all()


def test_csv_export(
    tmp_path: Path,
) -> None:
    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    output_path = (
        tmp_path
        / "feature_importance.csv"
    )

    csv_path = analyzer.to_csv(
        output_path
    )

    assert csv_path.exists()
    assert csv_path.is_file()
    assert csv_path.suffix == ".csv"
    assert csv_path.stat().st_size > 0

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

    assert len(
        dataframe
    ) == len(
        analyzer.model.feature_columns
    )


def test_json_export(
    tmp_path: Path,
) -> None:
    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    output_path = (
        tmp_path
        / "feature_importance.json"
    )

    json_path = analyzer.to_json(
        output_path
    )

    assert json_path.exists()
    assert json_path.is_file()
    assert json_path.suffix == ".json"
    assert json_path.stat().st_size > 0

    with json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    assert isinstance(
        payload,
        list,
    )

    assert len(
        payload
    ) == len(
        analyzer.model.feature_columns
    )

    assert payload

    assert set(
        payload[
            0
        ]
    ) == {
        "rank",
        "feature",
        "importance",
    }


def test_text_report() -> None:
    analyzer = FeatureImportanceAnalyzer(
        build_trained_model()
    )

    report = analyzer.to_text()

    assert isinstance(
        report,
        str,
    )

    assert report.strip()

    assert (
        "PredixaAI Feature Importance Report"
        in report
    )

    for feature_name in (
        analyzer.model.feature_columns
    ):
        assert feature_name in report


def test_invalid_model() -> None:
    with pytest.raises(
        ValueError,
    ):
        FeatureImportanceAnalyzer(
            object()
        )