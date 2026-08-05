from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
)
from app.ai.v7.ranking_dataset import (
    V7RankingDataset,
)
from app.ai.v7.ranking_model import (
    V7RankingModel,
)


EXPECTED_PRODUCTION_FEATURES = (
    "history_size",
    "average_sum",
    "average_even_count",
    "average_consecutive_pairs",
    "rate_20",
    "rate_50",
    "rate_100",
    "recency",
    "recency_ratio",
    "short_vs_long",
    "frequency_volatility",
)


def build_complete_feature_dictionary() -> dict[
    str,
    int | float,
]:
    """Build one complete assembler-compatible prediction dictionary."""

    features: dict[
        str,
        int | float,
    ] = {
        "history_size": 100,
        "average_sum": 125.0,
        "average_even_count": 2.5,
        "average_consecutive_pairs": 0.4,
    }

    for family in (
        V7RankingDataset
        .CANDIDATE_FEATURES
    ):
        for number in range(
            V7RankingDataset.NUMBER_MIN,
            V7RankingDataset.NUMBER_MAX + 1,
        ):
            features[
                f"{family}_{number}"
            ] = (
                number / 100.0
            )

    assert (
        len(features)
        == V7RankingDataset
        .EXPECTED_FULL_FEATURE_COUNT
    )

    return features


def build_training_dataset() -> pd.DataFrame:
    """Build a minimal valid binary training dataset."""

    rows: list[
        dict[str, Any]
    ] = []

    for target in (
        0,
        1,
    ):
        row: dict[
            str,
            Any,
        ] = {
            feature: (
                float(index + target)
            )
            for index, feature
            in enumerate(
                EXPECTED_PRODUCTION_FEATURES
            )
        }
        row["target"] = target
        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def test_production_feature_contract_is_exactly_eleven_features() -> None:
    features = tuple(
        V7RankingDataset
        .feature_columns()
    )

    assert features == (
        EXPECTED_PRODUCTION_FEATURES
    )
    assert len(features) == 11
    assert len(set(features)) == 11


def test_rate_10_is_pruned_but_short_vs_long_is_retained() -> None:
    features = set(
        V7RankingDataset
        .feature_columns()
    )

    assert "rate_10" not in features
    assert "short_vs_long" in features


def test_rate_10_remains_in_engineered_candidate_schema() -> None:
    """
    Feature engineering remains backward-compatible.

    rate_10 can still be generated and exported, but it must not enter the
    Random Forest feature matrix.
    """

    assert (
        "rate_10"
        in V7RankingDataset
        .CANDIDATE_FEATURES
    )
    assert (
        "rate_10"
        not in V7RankingDataset
        .MODEL_FEATURES
    )


def test_pruned_model_feature_marker() -> None:
    assert hasattr(
        V7RankingDataset,
        "PRUNED_MODEL_FEATURES",
    )
    assert (
        V7RankingDataset
        .PRUNED_MODEL_FEATURES
        == (
            "rate_10",
        )
    )


def test_ranking_model_uses_dynamic_eleven_feature_contract() -> None:
    assert (
        V7RankingModel
        .EXPECTED_FEATURE_COUNT
        == 11
    )

    model = V7RankingModel(
        n_estimators=1,
        max_depth=2,
        random_state=7,
    )

    assert tuple(
        model.feature_columns
    ) == EXPECTED_PRODUCTION_FEATURES
    assert len(
        model.feature_columns
    ) == 11


def test_feature_family_configuration_matches_active_model() -> None:
    configured = tuple(
        feature
        for family_name
        in FEATURE_FAMILY_ORDER
        for feature
        in FEATURE_FAMILIES[
            family_name
        ]
    )

    assert set(
        configured
    ) == set(
        EXPECTED_PRODUCTION_FEATURES
    )
    assert len(configured) == len(
        set(configured)
    )
    assert "rate_10" not in configured


def test_prediction_dataset_retains_engineered_rate_10_column() -> None:
    model = V7RankingModel(
        n_estimators=1,
        max_depth=2,
        random_state=7,
    )

    prediction_dataset = (
        model
        .build_prediction_dataset(
            build_complete_feature_dictionary()
        )
    )

    assert len(
        prediction_dataset
    ) == 49
    assert (
        prediction_dataset[
            "candidate_number"
        ].tolist()
        == list(
            range(
                1,
                50,
            )
        )
    )

    assert (
        "rate_10"
        in prediction_dataset.columns
    )
    assert (
        "rate_10"
        not in model.feature_columns
    )

    model_matrix = (
        prediction_dataset[
            model.feature_columns
        ]
    )

    assert model_matrix.shape == (
        49,
        11,
    )


def test_training_validation_does_not_require_rate_10() -> None:
    model = V7RankingModel(
        n_estimators=1,
        max_depth=2,
        random_state=7,
    )

    dataset = (
        build_training_dataset()
    )

    assert "rate_10" not in dataset.columns

    model._validate_training_dataset(
        dataset
    )


def test_training_validation_still_requires_retained_features() -> None:
    model = V7RankingModel(
        n_estimators=1,
        max_depth=2,
        random_state=7,
    )

    dataset = (
        build_training_dataset()
        .drop(
            columns=[
                "short_vs_long",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Training dataset is missing columns"
        ),
    ):
        model._validate_training_dataset(
            dataset
        )


def test_prediction_result_reports_eleven_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = V7RankingModel(
        n_estimators=1,
        max_depth=2,
        random_state=7,
    )

    model.is_fitted = True

    monkeypatch.setattr(
        model,
        "_positive_probabilities",
        lambda X: pd.Series(
            [
                float(
                    50 - index
                )
                / 50.0
                for index in range(
                    1,
                    50,
                )
            ]
        ).to_numpy(),
    )

    result = model.predict_top_k(
        build_complete_feature_dictionary(),
        top_k=5,
    )

    assert result["feature_count"] == 11
    assert result["candidate_count"] == 49
    assert len(
        result["predicted_numbers"]
    ) == 5
