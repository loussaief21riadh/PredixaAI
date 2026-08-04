from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.ai.v7.explainability.feature_correlation_report import (
    V7FeatureCorrelationReport,
)
from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
)
from app.ai.v7.ranking_dataset import (
    V7RankingDataset,
)


MODEL_FEATURES = (
    V7RankingDataset.feature_columns()
)


def make_valid_dataset(
    target_count: int = 4,
) -> pd.DataFrame:
    """
    Build a valid synthetic V7 candidate-level dataset.

    Each target contains:
        - 49 candidate rows;
        - exactly 5 positive labels;
        - all V7 model features;
        - deterministic numeric feature values.
    """

    rows: list[dict[str, Any]] = []

    for target_offset in range(
        target_count
    ):
        target_draw_index = (
            target_offset + 1
        )

        target_draw_date = (
            pd.Timestamp("2026-01-01")
            + pd.Timedelta(
                days=target_offset,
            )
        ).date().isoformat()

        for candidate_number in range(
            1,
            50,
        ):
            normalized_candidate = (
                candidate_number
                / 49.0
            )

            rate_10 = (
                normalized_candidate
                + target_offset
                * 0.01
            )

            row: dict[str, Any] = {
                "candidate_number": (
                    candidate_number
                ),
                "target": int(
                    candidate_number
                    in {
                        1,
                        2,
                        3,
                        4,
                        5,
                    }
                ),
                "target_draw_index": (
                    target_draw_index
                ),
                "target_draw_date": (
                    target_draw_date
                ),
                "history_size": 100.0,
                "average_sum": (
                    100.0
                    + target_offset
                ),
                "average_even_count": (
                    2.0
                    + target_offset
                    * 0.1
                ),
                "average_consecutive_pairs": (
                    target_offset
                    * 0.05
                ),
                "rate_10": rate_10,
                "rate_20": (
                    normalized_candidate
                    * 0.80
                    + target_offset
                    * 0.02
                ),
                "rate_50": (
                    normalized_candidate
                    * 0.50
                    + target_offset
                    * 0.03
                ),
                "rate_100": (
                    normalized_candidate
                    * 0.25
                    + target_offset
                    * 0.04
                ),
                "recency": (
                    1.0
                    - normalized_candidate
                    + target_offset
                    * 0.001
                ),
                "recency_ratio": (
                    1.0
                    - normalized_candidate
                    + target_offset
                    * 0.001
                ),
                "short_vs_long": (
                    rate_10
                    * 2.0
                ),
                "frequency_volatility": (
                    abs(
                        0.5
                        - normalized_candidate
                    )
                    + target_offset
                    * 0.005
                ),
            }

            rows.append(
                row
            )

    dataset = pd.DataFrame(
        rows
    )

    return dataset


@pytest.mark.parametrize(
    (
        "overrides",
        "expected_message",
    ),
    [
        (
            {
                "window_size": 99,
            },
            "window_size must be at least 100",
        ),
        (
            {
                "max_training_targets": -1,
            },
            "max_training_targets cannot be negative",
        ),
        (
            {
                "correlation_threshold": 0.0,
            },
            "correlation_threshold must be greater than 0",
        ),
        (
            {
                "correlation_threshold": 1.1,
            },
            "correlation_threshold must be greater than 0",
        ),
        (
            {
                "top_pairs": 0,
            },
            "top_pairs must be at least 1",
        ),
    ],
)
def test_validate_parameters_rejects_invalid_values(
    overrides: dict[str, int | float],
    expected_message: str,
) -> None:
    parameters: dict[
        str,
        int | float
    ] = {
        "window_size": 100,
        "max_training_targets": 1500,
        "correlation_threshold": 0.80,
        "top_pairs": 20,
    }

    parameters.update(
        overrides
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        (
            V7FeatureCorrelationReport
            ._validate_parameters(
                window_size=int(
                    parameters[
                        "window_size"
                    ]
                ),
                max_training_targets=int(
                    parameters[
                        "max_training_targets"
                    ]
                ),
                correlation_threshold=float(
                    parameters[
                        "correlation_threshold"
                    ]
                ),
                top_pairs=int(
                    parameters[
                        "top_pairs"
                    ]
                ),
            )
        )


def test_validate_parameters_accepts_valid_values(
) -> None:
    (
        V7FeatureCorrelationReport
        ._validate_parameters(
            window_size=100,
            max_training_targets=1500,
            correlation_threshold=0.80,
            top_pairs=20,
        )
    )


def test_validate_feature_configuration(
) -> None:
    (
        V7FeatureCorrelationReport
        ._validate_feature_configuration()
    )

    configured_features = {
        feature_name
        for family_name in (
            FEATURE_FAMILY_ORDER
        )
        for feature_name in (
            FEATURE_FAMILIES[
                family_name
            ]
        )
    }

    assert configured_features == set(
        MODEL_FEATURES
    )


def test_validate_dataset_accepts_valid_dataset(
) -> None:
    dataset = make_valid_dataset()

    (
        V7FeatureCorrelationReport
        ._validate_dataset(
            dataset
        )
    )


def test_validate_dataset_rejects_missing_required_column(
) -> None:
    dataset = make_valid_dataset()

    dataset = dataset.drop(
        columns=[
            "target_draw_date",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Dataset is missing required columns",
    ):
        (
            V7FeatureCorrelationReport
            ._validate_dataset(
                dataset
            )
        )


def test_validate_dataset_rejects_missing_model_feature(
) -> None:
    dataset = make_valid_dataset()

    dataset = dataset.drop(
        columns=[
            "rate_10",
        ]
    )

    with pytest.raises(
        ValueError,
        match="Dataset is missing V7 model features",
    ):
        (
            V7FeatureCorrelationReport
            ._validate_dataset(
                dataset
            )
        )


def test_validate_dataset_rejects_missing_values(
) -> None:
    dataset = make_valid_dataset()

    dataset.loc[
        0,
        "rate_10",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="Dataset contains missing values",
    ):
        (
            V7FeatureCorrelationReport
            ._validate_dataset(
                dataset
            )
        )


def test_validate_dataset_rejects_non_finite_values(
) -> None:
    dataset = make_valid_dataset()

    dataset.loc[
        0,
        "rate_10",
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="non-finite values",
    ):
        (
            V7FeatureCorrelationReport
            ._validate_dataset(
                dataset
            )
        )


def test_validate_dataset_rejects_invalid_rows_per_target(
) -> None:
    dataset = make_valid_dataset()

    dataset = (
        dataset
        .drop(
            index=dataset.index[0]
        )
        .reset_index(
            drop=True
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Every target must contain exactly "
            "49 candidate rows"
        ),
    ):
        (
            V7FeatureCorrelationReport
            ._validate_dataset(
                dataset
            )
        )


def test_validate_dataset_rejects_invalid_positive_count(
) -> None:
    dataset = make_valid_dataset()

    dataset.loc[
        (
            dataset[
                "target_draw_index"
            ]
            == 1
        )
        & (
            dataset[
                "candidate_number"
            ]
            == 5
        ),
        "target",
    ] = 0

    with pytest.raises(
        ValueError,
        match=(
            "Every target must contain exactly "
            "5 positive labels"
        ),
    ):
        (
            V7FeatureCorrelationReport
            ._validate_dataset(
                dataset
            )
        )


def test_feature_dataframe_contains_only_model_features(
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    assert list(
        feature_dataframe.columns
    ) == MODEL_FEATURES

    assert len(
        feature_dataframe
    ) == len(
        dataset
    )

    assert all(
        pd.api.types.is_float_dtype(
            feature_dataframe[
                feature_name
            ]
        )
        for feature_name in MODEL_FEATURES
    )


@pytest.mark.parametrize(
    "method",
    [
        "pearson",
        "spearman",
    ],
)
def test_correlation_matrix(
    method: str,
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe=(
                feature_dataframe
            ),
            method=method,
        )
    )

    assert matrix.shape == (
        len(
            MODEL_FEATURES
        ),
        len(
            MODEL_FEATURES
        ),
    )

    assert list(
        matrix.columns
    ) == MODEL_FEATURES

    assert list(
        matrix.index
    ) == MODEL_FEATURES

    assert matrix.loc[
        "rate_10",
        "rate_10",
    ] == pytest.approx(
        1.0
    )

    assert matrix.loc[
        "history_size",
        "history_size",
    ] != matrix.loc[
        "history_size",
        "history_size",
    ]


def test_correlation_matrix_rejects_invalid_method(
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Correlation method must be "
            "pearson or spearman"
        ),
    ):
        (
            V7FeatureCorrelationReport
            ._correlation_matrix(
                feature_dataframe=(
                    feature_dataframe
                ),
                method="kendall",
            )
        )


def test_clean_float(
) -> None:
    assert (
        V7FeatureCorrelationReport
        ._clean_float(
            0.5
        )
        == pytest.approx(
            0.5
        )
    )

    assert (
        V7FeatureCorrelationReport
        ._clean_float(
            None
        )
        is None
    )

    assert (
        V7FeatureCorrelationReport
        ._clean_float(
            np.nan
        )
        is None
    )

    assert (
        V7FeatureCorrelationReport
        ._clean_float(
            np.inf
        )
        is None
    )


def test_matrix_to_dictionary_converts_nan_to_none(
) -> None:
    matrix = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                np.nan,
            ],
            "feature_b": [
                np.nan,
                1.0,
            ],
        },
        index=[
            "feature_a",
            "feature_b",
        ],
    )

    result = (
        V7FeatureCorrelationReport
        ._matrix_to_dictionary(
            matrix
        )
    )

    assert result[
        "feature_a"
    ][
        "feature_a"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "feature_a"
    ][
        "feature_b"
    ] is None

    assert result[
        "feature_b"
    ][
        "feature_a"
    ] is None


def test_build_pair_dataframe_detects_high_correlations(
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    pearson_matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe=(
                feature_dataframe
            ),
            method="pearson",
        )
    )

    spearman_matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe=(
                feature_dataframe
            ),
            method="spearman",
        )
    )

    pair_dataframe = (
        V7FeatureCorrelationReport
        ._build_pair_dataframe(
            pearson_matrix=(
                pearson_matrix
            ),
            spearman_matrix=(
                spearman_matrix
            ),
            correlation_threshold=0.80,
        )
    )

    expected_pair_count = (
        len(
            MODEL_FEATURES
        )
        * (
            len(
                MODEL_FEATURES
            )
            - 1
        )
        // 2
    )

    assert len(
        pair_dataframe
    ) == expected_pair_count

    recency_pair = pair_dataframe[
        (
            pair_dataframe[
                "feature_a"
            ]
            == "recency"
        )
        & (
            pair_dataframe[
                "feature_b"
            ]
            == "recency_ratio"
        )
    ]

    assert len(
        recency_pair
    ) == 1

    recency_row = recency_pair.iloc[0]

    assert recency_row[
        "pearson"
    ] == pytest.approx(
        1.0
    )

    assert recency_row[
        "spearman"
    ] == pytest.approx(
        1.0
    )

    assert bool(
        recency_row[
            "high_correlation"
        ]
    ) is True

    assert recency_row[
        "family_a"
    ] == "recency"

    assert recency_row[
        "family_b"
    ] == "recency"


def test_build_feature_summary_detects_constant_feature(
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    pearson_matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe,
            method="pearson",
        )
    )

    spearman_matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe,
            method="spearman",
        )
    )

    pair_dataframe = (
        V7FeatureCorrelationReport
        ._build_pair_dataframe(
            pearson_matrix=(
                pearson_matrix
            ),
            spearman_matrix=(
                spearman_matrix
            ),
            correlation_threshold=0.80,
        )
    )

    summary = (
        V7FeatureCorrelationReport
        ._build_feature_summary(
            feature_dataframe=(
                feature_dataframe
            ),
            pair_dataframe=(
                pair_dataframe
            ),
            correlation_threshold=0.80,
        )
    )

    history_row = summary[
        summary[
            "feature"
        ]
        == "history_size"
    ].iloc[0]

    assert history_row[
        "unique_values"
    ] == 1

    assert history_row[
        "standard_deviation"
    ] == pytest.approx(
        0.0
    )

    assert bool(
        history_row[
            "is_constant"
        ]
    ) is True

    recency_row = summary[
        summary[
            "feature"
        ]
        == "recency"
    ].iloc[0]

    assert recency_row[
        "strongest_pearson_partner"
    ] == "recency_ratio"

    assert recency_row[
        "strongest_pearson"
    ] == pytest.approx(
        1.0
    )


def test_build_family_summary(
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    pearson_matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe,
            method="pearson",
        )
    )

    spearman_matrix = (
        V7FeatureCorrelationReport
        ._correlation_matrix(
            feature_dataframe,
            method="spearman",
        )
    )

    pair_dataframe = (
        V7FeatureCorrelationReport
        ._build_pair_dataframe(
            pearson_matrix=(
                pearson_matrix
            ),
            spearman_matrix=(
                spearman_matrix
            ),
            correlation_threshold=0.80,
        )
    )

    family_summary = (
        V7FeatureCorrelationReport
        ._build_family_summary(
            pair_dataframe=(
                pair_dataframe
            ),
            correlation_threshold=0.80,
        )
    )

    assert not family_summary.empty

    assert {
        "family_a",
        "family_b",
        "relationship",
        "pair_count",
        "mean_absolute_pearson",
        "maximum_absolute_pearson",
        "mean_absolute_spearman",
        "maximum_absolute_spearman",
        "maximum_absolute_correlation",
        "high_correlation_pairs",
    }.issubset(
        set(
            family_summary.columns
        )
    )

    recency_family_row = family_summary[
        (
            family_summary[
                "family_a"
            ]
            == "recency"
        )
        & (
            family_summary[
                "family_b"
            ]
            == "recency"
        )
    ].iloc[0]

    assert recency_family_row[
        "relationship"
    ] == "within_family"

    assert recency_family_row[
        "maximum_absolute_correlation"
    ] == pytest.approx(
        1.0
    )

    assert recency_family_row[
        "high_correlation_pairs"
    ] >= 1


def test_build_target_association(
) -> None:
    dataset = make_valid_dataset()

    feature_dataframe = (
        V7FeatureCorrelationReport
        ._feature_dataframe(
            dataset
        )
    )

    target_association = (
        V7FeatureCorrelationReport
        ._build_target_association(
            dataset=dataset,
            feature_dataframe=(
                feature_dataframe
            ),
        )
    )

    assert len(
        target_association
    ) == len(
        MODEL_FEATURES
    )

    assert set(
        target_association[
            "feature"
        ]
    ) == set(
        MODEL_FEATURES
    )

    history_row = target_association[
        target_association[
            "feature"
        ]
        == "history_size"
    ].iloc[0]

    assert pd.isna(
        history_row[
            "pearson_with_target"
        ]
    ) or (
        history_row[
            "pearson_with_target"
        ]
        is None
    )

    assert pd.isna(
        history_row[
            "spearman_with_target"
        ]
    ) or (
        history_row[
            "spearman_with_target"
        ]
        is None
    )


def test_json_safe(
) -> None:
    payload = {
        "integer": np.int64(
            5
        ),
        "float": np.float64(
            0.5
        ),
        "nan": np.nan,
        "infinity": np.inf,
        "items": [
            np.int64(
                2
            ),
            np.float64(
                0.25
            ),
        ],
    }

    safe_payload = (
        V7FeatureCorrelationReport
        ._json_safe(
            payload
        )
    )

    assert safe_payload[
        "integer"
    ] == 5

    assert safe_payload[
        "float"
    ] == pytest.approx(
        0.5
    )

    assert safe_payload[
        "nan"
    ] is None

    assert safe_payload[
        "infinity"
    ] is None

    assert safe_payload[
        "items"
    ] == [
        2,
        pytest.approx(
            0.25
        ),
    ]


def test_export_methods(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        [
            {
                "feature_a": "recency",
                "feature_b": (
                    "recency_ratio"
                ),
                "pearson": 1.0,
            }
        ]
    )

    csv_path = (
        tmp_path
        / "correlation.csv"
    )

    generated_csv = (
        V7FeatureCorrelationReport
        ._write_dataframe(
            dataframe=dataframe,
            output_path=csv_path,
        )
    )

    assert generated_csv == (
        csv_path.resolve()
    )

    assert csv_path.exists()

    exported_dataframe = pd.read_csv(
        csv_path
    )

    assert exported_dataframe[
        "feature_a"
    ].tolist() == [
        "recency"
    ]

    result: dict[str, Any] = {
        "status": "success",
        "version": (
            V7FeatureCorrelationReport
            .VERSION
        ),
        "dataset_targets": 4,
        "dataset_rows": 196,
        "feature_count": 12,
        "correlation_threshold": 0.80,
        "high_correlation_pair_count": 1,
        "constant_features": [
            "history_size",
        ],
        "top_pairs": [
            {
                "feature_a": "recency",
                "feature_b": (
                    "recency_ratio"
                ),
                "pearson": 1.0,
                "spearman": 1.0,
                "max_absolute_correlation": (
                    1.0
                ),
                "high_correlation": True,
            }
        ],
        "family_summary": [
            {
                "family_a": "recency",
                "family_b": "recency",
                "pair_count": 1,
                "mean_absolute_pearson": (
                    1.0
                ),
                "maximum_absolute_pearson": (
                    1.0
                ),
                "mean_absolute_spearman": (
                    1.0
                ),
                "maximum_absolute_spearman": (
                    1.0
                ),
                "high_correlation_pairs": 1,
            }
        ],
    }

    json_path = (
        tmp_path
        / "correlation.json"
    )

    generated_json = (
        V7FeatureCorrelationReport
        ._write_json(
            result=result,
            output_path=json_path,
        )
    )

    assert generated_json == (
        json_path.resolve()
    )

    exported_json = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert exported_json[
        "status"
    ] == "success"

    text_path = (
        tmp_path
        / "correlation.txt"
    )

    generated_text = (
        V7FeatureCorrelationReport
        ._write_text(
            result=result,
            output_path=text_path,
        )
    )

    assert generated_text == (
        text_path.resolve()
    )

    exported_text = (
        text_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "PREDIXA AI V7 FEATURE "
        "CORRELATION REPORT"
    ) in exported_text

    assert "recency_ratio" in exported_text

    assert "history_size" in exported_text

    assert (
        "High absolute correlation"
        in exported_text
    )
