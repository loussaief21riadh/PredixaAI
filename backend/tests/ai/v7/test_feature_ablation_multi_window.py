from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.ai.v7.explainability.feature_ablation_multi_window import (
    V7FeatureAblationMultiWindowReport,
)
from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
)
from app.ai.v7.ranking_dataset import V7RankingDataset


MODEL_FEATURES = V7RankingDataset.feature_columns()


def make_dataset(
    target_count: int,
    start_target_index: int = 1,
) -> pd.DataFrame:
    """
    Build a valid synthetic candidate-level ranking dataset.

    Every target contains:
        - 49 candidates;
        - 5 positive labels;
        - all 12 V7 model features.
    """

    rows: list[dict[str, Any]] = []

    start_date = pd.Timestamp("2020-01-01")

    for target_offset in range(target_count):
        target_draw_index = (
            start_target_index
            + target_offset
        )

        target_draw_date = (
            start_date
            + pd.Timedelta(
                days=target_offset,
            )
        ).date().isoformat()

        for candidate_number in range(
            1,
            50,
        ):
            candidate_score = (
                50.0
                - float(candidate_number)
            ) / 49.0

            row: dict[str, Any] = {
                "candidate_number": (
                    candidate_number
                ),
                "target": (
                    1
                    if candidate_number <= 5
                    else 0
                ),
                "target_draw_index": (
                    target_draw_index
                ),
                "target_draw_date": (
                    target_draw_date
                ),
            }

            for (
                feature_position,
                feature_name,
            ) in enumerate(
                MODEL_FEATURES
            ):
                row[feature_name] = (
                    candidate_score
                    + (
                        target_offset
                        * 0.0001
                    )
                    + (
                        feature_position
                        * 0.001
                    )
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def make_experiment_result(
    experiment: str,
    average_hits: float,
    delta: float,
    evaluated_targets: int = 40,
    feature_count: int = 12,
) -> dict[str, Any]:
    total_hits = int(
        round(
            average_hits
            * evaluated_targets
        )
    )

    return {
        "experiment": experiment,
        "removed_family": (
            None
            if experiment == "baseline"
            else experiment.replace(
                "without_",
                "",
                1,
            )
        ),
        "feature_count": feature_count,
        "features": MODEL_FEATURES[
            :feature_count
        ],
        "training_rows": 4900,
        "validation_rows": (
            evaluated_targets
            * 49
        ),
        "evaluated_targets": (
            evaluated_targets
        ),
        "total_hits": total_hits,
        "average_hits_at_5": (
            average_hits
        ),
        "precision_at_k": (
            total_hits
            / (
                evaluated_targets
                * 5
            )
        ),
        "at_least_1_hit_rate": 0.40,
        "at_least_2_hit_rate": 0.05,
        "hit_distribution": {
            0: 24,
            1: 12,
            2: 4,
            3: 0,
            4: 0,
            5: 0,
        },
        "training_seconds": 1.25,
        "evaluation_seconds": 0.50,
        "delta_vs_baseline": delta,
        "conclusion": (
            "reference"
            if experiment == "baseline"
            else (
                "improved_without_family"
                if delta > 0
                else (
                    "degraded_without_family"
                    if delta < 0
                    else "unchanged_without_family"
                )
            )
        ),
        "details": [],
    }


@pytest.mark.parametrize(
    (
        "parameter_overrides",
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
                "max_training_targets": 99,
            },
            "max_training_targets must be at least",
        ),
        (
            {
                "validation_targets": 4,
            },
            "validation_targets must be at least 5",
        ),
        (
            {
                "windows": 1,
            },
            "windows must be at least 2",
        ),
        (
            {
                "top_k": 0,
            },
            "top_k must be between 1 and 49",
        ),
        (
            {
                "top_k": 50,
            },
            "top_k must be between 1 and 49",
        ),
        (
            {
                "purge_targets": -1,
            },
            "purge_targets cannot be negative",
        ),
    ],
)
def test_validate_parameters_rejects_invalid_values(
    parameter_overrides: dict[str, int],
    expected_message: str,
) -> None:
    parameters = {
        "window_size": 100,
        "max_training_targets": 1000,
        "validation_targets": 40,
        "windows": 3,
        "top_k": 5,
        "purge_targets": 1,
    }

    parameters.update(
        parameter_overrides
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        (
            V7FeatureAblationMultiWindowReport
            ._validate_parameters(
                **parameters
            )
        )


def test_validate_parameters_accepts_valid_values(
) -> None:
    (
        V7FeatureAblationMultiWindowReport
        ._validate_parameters(
            window_size=100,
            max_training_targets=1000,
            validation_targets=40,
            windows=3,
            top_k=5,
            purge_targets=1,
        )
    )


def test_validate_dataset_accepts_valid_dataset(
) -> None:
    dataset = make_dataset(
        target_count=5
    )

    (
        V7FeatureAblationMultiWindowReport
        ._validate_dataset(
            dataset
        )
    )


def test_validate_dataset_rejects_missing_candidate(
) -> None:
    dataset = make_dataset(
        target_count=2
    )

    invalid_dataset = (
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
            V7FeatureAblationMultiWindowReport
            ._validate_dataset(
                invalid_dataset
            )
        )


def test_validate_dataset_rejects_invalid_positive_count(
) -> None:
    dataset = make_dataset(
        target_count=2
    )

    target_mask = (
        dataset[
            "target_draw_index"
        ]
        == 1
    )

    candidate_mask = (
        dataset[
            "candidate_number"
        ]
        == 5
    )

    dataset.loc[
        target_mask
        & candidate_mask,
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
            V7FeatureAblationMultiWindowReport
            ._validate_dataset(
                dataset
            )
        )


def test_feature_configuration_and_feature_sets(
) -> None:
    (
        V7FeatureAblationMultiWindowReport
        ._validate_feature_configuration()
    )

    feature_sets = (
        V7FeatureAblationMultiWindowReport
        ._feature_sets()
    )

    assert list(
        feature_sets
    ) == [
        "baseline",
        "without_global",
        "without_frequency",
        "without_recency",
        "without_trend",
        "without_volatility",
    ]

    assert feature_sets[
        "baseline"
    ] == MODEL_FEATURES

    for family_name in (
        FEATURE_FAMILY_ORDER
    ):
        experiment_name = (
            f"without_{family_name}"
        )

        retained_features = set(
            feature_sets[
                experiment_name
            ]
        )

        removed_features = set(
            FEATURE_FAMILIES[
                family_name
            ]
        )

        assert (
            retained_features
            & removed_features
        ) == set()

        assert retained_features == (
            set(
                MODEL_FEATURES
            )
            - removed_features
        )


def test_build_windows_applies_temporal_split_and_purge(
) -> None:
    dataset = make_dataset(
        target_count=111
    )

    windows = (
        V7FeatureAblationMultiWindowReport
        ._build_windows(
            dataset=dataset,
            windows=2,
            validation_targets=5,
            max_training_targets=100,
            purge_targets=1,
        )
    )

    assert len(
        windows
    ) == 2

    first_window = windows[0]
    second_window = windows[1]

    assert first_window[
        "training_target_indices"
    ] == list(
        range(
            1,
            101,
        )
    )

    assert first_window[
        "validation_target_indices"
    ] == list(
        range(
            102,
            107,
        )
    )

    assert 101 not in first_window[
        "training_target_indices"
    ]

    assert 101 not in first_window[
        "validation_target_indices"
    ]

    assert second_window[
        "training_target_indices"
    ] == list(
        range(
            6,
            106,
        )
    )

    assert second_window[
        "validation_target_indices"
    ] == list(
        range(
            107,
            112,
        )
    )

    assert 106 not in second_window[
        "training_target_indices"
    ]

    assert 106 not in second_window[
        "validation_target_indices"
    ]

    for window in windows:
        assert window[
            "training_targets"
        ] == 100

        assert window[
            "validation_targets"
        ] == 5

        assert window[
            "training_rows"
        ] == 4900

        assert window[
            "validation_rows"
        ] == 245

        assert max(
            window[
                "training_target_indices"
            ]
        ) < min(
            window[
                "validation_target_indices"
            ]
        )


def test_build_windows_rejects_insufficient_targets(
) -> None:
    dataset = make_dataset(
        target_count=100
    )

    with pytest.raises(
        ValueError,
        match=(
            "Not enough dataset targets for "
            "multi-window ablation"
        ),
    ):
        (
            V7FeatureAblationMultiWindowReport
            ._build_windows(
                dataset=dataset,
                windows=2,
                validation_targets=5,
                max_training_targets=100,
                purge_targets=1,
            )
        )


def test_evaluate_experiment_ranks_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DeterministicModel:
        def __init__(
            self,
        ) -> None:
            self.classes_ = np.array(
                [
                    0,
                    1,
                ]
            )

        def fit(
            self,
            training_x: pd.DataFrame,
            training_y: pd.Series,
        ) -> DeterministicModel:
            assert not training_x.empty
            assert set(
                training_y.unique()
            ) == {
                0,
                1,
            }

            return self

        def predict_proba(
            self,
            features: pd.DataFrame,
        ) -> np.ndarray:
            raw_scores = (
                features
                .iloc[
                    :,
                    0,
                ]
                .astype(float)
                .to_numpy()
            )

            minimum = float(
                np.min(
                    raw_scores
                )
            )

            maximum = float(
                np.max(
                    raw_scores
                )
            )

            positive_scores = (
                raw_scores
                - minimum
            ) / (
                maximum
                - minimum
            )

            return np.column_stack(
                (
                    1.0
                    - positive_scores,
                    positive_scores,
                )
            )

    monkeypatch.setattr(
        V7FeatureAblationMultiWindowReport,
        "_build_model",
        staticmethod(
            lambda: DeterministicModel()
        ),
    )

    dataset = make_dataset(
        target_count=5
    )

    training_dataset = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(
                [
                    1,
                    2,
                    3,
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation_dataset = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(
                [
                    4,
                    5,
                ]
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    result = (
        V7FeatureAblationMultiWindowReport
        ._evaluate_experiment(
            experiment_name="baseline",
            feature_columns=MODEL_FEATURES,
            training_dataset=(
                training_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            top_k=5,
        )
    )

    assert result[
        "experiment"
    ] == "baseline"

    assert result[
        "evaluated_targets"
    ] == 2

    assert result[
        "total_hits"
    ] == 10

    assert result[
        "average_hits_at_5"
    ] == pytest.approx(
        5.0
    )

    assert result[
        "precision_at_k"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "at_least_1_hit_rate"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "at_least_2_hit_rate"
    ] == pytest.approx(
        1.0
    )

    assert result[
        "hit_distribution"
    ][5] == 2

    assert len(
        result[
            "details"
        ]
    ) == 2

    for detail in result[
        "details"
    ]:
        assert detail[
            "predicted_top_k"
        ] == [
            1,
            2,
            3,
            4,
            5,
        ]

        assert detail[
            "actual_numbers"
        ] == [
            1,
            2,
            3,
            4,
            5,
        ]

        assert detail[
            "hits"
        ] == 5


def test_add_window_comparison(
) -> None:
    experiments = [
        make_experiment_result(
            experiment="baseline",
            average_hits=0.50,
            delta=0.0,
            feature_count=12,
        ),
        make_experiment_result(
            experiment="without_recency",
            average_hits=0.40,
            delta=0.0,
            feature_count=10,
        ),
        make_experiment_result(
            experiment="without_volatility",
            average_hits=0.60,
            delta=0.0,
            feature_count=11,
        ),
        make_experiment_result(
            experiment="without_trend",
            average_hits=0.50,
            delta=0.0,
            feature_count=11,
        ),
    ]

    (
        V7FeatureAblationMultiWindowReport
        ._add_window_comparison(
            experiments
        )
    )

    indexed = {
        experiment[
            "experiment"
        ]: experiment
        for experiment in experiments
    }

    assert indexed[
        "baseline"
    ][
        "delta_vs_baseline"
    ] == pytest.approx(
        0.0
    )

    assert indexed[
        "baseline"
    ][
        "conclusion"
    ] == "reference"

    assert indexed[
        "without_recency"
    ][
        "delta_vs_baseline"
    ] == pytest.approx(
        -0.10
    )

    assert indexed[
        "without_recency"
    ][
        "conclusion"
    ] == "degraded_without_family"

    assert indexed[
        "without_volatility"
    ][
        "delta_vs_baseline"
    ] == pytest.approx(
        0.10
    )

    assert indexed[
        "without_volatility"
    ][
        "conclusion"
    ] == "improved_without_family"

    assert indexed[
        "without_trend"
    ][
        "delta_vs_baseline"
    ] == pytest.approx(
        0.0
    )

    assert indexed[
        "without_trend"
    ][
        "conclusion"
    ] == "unchanged_without_family"


def test_aggregate_experiments(
) -> None:
    baseline_values = [
        0.40,
        0.60,
        0.40,
    ]

    recency_values = [
        0.30,
        0.40,
        0.35,
    ]

    windows_results: list[
        dict[str, Any]
    ] = []

    for (
        window_number,
        (
            baseline_hits,
            recency_hits,
        ),
    ) in enumerate(
        zip(
            baseline_values,
            recency_values,
        ),
        start=1,
    ):
        windows_results.append(
            {
                "window_number": (
                    window_number
                ),
                "experiments": [
                    make_experiment_result(
                        experiment="baseline",
                        average_hits=(
                            baseline_hits
                        ),
                        delta=0.0,
                        feature_count=12,
                    ),
                    make_experiment_result(
                        experiment=(
                            "without_recency"
                        ),
                        average_hits=(
                            recency_hits
                        ),
                        delta=(
                            recency_hits
                            - baseline_hits
                        ),
                        feature_count=10,
                    ),
                ],
            }
        )

    aggregates = (
        V7FeatureAblationMultiWindowReport
        ._aggregate_experiments(
            windows_results=(
                windows_results
            ),
            experiment_names=[
                "baseline",
                "without_recency",
            ],
            top_k=5,
        )
    )

    indexed = {
        aggregate[
            "experiment"
        ]: aggregate
        for aggregate in aggregates
    }

    baseline = indexed[
        "baseline"
    ]

    recency = indexed[
        "without_recency"
    ]

    assert baseline[
        "window_mean_hits_at_5"
    ] == pytest.approx(
        0.466667,
        abs=0.000001,
    )

    assert baseline[
        "window_std_hits_at_5"
    ] == pytest.approx(
        0.094281,
        abs=0.000001,
    )

    assert baseline[
        "conclusion"
    ] == "reference"

    assert recency[
        "window_mean_hits_at_5"
    ] == pytest.approx(
        0.35
    )

    assert recency[
        "mean_delta_vs_baseline"
    ] == pytest.approx(
        -0.116667,
        abs=0.000001,
    )

    assert recency[
        "improved_windows"
    ] == 0

    assert recency[
        "degraded_windows"
    ] == 3

    assert recency[
        "unchanged_windows"
    ] == 0

    assert recency[
        "conclusion"
    ] == "family_likely_useful"

    assert recency[
        "random_expectation"
    ] == pytest.approx(
        25 / 49,
        abs=0.000001,
    )


def test_dataframe_builders(
) -> None:
    aggregates = [
        {
            "experiment": "baseline",
            "removed_family": None,
            "feature_count": 12,
            "windows": 3,
            "total_evaluated_targets": 120,
            "total_hits": 56,
            "window_mean_hits_at_5": 0.466667,
            "window_std_hits_at_5": 0.094281,
            "minimum_hits_at_5": 0.40,
            "maximum_hits_at_5": 0.60,
            "mean_delta_vs_baseline": 0.0,
            "minimum_delta_vs_baseline": 0.0,
            "maximum_delta_vs_baseline": 0.0,
            "improved_windows": 0,
            "degraded_windows": 0,
            "unchanged_windows": 3,
            "average_training_seconds": 1.5,
            "average_evaluation_seconds": 0.5,
            "random_expectation": 0.510204,
            "mean_lift_vs_random": -0.043537,
            "conclusion": "reference",
        }
    ]

    aggregate_dataframe = (
        V7FeatureAblationMultiWindowReport
        ._aggregate_dataframe(
            aggregates
        )
    )

    assert len(
        aggregate_dataframe
    ) == 1

    assert aggregate_dataframe[
        "experiment"
    ].tolist() == [
        "baseline"
    ]

    window_results = [
        {
            "window_number": 1,
            "training_first_date": (
                "2020-01-01"
            ),
            "training_last_date": (
                "2020-04-09"
            ),
            "validation_first_date": (
                "2020-04-11"
            ),
            "validation_last_date": (
                "2020-04-15"
            ),
            "training_targets": 100,
            "validation_targets": 5,
            "experiments": [
                make_experiment_result(
                    experiment="baseline",
                    average_hits=0.40,
                    delta=0.0,
                    feature_count=12,
                )
            ],
        }
    ]

    window_dataframe = (
        V7FeatureAblationMultiWindowReport
        ._window_dataframe(
            window_results
        )
    )

    assert len(
        window_dataframe
    ) == 1

    assert window_dataframe[
        "window_number"
    ].tolist() == [
        1
    ]

    assert window_dataframe[
        "experiment"
    ].tolist() == [
        "baseline"
    ]


def test_export_methods(
    tmp_path: Path,
) -> None:
    aggregate = {
        "experiment": "baseline",
        "removed_family": None,
        "feature_count": 12,
        "windows": 1,
        "total_evaluated_targets": 40,
        "total_hits": 16,
        "window_mean_hits_at_5": 0.40,
        "window_std_hits_at_5": 0.0,
        "minimum_hits_at_5": 0.40,
        "maximum_hits_at_5": 0.40,
        "mean_delta_vs_baseline": 0.0,
        "minimum_delta_vs_baseline": 0.0,
        "maximum_delta_vs_baseline": 0.0,
        "improved_windows": 0,
        "degraded_windows": 0,
        "unchanged_windows": 1,
        "average_training_seconds": 1.0,
        "average_evaluation_seconds": 0.5,
        "random_expectation": 0.510204,
        "mean_lift_vs_random": -0.110204,
        "conclusion": "reference",
    }

    baseline_experiment = (
        make_experiment_result(
            experiment="baseline",
            average_hits=0.40,
            delta=0.0,
            feature_count=12,
        )
    )

    result = {
        "status": "success",
        "version": (
            V7FeatureAblationMultiWindowReport
            .VERSION
        ),
        "windows": 1,
        "validation_targets_per_window": 40,
        "max_training_targets": 1000,
        "purge_targets": 1,
        "window_size": 100,
        "top_k": 5,
        "aggregates": [
            aggregate
        ],
        "window_results": [
            {
                "window_number": 1,
                "training_targets": 1000,
                "validation_targets": 40,
                "training_first_date": (
                    "2020-01-01"
                ),
                "training_last_date": (
                    "2025-09-30"
                ),
                "validation_first_date": (
                    "2025-10-06"
                ),
                "validation_last_date": (
                    "2026-01-05"
                ),
                "experiments": [
                    baseline_experiment
                ],
            }
        ],
    }

    dataframe = pd.DataFrame(
        [
            aggregate
        ]
    )

    csv_path = (
        tmp_path
        / "report.csv"
    )

    json_path = (
        tmp_path
        / "report.json"
    )

    text_path = (
        tmp_path
        / "report.txt"
    )

    generated_csv = (
        V7FeatureAblationMultiWindowReport
        ._write_csv(
            dataframe=dataframe,
            output_path=csv_path,
        )
    )

    generated_json = (
        V7FeatureAblationMultiWindowReport
        ._write_json(
            result=result,
            output_path=json_path,
        )
    )

    generated_text = (
        V7FeatureAblationMultiWindowReport
        ._write_text(
            result=result,
            output_path=text_path,
        )
    )

    assert generated_csv == (
        csv_path.resolve()
    )

    assert generated_json == (
        json_path.resolve()
    )

    assert generated_text == (
        text_path.resolve()
    )

    exported_csv = pd.read_csv(
        csv_path
    )

    assert len(
        exported_csv
    ) == 1

    assert exported_csv[
        "experiment"
    ].tolist() == [
        "baseline"
    ]

    exported_json = json.loads(
        json_path.read_text(
            encoding="utf-8"
        )
    )

    assert exported_json[
        "status"
    ] == "success"

    assert exported_json[
        "windows"
    ] == 1

    exported_text = (
        text_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        "PREDIXA AI V7 FEATURE FAMILY "
        "ABLATION - MULTI-WINDOW"
    ) in exported_text

    assert "baseline" in exported_text

    assert (
        "A negative mean delta"
    ) in exported_text
