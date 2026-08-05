from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.ai.v7.explainability import (
    production_benchmark as benchmark,
)
from app.ai.v7.explainability.feature_ablation_runner import (
    FeatureAblationRunResult,
    TargetEvaluation,
)


PRODUCTION_FEATURES = (
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

REFERENCE_FEATURES = (
    "history_size",
    "average_sum",
    "average_even_count",
    "average_consecutive_pairs",
    "rate_10",
    "rate_20",
    "rate_50",
    "rate_100",
    "recency",
    "recency_ratio",
    "short_vs_long",
    "frequency_volatility",
)

GLOBAL_FEATURES = (
    "history_size",
    "average_sum",
    "average_even_count",
    "average_consecutive_pairs",
)

CANDIDATE_FEATURES = (
    "rate_10",
    "rate_20",
    "rate_50",
    "rate_100",
    "recency",
    "recency_ratio",
    "short_vs_long",
    "frequency_volatility",
)


@pytest.fixture(autouse=True)
def patch_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep unit tests independent from unrelated future feature changes."""

    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "GLOBAL_FEATURES",
        GLOBAL_FEATURES,
    )
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "CANDIDATE_FEATURES",
        CANDIDATE_FEATURES,
    )
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "MODEL_FEATURES",
        PRODUCTION_FEATURES,
    )
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "PRUNED_MODEL_FEATURES",
        (
            "rate_10",
        ),
        raising=False,
    )
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "EXPECTED_FULL_FEATURE_COUNT",
        396,
    )
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "feature_columns",
        classmethod(
            lambda cls: (
                PRODUCTION_FEATURES
            )
        ),
    )
    monkeypatch.setattr(
        benchmark.V7RankingModel,
        "EXPECTED_FEATURE_COUNT",
        11,
    )
    monkeypatch.setattr(
        benchmark,
        "FEATURE_FAMILY_ORDER",
        (
            "global",
            "frequency",
            "recency",
            "trend",
            "volatility",
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "FEATURE_FAMILIES",
        {
            "global": (
                "history_size",
                "average_sum",
                "average_even_count",
                "average_consecutive_pairs",
            ),
            "frequency": (
                "rate_20",
                "rate_50",
                "rate_100",
            ),
            "recency": (
                "recency",
                "recency_ratio",
            ),
            "trend": (
                "short_vs_long",
            ),
            "volatility": (
                "frequency_volatility",
            ),
        },
    )


def make_config(
    tmp_path: Path,
    **overrides: Any,
) -> (
    benchmark.ProductionBenchmarkConfig
):
    values: dict[
        str,
        Any,
    ] = {
        "output_directory": (
            tmp_path
            / "production_benchmark"
        ),
        "window_size": 100,
        "max_training_targets": 1500,
        "validation_targets": 5,
        "top_k": 5,
        "purge_targets": 1,
        "accuracy_tolerance": 0.0,
        "maximum_runtime_ratio": 2.0,
    }
    values.update(
        overrides
    )

    return (
        benchmark.ProductionBenchmarkConfig(
            **values
        )
    )


def make_dataset(
    target_indices: list[int],
) -> pd.DataFrame:
    """Build valid candidate-level rows for deterministic tests."""

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for target_index in (
        target_indices
    ):
        actual = {
            1,
            2,
            3,
            4,
            5,
        }

        for candidate in range(
            1,
            50,
        ):
            row: dict[
                str,
                Any,
            ] = {
                "candidate_number": (
                    candidate
                ),
                "target": int(
                    candidate
                    in actual
                ),
                "target_draw_index": (
                    target_index
                ),
                "target_draw_date": (
                    f"2026-01-{target_index:02d}"
                ),
                "history_size": 100,
                "average_sum": 125.0,
                "average_even_count": 2.5,
                "average_consecutive_pairs": 0.5,
            }

            for family_index, family in enumerate(
                CANDIDATE_FEATURES,
                start=1,
            ):
                row[
                    family
                ] = (
                    candidate
                    / 100.0
                    + family_index
                    / 1000.0
                    + target_index
                    / 10000.0
                )

            rows.append(
                row
            )

    return pd.DataFrame(
        rows
    )


def make_target_evaluation(
    target_index: int,
    *,
    hits: int,
    selected: tuple[int, ...] = (
        1,
        2,
        6,
        7,
        8,
    ),
) -> TargetEvaluation:
    return TargetEvaluation(
        target_draw_index=(
            target_index
        ),
        target_draw_date=(
            f"2026-01-{target_index:02d}"
        ),
        selected_numbers=(
            selected
        ),
        actual_numbers=(
            1,
            2,
            3,
            4,
            5,
        ),
        hits=hits,
    )


def make_run(
    *,
    experiment_name: str,
    features: tuple[str, ...],
    mean_hits: float,
    target_hit_rate: float,
    total_seconds: float,
    removed_features: tuple[str, ...],
) -> FeatureAblationRunResult:
    evaluations = tuple(
        make_target_evaluation(
            index,
            hits=(
                1
                if index
                <= 3
                else 0
            ),
        )
        for index
        in range(
            1,
            6,
        )
    )

    return FeatureAblationRunResult(
        experiment_name=(
            experiment_name
        ),
        feature_columns=features,
        feature_count=len(
            features
        ),
        removed_features=(
            removed_features
        ),
        top_k=5,
        training_rows=490,
        validation_rows=245,
        training_targets=10,
        validation_targets=5,
        fit_seconds=(
            total_seconds
            * 0.8
        ),
        prediction_seconds=(
            total_seconds
            * 0.2
        ),
        total_seconds=(
            total_seconds
        ),
        total_hits=int(
            round(
                mean_hits
                * 5
            )
        ),
        mean_hits_at_k=(
            mean_hits
        ),
        normalized_hits_at_k=(
            mean_hits
            / 5
        ),
        targets_with_at_least_one_hit=int(
            round(
                target_hit_rate
                * 5
            )
        ),
        target_hit_rate=(
            target_hit_rate
        ),
        target_evaluations=(
            evaluations
        ),
    )


def sample_contract(
) -> (
    benchmark.ProductionContractCheck
):
    return (
        benchmark.ProductionContractCheck(
            passed=True,
            production_features=(
                PRODUCTION_FEATURES
            ),
            reference_features=(
                REFERENCE_FEATURES
            ),
            pruned_features=(
                "rate_10",
            ),
            feature_family_order=(
                "global",
                "frequency",
                "recency",
                "trend",
                "volatility",
            ),
            configured_family_features=(
                PRODUCTION_FEATURES
            ),
            production_feature_count=11,
            reference_feature_count=12,
            expected_model_feature_count=11,
            full_engineered_feature_count=396,
            rate_10_engineered=True,
            rate_10_active=False,
            short_vs_long_active=True,
            family_contract_matches=True,
        )
    )


def sample_dataset_summary(
) -> (
    benchmark.BenchmarkDatasetSummary
):
    return (
        benchmark.BenchmarkDatasetSummary(
            draw_count=2780,
            dataset_rows=73500,
            dataset_targets=1500,
            training_rows=68551,
            training_targets=1399,
            validation_rows=4900,
            validation_targets=100,
            purge_targets=1,
            first_training_target=1280,
            last_training_target=2678,
            purged_target_indices=(
                2679,
            ),
            first_validation_target=2680,
            last_validation_target=2779,
        )
    )


def sample_smoke(
) -> (
    benchmark.ProductionSmokeResult
):
    return (
        benchmark.ProductionSmokeResult(
            passed=True,
            target_draw_index=2779,
            target_draw_date="2026-08-01",
            feature_dictionary_count=396,
            engineered_rate_10_present=True,
            model_feature_count=11,
            candidate_count=49,
            top_k=5,
            predicted_numbers=(
                1,
                2,
                6,
                7,
                8,
            ),
            actual_numbers=(
                1,
                2,
                3,
                4,
                5,
            ),
            hits=2,
            probability_count=49,
            ranking_is_sorted=True,
            unique_ranking_numbers=True,
            finite_probabilities=True,
            probability_bounds_valid=True,
        )
    )


def sample_report(
    *,
    ready: bool = True,
) -> (
    benchmark.ProductionBenchmarkReport
):
    reference = make_run(
        experiment_name=(
            "historical_reference_12_features"
        ),
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.46,
        target_hit_rate=0.46,
        total_seconds=2.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name=(
            "production_11_features"
        ),
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.49,
        target_hit_rate=0.49,
        total_seconds=1.8,
        removed_features=(
            "rate_10",
        ),
    )
    comparison = (
        benchmark.compare_benchmark_runs(
            reference_run=(
                reference
            ),
            production_run=(
                production
            ),
            accuracy_tolerance=0.0,
            maximum_runtime_ratio=2.0,
        )
    )

    return (
        benchmark.ProductionBenchmarkReport(
            status="success",
            version=(
                benchmark.VERSION
            ),
            protocol="test protocol",
            ready_for_production=(
                ready
            ),
            recommendation=(
                "READY_FOR_PRODUCTION"
                if ready
                else "REVIEW_PRODUCTION_PRUNING"
            ),
            window_size=100,
            max_training_targets=1500,
            validation_targets=100,
            top_k=5,
            purge_targets=1,
            accuracy_tolerance=0.0,
            maximum_runtime_ratio=2.0,
            contract=(
                sample_contract()
            ),
            dataset=(
                sample_dataset_summary()
            ),
            reference_run=(
                reference
            ),
            production_run=(
                production
            ),
            comparison=(
                comparison
            ),
            smoke=(
                sample_smoke()
            ),
        )
    )


def test_validate_config_returns_self(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path
    )

    assert (
        config.validated()
        is config
    )


@pytest.mark.parametrize(
    (
        "overrides",
        "message",
    ),
    [
        (
            {
                "output_directory": (
                    "invalid"
                ),
            },
            (
                "output_directory must be "
                "a pathlib.Path"
            ),
        ),
        (
            {
                "window_size": 99,
            },
            (
                "window_size must be "
                "at least 100"
            ),
        ),
        (
            {
                "max_training_targets": -1,
            },
            (
                "max_training_targets "
                "cannot be negative"
            ),
        ),
        (
            {
                "validation_targets": 4,
            },
            (
                "validation_targets must "
                "be at least 5"
            ),
        ),
        (
            {
                "top_k": 0,
            },
            (
                "top_k must be between "
                "1 and 49"
            ),
        ),
        (
            {
                "top_k": 50,
            },
            (
                "top_k must be between "
                "1 and 49"
            ),
        ),
        (
            {
                "purge_targets": -1,
            },
            (
                "purge_targets cannot "
                "be negative"
            ),
        ),
        (
            {
                "accuracy_tolerance": (
                    float(
                        "nan"
                    )
                ),
            },
            (
                "accuracy_tolerance must "
                "be finite"
            ),
        ),
        (
            {
                "accuracy_tolerance": -0.1,
            },
            (
                "accuracy_tolerance cannot "
                "be negative"
            ),
        ),
        (
            {
                "maximum_runtime_ratio": (
                    float(
                        "inf"
                    )
                ),
            },
            (
                "maximum_runtime_ratio must "
                "be finite"
            ),
        ),
        (
            {
                "maximum_runtime_ratio": 0,
            },
            (
                "maximum_runtime_ratio must "
                "be positive"
            ),
        ),
    ],
)
def test_validate_config_rejects_invalid_values(
    tmp_path: Path,
    overrides: dict[
        str,
        Any,
    ],
    message: str,
) -> None:
    with pytest.raises(
        benchmark.BenchmarkConfigurationError,
        match=message,
    ):
        make_config(
            tmp_path,
            **overrides,
        ).validated()


def test_normalise_feature_sequence() -> None:
    result = (
        benchmark._normalise_feature_sequence(
            (
                " rate_10 ",
                "rate_20",
            ),
            field_name="features",
        )
    )

    assert result == (
        "rate_10",
        "rate_20",
    )


@pytest.mark.parametrize(
    (
        "values",
        "message",
    ),
    [
        (
            "rate_10",
            (
                "features must be "
                "a sequence"
            ),
        ),
        (
            (
                "rate_10",
                7,
            ),
            (
                "features must contain "
                "strings"
            ),
        ),
        (
            (
                "rate_10",
                " ",
            ),
            (
                "features contains "
                "an empty feature"
            ),
        ),
        (
            (
                "rate_10",
                "rate_10",
            ),
            (
                "features contains "
                "duplicate feature"
            ),
        ),
        (
            (),
            (
                "features cannot be empty"
            ),
        ),
    ],
)
def test_normalise_feature_sequence_rejects_invalid_values(
    values: Any,
    message: str,
) -> None:
    with pytest.raises(
        benchmark.BenchmarkContractError,
        match=message,
    ):
        benchmark._normalise_feature_sequence(
            values,
            field_name="features",
        )


def test_reference_feature_columns() -> None:
    assert (
        benchmark.reference_feature_columns()
        == REFERENCE_FEATURES
    )


def test_validate_production_contract() -> None:
    result = (
        benchmark.validate_production_contract()
    )

    assert result.passed is True
    assert (
        result.production_features
        == PRODUCTION_FEATURES
    )
    assert (
        result.reference_features
        == REFERENCE_FEATURES
    )
    assert result.pruned_features == (
        "rate_10",
    )
    assert result.rate_10_engineered is True
    assert result.rate_10_active is False
    assert result.short_vs_long_active is True


def test_validate_production_contract_rejects_active_rate10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "feature_columns",
        classmethod(
            lambda cls: (
                REFERENCE_FEATURES
            )
        ),
    )

    with pytest.raises(
        benchmark.BenchmarkContractError,
        match=(
            "Invalid V7 production contract"
        ),
    ):
        benchmark.validate_production_contract()


def test_validate_production_contract_rejects_family_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = dict(
        benchmark.FEATURE_FAMILIES
    )
    families[
        "trend"
    ] = ()

    monkeypatch.setattr(
        benchmark,
        "FEATURE_FAMILIES",
        families,
    )

    with pytest.raises(
        benchmark.BenchmarkContractError,
        match=(
            "feature-family configuration "
            "does not match production"
        ),
    ):
        benchmark.validate_production_contract()


def test_target_indices() -> None:
    dataset = make_dataset(
        [
            3,
            1,
            2,
        ]
    )

    assert (
        benchmark._target_indices(
            dataset
        )
        == (
            1,
            2,
            3,
        )
    )


def test_target_indices_rejects_missing_column() -> None:
    with pytest.raises(
        benchmark.BenchmarkDatasetError,
        match=(
            "dataset is missing "
            "target_draw_index"
        ),
    ):
        benchmark._target_indices(
            pd.DataFrame(
                {
                    "value": [
                        1,
                    ],
                }
            )
        )


def test_split_dataset_with_purge() -> None:
    dataset = make_dataset(
        list(
            range(
                1,
                11,
            )
        )
    )

    (
        training,
        validation,
        training_indices,
        purged_indices,
        validation_indices,
    ) = (
        benchmark.split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=1,
        )
    )

    assert training_indices == (
        1,
        2,
        3,
        4,
        5,
        6,
    )
    assert purged_indices == (
        7,
    )
    assert validation_indices == (
        8,
        9,
        10,
    )
    assert (
        training[
            "target_draw_index"
        ].nunique()
        == 6
    )
    assert (
        validation[
            "target_draw_index"
        ].nunique()
        == 3
    )


def test_split_dataset_with_zero_purge() -> None:
    dataset = make_dataset(
        list(
            range(
                1,
                9,
            )
        )
    )

    (
        _training,
        _validation,
        training_indices,
        purged_indices,
        validation_indices,
    ) = (
        benchmark.split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=0,
        )
    )

    assert training_indices == (
        1,
        2,
        3,
        4,
        5,
    )
    assert purged_indices == ()
    assert validation_indices == (
        6,
        7,
        8,
    )


def test_split_dataset_with_purge_rejects_insufficient_targets() -> None:
    with pytest.raises(
        benchmark.BenchmarkDatasetError,
        match=(
            "Not enough targets"
        ),
    ):
        benchmark.split_dataset_with_purge(
            dataset=make_dataset(
                [
                    1,
                    2,
                    3,
                ]
            ),
            validation_targets=3,
            purge_targets=1,
        )


def test_prepare_benchmark_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def __init__(
            self,
        ) -> None:
            self.closed = False

        def close(
            self,
        ) -> None:
            self.closed = True

    database = (
        FakeDatabase()
    )
    dataset = make_dataset(
        list(
            range(
                1,
                11,
            )
        )
    )

    monkeypatch.setattr(
        benchmark,
        "SessionLocal",
        lambda: database,
    )
    monkeypatch.setattr(
        benchmark.V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: list(
                range(
                    100
                )
            )
        ),
    )
    monkeypatch.setattr(
        benchmark.V7RankingDataset,
        "build_from_draws",
        lambda self, **kwargs: (
            dataset,
            list(
                range(
                    10
                )
            ),
        ),
    )
    monkeypatch.setattr(
        benchmark.V7FeatureAblationReport,
        "_validate_dataset",
        classmethod(
            lambda cls, frame: None
        ),
    )

    (
        training,
        validation,
        summary,
    ) = (
        benchmark.prepare_benchmark_datasets(
            make_config(
                tmp_path,
                validation_targets=5,
            ),
            sample_contract(),
        )
    )

    assert database.closed is True
    assert summary.draw_count == 100
    assert summary.dataset_targets == 10
    assert summary.training_targets == 4
    assert summary.validation_targets == 5
    assert summary.purged_target_indices == (
        5,
    )
    assert len(
        training
    ) == 4 * 49
    assert len(
        validation
    ) == 5 * 49


def test_prepare_benchmark_datasets_wraps_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def __init__(
            self,
        ) -> None:
            self.closed = False

        def close(
            self,
        ) -> None:
            self.closed = True

    database = (
        FakeDatabase()
    )

    monkeypatch.setattr(
        benchmark,
        "SessionLocal",
        lambda: database,
    )
    monkeypatch.setattr(
        benchmark.V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: (
                (_ for _ in ())
                .throw(
                    RuntimeError(
                        "database failed"
                    )
                )
            )
        ),
    )

    with pytest.raises(
        benchmark.BenchmarkDatasetError,
        match=(
            "Unable to build the V7 "
            "benchmark dataset"
        ),
    ):
        benchmark.prepare_benchmark_datasets(
            make_config(
                tmp_path
            ),
            sample_contract(),
        )

    assert database.closed is True


class FakeClassifier:
    """Deterministic probability model for evaluation tests."""

    def __init__(
        self,
    ) -> None:
        self.classes_ = np.asarray(
            [
                0,
                1,
            ]
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "FakeClassifier":
        assert not X.empty
        assert set(
            y.unique()
        ) == {
            0,
            1,
        }
        return self

    def predict_proba(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        candidate_scores = (
            np.linspace(
                0.99,
                0.01,
                len(
                    X
                ),
            )
        )

        return np.column_stack(
            (
                1.0
                - candidate_scores,
                candidate_scores,
            )
        )


def test_positive_probabilities() -> None:
    model = (
        FakeClassifier()
    )
    frame = pd.DataFrame(
        {
            "value": [
                1.0,
                2.0,
            ],
        }
    )

    result = (
        benchmark._positive_probabilities(
            model,
            frame,
        )
    )

    assert result.shape == (
        2,
    )
    assert np.isfinite(
        result
    ).all()


def test_positive_probabilities_supports_zero_only_class() -> None:
    model = SimpleNamespace(
        classes_=np.asarray(
            [
                0,
            ]
        ),
        predict_proba=lambda X: (
            np.ones(
                (
                    len(
                        X
                    ),
                    1,
                )
            )
        ),
    )

    result = (
        benchmark._positive_probabilities(
            model,
            pd.DataFrame(
                {
                    "value": [
                        1,
                        2,
                    ],
                }
            ),
        )
    )

    assert np.array_equal(
        result,
        np.zeros(
            2
        ),
    )


def test_evaluate_target_groups() -> None:
    validation = make_dataset(
        [
            10,
        ]
    )
    probabilities = np.linspace(
        0.99,
        0.01,
        49,
    )

    evaluations = (
        benchmark._evaluate_target_groups(
            validation_dataset=(
                validation
            ),
            probabilities=(
                probabilities
            ),
            top_k=5,
        )
    )

    assert len(
        evaluations
    ) == 1
    assert (
        evaluations[0]
        .selected_numbers
        == (
            1,
            2,
            3,
            4,
            5,
        )
    )
    assert evaluations[0].hits == 5


def test_evaluate_feature_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(
                1,
                9,
            )
        )
    )
    (
        training,
        validation,
        *_rest,
    ) = (
        benchmark.split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=1,
        )
    )

    monkeypatch.setattr(
        benchmark.V7FeatureAblationReport,
        "_build_model",
        staticmethod(
            lambda: (
                FakeClassifier()
            )
        ),
    )

    result = (
        benchmark.evaluate_feature_contract(
            experiment_name=(
                "production_11_features"
            ),
            feature_columns=(
                PRODUCTION_FEATURES
            ),
            reference_features=(
                REFERENCE_FEATURES
            ),
            training_dataset=(
                training
            ),
            validation_dataset=(
                validation
            ),
            top_k=5,
        )
    )

    assert result.feature_count == 11
    assert result.removed_features == (
        "rate_10",
    )
    assert result.validation_targets == 3
    assert len(
        result.target_evaluations
    ) == 3
    assert result.total_hits == 15
    assert result.mean_hits_at_k == pytest.approx(
        5.0
    )
    assert result.target_hit_rate == pytest.approx(
        1.0
    )


def test_evaluate_feature_contract_rejects_unknown_feature() -> None:
    dataset = make_dataset(
        list(
            range(
                1,
                9,
            )
        )
    )
    (
        training,
        validation,
        *_rest,
    ) = (
        benchmark.split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=1,
        )
    )

    with pytest.raises(
        benchmark.BenchmarkEvaluationError,
        match=(
            "outside the engineered "
            "reference contract"
        ),
    ):
        benchmark.evaluate_feature_contract(
            experiment_name="invalid",
            feature_columns=(
                "unknown_feature",
            ),
            reference_features=(
                REFERENCE_FEATURES
            ),
            training_dataset=(
                training
            ),
            validation_dataset=(
                validation
            ),
            top_k=5,
        )


def test_build_full_feature_dictionary() -> None:
    target = make_dataset(
        [
            1,
        ]
    )

    result = (
        benchmark.build_full_feature_dictionary(
            target
        )
    )

    assert len(
        result
    ) == 396
    assert result[
        "history_size"
    ] == 100
    assert "rate_10_1" in result
    assert "rate_10_49" in result
    assert "frequency_volatility_49" in result


def test_build_full_feature_dictionary_rejects_missing_candidate() -> None:
    target = make_dataset(
        [
            1,
        ]
    ).iloc[
        :-1
    ]

    with pytest.raises(
        benchmark.ProductionSmokeError,
        match=(
            "candidates 1 through 49"
        ),
    ):
        benchmark.build_full_feature_dictionary(
            target
        )


def test_build_full_feature_dictionary_rejects_varying_global() -> None:
    target = make_dataset(
        [
            1,
        ]
    )
    target.loc[
        1,
        "history_size",
    ] = 101

    with pytest.raises(
        benchmark.ProductionSmokeError,
        match=(
            "Global feature history_size "
            "varies"
        ),
    ):
        benchmark.build_full_feature_dictionary(
            target
        )


def test_ranking_sorted() -> None:
    assert (
        benchmark._ranking_sorted(
            (
                {
                    "number": 1,
                    "score": 0.9,
                },
                {
                    "number": 2,
                    "score": 0.9,
                },
                {
                    "number": 3,
                    "score": 0.8,
                },
            )
        )
        is True
    )

    assert (
        benchmark._ranking_sorted(
            (
                {
                    "number": 2,
                    "score": 0.9,
                },
                {
                    "number": 1,
                    "score": 0.9,
                },
            )
        )
        is False
    )


class FakeProductionModel:
    """Public-path compatible deterministic production model."""

    EXPECTED_FEATURE_COUNT = 11
    VERSION = "TEST-V7"

    def fit(
        self,
        dataset: pd.DataFrame,
    ) -> "FakeProductionModel":
        assert not dataset.empty
        return self

    def predict_top_k(
        self,
        *,
        features: dict[
            str,
            Any,
        ],
        top_k: int,
    ) -> dict[
        str,
        Any,
    ]:
        assert len(
            features
        ) == 396

        ranking = [
            {
                "number": number,
                "score": (
                    50
                    - number
                )
                / 50.0,
            }
            for number
            in range(
                1,
                50,
            )
        ]

        return {
            "version": self.VERSION,
            "top_k": top_k,
            "predicted_numbers": [
                item[
                    "number"
                ]
                for item
                in ranking[
                    :top_k
                ]
            ],
            "ranking": ranking,
            "probabilities": {
                item[
                    "number"
                ]: item[
                    "score"
                ]
                for item
                in ranking
            },
            "feature_count": 11,
            "candidate_count": 49,
        }


def test_run_production_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(
                1,
                9,
            )
        )
    )
    (
        training,
        validation,
        *_rest,
    ) = (
        benchmark.split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=1,
        )
    )

    monkeypatch.setattr(
        benchmark,
        "V7RankingModel",
        FakeProductionModel,
    )

    result = (
        benchmark.run_production_smoke(
            training_dataset=(
                training
            ),
            validation_dataset=(
                validation
            ),
            top_k=5,
        )
    )

    assert result.passed is True
    assert result.feature_dictionary_count == 396
    assert result.model_feature_count == 11
    assert result.candidate_count == 49
    assert result.predicted_numbers == (
        1,
        2,
        3,
        4,
        5,
    )
    assert result.hits == 5


def test_run_production_smoke_wraps_prediction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def fit(
            self,
            dataset: pd.DataFrame,
        ) -> None:
            raise RuntimeError(
                "fit failed"
            )

    dataset = make_dataset(
        list(
            range(
                1,
                9,
            )
        )
    )
    (
        training,
        validation,
        *_rest,
    ) = (
        benchmark.split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=1,
        )
    )

    monkeypatch.setattr(
        benchmark,
        "V7RankingModel",
        FailingModel,
    )

    with pytest.raises(
        benchmark.ProductionSmokeError,
        match=(
            "Public production fit/predict "
            "smoke path failed"
        ),
    ):
        benchmark.run_production_smoke(
            training_dataset=(
                training
            ),
            validation_dataset=(
                validation
            ),
            top_k=5,
        )


def test_compare_benchmark_runs_accepts_improvement() -> None:
    reference = make_run(
        experiment_name="reference",
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.46,
        target_hit_rate=0.46,
        total_seconds=2.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name="production",
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.49,
        target_hit_rate=0.49,
        total_seconds=1.8,
        removed_features=(
            "rate_10",
        ),
    )

    comparison = (
        benchmark.compare_benchmark_runs(
            reference_run=(
                reference
            ),
            production_run=(
                production
            ),
            accuracy_tolerance=0.0,
            maximum_runtime_ratio=2.0,
        )
    )

    assert comparison.removed_features == (
        "rate_10",
    )
    assert comparison.absolute_hits_delta == pytest.approx(
        0.03
    )
    assert comparison.accuracy_accepted is True
    assert comparison.runtime_accepted is True
    assert comparison.runtime_ratio == pytest.approx(
        0.9
    )


def test_compare_benchmark_runs_rejects_accuracy_regression() -> None:
    reference = make_run(
        experiment_name="reference",
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=1.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name="production",
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.40,
        target_hit_rate=0.40,
        total_seconds=1.0,
        removed_features=(
            "rate_10",
        ),
    )

    comparison = (
        benchmark.compare_benchmark_runs(
            reference_run=(
                reference
            ),
            production_run=(
                production
            ),
            accuracy_tolerance=0.0,
            maximum_runtime_ratio=2.0,
        )
    )

    assert comparison.accuracy_accepted is False


def test_compare_benchmark_runs_applies_tolerance() -> None:
    reference = make_run(
        experiment_name="reference",
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=1.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name="production",
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.49,
        target_hit_rate=0.49,
        total_seconds=1.0,
        removed_features=(
            "rate_10",
        ),
    )

    comparison = (
        benchmark.compare_benchmark_runs(
            reference_run=(
                reference
            ),
            production_run=(
                production
            ),
            accuracy_tolerance=0.01,
            maximum_runtime_ratio=2.0,
        )
    )

    assert comparison.accuracy_accepted is True


def test_compare_benchmark_runs_rejects_runtime_regression() -> None:
    reference = make_run(
        experiment_name="reference",
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=1.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name="production",
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=3.0,
        removed_features=(
            "rate_10",
        ),
    )

    comparison = (
        benchmark.compare_benchmark_runs(
            reference_run=(
                reference
            ),
            production_run=(
                production
            ),
            accuracy_tolerance=0.0,
            maximum_runtime_ratio=2.0,
        )
    )

    assert comparison.runtime_accepted is False


def test_run_production_benchmark_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = make_run(
        experiment_name=(
            "historical_reference_12_features"
        ),
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.46,
        target_hit_rate=0.46,
        total_seconds=2.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name=(
            "production_11_features"
        ),
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.49,
        target_hit_rate=0.49,
        total_seconds=1.8,
        removed_features=(
            "rate_10",
        ),
    )

    monkeypatch.setattr(
        benchmark,
        "validate_production_contract",
        sample_contract,
    )
    monkeypatch.setattr(
        benchmark,
        "prepare_benchmark_datasets",
        lambda config, contract: (
            pd.DataFrame(),
            pd.DataFrame(),
            sample_dataset_summary(),
        ),
    )

    calls = iter(
        [
            reference,
            production,
        ]
    )

    monkeypatch.setattr(
        benchmark,
        "evaluate_feature_contract",
        lambda **kwargs: next(
            calls
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "run_production_smoke",
        lambda **kwargs: (
            sample_smoke()
        ),
    )

    report = (
        benchmark.run_production_benchmark(
            make_config(
                tmp_path
            )
        )
    )

    assert report.ready_for_production is True
    assert report.recommendation == (
        "READY_FOR_PRODUCTION"
    )
    assert report.comparison.absolute_hits_delta == pytest.approx(
        0.03
    )


def test_run_production_benchmark_requires_accuracy_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = make_run(
        experiment_name=(
            "historical_reference_12_features"
        ),
        features=(
            REFERENCE_FEATURES
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=1.0,
        removed_features=(),
    )
    production = make_run(
        experiment_name=(
            "production_11_features"
        ),
        features=(
            PRODUCTION_FEATURES
        ),
        mean_hits=0.40,
        target_hit_rate=0.40,
        total_seconds=1.0,
        removed_features=(
            "rate_10",
        ),
    )

    monkeypatch.setattr(
        benchmark,
        "validate_production_contract",
        sample_contract,
    )
    monkeypatch.setattr(
        benchmark,
        "prepare_benchmark_datasets",
        lambda config, contract: (
            pd.DataFrame(),
            pd.DataFrame(),
            sample_dataset_summary(),
        ),
    )

    calls = iter(
        [
            reference,
            production,
        ]
    )

    monkeypatch.setattr(
        benchmark,
        "evaluate_feature_contract",
        lambda **kwargs: next(
            calls
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "run_production_smoke",
        lambda **kwargs: (
            sample_smoke()
        ),
    )

    report = (
        benchmark.run_production_benchmark(
            make_config(
                tmp_path
            )
        )
    )

    assert report.ready_for_production is False
    assert report.recommendation == (
        "REVIEW_PRODUCTION_PRUNING"
    )


def test_json_safe() -> None:
    result = (
        benchmark._json_safe(
            {
                "path": Path(
                    "/tmp/report"
                ),
                "tuple": (
                    1,
                    2,
                ),
                "nan": float(
                    "nan"
                ),
                "numpy": np.int64(
                    7
                ),
            }
        )
    )

    assert result == {
        "path": "/tmp/report",
        "tuple": [
            1,
            2,
        ],
        "nan": None,
        "numpy": 7,
    }


def test_write_csv(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "report.csv"
    )

    resolved = (
        benchmark._write_csv(
            path=path,
            fieldnames=(
                "feature",
                "delta",
            ),
            rows=(
                {
                    "feature": (
                        "rate_10"
                    ),
                    "delta": 0.03,
                },
            ),
        )
    )

    assert resolved == (
        path.resolve()
    )

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    assert rows == [
        {
            "feature": (
                "rate_10"
            ),
            "delta": "0.03",
        }
    ]


def test_target_comparison_rows() -> None:
    rows = (
        benchmark._target_comparison_rows(
            sample_report()
        )
    )

    assert len(
        rows
    ) == 5
    assert rows[0][
        "reference_hits"
    ] == 1
    assert rows[0][
        "production_hits"
    ] == 1


def test_export_production_benchmark(
    tmp_path: Path,
) -> None:
    report = (
        sample_report()
    )

    files = (
        benchmark.export_production_benchmark(
            report=report,
            output_directory=(
                tmp_path
                / "benchmark"
            ),
        )
    )

    assert set(
        files
    ) == {
        "json",
        "text",
        "targets_csv",
        "recommendation",
    }

    payload = json.loads(
        files[
            "json"
        ].read_text(
            encoding="utf-8"
        )
    )
    text = files[
        "text"
    ].read_text(
        encoding="utf-8"
    )
    recommendation = files[
        "recommendation"
    ].read_text(
        encoding="utf-8"
    )

    with files[
        "targets_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(
                file
            )
        )

    assert payload[
        "ready_for_production"
    ] is True
    assert (
        "FINAL PRODUCTION BENCHMARK"
        in text
    )
    assert (
        "recommendation="
        "READY_FOR_PRODUCTION"
        in recommendation
    )
    assert len(
        rows
    ) == 5


def test_print_production_benchmark(
    capsys: pytest.CaptureFixture[
        str
    ],
    tmp_path: Path,
) -> None:
    report = (
        sample_report()
    )

    benchmark.print_production_benchmark(
        report=report,
        generated_files={
            "json": (
                tmp_path
                / "report.json"
            ),
        },
    )

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        "READY_FOR_PRODUCTION"
        in output
    )
    assert (
        "SUCCESS"
        in output
    )


def test_build_argument_parser_defaults() -> None:
    arguments = (
        benchmark.build_argument_parser()
        .parse_args(
            []
        )
    )

    assert arguments.window_size == 100
    assert arguments.max_training_targets == 1500
    assert arguments.validation_targets == 100
    assert arguments.top_k == 5
    assert arguments.purge_targets == 1
    assert arguments.accuracy_tolerance == pytest.approx(
        0.0
    )
    assert arguments.maximum_runtime_ratio == pytest.approx(
        2.0
    )


def test_build_argument_parser_overrides() -> None:
    arguments = (
        benchmark.build_argument_parser()
        .parse_args(
            [
                "--validation-targets",
                "50",
                "--top-k",
                "7",
                "--purge-targets",
                "2",
                "--accuracy-tolerance",
                "0.01",
                "--maximum-runtime-ratio",
                "1.5",
            ]
        )
    )

    assert arguments.validation_targets == 50
    assert arguments.top_k == 7
    assert arguments.purge_targets == 2
    assert arguments.accuracy_tolerance == pytest.approx(
        0.01
    )
    assert arguments.maximum_runtime_ratio == pytest.approx(
        1.5
    )


def test_main_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = (
        sample_report()
    )
    captured: dict[
        str,
        Any,
    ] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "production_benchmark",
            "--output-directory",
            str(
                tmp_path
                / "output"
            ),
            "--validation-targets",
            "5",
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "run_production_benchmark",
        lambda config: (
            captured.setdefault(
                "config",
                config,
            )
            and report
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "export_production_benchmark",
        lambda **kwargs: {
            "json": (
                tmp_path
                / "report.json"
            ),
        },
    )
    monkeypatch.setattr(
        benchmark,
        "print_production_benchmark",
        lambda **kwargs: (
            captured.update(
                {
                    "printed": (
                        kwargs
                    ),
                }
            )
        ),
    )

    assert benchmark.main() == 0
    assert (
        captured[
            "config"
        ].validation_targets
        == 5
    )
    assert (
        captured[
            "printed"
        ][
            "report"
        ]
        is report
    )


def test_main_returns_one_on_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[
        str
    ],
) -> None:
    error = (
        benchmark.BenchmarkEvaluationError(
            "benchmark failed"
        )
    )
    error.__cause__ = RuntimeError(
        "underlying failure"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "production_benchmark",
            "--output-directory",
            str(
                tmp_path
                / "output"
            ),
            "--validation-targets",
            "5",
        ],
    )
    monkeypatch.setattr(
        benchmark,
        "run_production_benchmark",
        lambda config: (
            (_ for _ in ())
            .throw(
                error
            )
        ),
    )

    assert benchmark.main() == 1

    output = (
        capsys
        .readouterr()
        .out
    )

    assert (
        "ERROR: benchmark failed"
        in output
    )
    assert (
        "CAUSE: RuntimeError: "
        "underlying failure"
        in output
    )
