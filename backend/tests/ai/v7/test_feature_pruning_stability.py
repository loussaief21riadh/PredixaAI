from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.ai.v7.explainability import feature_pruning_stability as stability
from app.ai.v7.explainability.feature_ablation_runner import (
    FeatureAblationComparison,
    FeatureAblationRunResult,
)


MODEL_FEATURES = (
    "rate_10",
    "recency",
    "recency_ratio",
    "short_vs_long",
    "other_feature",
)


@pytest.fixture(autouse=True)
def patch_model_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep tests independent from the production feature list."""

    monkeypatch.setattr(
        stability.V7RankingDataset,
        "feature_columns",
        classmethod(
            lambda cls: MODEL_FEATURES
        ),
    )


def make_config(
    tmp_path: Path,
    **overrides: Any,
) -> stability.StabilityConfig:
    """Build one valid Sprint 4 configuration."""

    values: dict[str, Any] = {
        "output_directory": (
            tmp_path
            / "feature_pruning_stability"
        ),
        "removal_sequence": (
            "short_vs_long",
            "rate_10",
        ),
        "window_size": 100,
        "max_training_targets": 1500,
        "validation_targets": 5,
        "top_k": 5,
        "purge_targets": 1,
        "tolerance": 0.0,
        "window_count": 5,
        "window_step_targets": 5,
        "minimum_stability_rate": 0.80,
        "minimum_training_targets": 5,
    }

    values.update(overrides)

    return stability.StabilityConfig(
        **values
    )


def make_dataset(
    target_indices: list[int],
    *,
    rows_per_target: int = 2,
) -> pd.DataFrame:
    """Build a small chronological ranking-like dataset."""

    rows: list[dict[str, Any]] = []

    for target_index in target_indices:
        for candidate in range(
            1,
            rows_per_target + 1,
        ):
            rows.append(
                {
                    "target_draw_index": target_index,
                    "candidate_number": candidate,
                    "target": int(
                        candidate == 1
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_window(
    window_id: int = 1,
    *,
    training: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
    ),
    purged: tuple[int, ...] = (6,),
    validation: tuple[int, ...] = (
        7,
        8,
        9,
        10,
        11,
    ),
) -> stability.TemporalWindow:
    """Build one temporal window."""

    return stability.TemporalWindow(
        window_id=window_id,
        training_target_indices=training,
        purged_target_indices=purged,
        validation_target_indices=validation,
    )


def make_run_result(
    *,
    experiment_name: str,
    feature_columns: tuple[str, ...],
    mean_hits: float,
    target_hit_rate: float,
    total_seconds: float = 1.0,
    validation_targets: int = 5,
    top_k: int = 5,
) -> FeatureAblationRunResult:
    """Build one deterministic model-run result."""

    total_hits = int(
        round(
            mean_hits
            * validation_targets
        )
    )
    targets_with_hit = int(
        round(
            target_hit_rate
            * validation_targets
        )
    )

    return FeatureAblationRunResult(
        experiment_name=experiment_name,
        feature_columns=feature_columns,
        feature_count=len(
            feature_columns
        ),
        removed_features=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature not in feature_columns
        ),
        top_k=top_k,
        training_rows=100,
        validation_rows=50,
        training_targets=10,
        validation_targets=validation_targets,
        fit_seconds=0.8,
        prediction_seconds=0.1,
        total_seconds=total_seconds,
        total_hits=total_hits,
        mean_hits_at_k=mean_hits,
        normalized_hits_at_k=(
            mean_hits / top_k
        ),
        targets_with_at_least_one_hit=(
            targets_with_hit
        ),
        target_hit_rate=target_hit_rate,
        target_evaluations=(),
    )


def make_comparison(
    baseline: FeatureAblationRunResult,
    candidate: FeatureAblationRunResult,
    *,
    tolerance: float = 0.0,
) -> FeatureAblationComparison:
    """Build a comparison consistent with production acceptance rules."""

    delta = (
        candidate.mean_hits_at_k
        - baseline.mean_hits_at_k
    )
    accepted = (
        candidate.mean_hits_at_k
        >= (
            baseline.mean_hits_at_k
            - tolerance
        )
    )
    relative_delta = (
        delta
        / baseline.mean_hits_at_k
        if baseline.mean_hits_at_k
        else None
    )

    return FeatureAblationComparison(
        baseline_experiment=baseline.experiment_name,
        candidate_experiment=candidate.experiment_name,
        baseline_features=baseline.feature_columns,
        candidate_features=candidate.feature_columns,
        removed_features=tuple(
            feature
            for feature in baseline.feature_columns
            if feature not in candidate.feature_columns
        ),
        baseline_mean_hits_at_k=baseline.mean_hits_at_k,
        candidate_mean_hits_at_k=candidate.mean_hits_at_k,
        absolute_delta=delta,
        relative_delta=relative_delta,
        accepted=accepted,
        tolerance=tolerance,
    )


def make_window_result(
    *,
    window_id: int,
    baseline_score: float,
    after_short_score: float,
    final_score: float,
    tolerance: float = 0.0,
) -> stability.StabilityWindowResult:
    """Build a complete two-step window result."""

    baseline = make_run_result(
        experiment_name=(
            f"stability_w{window_id}_baseline"
        ),
        feature_columns=MODEL_FEATURES,
        mean_hits=baseline_score,
        target_hit_rate=0.40,
    )

    after_short_features = tuple(
        feature
        for feature in MODEL_FEATURES
        if feature != "short_vs_long"
    )

    after_short = make_run_result(
        experiment_name=(
            f"stability_w{window_id}_"
            "s1_without_short_vs_long"
        ),
        feature_columns=after_short_features,
        mean_hits=after_short_score,
        target_hit_rate=0.45,
    )

    final_features = tuple(
        feature
        for feature in after_short_features
        if feature != "rate_10"
    )

    final_run = make_run_result(
        experiment_name=(
            f"stability_w{window_id}_"
            "s2_without_rate_10"
        ),
        feature_columns=final_features,
        mean_hits=final_score,
        target_hit_rate=0.50,
    )

    comparison_1 = make_comparison(
        baseline,
        after_short,
        tolerance=tolerance,
    )
    comparison_2 = make_comparison(
        after_short,
        final_run,
        tolerance=tolerance,
    )

    delta = (
        final_run.mean_hits_at_k
        - baseline.mean_hits_at_k
    )
    relative_delta = (
        delta / baseline.mean_hits_at_k
        if baseline.mean_hits_at_k
        else None
    )

    window = make_window(
        window_id=window_id,
        training=tuple(
            range(
                1,
                6 + window_id,
            )
        ),
        purged=(
            6 + window_id,
        ),
        validation=tuple(
            range(
                7 + window_id,
                12 + window_id,
            )
        ),
    )

    return stability.StabilityWindowResult(
        window=window,
        training_rows=100,
        validation_rows=50,
        baseline_run=baseline,
        step_results=(
            stability.StabilityStepResult(
                window_id=window_id,
                step_number=1,
                removed_feature="short_vs_long",
                baseline_features=MODEL_FEATURES,
                candidate_features=after_short_features,
                baseline_run=baseline,
                candidate_run=after_short,
                comparison=comparison_1,
                decision=(
                    "ACCEPT"
                    if comparison_1.accepted
                    else "REJECT"
                ),
            ),
            stability.StabilityStepResult(
                window_id=window_id,
                step_number=2,
                removed_feature="rate_10",
                baseline_features=after_short_features,
                candidate_features=final_features,
                baseline_run=after_short,
                candidate_run=final_run,
                comparison=comparison_2,
                decision=(
                    "ACCEPT"
                    if comparison_2.accepted
                    else "REJECT"
                ),
            ),
        ),
        final_run=final_run,
        final_features=final_features,
        all_steps_accepted=(
            comparison_1.accepted
            and comparison_2.accepted
        ),
        total_absolute_delta=delta,
        total_relative_delta=relative_delta,
    )


def sample_dataset_summary() -> (
    stability.StabilityDatasetSummary
):
    """Return representative production-like dataset metadata."""

    return stability.StabilityDatasetSummary(
        draw_count=2780,
        dataset_rows=73500,
        dataset_targets=1500,
        first_dataset_target=1280,
        last_dataset_target=2779,
        requested_window_count=5,
        generated_window_count=5,
        validation_targets_per_window=100,
        window_step_targets=100,
        purge_targets_per_window=1,
        minimum_training_targets=500,
    )


def make_stable_window_results() -> tuple[
    stability.StabilityWindowResult,
    ...,
]:
    """Return five windows meeting an 80% stability threshold."""

    return (
        make_window_result(
            window_id=1,
            baseline_score=0.46,
            after_short_score=0.48,
            final_score=0.50,
        ),
        make_window_result(
            window_id=2,
            baseline_score=0.47,
            after_short_score=0.48,
            final_score=0.49,
        ),
        make_window_result(
            window_id=3,
            baseline_score=0.50,
            after_short_score=0.53,
            final_score=0.54,
        ),
        make_window_result(
            window_id=4,
            baseline_score=0.49,
            after_short_score=0.48,
            final_score=0.50,
        ),
        make_window_result(
            window_id=5,
            baseline_score=0.52,
            after_short_score=0.53,
            final_score=0.51,
        ),
    )


def make_report(
    *,
    stable_report: bool = True,
) -> stability.FeaturePruningStabilityReport:
    """Build a complete exportable report."""

    windows = make_stable_window_results()

    aggregates = (
        stability.aggregate_step_results(
            window_results=windows,
            removal_sequence=(
                "short_vs_long",
                "rate_10",
            ),
            minimum_stability_rate=0.80,
        )
    )

    final_accepted = sum(
        result.final_run.mean_hits_at_k
        >= result.baseline_run.mean_hits_at_k
        for result in windows
    )
    final_rate = (
        final_accepted / len(windows)
    )
    baseline_scores = [
        result.baseline_run.mean_hits_at_k
        for result in windows
    ]
    final_scores = [
        result.final_run.mean_hits_at_k
        for result in windows
    ]
    deltas = [
        result.total_absolute_delta
        for result in windows
    ]

    is_stable = (
        stable_report
        and all(
            aggregate.stable
            for aggregate in aggregates
        )
        and final_rate >= 0.80
    )

    return stability.FeaturePruningStabilityReport(
        status="success",
        version=stability.VERSION,
        protocol="test multi-window protocol",
        stable=is_stable,
        recommendation=(
            "ACCEPT_CUMULATIVE_PRUNING"
            if is_stable
            else "REJECT_OR_REVIEW_CUMULATIVE_PRUNING"
        ),
        removal_sequence=(
            "short_vs_long",
            "rate_10",
        ),
        final_features=(
            "recency",
            "recency_ratio",
            "other_feature",
        ),
        tolerance=0.0,
        minimum_stability_rate=0.80,
        window_size=100,
        max_training_targets=1500,
        validation_targets=100,
        top_k=5,
        purge_targets=1,
        window_count=5,
        window_step_targets=100,
        minimum_training_targets=500,
        dataset=sample_dataset_summary(),
        windows=windows,
        step_aggregates=aggregates,
        final_model_accepted_window_count=final_accepted,
        final_model_acceptance_rate=final_rate,
        mean_baseline_hits_at_k=(
            sum(baseline_scores)
            / len(baseline_scores)
        ),
        mean_final_hits_at_k=(
            sum(final_scores)
            / len(final_scores)
        ),
        mean_total_absolute_delta=(
            sum(deltas)
            / len(deltas)
        ),
        minimum_total_absolute_delta=min(
            deltas
        ),
        maximum_total_absolute_delta=max(
            deltas
        ),
    )


def test_config_validated_returns_self(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path
    )

    assert config.validated() is config


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "output_directory": "not-a-path",
            },
            (
                "output_directory must be "
                "a pathlib.Path"
            ),
        ),
        (
            {
                "removal_sequence": (),
            },
            "removal_sequence cannot be empty",
        ),
        (
            {
                "removal_sequence": (
                    "rate_10",
                    "rate_10",
                ),
            },
            (
                "duplicate feature in "
                "removal_sequence"
            ),
        ),
        (
            {
                "removal_sequence": (
                    "unknown_feature",
                ),
            },
            "unknown removal features",
        ),
        (
            {
                "removal_sequence": (
                    "rate_10",
                    "recency",
                    "recency_ratio",
                    "short_vs_long",
                    "other_feature",
                ),
            },
            (
                "removal_sequence must leave "
                "at least one model feature"
            ),
        ),
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
            (
                "max_training_targets cannot "
                "be negative"
            ),
        ),
        (
            {
                "validation_targets": 4,
            },
            (
                "validation_targets must be "
                "at least 5"
            ),
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
            (
                "purge_targets cannot be "
                "negative"
            ),
        ),
        (
            {
                "tolerance": float("nan"),
            },
            "tolerance must be finite",
        ),
        (
            {
                "tolerance": -0.01,
            },
            "tolerance cannot be negative",
        ),
        (
            {
                "window_count": 1,
            },
            "window_count must be at least 2",
        ),
        (
            {
                "window_step_targets": 0,
            },
            (
                "window_step_targets must "
                "be at least 1"
            ),
        ),
        (
            {
                "minimum_stability_rate": (
                    float("inf")
                ),
            },
            (
                "minimum_stability_rate must "
                "be finite"
            ),
        ),
        (
            {
                "minimum_stability_rate": 0.0,
            },
            (
                "minimum_stability_rate must "
                "be greater than 0 and at most 1"
            ),
        ),
        (
            {
                "minimum_stability_rate": 1.1,
            },
            (
                "minimum_stability_rate must "
                "be greater than 0 and at most 1"
            ),
        ),
        (
            {
                "minimum_training_targets": 0,
            },
            (
                "minimum_training_targets must "
                "be at least 1"
            ),
        ),
    ],
)
def test_validate_config_rejects_invalid_values(
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    config = make_config(
        tmp_path,
        **overrides,
    )

    with pytest.raises(
        stability.StabilityConfigurationError,
        match=message,
    ):
        stability.validate_config(
            config
        )


@pytest.mark.parametrize(
    ("values", "field_name", "allow_empty", "message"),
    [
        (
            "rate_10",
            "features",
            False,
            (
                "features must be a sequence "
                "of feature names"
            ),
        ),
        (
            (
                "rate_10",
                7,
            ),
            "features",
            False,
            (
                "features values must be strings"
            ),
        ),
        (
            (
                "rate_10",
                " ",
            ),
            "features",
            False,
            (
                "features values cannot be empty"
            ),
        ),
        (
            (),
            "features",
            False,
            "features cannot be empty",
        ),
    ],
)
def test_normalise_feature_sequence_rejects_invalid_values(
    values: Any,
    field_name: str,
    allow_empty: bool,
    message: str,
) -> None:
    with pytest.raises(
        stability.StabilityConfigurationError,
        match=message,
    ):
        stability._normalise_feature_sequence(
            values,
            field_name=field_name,
            allow_empty=allow_empty,
        )


def test_normalise_feature_sequence_strips_and_preserves_order() -> None:
    result = (
        stability
        ._normalise_feature_sequence(
            (
                " short_vs_long ",
                "rate_10",
            ),
            field_name="removals",
            allow_empty=False,
        )
    )

    assert result == (
        "short_vs_long",
        "rate_10",
    )


def test_normalise_feature_sequence_allows_empty_when_requested() -> None:
    assert (
        stability
        ._normalise_feature_sequence(
            (),
            field_name="features",
            allow_empty=True,
        )
        == ()
    )


def test_temporal_window_properties() -> None:
    window = make_window()

    assert window.training_target_count == 5
    assert window.validation_target_count == 5
    assert window.first_training_target == 1
    assert window.last_training_target == 5
    assert window.first_validation_target == 7
    assert window.last_validation_target == 11


def test_build_temporal_windows_newest_to_oldest() -> None:
    windows = (
        stability
        .build_temporal_windows(
            target_indices=tuple(
                range(1, 21)
            ),
            validation_targets=5,
            purge_targets=1,
            window_count=3,
            window_step_targets=3,
            minimum_training_targets=5,
        )
    )

    assert len(windows) == 3

    assert windows[0].training_target_indices == tuple(
        range(1, 15)
    )
    assert windows[0].purged_target_indices == (15,)
    assert windows[0].validation_target_indices == (
        16,
        17,
        18,
        19,
        20,
    )

    assert windows[1].training_target_indices == tuple(
        range(1, 12)
    )
    assert windows[1].purged_target_indices == (12,)
    assert windows[1].validation_target_indices == (
        13,
        14,
        15,
        16,
        17,
    )

    assert windows[2].training_target_indices == tuple(
        range(1, 9)
    )
    assert windows[2].purged_target_indices == (9,)
    assert windows[2].validation_target_indices == (
        10,
        11,
        12,
        13,
        14,
    )


def test_build_temporal_windows_deduplicates_and_sorts() -> None:
    windows = (
        stability
        .build_temporal_windows(
            target_indices=(
                5,
                3,
                1,
                2,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
            ),
            validation_targets=5,
            purge_targets=1,
            window_count=2,
            window_step_targets=2,
            minimum_training_targets=5,
        )
    )

    assert windows[0].validation_target_indices == (
        11,
        12,
        13,
        14,
        15,
    )
    assert windows[1].validation_target_indices == (
        9,
        10,
        11,
        12,
        13,
    )


def test_build_temporal_windows_supports_zero_purge() -> None:
    windows = (
        stability
        .build_temporal_windows(
            target_indices=tuple(
                range(1, 16)
            ),
            validation_targets=5,
            purge_targets=0,
            window_count=2,
            window_step_targets=3,
            minimum_training_targets=5,
        )
    )

    assert windows[0].purged_target_indices == ()
    assert windows[0].training_target_indices[-1] == 10
    assert windows[0].validation_target_indices[0] == 11


def test_build_temporal_windows_rejects_empty_targets() -> None:
    with pytest.raises(
        stability.StabilityDatasetError,
        match="no target indices are available",
    ):
        stability.build_temporal_windows(
            target_indices=(),
            validation_targets=5,
            purge_targets=1,
            window_count=2,
            window_step_targets=5,
            minimum_training_targets=5,
        )


def test_build_temporal_windows_rejects_string_sequence() -> None:
    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "target_indices must be a sequence"
        ),
    ):
        stability.build_temporal_windows(
            target_indices="12345",
            validation_targets=5,
            purge_targets=1,
            window_count=2,
            window_step_targets=5,
            minimum_training_targets=5,
        )


def test_build_temporal_windows_rejects_invalid_index() -> None:
    with pytest.raises(
        stability.StabilityDatasetError,
        match="invalid target index",
    ):
        stability.build_temporal_windows(
            target_indices=(
                1,
                2,
                "invalid",
            ),
            validation_targets=5,
            purge_targets=1,
            window_count=2,
            window_step_targets=5,
            minimum_training_targets=1,
        )


def test_build_temporal_windows_rejects_insufficient_targets() -> None:
    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "Not enough chronological targets "
            "to build 3 windows"
        ),
    ):
        stability.build_temporal_windows(
            target_indices=tuple(
                range(1, 15)
            ),
            validation_targets=5,
            purge_targets=1,
            window_count=3,
            window_step_targets=5,
            minimum_training_targets=5,
        )


def test_prepare_dataset_and_windows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    database = FakeDatabase()
    dataset = make_dataset(
        list(
            range(1, 31)
        )
    )
    metadata = list(
        range(30)
    )
    draws = list(
        range(100)
    )

    monkeypatch.setattr(
        stability,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: draws
        ),
    )

    monkeypatch.setattr(
        stability.V7RankingDataset,
        "build_from_draws",
        lambda self, **kwargs: (
            dataset,
            metadata,
        ),
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
        "_validate_dataset",
        classmethod(
            lambda cls, frame: None
        ),
    )

    config = make_config(
        tmp_path,
        window_count=3,
        window_step_targets=5,
        minimum_training_targets=5,
    )

    returned_dataset, summary, windows = (
        stability
        .prepare_dataset_and_windows(
            config
        )
    )

    assert database.closed is True
    assert returned_dataset.equals(
        dataset
    )
    assert summary.draw_count == 100
    assert summary.dataset_rows == len(
        dataset
    )
    assert summary.dataset_targets == 30
    assert summary.first_dataset_target == 1
    assert summary.last_dataset_target == 30
    assert summary.generated_window_count == 3
    assert len(windows) == 3


def test_prepare_dataset_and_windows_wraps_build_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    database = FakeDatabase()

    monkeypatch.setattr(
        stability,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
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
        stability.StabilityDatasetError,
        match=(
            "Unable to build the V7 "
            "ranking dataset"
        ),
    ):
        stability.prepare_dataset_and_windows(
            make_config(
                tmp_path
            )
        )

    assert database.closed is True


def test_prepare_dataset_and_windows_wraps_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def close(self) -> None:
            return None

    dataset = make_dataset(
        list(
            range(1, 31)
        )
    )

    monkeypatch.setattr(
        stability,
        "SessionLocal",
        lambda: FakeDatabase(),
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: list(
                range(100)
            )
        ),
    )

    monkeypatch.setattr(
        stability.V7RankingDataset,
        "build_from_draws",
        lambda self, **kwargs: (
            dataset,
            list(
                range(30)
            ),
        ),
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
        "_validate_dataset",
        classmethod(
            lambda cls, frame: (
                (_ for _ in ())
                .throw(
                    ValueError(
                        "invalid dataset"
                    )
                )
            )
        ),
    )

    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "generated V7 ranking dataset "
            "is invalid"
        ),
    ):
        stability.prepare_dataset_and_windows(
            make_config(
                tmp_path
            )
        )


def test_prepare_dataset_and_windows_rejects_missing_target_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        stability,
        "SessionLocal",
        lambda: FakeDatabase(),
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: list(
                range(10)
            )
        ),
    )

    monkeypatch.setattr(
        stability.V7RankingDataset,
        "build_from_draws",
        lambda self, **kwargs: (
            pd.DataFrame(
                {
                    "candidate_number": [
                        1,
                    ],
                }
            ),
            [
                1,
            ],
        ),
    )

    monkeypatch.setattr(
        stability.V7FeatureAblationReport,
        "_validate_dataset",
        classmethod(
            lambda cls, frame: None
        ),
    )

    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "dataset is missing "
            "target_draw_index"
        ),
    ):
        stability.prepare_dataset_and_windows(
            make_config(
                tmp_path
            )
        )


def test_split_dataset_for_window() -> None:
    dataset = make_dataset(
        list(
            range(1, 12)
        )
    )
    window = make_window()

    training, validation = (
        stability
        .split_dataset_for_window(
            dataset,
            window,
        )
    )

    assert (
        training[
            "target_draw_index"
        ].nunique()
        == 5
    )
    assert (
        validation[
            "target_draw_index"
        ].nunique()
        == 5
    )
    assert set(
        training[
            "target_draw_index"
        ]
    ) == {
        1,
        2,
        3,
        4,
        5,
    }
    assert set(
        validation[
            "target_draw_index"
        ]
    ) == {
        7,
        8,
        9,
        10,
        11,
    }


def test_split_dataset_for_window_rejects_non_dataframe() -> None:
    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "dataset must be a pandas DataFrame"
        ),
    ):
        stability.split_dataset_for_window(
            [],
            make_window(),
        )


def test_split_dataset_for_window_rejects_missing_column() -> None:
    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "dataset is missing "
            "target_draw_index"
        ),
    ):
        stability.split_dataset_for_window(
            pd.DataFrame(
                {
                    "value": [
                        1,
                    ],
                }
            ),
            make_window(),
        )


def test_split_dataset_for_window_rejects_incomplete_training_targets() -> None:
    dataset = make_dataset(
        [
            1,
            2,
            3,
            4,
            7,
            8,
            9,
            10,
            11,
        ]
    )

    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "window 1 training targets "
            "are incomplete"
        ),
    ):
        stability.split_dataset_for_window(
            dataset,
            make_window(),
        )


def test_split_dataset_for_window_rejects_incomplete_validation_targets() -> None:
    dataset = make_dataset(
        [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
        ]
    )

    with pytest.raises(
        stability.StabilityDatasetError,
        match=(
            "window 1 validation targets "
            "are incomplete"
        ),
    ):
        stability.split_dataset_for_window(
            dataset,
            make_window(),
        )


def test_evaluate_window_runs_cumulative_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(1, 12)
        )
    )

    scores = {
        "stability_w1_baseline": (
            0.46,
            0.40,
        ),
        (
            "stability_w1_s1_"
            "without_short_vs_long"
        ): (
            0.54,
            0.46,
        ),
        (
            "stability_w1_s2_"
            "without_rate_10"
        ): (
            0.63,
            0.52,
        ),
    }

    calls: list[
        tuple[str, tuple[str, ...]]
    ] = []

    def fake_run(
        *,
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        config: Any,
    ) -> FeatureAblationRunResult:
        assert not training_dataset.empty
        assert not validation_dataset.empty

        calls.append(
            (
                config.experiment_name,
                tuple(
                    config.feature_columns
                ),
            )
        )

        mean_hits, hit_rate = scores[
            config.experiment_name
        ]

        return make_run_result(
            experiment_name=(
                config.experiment_name
            ),
            feature_columns=tuple(
                config.feature_columns
            ),
            mean_hits=mean_hits,
            target_hit_rate=hit_rate,
        )

    monkeypatch.setattr(
        stability,
        "run_feature_subset",
        fake_run,
    )

    result = (
        stability
        .evaluate_window(
            dataset=dataset,
            window=make_window(),
            config=make_config(
                tmp_path
            ),
        )
    )

    assert len(calls) == 3
    assert result.baseline_run.mean_hits_at_k == pytest.approx(
        0.46
    )
    assert len(result.step_results) == 2
    assert result.step_results[0].removed_feature == (
        "short_vs_long"
    )
    assert result.step_results[1].removed_feature == (
        "rate_10"
    )
    assert all(
        step.decision == "ACCEPT"
        for step in result.step_results
    )
    assert result.all_steps_accepted is True
    assert result.final_features == (
        "recency",
        "recency_ratio",
        "other_feature",
    )
    assert result.total_absolute_delta == pytest.approx(
        0.17
    )


def test_evaluate_window_continues_after_rejected_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(1, 12)
        )
    )

    scores = iter(
        [
            0.50,
            0.48,
            0.52,
        ]
    )

    monkeypatch.setattr(
        stability,
        "run_feature_subset",
        lambda **kwargs: make_run_result(
            experiment_name=(
                kwargs[
                    "config"
                ].experiment_name
            ),
            feature_columns=tuple(
                kwargs[
                    "config"
                ].feature_columns
            ),
            mean_hits=next(
                scores
            ),
            target_hit_rate=0.40,
        ),
    )

    result = (
        stability
        .evaluate_window(
            dataset=dataset,
            window=make_window(),
            config=make_config(
                tmp_path
            ),
        )
    )

    assert (
        result.step_results[0].decision
        == "REJECT"
    )
    assert (
        result.step_results[1].decision
        == "ACCEPT"
    )
    assert result.all_steps_accepted is False
    assert result.final_run.mean_hits_at_k == pytest.approx(
        0.52
    )


def test_evaluate_window_handles_zero_baseline_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(1, 12)
        )
    )

    monkeypatch.setattr(
        stability,
        "run_feature_subset",
        lambda **kwargs: make_run_result(
            experiment_name=(
                kwargs[
                    "config"
                ].experiment_name
            ),
            feature_columns=tuple(
                kwargs[
                    "config"
                ].feature_columns
            ),
            mean_hits=0.0,
            target_hit_rate=0.0,
        ),
    )

    result = (
        stability
        .evaluate_window(
            dataset=dataset,
            window=make_window(),
            config=make_config(
                tmp_path
            ),
        )
    )

    assert result.total_absolute_delta == pytest.approx(
        0.0
    )
    assert result.total_relative_delta is None


def test_evaluate_window_wraps_baseline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(1, 12)
        )
    )

    monkeypatch.setattr(
        stability,
        "run_feature_subset",
        lambda **kwargs: (
            (_ for _ in ())
            .throw(
                RuntimeError(
                    "baseline failed"
                )
            )
        ),
    )

    with pytest.raises(
        stability.StabilityEvaluationError,
        match=(
            "Baseline evaluation failed "
            "for window 1"
        ),
    ) as captured:
        stability.evaluate_window(
            dataset=dataset,
            window=make_window(),
            config=make_config(
                tmp_path
            ),
        )

    assert isinstance(
        captured.value.__cause__,
        RuntimeError,
    )


def test_evaluate_window_wraps_step_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset(
        list(
            range(1, 12)
        )
    )
    calls = 0

    def fake_run(
        **kwargs: Any,
    ) -> FeatureAblationRunResult:
        nonlocal calls
        calls += 1

        if calls == 1:
            return make_run_result(
                experiment_name=(
                    kwargs[
                        "config"
                    ].experiment_name
                ),
                feature_columns=tuple(
                    kwargs[
                        "config"
                    ].feature_columns
                ),
                mean_hits=0.50,
                target_hit_rate=0.40,
            )

        raise RuntimeError(
            "step failed"
        )

    monkeypatch.setattr(
        stability,
        "run_feature_subset",
        fake_run,
    )

    with pytest.raises(
        stability.StabilityEvaluationError,
        match=(
            "window 1, step 1, feature "
            "short_vs_long"
        ),
    ) as captured:
        stability.evaluate_window(
            dataset=dataset,
            window=make_window(),
            config=make_config(
                tmp_path
            ),
        )

    assert isinstance(
        captured.value.__cause__,
        RuntimeError,
    )


def test_mean() -> None:
    assert stability._mean(
        [
            1.0,
            2.0,
            3.0,
        ]
    ) == pytest.approx(
        2.0
    )


def test_mean_rejects_empty_sequence() -> None:
    with pytest.raises(
        stability.StabilityEvaluationError,
        match=(
            "cannot calculate a mean "
            "from an empty sequence"
        ),
    ):
        stability._mean(
            []
        )


def test_aggregate_step_results() -> None:
    windows = (
        make_stable_window_results()
    )

    aggregates = (
        stability
        .aggregate_step_results(
            window_results=windows,
            removal_sequence=(
                "short_vs_long",
                "rate_10",
            ),
            minimum_stability_rate=0.80,
        )
    )

    assert len(aggregates) == 2

    first = aggregates[0]
    second = aggregates[1]

    assert first.removed_feature == (
        "short_vs_long"
    )
    assert first.accepted_window_count == 4
    assert first.acceptance_rate == pytest.approx(
        0.80
    )
    assert first.stable is True

    assert second.removed_feature == "rate_10"
    assert second.accepted_window_count == 4
    assert second.acceptance_rate == pytest.approx(
        0.80
    )
    assert second.stable is True


def test_aggregate_step_results_marks_unstable() -> None:
    windows = (
        make_window_result(
            window_id=1,
            baseline_score=0.50,
            after_short_score=0.49,
            final_score=0.48,
        ),
        make_window_result(
            window_id=2,
            baseline_score=0.50,
            after_short_score=0.49,
            final_score=0.51,
        ),
        make_window_result(
            window_id=3,
            baseline_score=0.50,
            after_short_score=0.51,
            final_score=0.52,
        ),
        make_window_result(
            window_id=4,
            baseline_score=0.50,
            after_short_score=0.49,
            final_score=0.48,
        ),
        make_window_result(
            window_id=5,
            baseline_score=0.50,
            after_short_score=0.51,
            final_score=0.50,
        ),
    )

    aggregates = (
        stability
        .aggregate_step_results(
            window_results=windows,
            removal_sequence=(
                "short_vs_long",
                "rate_10",
            ),
            minimum_stability_rate=0.80,
        )
    )

    assert aggregates[0].accepted_window_count == 2
    assert aggregates[0].stable is False


def test_aggregate_step_results_rejects_empty_windows() -> None:
    with pytest.raises(
        stability.StabilityEvaluationError,
        match="window_results cannot be empty",
    ):
        stability.aggregate_step_results(
            window_results=(),
            removal_sequence=(
                "rate_10",
            ),
            minimum_stability_rate=0.80,
        )


def test_aggregate_step_results_rejects_missing_step() -> None:
    window = make_window_result(
        window_id=1,
        baseline_score=0.50,
        after_short_score=0.51,
        final_score=0.52,
    )

    with pytest.raises(
        stability.StabilityEvaluationError,
        match=(
            "missing step 1 "
            r"\(recency\) for window 1"
        ),
    ):
        stability.aggregate_step_results(
            window_results=(
                window,
            ),
            removal_sequence=(
                "recency",
            ),
            minimum_stability_rate=0.80,
        )


def test_run_stability_validation_accepts_stable_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = tuple(
        result.window
        for result
        in make_stable_window_results()
    )

    monkeypatch.setattr(
        stability,
        "prepare_dataset_and_windows",
        lambda config: (
            pd.DataFrame(),
            sample_dataset_summary(),
            windows,
        ),
    )

    results_by_id = {
        result.window.window_id: result
        for result
        in make_stable_window_results()
    }

    monkeypatch.setattr(
        stability,
        "evaluate_window",
        lambda **kwargs: results_by_id[
            kwargs["window"].window_id
        ],
    )

    report = (
        stability
        .run_stability_validation(
            make_config(
                tmp_path
            )
        )
    )

    assert report.status == "success"
    assert report.stable is True
    assert report.recommendation == (
        "ACCEPT_CUMULATIVE_PRUNING"
    )
    assert report.final_model_accepted_window_count == 4
    assert report.final_model_acceptance_rate == pytest.approx(
        0.80
    )
    assert all(
        aggregate.stable
        for aggregate in report.step_aggregates
    )
    assert report.final_features == (
        "recency",
        "recency_ratio",
        "other_feature",
    )


def test_run_stability_validation_rejects_unstable_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = (
        make_window_result(
            window_id=1,
            baseline_score=0.50,
            after_short_score=0.49,
            final_score=0.48,
        ),
        make_window_result(
            window_id=2,
            baseline_score=0.50,
            after_short_score=0.49,
            final_score=0.49,
        ),
        make_window_result(
            window_id=3,
            baseline_score=0.50,
            after_short_score=0.51,
            final_score=0.52,
        ),
        make_window_result(
            window_id=4,
            baseline_score=0.50,
            after_short_score=0.49,
            final_score=0.48,
        ),
        make_window_result(
            window_id=5,
            baseline_score=0.50,
            after_short_score=0.51,
            final_score=0.50,
        ),
    )

    monkeypatch.setattr(
        stability,
        "prepare_dataset_and_windows",
        lambda config: (
            pd.DataFrame(),
            sample_dataset_summary(),
            tuple(
                result.window
                for result in results
            ),
        ),
    )

    results_by_id = {
        result.window.window_id: result
        for result in results
    }

    monkeypatch.setattr(
        stability,
        "evaluate_window",
        lambda **kwargs: results_by_id[
            kwargs["window"].window_id
        ],
    )

    report = (
        stability
        .run_stability_validation(
            make_config(
                tmp_path
            )
        )
    )

    assert report.stable is False
    assert report.recommendation == (
        "REJECT_OR_REVIEW_CUMULATIVE_PRUNING"
    )
    assert any(
        not aggregate.stable
        for aggregate in report.step_aggregates
    )


def test_run_stability_validation_requires_nonnegative_mean_delta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results = tuple(
        make_window_result(
            window_id=index,
            baseline_score=0.50,
            after_short_score=0.50,
            final_score=0.49,
            tolerance=0.02,
        )
        for index in range(
            1,
            6,
        )
    )

    monkeypatch.setattr(
        stability,
        "prepare_dataset_and_windows",
        lambda config: (
            pd.DataFrame(),
            sample_dataset_summary(),
            tuple(
                result.window
                for result in results
            ),
        ),
    )

    results_by_id = {
        result.window.window_id: result
        for result in results
    }

    monkeypatch.setattr(
        stability,
        "evaluate_window",
        lambda **kwargs: results_by_id[
            kwargs["window"].window_id
        ],
    )

    report = (
        stability
        .run_stability_validation(
            make_config(
                tmp_path,
                tolerance=0.0,
            )
        )
    )

    assert report.mean_total_absolute_delta < 0
    assert report.stable is False


def test_json_safe_converts_nested_values() -> None:
    payload = stability._json_safe(
        {
            "path": Path("/tmp/report"),
            "tuple": (
                1,
                Path("/tmp/a"),
            ),
            "values": [
                float("nan"),
                float("inf"),
            ],
        }
    )

    assert payload == {
        "path": "/tmp/report",
        "tuple": [
            1,
            "/tmp/a",
        ],
        "values": [
            None,
            None,
        ],
    }


def test_write_csv(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "report.csv"
    )

    resolved = (
        stability
        ._write_csv(
            path=path,
            fieldnames=(
                "feature",
                "delta",
            ),
            rows=(
                {
                    "feature": "rate_10",
                    "delta": 0.014,
                },
            ),
        )
    )

    assert resolved == path.resolve()

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert rows == [
        {
            "feature": "rate_10",
            "delta": "0.014",
        }
    ]


def test_export_stability_report(
    tmp_path: Path,
) -> None:
    report = make_report()

    files = (
        stability
        .export_stability_report(
            report=report,
            output_directory=(
                tmp_path
                / "stability"
            ),
        )
    )

    assert set(files) == {
        "json",
        "text",
        "windows_csv",
        "steps_csv",
        "recommendation",
    }

    payload = json.loads(
        files["json"]
        .read_text(
            encoding="utf-8"
        )
    )
    text = (
        files["text"]
        .read_text(
            encoding="utf-8"
        )
    )
    recommendation = (
        files["recommendation"]
        .read_text(
            encoding="utf-8"
        )
    )

    with files[
        "windows_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        window_rows = list(
            csv.DictReader(file)
        )

    with files[
        "steps_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        step_rows = list(
            csv.DictReader(file)
        )

    assert payload["stable"] is True
    assert payload["recommendation"] == (
        "ACCEPT_CUMULATIVE_PRUNING"
    )
    assert payload["removal_sequence"] == [
        "short_vs_long",
        "rate_10",
    ]
    assert len(window_rows) == 5
    assert len(step_rows) == 10
    assert (
        "PREDIXA AI V7 FEATURE PRUNING "
        "MULTI-WINDOW STABILITY"
        in text
    )
    assert (
        "recommendation="
        "ACCEPT_CUMULATIVE_PRUNING"
        in recommendation
    )
    assert "stable=true" in recommendation


def test_print_stability_report(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    report = make_report()

    stability.print_stability_report(
        report=report,
        generated_files={
            "json": (
                tmp_path
                / "report.json"
            ),
        },
    )

    output = (
        capsys.readouterr().out
    )

    assert (
        "PREDIXA AI V7 FEATURE PRUNING "
        "MULTI-WINDOW STABILITY"
        in output
    )
    assert (
        "ACCEPT_CUMULATIVE_PRUNING"
        in output
    )
    assert "short_vs_long -> rate_10" in output
    assert "SUCCESS" in output


def test_build_argument_parser_defaults() -> None:
    arguments = (
        stability
        .build_argument_parser()
        .parse_args([])
    )

    assert arguments.removals == [
        "short_vs_long",
        "rate_10",
    ]
    assert arguments.window_count == 5
    assert arguments.window_step_targets == 100
    assert (
        arguments.minimum_stability_rate
        == pytest.approx(0.80)
    )
    assert (
        arguments.minimum_training_targets
        == 500
    )


def test_build_argument_parser_accepts_overrides() -> None:
    arguments = (
        stability
        .build_argument_parser()
        .parse_args(
            [
                "--removals",
                "rate_10",
                "--window-count",
                "7",
                "--window-step-targets",
                "50",
                "--minimum-stability-rate",
                "0.90",
                "--minimum-training-targets",
                "700",
            ]
        )
    )

    assert arguments.removals == [
        "rate_10",
    ]
    assert arguments.window_count == 7
    assert arguments.window_step_targets == 50
    assert (
        arguments.minimum_stability_rate
        == pytest.approx(0.90)
    )
    assert (
        arguments.minimum_training_targets
        == 700
    )


def test_main_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = make_report()
    output_directory = (
        tmp_path
        / "output"
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_stability",
            "--output-directory",
            str(output_directory),
            "--removals",
            "rate_10",
            "--validation-targets",
            "5",
            "--window-count",
            "2",
            "--minimum-training-targets",
            "5",
        ],
    )

    monkeypatch.setattr(
        stability,
        "run_stability_validation",
        lambda config: (
            captured.setdefault(
                "config",
                config,
            )
            and report
        ),
    )

    monkeypatch.setattr(
        stability,
        "export_stability_report",
        lambda **kwargs: {
            "json": (
                output_directory
                / "report.json"
            ),
        },
    )

    monkeypatch.setattr(
        stability,
        "print_stability_report",
        lambda **kwargs: captured.update(
            {
                "printed": kwargs,
            }
        ),
    )

    assert stability.main() == 0
    assert captured["config"].removal_sequence == (
        "rate_10",
    )
    assert (
        captured["printed"]["report"]
        is report
    )


def test_main_returns_one_on_stability_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = (
        stability
        .StabilityEvaluationError(
            "stability failed"
        )
    )
    error.__cause__ = RuntimeError(
        "underlying failure"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_stability",
            "--output-directory",
            str(
                tmp_path
                / "output"
            ),
            "--validation-targets",
            "5",
            "--window-count",
            "2",
            "--minimum-training-targets",
            "5",
        ],
    )

    monkeypatch.setattr(
        stability,
        "run_stability_validation",
        lambda config: (
            (_ for _ in ())
            .throw(error)
        ),
    )

    assert stability.main() == 1

    output = (
        capsys.readouterr().out
    )

    assert (
        "PREDIXA AI V7 FEATURE PRUNING "
        "MULTI-WINDOW STABILITY"
        in output
    )
    assert (
        "ERROR: stability failed"
        in output
    )
    assert (
        "CAUSE: RuntimeError: "
        "underlying failure"
        in output
    )
