from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from app.ai.v7.explainability import feature_pruning_greedy as greedy
from app.ai.v7.explainability import feature_pruning_optimizer as optimizer
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
        greedy.V7RankingDataset,
        "feature_columns",
        classmethod(
            lambda cls: MODEL_FEATURES
        ),
    )


def make_config(
    tmp_path: Path,
    **overrides: Any,
) -> greedy.GreedyPruningConfig:
    """Build one valid cumulative-pruning configuration."""

    values: dict[str, Any] = {
        "correlation_report": (
            tmp_path
            / "feature_correlation_report.json"
        ),
        "output_directory": (
            tmp_path
            / "feature_pruning_greedy"
        ),
        "correlation_threshold": 0.80,
        "minimum_group_size": 2,
        "maximum_candidates_per_group": 1,
        "window_size": 100,
        "max_training_targets": 1500,
        "validation_targets": 5,
        "top_k": 5,
        "purge_targets": 1,
        "tolerance": 0.0,
        "candidate_scope": (
            "all_group_features"
        ),
        "explicit_features": (),
        "maximum_iterations": 10,
        "minimum_features": 1,
    }

    values.update(overrides)

    return greedy.GreedyPruningConfig(
        **values
    )


def sample_group() -> optimizer.FeatureGroup:
    """Return the connected high-correlation group."""

    return optimizer.FeatureGroup(
        group_id=1,
        features=(
            "rate_10",
            "recency",
            "recency_ratio",
            "short_vs_long",
        ),
        pair_count=4,
        maximum_absolute_correlation=1.0,
    )


def sample_plan() -> optimizer.PruningPlan:
    """Build a representative Sprint 2 plan."""

    return optimizer.PruningPlan(
        status="success",
        correlation_threshold=0.80,
        pair_count=4,
        reported_total_pair_count=66,
        high_pair_count=4,
        reported_high_pair_count=4,
        constant_features=(
            "history_size",
        ),
        group_count=1,
        candidate_count=1,
        feature_groups=(
            sample_group(),
        ),
        pruning_candidates=(
            optimizer.PruningCandidate(
                group_id=1,
                feature="rate_10",
                retained_features=(
                    "recency",
                    "recency_ratio",
                    "short_vs_long",
                ),
                reason="test candidate",
                correlation_link_count=3,
                cumulative_absolute_correlation=2.58,
                maximum_absolute_correlation=0.95,
            ),
        ),
    )


def candidate_spec(
    feature: str,
    *,
    links: int = 1,
    cumulative: float = 0.90,
    maximum: float = 0.90,
    source: str = "all_group_features",
) -> optimizer.EvaluationCandidateSpec:
    """Build one cumulative-removal candidate."""

    return optimizer.EvaluationCandidateSpec(
        feature=feature,
        group_ids=(1,),
        correlation_link_count=links,
        cumulative_absolute_correlation=(
            cumulative
        ),
        maximum_absolute_correlation=(
            maximum
        ),
        source=source,
    )


def sample_candidates() -> tuple[
    optimizer.EvaluationCandidateSpec,
    ...,
]:
    """Return the four candidates observed in the real report."""

    return (
        candidate_spec(
            "rate_10",
            links=3,
            cumulative=2.58,
            maximum=0.95,
        ),
        candidate_spec(
            "recency",
            links=2,
            cumulative=1.82,
            maximum=1.0,
        ),
        candidate_spec(
            "recency_ratio",
            links=2,
            cumulative=1.81,
            maximum=1.0,
        ),
        candidate_spec(
            "short_vs_long",
            links=1,
            cumulative=0.95,
            maximum=0.95,
        ),
    )


def sample_dataset_summary() -> (
    optimizer.TemporalDatasetSummary
):
    """Return representative chronological split metadata."""

    return optimizer.TemporalDatasetSummary(
        draw_count=2780,
        dataset_rows=73500,
        dataset_targets=1500,
        training_rows=68551,
        training_targets=1399,
        validation_rows=4900,
        validation_targets=100,
        purged_targets=1,
        first_training_target=1280,
        last_training_target=2678,
        purged_target_indices=(2679,),
        first_validation_target=2680,
        last_validation_target=2779,
    )


def make_run_result(
    *,
    experiment_name: str,
    feature_columns: tuple[str, ...],
    mean_hits: float,
    target_hit_rate: float,
    total_seconds: float = 1.0,
    validation_targets: int = 100,
    top_k: int = 5,
) -> FeatureAblationRunResult:
    """Build a deterministic runner result."""

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
            if feature
            not in feature_columns
        ),
        top_k=top_k,
        training_rows=100,
        validation_rows=50,
        training_targets=10,
        validation_targets=(
            validation_targets
        ),
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
        target_hit_rate=(
            target_hit_rate
        ),
        target_evaluations=(),
    )


def make_comparison(
    baseline: FeatureAblationRunResult,
    candidate: FeatureAblationRunResult,
    *,
    accepted: bool,
    tolerance: float = 0.0,
) -> FeatureAblationComparison:
    """Build one comparison with consistent metrics."""

    delta = (
        candidate.mean_hits_at_k
        - baseline.mean_hits_at_k
    )

    relative_delta = (
        delta
        / baseline.mean_hits_at_k
        if baseline.mean_hits_at_k
        else None
    )

    return FeatureAblationComparison(
        baseline_experiment=(
            baseline.experiment_name
        ),
        candidate_experiment=(
            candidate.experiment_name
        ),
        baseline_features=(
            baseline.feature_columns
        ),
        candidate_features=(
            candidate.feature_columns
        ),
        removed_features=tuple(
            feature
            for feature
            in baseline.feature_columns
            if feature
            not in candidate.feature_columns
        ),
        baseline_mean_hits_at_k=(
            baseline.mean_hits_at_k
        ),
        candidate_mean_hits_at_k=(
            candidate.mean_hits_at_k
        ),
        absolute_delta=delta,
        relative_delta=relative_delta,
        accepted=accepted,
        tolerance=tolerance,
    )


def make_candidate_evaluation(
    *,
    iteration: int,
    feature: str,
    baseline: FeatureAblationRunResult,
    run: FeatureAblationRunResult,
    accepted: bool,
    selected: bool = False,
    links: int = 1,
    maximum: float = 0.90,
) -> greedy.GreedyCandidateEvaluation:
    """Build one complete greedy candidate evaluation."""

    return greedy.GreedyCandidateEvaluation(
        iteration=iteration,
        feature=feature,
        group_ids=(1,),
        source="all_group_features",
        correlation_link_count=links,
        cumulative_absolute_correlation=(
            maximum
            * max(links, 1)
        ),
        maximum_absolute_correlation=maximum,
        active_features_before=(
            baseline.feature_columns
        ),
        candidate_features=(
            run.feature_columns
        ),
        decision=(
            "ACCEPT"
            if accepted
            else "REJECT"
        ),
        selected_for_removal=selected,
        run=run,
        comparison=make_comparison(
            baseline,
            run,
            accepted=accepted,
        ),
    )


def sample_report() -> greedy.GreedyPruningReport:
    """Build a report matching the successful real execution shape."""

    initial = make_run_result(
        experiment_name="greedy_baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.46,
        target_hit_rate=0.40,
        total_seconds=2.5,
    )

    after_short_features = tuple(
        feature
        for feature in MODEL_FEATURES
        if feature != "short_vs_long"
    )

    after_short = make_run_result(
        experiment_name=(
            "greedy_i1_without_short_vs_long"
        ),
        feature_columns=after_short_features,
        mean_hits=0.54,
        target_hit_rate=0.46,
        total_seconds=2.4,
    )

    after_rate_features = tuple(
        feature
        for feature in after_short_features
        if feature != "rate_10"
    )

    after_rate = make_run_result(
        experiment_name=(
            "greedy_i2_without_rate_10"
        ),
        feature_columns=after_rate_features,
        mean_hits=0.63,
        target_hit_rate=0.52,
        total_seconds=2.3,
    )

    rejected_recency = make_run_result(
        experiment_name=(
            "greedy_i3_without_recency"
        ),
        feature_columns=tuple(
            feature
            for feature in after_rate_features
            if feature != "recency"
        ),
        mean_hits=0.60,
        target_hit_rate=0.50,
        total_seconds=2.2,
    )

    iteration_1_evaluation = (
        make_candidate_evaluation(
            iteration=1,
            feature="short_vs_long",
            baseline=initial,
            run=after_short,
            accepted=True,
            selected=True,
            maximum=0.95,
        )
    )

    iteration_2_evaluation = (
        make_candidate_evaluation(
            iteration=2,
            feature="rate_10",
            baseline=after_short,
            run=after_rate,
            accepted=True,
            selected=True,
            links=3,
            maximum=0.95,
        )
    )

    iteration_3_evaluation = (
        make_candidate_evaluation(
            iteration=3,
            feature="recency",
            baseline=after_rate,
            run=rejected_recency,
            accepted=False,
            selected=False,
            links=2,
            maximum=1.0,
        )
    )

    return greedy.GreedyPruningReport(
        status="success",
        version=greedy.VERSION,
        protocol="test greedy protocol",
        stop_reason="no_acceptable_removal",
        correlation_threshold=0.80,
        candidate_scope=(
            "all_group_features"
        ),
        tolerance=0.0,
        window_size=100,
        max_training_targets=1500,
        validation_targets=100,
        top_k=5,
        purge_targets=1,
        maximum_iterations=10,
        minimum_features=1,
        dataset=sample_dataset_summary(),
        plan=sample_plan(),
        initial_candidates=(
            sample_candidates()
        ),
        initial_baseline=initial,
        iterations=(
            greedy.GreedyPruningIteration(
                iteration=1,
                baseline_experiment=(
                    initial.experiment_name
                ),
                baseline_features=(
                    initial.feature_columns
                ),
                baseline_mean_hits_at_k=0.46,
                baseline_target_hit_rate=0.40,
                evaluated_candidate_count=4,
                accepted_candidate_count=4,
                selected_feature=(
                    "short_vs_long"
                ),
                selected_delta=0.08,
                selected_mean_hits_at_k=0.54,
                selected_target_hit_rate=0.46,
                active_features_after=(
                    after_short_features
                ),
                candidate_evaluations=(
                    iteration_1_evaluation,
                ),
            ),
            greedy.GreedyPruningIteration(
                iteration=2,
                baseline_experiment=(
                    after_short.experiment_name
                ),
                baseline_features=(
                    after_short.feature_columns
                ),
                baseline_mean_hits_at_k=0.54,
                baseline_target_hit_rate=0.46,
                evaluated_candidate_count=3,
                accepted_candidate_count=1,
                selected_feature="rate_10",
                selected_delta=0.09,
                selected_mean_hits_at_k=0.63,
                selected_target_hit_rate=0.52,
                active_features_after=(
                    after_rate_features
                ),
                candidate_evaluations=(
                    iteration_2_evaluation,
                ),
            ),
            greedy.GreedyPruningIteration(
                iteration=3,
                baseline_experiment=(
                    after_rate.experiment_name
                ),
                baseline_features=(
                    after_rate.feature_columns
                ),
                baseline_mean_hits_at_k=0.63,
                baseline_target_hit_rate=0.52,
                evaluated_candidate_count=2,
                accepted_candidate_count=0,
                selected_feature=None,
                selected_delta=None,
                selected_mean_hits_at_k=None,
                selected_target_hit_rate=None,
                active_features_after=(
                    after_rate_features
                ),
                candidate_evaluations=(
                    iteration_3_evaluation,
                ),
            ),
        ),
        accepted_removal_sequence=(
            "short_vs_long",
            "rate_10",
        ),
        final_features=after_rate_features,
        final_feature_count=len(
            after_rate_features
        ),
        final_run=after_rate,
        initial_mean_hits_at_k=0.46,
        final_mean_hits_at_k=0.63,
        total_absolute_delta=0.17,
        total_relative_delta=(
            0.17 / 0.46
        ),
    )


def test_config_to_pruning_config_preserves_shared_fields(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        explicit_features=(
            "rate_10",
            "recency",
        ),
        tolerance=0.02,
    )

    pruning = (
        config.to_pruning_config()
    )

    assert isinstance(
        pruning,
        optimizer.PruningConfig,
    )
    assert (
        pruning.correlation_report
        == config.correlation_report
    )
    assert (
        pruning.output_directory
        == config.output_directory
    )
    assert (
        pruning.validation_targets
        == 5
    )
    assert pruning.tolerance == pytest.approx(
        0.02
    )
    assert pruning.explicit_features == (
        "rate_10",
        "recency",
    )
    assert pruning.plan_only is False


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
                "validation_targets": 4,
            },
            (
                "validation_targets must "
                "be at least 5"
            ),
        ),
        (
            {
                "maximum_iterations": 0,
            },
            (
                "maximum_iterations must "
                "be at least 1"
            ),
        ),
        (
            {
                "minimum_features": 0,
            },
            (
                "minimum_features must be "
                "between 1 and 5"
            ),
        ),
        (
            {
                "minimum_features": 6,
            },
            (
                "minimum_features must be "
                "between 1 and 5"
            ),
        ),
    ],
)
def test_validate_greedy_config_rejects_invalid_values(
    tmp_path: Path,
    overrides: dict[str, Any],
    message: str,
) -> None:
    config = make_config(
        tmp_path,
        **overrides,
    )

    with pytest.raises(
        greedy.GreedyConfigurationError,
        match=message,
    ):
        greedy.validate_greedy_config(
            config
        )


def test_validate_greedy_config_wraps_sprint2_error(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        top_k=50,
    )

    with pytest.raises(
        greedy.GreedyConfigurationError,
        match=(
            "top_k must be between "
            "1 and 49"
        ),
    ) as captured:
        greedy.validate_greedy_config(
            config
        )

    assert isinstance(
        captured.value.__cause__,
        optimizer.FeaturePruningError,
    )


def test_normalise_active_features_strips_and_preserves_order() -> None:
    result = (
        greedy
        ._normalise_active_features(
            (
                " rate_10 ",
                "recency",
                "other_feature",
            )
        )
    )

    assert result == (
        "rate_10",
        "recency",
        "other_feature",
    )


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (
            "rate_10",
            (
                "active features must be "
                "a sequence of names"
            ),
        ),
        (
            (
                "rate_10",
                7,
            ),
            (
                "active feature names must "
                "be strings"
            ),
        ),
        (
            (
                "rate_10",
                " ",
            ),
            (
                "active feature names cannot "
                "be empty"
            ),
        ),
        (
            (
                "rate_10",
                "rate_10",
            ),
            (
                "duplicate active feature: "
                "rate_10"
            ),
        ),
        (
            (),
            (
                "at least one active feature "
                "is required"
            ),
        ),
        (
            (
                "unknown_feature",
            ),
            (
                "unknown active V7 features"
            ),
        ),
    ],
)
def test_normalise_active_features_rejects_invalid_sequences(
    features: Any,
    message: str,
) -> None:
    with pytest.raises(
        greedy.GreedyConfigurationError,
        match=message,
    ):
        greedy._normalise_active_features(
            features
        )


def test_build_subset_config() -> None:
    config = (
        greedy
        .build_subset_config(
            experiment_name=(
                "greedy_i1_without_rate_10"
            ),
            feature_columns=(
                "recency",
                "recency_ratio",
            ),
            top_k=5,
        )
    )

    assert (
        config.experiment_name
        == "greedy_i1_without_rate_10"
    )
    assert config.feature_columns == (
        "recency",
        "recency_ratio",
    )
    assert config.top_k == 5


def test_candidate_order_key_prioritises_delta() -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    low_delta_run = make_run_result(
        experiment_name="low",
        feature_columns=MODEL_FEATURES[:-1],
        mean_hits=0.41,
        target_hit_rate=0.90,
    )

    high_delta_run = make_run_result(
        experiment_name="high",
        feature_columns=MODEL_FEATURES[:-1],
        mean_hits=0.50,
        target_hit_rate=0.10,
    )

    low = make_candidate_evaluation(
        iteration=1,
        feature="other_feature",
        baseline=baseline,
        run=low_delta_run,
        accepted=True,
    )

    high = make_candidate_evaluation(
        iteration=1,
        feature="short_vs_long",
        baseline=baseline,
        run=high_delta_run,
        accepted=True,
    )

    assert (
        greedy._candidate_order_key(
            high
        )
        > greedy._candidate_order_key(
            low
        )
    )


def test_candidate_order_key_uses_target_hit_rate_as_tie_breaker() -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    low_rate = make_run_result(
        experiment_name="low_rate",
        feature_columns=MODEL_FEATURES[:-1],
        mean_hits=0.50,
        target_hit_rate=0.40,
    )

    high_rate = make_run_result(
        experiment_name="high_rate",
        feature_columns=MODEL_FEATURES[:-1],
        mean_hits=0.50,
        target_hit_rate=0.60,
    )

    low = make_candidate_evaluation(
        iteration=1,
        feature="other_feature",
        baseline=baseline,
        run=low_rate,
        accepted=True,
    )

    high = make_candidate_evaluation(
        iteration=1,
        feature="short_vs_long",
        baseline=baseline,
        run=high_rate,
        accepted=True,
    )

    assert (
        greedy._candidate_order_key(
            high
        )
        > greedy._candidate_order_key(
            low
        )
    )


def test_candidate_order_key_prefers_faster_runtime_after_metric_ties() -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    slow_run = make_run_result(
        experiment_name="slow",
        feature_columns=MODEL_FEATURES[:-1],
        mean_hits=0.50,
        target_hit_rate=0.60,
        total_seconds=3.0,
    )

    fast_run = make_run_result(
        experiment_name="fast",
        feature_columns=MODEL_FEATURES[:-1],
        mean_hits=0.50,
        target_hit_rate=0.60,
        total_seconds=1.0,
    )

    slow = make_candidate_evaluation(
        iteration=1,
        feature="other_feature",
        baseline=baseline,
        run=slow_run,
        accepted=True,
    )

    fast = make_candidate_evaluation(
        iteration=1,
        feature="short_vs_long",
        baseline=baseline,
        run=fast_run,
        accepted=True,
    )

    assert (
        greedy._candidate_order_key(
            fast
        )
        > greedy._candidate_order_key(
            slow
        )
    )


def test_evaluate_iteration_candidates_selects_best_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.46,
        target_hit_rate=0.40,
    )

    outcomes = {
        "rate_10": (
            0.49,
            0.40,
            2.0,
        ),
        "short_vs_long": (
            0.54,
            0.46,
            2.4,
        ),
        "recency": (
            0.54,
            0.45,
            2.3,
        ),
    }

    def fake_run(
        *,
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        config: Any,
    ) -> FeatureAblationRunResult:
        del training_dataset
        del validation_dataset

        feature = next(
            candidate
            for candidate
            in outcomes
            if (
                config.experiment_name
                .endswith(candidate)
            )
        )

        mean_hits, hit_rate, seconds = (
            outcomes[feature]
        )

        return make_run_result(
            experiment_name=(
                config.experiment_name
            ),
            feature_columns=(
                config.feature_columns
            ),
            mean_hits=mean_hits,
            target_hit_rate=hit_rate,
            total_seconds=seconds,
        )

    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        fake_run,
    )

    evaluations, selected = (
        greedy
        ._evaluate_iteration_candidates(
            iteration=1,
            baseline=baseline,
            active_features=MODEL_FEATURES,
            remaining_candidates=(
                candidate_spec(
                    "rate_10",
                    links=3,
                    maximum=0.95,
                ),
                candidate_spec(
                    "short_vs_long",
                    links=1,
                    maximum=0.95,
                ),
                candidate_spec(
                    "recency",
                    links=2,
                    maximum=1.0,
                ),
            ),
            training_dataset=(
                pd.DataFrame()
            ),
            validation_dataset=(
                pd.DataFrame()
            ),
            config=make_config(
                tmp_path
            ),
        )
    )

    assert len(evaluations) == 3
    assert selected is not None
    assert (
        selected.feature
        == "short_vs_long"
    )
    assert selected.selected_for_removal is True
    assert (
        sum(
            evaluation.selected_for_removal
            for evaluation in evaluations
        )
        == 1
    )
    assert all(
        evaluation.decision == "ACCEPT"
        for evaluation in evaluations
    )
    assert (
        "short_vs_long"
        not in selected.candidate_features
    )


def test_evaluate_iteration_candidates_returns_none_when_all_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.63,
        target_hit_rate=0.50,
    )

    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        lambda **kwargs: make_run_result(
            experiment_name=(
                kwargs[
                    "config"
                ].experiment_name
            ),
            feature_columns=(
                kwargs[
                    "config"
                ].feature_columns
            ),
            mean_hits=0.60,
            target_hit_rate=0.48,
        ),
    )

    evaluations, selected = (
        greedy
        ._evaluate_iteration_candidates(
            iteration=3,
            baseline=baseline,
            active_features=MODEL_FEATURES,
            remaining_candidates=(
                candidate_spec(
                    "recency"
                ),
                candidate_spec(
                    "recency_ratio"
                ),
            ),
            training_dataset=(
                pd.DataFrame()
            ),
            validation_dataset=(
                pd.DataFrame()
            ),
            config=make_config(
                tmp_path
            ),
        )
    )

    assert len(evaluations) == 2
    assert selected is None
    assert all(
        evaluation.decision == "REJECT"
        for evaluation in evaluations
    )
    assert not any(
        evaluation.selected_for_removal
        for evaluation in evaluations
    )


def test_evaluate_iteration_candidates_skips_inactive_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=(
            "recency",
            "other_feature",
        ),
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    calls: list[str] = []

    def fake_run(
        **kwargs: Any,
    ) -> FeatureAblationRunResult:
        config = kwargs["config"]
        calls.append(
            config.experiment_name
        )

        return make_run_result(
            experiment_name=(
                config.experiment_name
            ),
            feature_columns=(
                config.feature_columns
            ),
            mean_hits=0.40,
            target_hit_rate=0.40,
        )

    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        fake_run,
    )

    evaluations, selected = (
        greedy
        ._evaluate_iteration_candidates(
            iteration=1,
            baseline=baseline,
            active_features=(
                "recency",
                "other_feature",
            ),
            remaining_candidates=(
                candidate_spec(
                    "rate_10"
                ),
                candidate_spec(
                    "recency"
                ),
            ),
            training_dataset=(
                pd.DataFrame()
            ),
            validation_dataset=(
                pd.DataFrame()
            ),
            config=make_config(
                tmp_path
            ),
        )
    )

    assert len(evaluations) == 1
    assert calls == [
        "greedy_i1_without_recency",
    ]
    assert selected is not None
    assert selected.feature == "recency"


def test_evaluate_iteration_candidates_respects_minimum_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        lambda **kwargs: (
            (_ for _ in ())
            .throw(
                AssertionError(
                    "model must not run"
                )
            )
        ),
    )

    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    evaluations, selected = (
        greedy
        ._evaluate_iteration_candidates(
            iteration=1,
            baseline=baseline,
            active_features=MODEL_FEATURES,
            remaining_candidates=(
                candidate_spec(
                    "rate_10"
                ),
            ),
            training_dataset=(
                pd.DataFrame()
            ),
            validation_dataset=(
                pd.DataFrame()
            ),
            config=make_config(
                tmp_path,
                minimum_features=5,
            ),
        )
    )

    assert evaluations == ()
    assert selected is None


def test_evaluate_iteration_candidates_wraps_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        lambda **kwargs: (
            (_ for _ in ())
            .throw(
                RuntimeError(
                    "fit failed"
                )
            )
        ),
    )

    with pytest.raises(
        greedy.GreedyEvaluationError,
        match=(
            "iteration 2 for feature "
            "rate_10"
        ),
    ) as captured:
        greedy._evaluate_iteration_candidates(
            iteration=2,
            baseline=baseline,
            active_features=MODEL_FEATURES,
            remaining_candidates=(
                candidate_spec(
                    "rate_10"
                ),
            ),
            training_dataset=(
                pd.DataFrame()
            ),
            validation_dataset=(
                pd.DataFrame()
            ),
            config=make_config(
                tmp_path
            ),
        )

    assert isinstance(
        captured.value.__cause__,
        RuntimeError,
    )


def install_common_run_patches(
    monkeypatch: pytest.MonkeyPatch,
    candidates: tuple[
        optimizer.EvaluationCandidateSpec,
        ...,
] | None = None,
) -> None:
    """Patch planning and temporal dataset construction."""

    monkeypatch.setattr(
        greedy,
        "build_pruning_plan",
        lambda config: (
            sample_plan(),
            (),
        ),
    )

    monkeypatch.setattr(
        greedy,
        "build_evaluation_candidates",
        lambda **kwargs: (
            sample_candidates()
            if candidates is None
            else candidates
        ),
    )

    monkeypatch.setattr(
        greedy,
        "prepare_temporal_datasets",
        lambda config: (
            pd.DataFrame(
                {
                    "target_draw_index": [
                        1,
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "target_draw_index": [
                        2,
                    ],
                }
            ),
            sample_dataset_summary(),
        ),
    )


def test_run_greedy_pruning_reproduces_cumulative_sequence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch
    )

    score_by_features = {
        MODEL_FEATURES: (
            0.46,
            0.40,
        ),
        tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "rate_10"
        ): (
            0.49,
            0.40,
        ),
        tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "recency"
        ): (
            0.54,
            0.45,
        ),
        tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "recency_ratio"
        ): (
            0.54,
            0.45,
        ),
        tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "short_vs_long"
        ): (
            0.54,
            0.46,
        ),
        (
            "recency",
            "recency_ratio",
            "other_feature",
        ): (
            0.63,
            0.52,
        ),
        (
            "rate_10",
            "recency_ratio",
            "other_feature",
        ): (
            0.50,
            0.43,
        ),
        (
            "rate_10",
            "recency",
            "other_feature",
        ): (
            0.51,
            0.44,
        ),
        (
            "recency_ratio",
            "other_feature",
        ): (
            0.60,
            0.49,
        ),
        (
            "recency",
            "other_feature",
        ): (
            0.61,
            0.50,
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
        del training_dataset
        del validation_dataset

        features = tuple(
            config.feature_columns
        )
        calls.append(
            (
                config.experiment_name,
                features,
            )
        )

        mean_hits, hit_rate = (
            score_by_features[
                features
            ]
        )

        return make_run_result(
            experiment_name=(
                config.experiment_name
            ),
            feature_columns=features,
            mean_hits=mean_hits,
            target_hit_rate=hit_rate,
        )

    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        fake_run,
    )

    report = (
        greedy
        .run_greedy_pruning(
            make_config(
                tmp_path
            )
        )
    )

    assert report.status == "success"
    assert (
        report.stop_reason
        == "no_acceptable_removal"
    )
    assert (
        report.accepted_removal_sequence
        == (
            "short_vs_long",
            "rate_10",
        )
    )
    assert report.final_features == (
        "recency",
        "recency_ratio",
        "other_feature",
    )
    assert report.final_feature_count == 3
    assert (
        report.initial_mean_hits_at_k
        == pytest.approx(0.46)
    )
    assert (
        report.final_mean_hits_at_k
        == pytest.approx(0.63)
    )
    assert (
        report.total_absolute_delta
        == pytest.approx(0.17)
    )
    assert (
        report.total_relative_delta
        == pytest.approx(
            0.17 / 0.46
        )
    )
    assert len(report.iterations) == 3
    assert (
        report.iterations[0]
        .selected_feature
        == "short_vs_long"
    )
    assert (
        report.iterations[1]
        .selected_feature
        == "rate_10"
    )
    assert (
        report.iterations[2]
        .selected_feature
        is None
    )
    assert (
        report.iterations[0]
        .evaluated_candidate_count
        == 4
    )
    assert (
        report.iterations[1]
        .evaluated_candidate_count
        == 3
    )
    assert (
        report.iterations[2]
        .evaluated_candidate_count
        == 2
    )
    assert len(calls) == 10
    assert calls[0][0] == "greedy_baseline"


def test_run_greedy_pruning_stops_when_no_candidates_remain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch,
        candidates=(
            candidate_spec(
                "rate_10"
            ),
        ),
    )

    def fake_run(
        **kwargs: Any,
    ) -> FeatureAblationRunResult:
        config = kwargs["config"]
        features = tuple(
            config.feature_columns
        )

        return make_run_result(
            experiment_name=(
                config.experiment_name
            ),
            feature_columns=features,
            mean_hits=(
                0.50
                if "rate_10"
                not in features
                else 0.40
            ),
            target_hit_rate=0.50,
        )

    monkeypatch.setattr(
        greedy,
        "run_feature_subset",
        fake_run,
    )

    report = (
        greedy
        .run_greedy_pruning(
            make_config(
                tmp_path
            )
        )
    )

    assert (
        report.stop_reason
        == "no_candidates_remaining"
    )
    assert (
        report.accepted_removal_sequence
        == (
            "rate_10",
        )
    )
    assert len(report.iterations) == 1


def test_run_greedy_pruning_stops_at_minimum_feature_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch,
        candidates=(
            candidate_spec(
                "rate_10"
            ),
        ),
    )

    monkeypatch.setattr(
        greedy,
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
            mean_hits=0.50,
            target_hit_rate=0.50,
        ),
    )

    report = (
        greedy
        .run_greedy_pruning(
            make_config(
                tmp_path,
                minimum_features=4,
            )
        )
    )

    assert (
        report.stop_reason
        == "minimum_feature_count_reached"
    )
    assert report.final_feature_count == 4
    assert (
        report.accepted_removal_sequence
        == (
            "rate_10",
        )
    )


def test_run_greedy_pruning_stops_at_iteration_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch
    )

    monkeypatch.setattr(
        greedy,
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
            mean_hits=(
                0.50
                if (
                    kwargs[
                        "config"
                    ].experiment_name
                    != "greedy_baseline"
                )
                else 0.40
            ),
            target_hit_rate=0.50,
        ),
    )

    report = (
        greedy
        .run_greedy_pruning(
            make_config(
                tmp_path,
                maximum_iterations=1,
            )
        )
    )

    assert (
        report.stop_reason
        == "maximum_iterations_reached"
    )
    assert len(report.iterations) == 1
    assert len(
        report.accepted_removal_sequence
    ) == 1


def test_run_greedy_pruning_reports_explicit_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch,
        candidates=(
            candidate_spec(
                "rate_10",
                source="explicit",
            ),
        ),
    )

    monkeypatch.setattr(
        greedy,
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
            mean_hits=0.40,
            target_hit_rate=0.40,
        ),
    )

    report = (
        greedy
        .run_greedy_pruning(
            make_config(
                tmp_path,
                explicit_features=(
                    "rate_10",
                ),
            )
        )
    )

    assert (
        report.candidate_scope
        == "explicit"
    )


def test_run_greedy_pruning_handles_zero_initial_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch,
        candidates=(
            candidate_spec(
                "rate_10"
            ),
        ),
    )

    monkeypatch.setattr(
        greedy,
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

    report = (
        greedy
        .run_greedy_pruning(
            make_config(
                tmp_path
            )
        )
    )

    assert (
        report.total_absolute_delta
        == pytest.approx(0.0)
    )
    assert (
        report.total_relative_delta
        is None
    )


def test_run_greedy_pruning_rejects_empty_initial_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch,
        candidates=(),
    )

    with pytest.raises(
        greedy.GreedyEvaluationError,
        match=(
            "No feature was selected "
            "for cumulative evaluation"
        ),
    ):
        greedy.run_greedy_pruning(
            make_config(
                tmp_path
            )
        )


def test_run_greedy_pruning_wraps_planning_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        greedy,
        "build_pruning_plan",
        lambda config: (
            (_ for _ in ())
            .throw(
                RuntimeError(
                    "planning failed"
                )
            )
        ),
    )

    with pytest.raises(
        greedy.GreedyEvaluationError,
        match=(
            "Unable to build the initial "
            "pruning plan"
        ),
    ) as captured:
        greedy.run_greedy_pruning(
            make_config(
                tmp_path
            )
        )

    assert isinstance(
        captured.value.__cause__,
        RuntimeError,
    )


def test_run_greedy_pruning_wraps_dataset_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        greedy,
        "build_pruning_plan",
        lambda config: (
            sample_plan(),
            (),
        ),
    )

    monkeypatch.setattr(
        greedy,
        "build_evaluation_candidates",
        lambda **kwargs: (
            sample_candidates()
        ),
    )

    monkeypatch.setattr(
        greedy,
        "prepare_temporal_datasets",
        lambda config: (
            (_ for _ in ())
            .throw(
                RuntimeError(
                    "dataset failed"
                )
            )
        ),
    )

    with pytest.raises(
        greedy.GreedyEvaluationError,
        match=(
            "Unable to prepare cumulative "
            "temporal datasets"
        ),
    ) as captured:
        greedy.run_greedy_pruning(
            make_config(
                tmp_path
            )
        )

    assert isinstance(
        captured.value.__cause__,
        RuntimeError,
    )


def test_run_greedy_pruning_wraps_baseline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_common_run_patches(
        monkeypatch
    )

    monkeypatch.setattr(
        greedy,
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
        greedy.GreedyEvaluationError,
        match=(
            "Unable to evaluate the "
            "greedy baseline"
        ),
    ) as captured:
        greedy.run_greedy_pruning(
            make_config(
                tmp_path
            )
        )

    assert isinstance(
        captured.value.__cause__,
        RuntimeError,
    )


def test_run_greedy_pruning_propagates_feature_pruning_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = optimizer.CorrelationReportError(
        "invalid report"
    )

    monkeypatch.setattr(
        greedy,
        "build_pruning_plan",
        lambda config: (
            (_ for _ in ())
            .throw(error)
        ),
    )

    with pytest.raises(
        optimizer.CorrelationReportError,
        match="invalid report",
    ):
        greedy.run_greedy_pruning(
            make_config(
                tmp_path
            )
        )


def test_json_safe_converts_nested_values() -> None:
    payload = greedy._json_safe(
        {
            "path": Path("/tmp/report"),
            "tuple": (
                1,
                Path("/tmp/a"),
            ),
            "list": [
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
        "list": [
            None,
            None,
        ],
    }


def test_write_json(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "report.json"
    )

    resolved = greedy._write_json(
        {
            "value": float("nan"),
            "path": Path("/tmp/x"),
        },
        path,
    )

    assert resolved == path.resolve()

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert payload == {
        "path": "/tmp/x",
        "value": None,
    }


def test_write_csv(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "report.csv"
    )

    resolved = greedy._write_csv(
        rows=(
            {
                "feature": "rate_10",
                "delta": 0.09,
            },
        ),
        path=path,
        fieldnames=(
            "feature",
            "delta",
        ),
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
            "delta": "0.09",
        }
    ]


def test_export_greedy_report(
    tmp_path: Path,
) -> None:
    report = sample_report()

    files = (
        greedy
        .export_greedy_report(
            report=report,
            output_directory=(
                tmp_path
                / "greedy"
            ),
        )
    )

    assert set(files) == {
        "json",
        "text",
        "iterations_csv",
        "candidates_csv",
        "final_features",
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

    final_features = (
        files["final_features"]
        .read_text(
            encoding="utf-8"
        )
        .splitlines()
    )

    with files[
        "iterations_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        iteration_rows = list(
            csv.DictReader(file)
        )

    with files[
        "candidates_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        candidate_rows = list(
            csv.DictReader(file)
        )

    assert (
        payload[
            "accepted_removal_sequence"
        ]
        == [
            "short_vs_long",
            "rate_10",
        ]
    )
    assert payload["stop_reason"] == (
        "no_acceptable_removal"
    )
    assert (
        payload["final_mean_hits_at_k"]
        == pytest.approx(0.63)
    )
    assert len(iteration_rows) == 3
    assert (
        iteration_rows[0][
            "selected_feature"
        ]
        == "short_vs_long"
    )
    assert len(candidate_rows) == 3
    assert (
        candidate_rows[0][
            "selected_for_removal"
        ]
        == "True"
    )
    assert (
        "PREDIXA AI V7 GREEDY "
        "CUMULATIVE FEATURE PRUNING"
        in text
    )
    assert (
        "short_vs_long -> rate_10"
        in text
    )
    assert final_features == list(
        report.final_features
    )


def test_export_greedy_report_handles_no_iterations(
    tmp_path: Path,
) -> None:
    report = sample_report()

    empty_report = (
        greedy.GreedyPruningReport(
            status=report.status,
            version=report.version,
            protocol=report.protocol,
            stop_reason=(
                "minimum_feature_count_reached"
            ),
            correlation_threshold=(
                report.correlation_threshold
            ),
            candidate_scope=(
                report.candidate_scope
            ),
            tolerance=report.tolerance,
            window_size=report.window_size,
            max_training_targets=(
                report.max_training_targets
            ),
            validation_targets=(
                report.validation_targets
            ),
            top_k=report.top_k,
            purge_targets=(
                report.purge_targets
            ),
            maximum_iterations=(
                report.maximum_iterations
            ),
            minimum_features=(
                report.minimum_features
            ),
            dataset=report.dataset,
            plan=report.plan,
            initial_candidates=(
                report.initial_candidates
            ),
            initial_baseline=(
                report.initial_baseline
            ),
            iterations=(),
            accepted_removal_sequence=(),
            final_features=(
                report.initial_baseline
                .feature_columns
            ),
            final_feature_count=(
                report.initial_baseline
                .feature_count
            ),
            final_run=(
                report.initial_baseline
            ),
            initial_mean_hits_at_k=(
                report.initial_mean_hits_at_k
            ),
            final_mean_hits_at_k=(
                report.initial_mean_hits_at_k
            ),
            total_absolute_delta=0.0,
            total_relative_delta=0.0,
        )
    )

    files = (
        greedy
        .export_greedy_report(
            report=empty_report,
            output_directory=(
                tmp_path
                / "empty"
            ),
        )
    )

    text = (
        files["text"]
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        "No greedy iteration was executed."
        in text
    )

    with files[
        "iterations_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert rows == []


def test_print_greedy_report(
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    report = sample_report()

    greedy.print_greedy_report(
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
        "PREDIXA AI V7 GREEDY "
        "CUMULATIVE FEATURE PRUNING"
        in output
    )
    assert (
        "short_vs_long -> rate_10"
        in output
    )
    assert "0.630000" in output
    assert "SUCCESS" in output


def test_build_argument_parser_defaults() -> None:
    arguments = (
        greedy
        .build_argument_parser()
        .parse_args([])
    )

    assert (
        arguments.correlation_threshold
        == pytest.approx(0.80)
    )
    assert (
        arguments.candidate_scope
        == "all_group_features"
    )
    assert (
        arguments.maximum_iterations
        == 10
    )
    assert arguments.minimum_features == 1
    assert arguments.features == ()


def test_build_argument_parser_accepts_overrides() -> None:
    arguments = (
        greedy
        .build_argument_parser()
        .parse_args(
            [
                "--features",
                "rate_10",
                "recency",
                "--maximum-iterations",
                "3",
                "--minimum-features",
                "4",
                "--tolerance",
                "0.02",
            ]
        )
    )

    assert arguments.features == [
        "rate_10",
        "recency",
    ]
    assert (
        arguments.maximum_iterations
        == 3
    )
    assert arguments.minimum_features == 4
    assert arguments.tolerance == pytest.approx(
        0.02
    )


def test_main_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = (
        tmp_path
        / "reports"
    )
    report = sample_report()

    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_greedy",
            "--correlation-report",
            str(
                tmp_path
                / "correlation.json"
            ),
            "--output-directory",
            str(output_directory),
            "--validation-targets",
            "5",
            "--maximum-iterations",
            "3",
        ],
    )

    captured: dict[
        str,
        Any,
    ] = {}

    monkeypatch.setattr(
        greedy,
        "run_greedy_pruning",
        lambda config: (
            captured.setdefault(
                "config",
                config,
            )
            and report
        ),
    )

    monkeypatch.setattr(
        greedy,
        "export_greedy_report",
        lambda **kwargs: {
            "json": (
                output_directory
                / "report.json"
            ),
        },
    )

    monkeypatch.setattr(
        greedy,
        "print_greedy_report",
        lambda **kwargs: captured.update(
            {
                "printed": kwargs,
            }
        ),
    )

    assert greedy.main() == 0
    assert (
        captured[
            "config"
        ].maximum_iterations
        == 3
    )
    assert (
        captured[
            "printed"
        ]["report"]
        is report
    )


def test_main_returns_one_on_greedy_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_greedy",
            "--correlation-report",
            str(
                tmp_path
                / "missing.json"
            ),
            "--validation-targets",
            "5",
        ],
    )

    error = (
        greedy
        .GreedyEvaluationError(
            "greedy failed"
        )
    )
    error.__cause__ = RuntimeError(
        "underlying failure"
    )

    monkeypatch.setattr(
        greedy,
        "run_greedy_pruning",
        lambda config: (
            (_ for _ in ())
            .throw(error)
        ),
    )

    assert greedy.main() == 1

    output = (
        capsys.readouterr().out
    )

    assert (
        "PREDIXA AI V7 GREEDY "
        "FEATURE PRUNING"
        in output
    )
    assert "ERROR: greedy failed" in output
    assert (
        "CAUSE: RuntimeError: "
        "underlying failure"
        in output
    )
