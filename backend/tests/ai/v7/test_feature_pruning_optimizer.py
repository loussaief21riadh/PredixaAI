from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

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
    """Keep all tests independent from the production feature list."""

    monkeypatch.setattr(
        optimizer.V7RankingDataset,
        "feature_columns",
        classmethod(
            lambda cls: MODEL_FEATURES
        ),
    )


def make_config(
    tmp_path: Path,
    **overrides: Any,
) -> optimizer.PruningConfig:
    """Build a valid configuration with temporary paths."""

    values: dict[str, Any] = {
        "correlation_report": (
            tmp_path
            / "feature_correlation_report.json"
        ),
        "output_directory": (
            tmp_path
            / "feature_pruning"
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
        "plan_only": False,
    }

    values.update(overrides)

    return optimizer.PruningConfig(
        **values
    )


def make_pair(
    feature_a: str,
    feature_b: str,
    correlation: float,
    *,
    pearson: float | None = None,
    spearman: float | None = None,
) -> optimizer.CorrelationPair:
    """Build one correlation pair."""

    return optimizer.CorrelationPair(
        feature_a=feature_a,
        feature_b=feature_b,
        pearson=(
            correlation
            if pearson is None
            else pearson
        ),
        spearman=(
            correlation
            if spearman is None
            else spearman
        ),
        maximum_absolute_correlation=abs(
            correlation
        ),
        high_correlation=(
            abs(correlation) >= 0.80
        ),
    )


def sample_pairs() -> tuple[
    optimizer.CorrelationPair,
    ...,
]:
    """Return the same connected pattern observed in PredixaAI."""

    return (
        make_pair(
            "recency",
            "recency_ratio",
            1.0,
        ),
        make_pair(
            "rate_10",
            "short_vs_long",
            0.95,
        ),
        make_pair(
            "rate_10",
            "recency",
            0.82,
        ),
        make_pair(
            "rate_10",
            "recency_ratio",
            0.81,
        ),
        make_pair(
            "recency",
            "short_vs_long",
            0.74,
        ),
    )


def sample_group() -> optimizer.FeatureGroup:
    """Return one connected high-correlation feature group."""

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


def sample_candidate() -> optimizer.PruningCandidate:
    """Return the deterministic Sprint 1 candidate."""

    return optimizer.PruningCandidate(
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
    )


def sample_plan() -> optimizer.PruningPlan:
    """Build a complete in-memory pruning plan."""

    return optimizer.PruningPlan(
        status="success",
        correlation_threshold=0.80,
        pair_count=5,
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
            sample_candidate(),
        ),
    )


def make_dataset(
    target_indices: list[int],
    *,
    rows_per_target: int = 2,
) -> pd.DataFrame:
    """Build a small dataset sufficient for temporal split tests."""

    rows: list[
        dict[str, Any]
    ] = []

    for target_index in target_indices:
        for candidate in range(
            1,
            rows_per_target + 1,
        ):
            rows.append(
                {
                    "target_draw_index": (
                        target_index
                    ),
                    "candidate_number": (
                        candidate
                    ),
                    "target": int(
                        candidate == 1
                    ),
                    "target_draw_date": (
                        f"2025-01-"
                        f"{target_index:02d}"
                    ),
                }
            )

    return pd.DataFrame(rows)


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
    """Build one runner result used by optimizer tests."""

    total_hits = int(
        round(
            mean_hits
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
        targets_with_at_least_one_hit=int(
            round(
                target_hit_rate
                * validation_targets
            )
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
    """Build one baseline/candidate comparison."""

    delta = (
        candidate.mean_hits_at_k
        - baseline.mean_hits_at_k
    )

    relative = (
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
        relative_delta=relative,
        accepted=accepted,
        tolerance=tolerance,
    )


def sample_dataset_summary() -> (
    optimizer.TemporalDatasetSummary
):
    """Return representative temporal split metadata."""

    return optimizer.TemporalDatasetSummary(
        draw_count=100,
        dataset_rows=500,
        dataset_targets=10,
        training_rows=200,
        training_targets=4,
        validation_rows=250,
        validation_targets=5,
        purged_targets=1,
        first_training_target=1,
        last_training_target=4,
        purged_target_indices=(5,),
        first_validation_target=6,
        last_validation_target=10,
    )


def sample_evaluation_report() -> (
    optimizer.PruningEvaluationReport
):
    """Build a complete report with one accept and one reject."""

    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
        total_seconds=2.0,
    )

    accepted_run = make_run_result(
        experiment_name=(
            "without_short_vs_long"
        ),
        feature_columns=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "short_vs_long"
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=1.8,
    )

    rejected_run = make_run_result(
        experiment_name=(
            "without_rate_10"
        ),
        feature_columns=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "rate_10"
        ),
        mean_hits=0.30,
        target_hit_rate=0.30,
        total_seconds=1.7,
    )

    accepted_comparison = (
        make_comparison(
            baseline,
            accepted_run,
            accepted=True,
        )
    )

    rejected_comparison = (
        make_comparison(
            baseline,
            rejected_run,
            accepted=False,
        )
    )

    return optimizer.PruningEvaluationReport(
        status="success",
        version=optimizer.VERSION,
        protocol="test protocol",
        correlation_threshold=0.80,
        candidate_scope=(
            "all_group_features"
        ),
        tolerance=0.0,
        window_size=100,
        max_training_targets=1500,
        validation_targets=5,
        top_k=5,
        purge_targets=1,
        dataset=sample_dataset_summary(),
        plan=sample_plan(),
        baseline=baseline,
        candidate_evaluations=(
            optimizer.CandidateEvaluation(
                feature="short_vs_long",
                group_ids=(1,),
                correlation_link_count=1,
                cumulative_absolute_correlation=0.95,
                maximum_absolute_correlation=0.95,
                source=(
                    "all_group_features"
                ),
                decision="ACCEPT",
                run=accepted_run,
                comparison=(
                    accepted_comparison
                ),
            ),
            optimizer.CandidateEvaluation(
                feature="rate_10",
                group_ids=(1,),
                correlation_link_count=3,
                cumulative_absolute_correlation=2.58,
                maximum_absolute_correlation=0.95,
                source=(
                    "all_group_features"
                ),
                decision="REJECT",
                run=rejected_run,
                comparison=(
                    rejected_comparison
                ),
            ),
        ),
        accepted_features=(
            "short_vs_long",
        ),
        rejected_features=(
            "rate_10",
        ),
        best_single_removal=(
            "short_vs_long"
        ),
        best_single_removal_delta=0.10,
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "correlation_report": (
                    "not-a-path"
                ),
            },
            (
                "correlation_report must "
                "be a pathlib.Path"
            ),
        ),
        (
            {
                "output_directory": (
                    "not-a-path"
                ),
            },
            (
                "output_directory must "
                "be a pathlib.Path"
            ),
        ),
        (
            {
                "correlation_threshold": (
                    float("nan")
                ),
            },
            (
                "correlation_threshold "
                "must be finite"
            ),
        ),
        (
            {
                "correlation_threshold": 0.0,
            },
            (
                "correlation_threshold must "
                "be greater than 0"
            ),
        ),
        (
            {
                "correlation_threshold": 1.1,
            },
            (
                "correlation_threshold must "
                "be greater than 0"
            ),
        ),
        (
            {
                "minimum_group_size": 1,
            },
            (
                "minimum_group_size must "
                "be at least 2"
            ),
        ),
        (
            {
                "maximum_candidates_per_group": 0,
            },
            (
                "maximum_candidates_per_group "
                "must be at least 1"
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
                "tolerance": (
                    float("inf")
                ),
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
                "candidate_scope": "invalid",
            },
            (
                "candidate_scope must "
                "be one of"
            ),
        ),
        (
            {
                "explicit_features": (
                    7,
                ),
            },
            (
                "explicit feature names "
                "must be strings"
            ),
        ),
        (
            {
                "explicit_features": (
                    " ",
                ),
            },
            (
                "explicit feature names "
                "cannot be empty"
            ),
        ),
        (
            {
                "explicit_features": (
                    "rate_10",
                    "rate_10",
                ),
            },
            (
                "duplicate explicit feature"
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
        optimizer.ConfigurationError,
        match=message,
    ):
        optimizer.validate_config(
            config
        )


def test_validate_config_accepts_valid_values(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path
    )

    assert config.validated() is config


def test_load_correlation_report_accepts_valid_json(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "report.json"
    )

    path.write_text(
        json.dumps(
            {
                "status": "success",
                "top_pairs": [],
            }
        ),
        encoding="utf-8",
    )

    report = (
        optimizer
        .load_correlation_report(
            path
        )
    )

    assert report["status"] == "success"


def test_load_correlation_report_rejects_missing_file(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        optimizer.CorrelationReportError,
        match="does not exist",
    ):
        optimizer.load_correlation_report(
            tmp_path
            / "missing.json"
        )


def test_load_correlation_report_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        optimizer.CorrelationReportError,
        match="is not a file",
    ):
        optimizer.load_correlation_report(
            tmp_path
        )


def test_load_correlation_report_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "invalid.json"
    )
    path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        optimizer.CorrelationReportError,
        match="Invalid JSON",
    ):
        optimizer.load_correlation_report(
            path
        )


def test_load_correlation_report_rejects_non_object(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "list.json"
    )
    path.write_text(
        "[]",
        encoding="utf-8",
    )

    with pytest.raises(
        optimizer.CorrelationReportError,
        match="must be a JSON object",
    ):
        optimizer.load_correlation_report(
            path
        )


def test_candidate_pair_collections_supports_direct_and_nested() -> None:
    direct = [
        {
            "feature_a": "a",
            "feature_b": "b",
            "pearson": 0.9,
        }
    ]
    nested = [
        {
            "feature_a": "c",
            "feature_b": "d",
            "pearson": 0.8,
        }
    ]

    collections = list(
        optimizer
        ._candidate_pair_collections(
            {
                "top_pairs": direct,
                "results": {
                    "pairs": nested,
                },
            }
        )
    )

    assert collections == [
        direct,
        nested,
    ]


def test_parse_pair_uses_max_absolute_correlation() -> None:
    pair = optimizer._parse_pair(
        {
            "feature_a": " a ",
            "feature_b": "b",
            "pearson": -0.60,
            "spearman": -0.85,
            "max_absolute_correlation": (
                0.85
            ),
            "high_correlation": True,
        },
        threshold=0.80,
    )

    assert pair.feature_a == "a"
    assert pair.feature_b == "b"
    assert pair.pearson == pytest.approx(
        -0.60
    )
    assert pair.spearman == pytest.approx(
        -0.85
    )
    assert (
        pair.maximum_absolute_correlation
        == pytest.approx(0.85)
    )
    assert pair.high_correlation is True


def test_parse_pair_derives_maximum_and_high_flag() -> None:
    pair = optimizer._parse_pair(
        {
            "feature_1": "a",
            "feature_2": "b",
            "pearson": -0.70,
            "spearman": -0.90,
        },
        threshold=0.80,
    )

    assert (
        pair.maximum_absolute_correlation
        == pytest.approx(0.90)
    )
    assert pair.high_correlation is True


@pytest.mark.parametrize(
    ("item", "message"),
    [
        (
            {
                "feature_a": "a",
                "feature_b": "a",
                "pearson": 0.9,
            },
            "same feature twice",
        ),
        (
            {
                "feature_a": "",
                "feature_b": "b",
                "pearson": 0.9,
            },
            "feature_a cannot be empty",
        ),
        (
            {
                "feature_a": "a",
                "feature_b": "b",
            },
            "No correlation value found",
        ),
        (
            {
                "feature_a": "a",
                "feature_b": "b",
                "pearson": 0.9,
                "high_correlation": "yes",
            },
            (
                "high_correlation must "
                "be a boolean"
            ),
        ),
    ],
)
def test_parse_pair_rejects_invalid_records(
    item: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(
        optimizer.CorrelationReportError,
        match=message,
    ):
        optimizer._parse_pair(
            item,
            threshold=0.80,
        )


def test_extract_correlation_pairs_deduplicates_and_sorts() -> None:
    report = {
        "top_pairs": [
            {
                "feature_a": "a",
                "feature_b": "b",
                "pearson": 0.85,
            },
            {
                "feature_a": "b",
                "feature_b": "a",
                "pearson": 0.95,
            },
            {
                "feature_a": "c",
                "feature_b": "d",
                "pearson": 0.90,
            },
        ]
    }

    pairs = (
        optimizer
        .extract_correlation_pairs(
            report,
            threshold=0.80,
        )
    )

    assert len(pairs) == 2
    assert (
        pairs[0]
        .maximum_absolute_correlation
        == pytest.approx(0.95)
    )
    assert {
        pairs[0].feature_a,
        pairs[0].feature_b,
    } == {
        "a",
        "b",
    }


def test_extract_correlation_pairs_rejects_missing_collection() -> None:
    with pytest.raises(
        optimizer.CorrelationReportError,
        match=(
            "No feature-pair collection"
        ),
    ):
        optimizer.extract_correlation_pairs(
            {},
            threshold=0.80,
        )


def test_build_feature_groups_builds_connected_component() -> None:
    groups = (
        optimizer
        .build_feature_groups(
            pairs=sample_pairs(),
            threshold=0.80,
            minimum_group_size=2,
        )
    )

    assert len(groups) == 1
    assert groups[0].features == (
        "rate_10",
        "recency",
        "recency_ratio",
        "short_vs_long",
    )
    assert groups[0].pair_count == 4
    assert (
        groups[0]
        .maximum_absolute_correlation
        == pytest.approx(1.0)
    )


def test_build_feature_groups_ignores_low_pairs() -> None:
    groups = (
        optimizer
        .build_feature_groups(
            pairs=(
                make_pair(
                    "a",
                    "b",
                    0.70,
                ),
            ),
            threshold=0.80,
        )
    )

    assert groups == ()


def test_feature_redundancy_scores() -> None:
    scores = (
        optimizer
        ._feature_redundancy_scores(
            sample_group(),
            tuple(
                pair
                for pair
                in sample_pairs()
                if (
                    pair
                    .maximum_absolute_correlation
                    >= 0.80
                )
            ),
        )
    )

    assert scores["rate_10"][0] == 3
    assert scores["recency"][0] == 2
    assert (
        scores["recency_ratio"][2]
        == pytest.approx(1.0)
    )


def test_select_pruning_candidates_prefers_most_connected_feature() -> None:
    high_pairs = tuple(
        pair
        for pair in sample_pairs()
        if (
            pair
            .maximum_absolute_correlation
            >= 0.80
        )
    )

    candidates = (
        optimizer
        .select_pruning_candidates(
            groups=(
                sample_group(),
            ),
            pairs=high_pairs,
            maximum_candidates_per_group=1,
        )
    )

    assert len(candidates) == 1
    assert (
        candidates[0].feature
        == "rate_10"
    )
    assert (
        candidates[0]
        .correlation_link_count
        == 3
    )


def test_select_pruning_candidates_can_select_multiple() -> None:
    high_pairs = tuple(
        pair
        for pair in sample_pairs()
        if (
            pair
            .maximum_absolute_correlation
            >= 0.80
        )
    )

    candidates = (
        optimizer
        .select_pruning_candidates(
            groups=(
                sample_group(),
            ),
            pairs=high_pairs,
            maximum_candidates_per_group=2,
        )
    )

    assert len(candidates) == 2
    assert candidates[0].feature == "rate_10"


def test_build_pruning_plan_reads_real_report_shape(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path
    )

    report = {
        "status": "success",
        "total_feature_pairs": 66,
        "high_correlation_pair_count": 4,
        "constant_features": [
            "history_size",
        ],
        "top_pairs": [
            {
                "feature_a": (
                    pair.feature_a
                ),
                "feature_b": (
                    pair.feature_b
                ),
                "pearson": pair.pearson,
                "spearman": pair.spearman,
                "max_absolute_correlation": (
                    pair
                    .maximum_absolute_correlation
                ),
                "high_correlation": (
                    pair.high_correlation
                ),
            }
            for pair in sample_pairs()
        ],
    }

    config.correlation_report.write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    plan, pairs = (
        optimizer
        .build_pruning_plan(
            config
        )
    )

    assert plan.status == "success"
    assert plan.pair_count == 5
    assert (
        plan.reported_total_pair_count
        == 66
    )
    assert plan.high_pair_count == 4
    assert (
        plan.reported_high_pair_count
        == 4
    )
    assert plan.constant_features == (
        "history_size",
    )
    assert plan.group_count == 1
    assert plan.candidate_count == 1
    assert (
        plan.pruning_candidates[0].feature
        == "rate_10"
    )
    assert len(pairs) == 5


def test_build_pruning_plan_rejects_invalid_constant_features(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path
    )

    config.correlation_report.write_text(
        json.dumps(
            {
                "constant_features": (
                    "history_size"
                ),
                "top_pairs": [
                    {
                        "feature_a": "a",
                        "feature_b": "b",
                        "pearson": 0.9,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        optimizer.CorrelationReportError,
        match=(
            "constant_features must "
            "be a list"
        ),
    ):
        optimizer.build_pruning_plan(
            config
        )


def test_normalise_explicit_features_preserves_order_and_deduplicates() -> None:
    result = (
        optimizer
        ._normalise_explicit_features(
            (
                " rate_10 ",
                "recency",
                "rate_10",
            )
        )
    )

    assert result == (
        "rate_10",
        "recency",
    )


def test_build_evaluation_candidates_explicit() -> None:
    config = optimizer.PruningConfig(
        explicit_features=(
            "short_vs_long",
            "other_feature",
        ),
        validation_targets=5,
    )

    specs = (
        optimizer
        .build_evaluation_candidates(
            config=config,
            plan=sample_plan(),
            pairs=sample_pairs(),
        )
    )

    assert [
        spec.feature
        for spec in specs
    ] == [
        "short_vs_long",
        "other_feature",
    ]

    assert specs[0].group_ids == (1,)
    assert (
        specs[0]
        .correlation_link_count
        == 1
    )
    assert specs[0].source == "explicit"
    assert specs[1].group_ids == ()
    assert (
        specs[1]
        .maximum_absolute_correlation
        == pytest.approx(0.0)
    )


def test_build_evaluation_candidates_rejects_unknown_explicit_feature() -> None:
    config = optimizer.PruningConfig(
        explicit_features=(
            "unknown_feature",
        ),
        validation_targets=5,
    )

    with pytest.raises(
        optimizer.ConfigurationError,
        match="Unknown explicit V7 features",
    ):
        optimizer.build_evaluation_candidates(
            config=config,
            plan=sample_plan(),
            pairs=sample_pairs(),
        )


def test_build_evaluation_candidates_selected_scope() -> None:
    config = optimizer.PruningConfig(
        candidate_scope="selected",
        validation_targets=5,
    )

    specs = (
        optimizer
        .build_evaluation_candidates(
            config=config,
            plan=sample_plan(),
            pairs=sample_pairs(),
        )
    )

    assert len(specs) == 1
    assert specs[0].feature == "rate_10"
    assert specs[0].source == "selected"


def test_build_evaluation_candidates_all_group_features() -> None:
    config = optimizer.PruningConfig(
        candidate_scope=(
            "all_group_features"
        ),
        validation_targets=5,
    )

    specs = (
        optimizer
        .build_evaluation_candidates(
            config=config,
            plan=sample_plan(),
            pairs=sample_pairs(),
        )
    )

    assert [
        spec.feature
        for spec in specs
    ] == [
        "rate_10",
        "recency",
        "recency_ratio",
        "short_vs_long",
    ]

    assert specs[0].correlation_link_count == 3
    assert all(
        spec.source
        == "all_group_features"
        for spec in specs
    )


def test_build_evaluation_candidates_rejects_unknown_report_feature() -> None:
    plan = replace(
        sample_plan(),
        feature_groups=(
            optimizer.FeatureGroup(
                group_id=1,
                features=(
                    "rate_10",
                    "unknown_feature",
                ),
                pair_count=1,
                maximum_absolute_correlation=0.9,
            ),
        ),
    )

    config = optimizer.PruningConfig(
        candidate_scope=(
            "all_group_features"
        ),
        validation_targets=5,
    )

    with pytest.raises(
        optimizer.ConfigurationError,
        match=(
            "Correlation report references "
            "an unknown V7 feature"
        ),
    ):
        optimizer.build_evaluation_candidates(
            config=config,
            plan=plan,
            pairs=(
                make_pair(
                    "rate_10",
                    "unknown_feature",
                    0.9,
                ),
            ),
        )


def test_split_dataset_with_purge() -> None:
    dataset = make_dataset(
        list(
            range(1, 11)
        )
    )

    (
        training,
        validation,
        training_indices,
        purged_indices,
        validation_indices,
    ) = (
        optimizer
        ._split_dataset_with_purge(
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
    assert purged_indices == (7,)
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
            range(1, 9)
        )
    )

    result = (
        optimizer
        ._split_dataset_with_purge(
            dataset=dataset,
            validation_targets=3,
            purge_targets=0,
        )
    )

    assert result[2] == (
        1,
        2,
        3,
        4,
        5,
    )
    assert result[3] == ()
    assert result[4] == (
        6,
        7,
        8,
    )


def test_split_dataset_with_purge_rejects_insufficient_targets() -> None:
    dataset = make_dataset(
        [
            1,
            2,
            3,
            4,
            5,
        ]
    )

    with pytest.raises(
        optimizer.DatasetPreparationError,
        match="Not enough targets",
    ):
        optimizer._split_dataset_with_purge(
            dataset=dataset,
            validation_targets=5,
            purge_targets=1,
        )


def test_prepare_temporal_datasets(
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
            range(1, 11)
        )
    )
    metadata = [
        {
            "target": index,
        }
        for index in range(1, 11)
    ]
    draws = list(
        range(100)
    )

    monkeypatch.setattr(
        optimizer,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        optimizer
        .V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: draws
        ),
    )

    monkeypatch.setattr(
        optimizer.V7RankingDataset,
        "build_from_draws",
        lambda self, **kwargs: (
            dataset,
            metadata,
        ),
    )

    monkeypatch.setattr(
        optimizer
        .V7FeatureAblationReport,
        "_validate_dataset",
        classmethod(
            lambda cls, frame: None
        ),
    )

    config = make_config(
        tmp_path,
        validation_targets=5,
        purge_targets=1,
    )

    training, validation, summary = (
        optimizer
        .prepare_temporal_datasets(
            config
        )
    )

    assert database.closed is True
    assert (
        training[
            "target_draw_index"
        ].nunique()
        == 4
    )
    assert (
        validation[
            "target_draw_index"
        ].nunique()
        == 5
    )
    assert summary.draw_count == 100
    assert summary.dataset_targets == 10
    assert summary.training_targets == 4
    assert summary.purged_target_indices == (
        5,
    )
    assert summary.first_validation_target == 6
    assert summary.last_validation_target == 10


def test_prepare_temporal_datasets_wraps_build_error(
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
        optimizer,
        "SessionLocal",
        lambda: database,
    )

    monkeypatch.setattr(
        optimizer
        .V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: (
                (_ for _ in ())
                .throw(
                    RuntimeError("database failed")
                )
            )
        ),
    )

    with pytest.raises(
        optimizer.DatasetPreparationError,
        match=(
            "Unable to build the V7 "
            "ranking dataset"
        ),
    ):
        optimizer.prepare_temporal_datasets(
            make_config(
                tmp_path
            )
        )

    assert database.closed is True


def test_prepare_temporal_datasets_wraps_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDatabase:
        def close(self) -> None:
            return None

    monkeypatch.setattr(
        optimizer,
        "SessionLocal",
        lambda: FakeDatabase(),
    )

    monkeypatch.setattr(
        optimizer
        .V7FeatureAblationReport,
        "_load_draws",
        classmethod(
            lambda cls, db: list(
                range(10)
            )
        ),
    )

    monkeypatch.setattr(
        optimizer.V7RankingDataset,
        "build_from_draws",
        lambda self, **kwargs: (
            make_dataset(
                list(
                    range(1, 11)
                )
            ),
            list(
                range(10)
            ),
        ),
    )

    monkeypatch.setattr(
        optimizer
        .V7FeatureAblationReport,
        "_validate_dataset",
        classmethod(
            lambda cls, frame: (
                (_ for _ in ())
                .throw(
                    ValueError("invalid dataset")
                )
            )
        ),
    )

    with pytest.raises(
        optimizer.DatasetPreparationError,
        match=(
            "generated V7 ranking "
            "dataset is invalid"
        ),
    ):
        optimizer.prepare_temporal_datasets(
            make_config(
                tmp_path
            )
        )


def test_evaluate_pruning_plan_accepts_and_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        tmp_path,
        validation_targets=5,
        tolerance=0.0,
    )

    specs = (
        optimizer.EvaluationCandidateSpec(
            feature="rate_10",
            group_ids=(1,),
            correlation_link_count=3,
            cumulative_absolute_correlation=2.58,
            maximum_absolute_correlation=0.95,
            source=(
                "all_group_features"
            ),
        ),
        optimizer.EvaluationCandidateSpec(
            feature="short_vs_long",
            group_ids=(1,),
            correlation_link_count=1,
            cumulative_absolute_correlation=0.95,
            maximum_absolute_correlation=0.95,
            source=(
                "all_group_features"
            ),
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "build_evaluation_candidates",
        lambda **kwargs: specs,
    )

    monkeypatch.setattr(
        optimizer,
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

    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
        total_seconds=2.0,
    )

    rate_run = make_run_result(
        experiment_name=(
            "without_rate_10"
        ),
        feature_columns=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "rate_10"
        ),
        mean_hits=0.30,
        target_hit_rate=0.30,
        total_seconds=1.5,
    )

    short_run = make_run_result(
        experiment_name=(
            "without_short_vs_long"
        ),
        feature_columns=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "short_vs_long"
        ),
        mean_hits=0.50,
        target_hit_rate=0.60,
        total_seconds=1.7,
    )

    def fake_run_feature_subset(
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        config: Any,
    ) -> FeatureAblationRunResult:
        del training_dataset
        del validation_dataset

        if (
            config.experiment_name
            == "baseline"
        ):
            return baseline

        if (
            config.experiment_name
            == "without_rate_10"
        ):
            return rate_run

        if (
            config.experiment_name
            == "without_short_vs_long"
        ):
            return short_run

        raise AssertionError(
            config.experiment_name
        )

    monkeypatch.setattr(
        optimizer,
        "run_feature_subset",
        fake_run_feature_subset,
    )

    report = (
        optimizer
        .evaluate_pruning_plan(
            config=config,
            plan=sample_plan(),
            pairs=sample_pairs(),
        )
    )

    assert report.status == "success"
    assert report.baseline is baseline
    assert report.accepted_features == (
        "short_vs_long",
    )
    assert report.rejected_features == (
        "rate_10",
    )
    assert (
        report.best_single_removal
        == "short_vs_long"
    )
    assert (
        report.best_single_removal_delta
        == pytest.approx(0.10)
    )
    assert [
        item.decision
        for item
        in report.candidate_evaluations
    ] == [
        "REJECT",
        "ACCEPT",
    ]


def test_evaluate_pruning_plan_uses_tie_breakers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        tmp_path,
        validation_targets=5,
    )

    specs = (
        optimizer.EvaluationCandidateSpec(
            feature="recency",
            group_ids=(1,),
            correlation_link_count=2,
            cumulative_absolute_correlation=1.82,
            maximum_absolute_correlation=1.0,
            source=(
                "all_group_features"
            ),
        ),
        optimizer.EvaluationCandidateSpec(
            feature="short_vs_long",
            group_ids=(1,),
            correlation_link_count=1,
            cumulative_absolute_correlation=0.95,
            maximum_absolute_correlation=0.95,
            source=(
                "all_group_features"
            ),
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "build_evaluation_candidates",
        lambda **kwargs: specs,
    )

    monkeypatch.setattr(
        optimizer,
        "prepare_temporal_datasets",
        lambda config: (
            pd.DataFrame(),
            pd.DataFrame(),
            sample_dataset_summary(),
        ),
    )

    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    recency_run = make_run_result(
        experiment_name=(
            "without_recency"
        ),
        feature_columns=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "recency"
        ),
        mean_hits=0.50,
        target_hit_rate=0.50,
        total_seconds=1.0,
    )

    short_run = make_run_result(
        experiment_name=(
            "without_short_vs_long"
        ),
        feature_columns=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature != "short_vs_long"
        ),
        mean_hits=0.50,
        target_hit_rate=0.60,
        total_seconds=2.0,
    )

    def fake_run(
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        config: Any,
    ) -> FeatureAblationRunResult:
        del training_dataset
        del validation_dataset

        return {
            "baseline": baseline,
            "without_recency": (
                recency_run
            ),
            "without_short_vs_long": (
                short_run
            ),
        }[
            config.experiment_name
        ]

    monkeypatch.setattr(
        optimizer,
        "run_feature_subset",
        fake_run,
    )

    report = (
        optimizer
        .evaluate_pruning_plan(
            config=config,
            plan=sample_plan(),
            pairs=sample_pairs(),
        )
    )

    assert (
        report.best_single_removal
        == "short_vs_long"
    )


def test_evaluate_pruning_plan_rejects_empty_candidate_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimizer,
        "build_evaluation_candidates",
        lambda **kwargs: (),
    )

    with pytest.raises(
        optimizer.EvaluationError,
        match=(
            "No feature was selected"
        ),
    ):
        optimizer.evaluate_pruning_plan(
            config=make_config(
                tmp_path
            ),
            plan=sample_plan(),
            pairs=sample_pairs(),
        )


def test_evaluate_pruning_plan_wraps_baseline_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        optimizer,
        "build_evaluation_candidates",
        lambda **kwargs: (
            optimizer.EvaluationCandidateSpec(
                feature="rate_10",
                group_ids=(1,),
                correlation_link_count=3,
                cumulative_absolute_correlation=2.58,
                maximum_absolute_correlation=0.95,
                source="selected",
            ),
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "prepare_temporal_datasets",
        lambda config: (
            pd.DataFrame(),
            pd.DataFrame(),
            sample_dataset_summary(),
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "run_feature_subset",
        lambda **kwargs: (
            (_ for _ in ())
            .throw(
                RuntimeError("fit failed")
            )
        ),
    )

    with pytest.raises(
        optimizer.EvaluationError,
        match="Baseline evaluation failed",
    ):
        optimizer.evaluate_pruning_plan(
            config=make_config(
                tmp_path
            ),
            plan=sample_plan(),
            pairs=sample_pairs(),
        )


def test_evaluate_pruning_plan_wraps_candidate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = (
        optimizer
        .EvaluationCandidateSpec(
            feature="rate_10",
            group_ids=(1,),
            correlation_link_count=3,
            cumulative_absolute_correlation=2.58,
            maximum_absolute_correlation=0.95,
            source="selected",
        )
    )

    monkeypatch.setattr(
        optimizer,
        "build_evaluation_candidates",
        lambda **kwargs: (
            spec,
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "prepare_temporal_datasets",
        lambda config: (
            pd.DataFrame(),
            pd.DataFrame(),
            sample_dataset_summary(),
        ),
    )

    baseline = make_run_result(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        mean_hits=0.40,
        target_hit_rate=0.40,
    )

    calls = 0

    def fake_run(
        **kwargs: Any,
    ) -> FeatureAblationRunResult:
        nonlocal calls
        calls += 1

        if calls == 1:
            return baseline

        raise RuntimeError(
            "candidate fit failed"
        )

    monkeypatch.setattr(
        optimizer,
        "run_feature_subset",
        fake_run,
    )

    with pytest.raises(
        optimizer.EvaluationError,
        match=(
            "Candidate evaluation failed "
            "for feature: rate_10"
        ),
    ):
        optimizer.evaluate_pruning_plan(
            config=make_config(
                tmp_path
            ),
            plan=sample_plan(),
            pairs=sample_pairs(),
        )


def test_json_safe_converts_paths_tuples_and_non_finite() -> None:
    payload = optimizer._json_safe(
        {
            "path": Path("/tmp/test"),
            "tuple": (
                1,
                2,
            ),
            "nan": float("nan"),
            "inf": float("inf"),
        }
    )

    assert payload == {
        "path": "/tmp/test",
        "tuple": [
            1,
            2,
        ],
        "nan": None,
        "inf": None,
    }


def test_export_pruning_plan(
    tmp_path: Path,
) -> None:
    files = (
        optimizer
        .export_pruning_plan(
            plan=sample_plan(),
            output_directory=(
                tmp_path
                / "reports"
            ),
        )
    )

    assert set(files) == {
        "plan_json",
        "plan_text",
    }

    json_payload = json.loads(
        files["plan_json"]
        .read_text(
            encoding="utf-8"
        )
    )

    text = (
        files["plan_text"]
        .read_text(
            encoding="utf-8"
        )
    )

    assert (
        json_payload["status"]
        == "success"
    )
    assert (
        json_payload[
            "reported_total_pair_count"
        ]
        == 66
    )
    assert (
        "PREDIXA AI V7 FEATURE "
        "PRUNING PLAN"
        in text
    )
    assert "rate_10" in text
    assert (
        "does not remove features"
        in text
    )


def test_export_evaluation_report(
    tmp_path: Path,
) -> None:
    report = (
        sample_evaluation_report()
    )

    files = (
        optimizer
        .export_evaluation_report(
            report=report,
            output_directory=(
                tmp_path
                / "reports"
            ),
        )
    )

    assert set(files) == {
        "evaluation_json",
        "evaluation_text",
        "evaluation_csv",
        "accepted_csv",
        "rejected_csv",
    }

    payload = json.loads(
        files["evaluation_json"]
        .read_text(
            encoding="utf-8"
        )
    )

    text = (
        files["evaluation_text"]
        .read_text(
            encoding="utf-8"
        )
    )

    with files[
        "evaluation_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        all_rows = list(
            csv.DictReader(file)
        )

    with files[
        "accepted_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        accepted_rows = list(
            csv.DictReader(file)
        )

    with files[
        "rejected_csv"
    ].open(
        encoding="utf-8",
        newline="",
    ) as file:
        rejected_rows = list(
            csv.DictReader(file)
        )

    assert (
        payload["best_single_removal"]
        == "short_vs_long"
    )
    assert (
        payload["accepted_features"]
        == [
            "short_vs_long",
        ]
    )
    assert len(all_rows) == 2
    assert len(accepted_rows) == 1
    assert (
        accepted_rows[0]["feature"]
        == "short_vs_long"
    )
    assert len(rejected_rows) == 1
    assert (
        rejected_rows[0]["feature"]
        == "rate_10"
    )
    assert (
        "SINGLE-FEATURE REMOVAL "
        "EXPERIMENTS"
        in text
    )
    assert (
        "accepted features were tested "
        "independently"
        in text
    )


def test_build_argument_parser_defaults() -> None:
    arguments = (
        optimizer
        .build_argument_parser()
        .parse_args([])
    )

    assert (
        arguments
        .correlation_threshold
        == pytest.approx(0.80)
    )
    assert (
        arguments
        .candidate_scope
        == "all_group_features"
    )
    assert arguments.plan_only is False
    assert arguments.features == ()


def test_build_argument_parser_accepts_explicit_features() -> None:
    arguments = (
        optimizer
        .build_argument_parser()
        .parse_args(
            [
                "--features",
                "rate_10",
                "recency",
                "--plan-only",
            ]
        )
    )

    assert arguments.features == [
        "rate_10",
        "recency",
    ]
    assert arguments.plan_only is True


def test_main_plan_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = (
        tmp_path
        / "correlation.json"
    )
    output_directory = (
        tmp_path
        / "output"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_optimizer",
            "--correlation-report",
            str(report_path),
            "--output-directory",
            str(output_directory),
            "--validation-targets",
            "5",
            "--plan-only",
        ],
    )

    monkeypatch.setattr(
        optimizer,
        "build_pruning_plan",
        lambda config: (
            sample_plan(),
            sample_pairs(),
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "export_pruning_plan",
        lambda **kwargs: {
            "plan_json": (
                output_directory
                / "plan.json"
            ),
        },
    )

    printed: dict[
        str,
        Any,
    ] = {}

    monkeypatch.setattr(
        optimizer,
        "print_pruning_plan",
        lambda **kwargs: printed.update(
            kwargs
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "evaluate_pruning_plan",
        lambda **kwargs: (
            (_ for _ in ())
            .throw(
                AssertionError(
                    "evaluation must not run"
                )
            )
        ),
    )

    assert optimizer.main() == 0
    assert (
        printed["plan"]
        == sample_plan()
    )


def test_main_full_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = (
        tmp_path
        / "correlation.json"
    )
    output_directory = (
        tmp_path
        / "output"
    )
    evaluation = (
        sample_evaluation_report()
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_optimizer",
            "--correlation-report",
            str(report_path),
            "--output-directory",
            str(output_directory),
            "--validation-targets",
            "5",
        ],
    )

    monkeypatch.setattr(
        optimizer,
        "build_pruning_plan",
        lambda config: (
            sample_plan(),
            sample_pairs(),
        ),
    )

    monkeypatch.setattr(
        optimizer,
        "export_pruning_plan",
        lambda **kwargs: {
            "plan_json": (
                output_directory
                / "plan.json"
            ),
        },
    )

    monkeypatch.setattr(
        optimizer,
        "evaluate_pruning_plan",
        lambda **kwargs: evaluation,
    )

    monkeypatch.setattr(
        optimizer,
        "export_evaluation_report",
        lambda **kwargs: {
            "evaluation_json": (
                output_directory
                / "evaluation.json"
            ),
        },
    )

    printed: dict[
        str,
        Any,
    ] = {}

    monkeypatch.setattr(
        optimizer,
        "print_evaluation_report",
        lambda **kwargs: printed.update(
            kwargs
        ),
    )

    assert optimizer.main() == 0
    assert (
        printed["report"]
        is evaluation
    )
    assert set(
        printed["generated_files"]
    ) == {
        "plan_json",
        "evaluation_json",
    }


def test_main_returns_one_on_feature_pruning_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "feature_pruning_optimizer",
            "--correlation-report",
            str(
                tmp_path
                / "missing.json"
            ),
            "--validation-targets",
            "5",
        ],
    )

    assert optimizer.main() == 1

    output = capsys.readouterr().out

    assert (
        "PREDIXA AI V7 FEATURE PRUNING"
        in output
    )
    assert "ERROR:" in output
