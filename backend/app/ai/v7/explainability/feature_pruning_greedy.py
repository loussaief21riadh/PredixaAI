from __future__ import annotations

"""
PredixaAI V7 - Greedy cumulative feature pruning, Sprint 3.

This module extends the independent Sprint 2 experiments with cumulative
single-feature removals.

Protocol
--------
1. Build the same V7 correlation plan used by Sprint 2.
2. Build one shared chronological dataset and purged validation split.
3. Evaluate the all-feature baseline.
4. At each iteration:
   - evaluate removing every remaining candidate from the current feature set;
   - accept candidates that stay within the configured Hits@K tolerance;
   - choose the best accepted candidate deterministically;
   - make that candidate the new cumulative baseline.
5. Stop when no removal is acceptable, no candidate remains, the minimum
   feature count is reached, or the iteration limit is reached.

Important
---------
- Every candidate in an iteration is compared against that iteration's
  current cumulative baseline.
- The dataset and temporal split are built once and reused unchanged.
- This module never changes V7RankingDataset or the production model.
- The output is a recommendation and does not mutate production features.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.ai.v7.explainability.feature_ablation_runner import (
    FeatureAblationComparison,
    FeatureAblationRunConfig,
    FeatureAblationRunResult,
    compare_feature_runs,
    run_feature_subset,
)
from app.ai.v7.explainability.feature_pruning_optimizer import (
    BACKEND_DIRECTORY,
    CANDIDATE_SCOPES,
    DEFAULT_CANDIDATE_SCOPE,
    DEFAULT_CORRELATION_REPORT,
    DEFAULT_CORRELATION_THRESHOLD,
    DEFAULT_MAX_TRAINING_TARGETS,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_PURGE_TARGETS,
    DEFAULT_TOLERANCE,
    DEFAULT_TOP_K,
    DEFAULT_VALIDATION_TARGETS,
    DEFAULT_WINDOW_SIZE,
    EvaluationCandidateSpec,
    FeaturePruningError,
    PruningConfig,
    PruningPlan,
    TemporalDatasetSummary,
    build_evaluation_candidates,
    build_pruning_plan,
    prepare_temporal_datasets,
)
from app.ai.v7.ranking_dataset import V7RankingDataset


VERSION = "V7-FEATURE-PRUNING-GREEDY-TEMPORAL-V1"

DEFAULT_GREEDY_OUTPUT_DIRECTORY = (
    BACKEND_DIRECTORY
    / "reports"
    / "v7"
    / "feature_pruning_greedy"
)

DEFAULT_MAX_ITERATIONS = 10
DEFAULT_MINIMUM_FEATURES = 1


class GreedyPruningError(FeaturePruningError):
    """Base exception for cumulative pruning failures."""


class GreedyConfigurationError(GreedyPruningError):
    """Raised when cumulative pruning parameters are invalid."""


class GreedyEvaluationError(GreedyPruningError):
    """Raised when a cumulative model evaluation fails."""


@dataclass(frozen=True)
class GreedyPruningConfig:
    """Complete Sprint 3 configuration."""

    correlation_report: Path = DEFAULT_CORRELATION_REPORT
    output_directory: Path = DEFAULT_GREEDY_OUTPUT_DIRECTORY
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    minimum_group_size: int = 2
    maximum_candidates_per_group: int = 1
    window_size: int = DEFAULT_WINDOW_SIZE
    max_training_targets: int = DEFAULT_MAX_TRAINING_TARGETS
    validation_targets: int = DEFAULT_VALIDATION_TARGETS
    top_k: int = DEFAULT_TOP_K
    purge_targets: int = DEFAULT_PURGE_TARGETS
    tolerance: float = DEFAULT_TOLERANCE
    candidate_scope: str = DEFAULT_CANDIDATE_SCOPE
    explicit_features: tuple[str, ...] = ()
    maximum_iterations: int = DEFAULT_MAX_ITERATIONS
    minimum_features: int = DEFAULT_MINIMUM_FEATURES

    def validated(self) -> "GreedyPruningConfig":
        """Validate and return this immutable configuration."""

        validate_greedy_config(self)
        return self

    def to_pruning_config(self) -> PruningConfig:
        """Build the compatible Sprint 2 planning configuration."""

        return PruningConfig(
            correlation_report=self.correlation_report,
            output_directory=self.output_directory,
            correlation_threshold=self.correlation_threshold,
            minimum_group_size=self.minimum_group_size,
            maximum_candidates_per_group=(
                self.maximum_candidates_per_group
            ),
            window_size=self.window_size,
            max_training_targets=self.max_training_targets,
            validation_targets=self.validation_targets,
            top_k=self.top_k,
            purge_targets=self.purge_targets,
            tolerance=self.tolerance,
            candidate_scope=self.candidate_scope,
            explicit_features=self.explicit_features,
            plan_only=False,
        )


@dataclass(frozen=True)
class GreedyCandidateEvaluation:
    """One feature-removal evaluation inside one greedy iteration."""

    iteration: int
    feature: str
    group_ids: tuple[int, ...]
    source: str
    correlation_link_count: int
    cumulative_absolute_correlation: float
    maximum_absolute_correlation: float
    active_features_before: tuple[str, ...]
    candidate_features: tuple[str, ...]
    decision: str
    selected_for_removal: bool
    run: FeatureAblationRunResult
    comparison: FeatureAblationComparison


@dataclass(frozen=True)
class GreedyPruningIteration:
    """Complete result of one cumulative pruning iteration."""

    iteration: int
    baseline_experiment: str
    baseline_features: tuple[str, ...]
    baseline_mean_hits_at_k: float
    baseline_target_hit_rate: float
    evaluated_candidate_count: int
    accepted_candidate_count: int
    selected_feature: str | None
    selected_delta: float | None
    selected_mean_hits_at_k: float | None
    selected_target_hit_rate: float | None
    active_features_after: tuple[str, ...]
    candidate_evaluations: tuple[GreedyCandidateEvaluation, ...]


@dataclass(frozen=True)
class GreedyPruningReport:
    """Complete cumulative pruning report."""

    status: str
    version: str
    protocol: str
    stop_reason: str
    correlation_threshold: float
    candidate_scope: str
    tolerance: float
    window_size: int
    max_training_targets: int
    validation_targets: int
    top_k: int
    purge_targets: int
    maximum_iterations: int
    minimum_features: int
    dataset: TemporalDatasetSummary
    plan: PruningPlan
    initial_candidates: tuple[EvaluationCandidateSpec, ...]
    initial_baseline: FeatureAblationRunResult
    iterations: tuple[GreedyPruningIteration, ...]
    accepted_removal_sequence: tuple[str, ...]
    final_features: tuple[str, ...]
    final_feature_count: int
    final_run: FeatureAblationRunResult
    initial_mean_hits_at_k: float
    final_mean_hits_at_k: float
    total_absolute_delta: float
    total_relative_delta: float | None


def validate_greedy_config(
    config: GreedyPruningConfig,
) -> None:
    """Validate planning, temporal, and cumulative parameters."""

    try:
        config.to_pruning_config().validated()
    except FeaturePruningError as exc:
        raise GreedyConfigurationError(
            str(exc)
        ) from exc

    if config.maximum_iterations < 1:
        raise GreedyConfigurationError(
            "maximum_iterations must be at least 1"
        )

    model_feature_count = len(
        V7RankingDataset.feature_columns()
    )

    if not 1 <= config.minimum_features <= model_feature_count:
        raise GreedyConfigurationError(
            "minimum_features must be between 1 and "
            f"{model_feature_count}"
        )


def _normalise_active_features(
    features: Sequence[str],
) -> tuple[str, ...]:
    """Validate an ordered active-feature sequence."""

    if isinstance(
        features,
        (str, bytes, bytearray),
    ):
        raise GreedyConfigurationError(
            "active features must be a sequence of names"
        )

    normalised: list[str] = []
    seen: set[str] = set()

    for value in features:
        if not isinstance(value, str):
            raise GreedyConfigurationError(
                "active feature names must be strings"
            )

        feature = value.strip()

        if not feature:
            raise GreedyConfigurationError(
                "active feature names cannot be empty"
            )

        if feature in seen:
            raise GreedyConfigurationError(
                f"duplicate active feature: {feature}"
            )

        seen.add(feature)
        normalised.append(feature)

    if not normalised:
        raise GreedyConfigurationError(
            "at least one active feature is required"
        )

    model_features = set(
        V7RankingDataset.feature_columns()
    )

    unknown = sorted(
        set(normalised)
        - model_features
    )

    if unknown:
        raise GreedyConfigurationError(
            f"unknown active V7 features: {unknown}"
        )

    return tuple(normalised)


def build_subset_config(
    experiment_name: str,
    feature_columns: Sequence[str],
    top_k: int,
) -> FeatureAblationRunConfig:
    """Build a validated arbitrary-subset runner configuration."""

    return FeatureAblationRunConfig(
        experiment_name=experiment_name,
        feature_columns=_normalise_active_features(
            feature_columns
        ),
        top_k=top_k,
    ).validated()


def _candidate_order_key(
    evaluation: GreedyCandidateEvaluation,
) -> tuple[
    float,
    float,
    float,
    int,
    float,
    str,
]:
    """
    Deterministically rank accepted candidates.

    Priority:
    1. larger Hits@K delta;
    2. larger target hit rate;
    3. faster total runtime;
    4. more correlation links;
    5. larger maximum correlation;
    6. lexical feature name.
    """

    return (
        evaluation.comparison.absolute_delta,
        evaluation.run.target_hit_rate,
        -evaluation.run.total_seconds,
        evaluation.correlation_link_count,
        evaluation.maximum_absolute_correlation,
        evaluation.feature,
    )


def _evaluate_iteration_candidates(
    *,
    iteration: int,
    baseline: FeatureAblationRunResult,
    active_features: tuple[str, ...],
    remaining_candidates: Sequence[EvaluationCandidateSpec],
    training_dataset: Any,
    validation_dataset: Any,
    config: GreedyPruningConfig,
) -> tuple[
    tuple[GreedyCandidateEvaluation, ...],
    GreedyCandidateEvaluation | None,
]:
    """Evaluate all valid single-feature removals for one iteration."""

    evaluations: list[
        GreedyCandidateEvaluation
    ] = []

    for candidate in remaining_candidates:
        if candidate.feature not in active_features:
            continue

        if len(active_features) <= config.minimum_features:
            continue

        candidate_features = tuple(
            feature
            for feature in active_features
            if feature != candidate.feature
        )

        if len(candidate_features) < config.minimum_features:
            continue

        experiment_name = (
            f"greedy_i{iteration}_without_"
            f"{candidate.feature}"
        )

        try:
            candidate_run = run_feature_subset(
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                config=build_subset_config(
                    experiment_name=experiment_name,
                    feature_columns=candidate_features,
                    top_k=config.top_k,
                ),
            )

            comparison = compare_feature_runs(
                baseline=baseline,
                candidate=candidate_run,
                tolerance=config.tolerance,
            )
        except Exception as exc:
            raise GreedyEvaluationError(
                "Greedy candidate evaluation failed "
                f"at iteration {iteration} for feature "
                f"{candidate.feature}"
            ) from exc

        evaluations.append(
            GreedyCandidateEvaluation(
                iteration=iteration,
                feature=candidate.feature,
                group_ids=candidate.group_ids,
                source=candidate.source,
                correlation_link_count=(
                    candidate.correlation_link_count
                ),
                cumulative_absolute_correlation=(
                    candidate.cumulative_absolute_correlation
                ),
                maximum_absolute_correlation=(
                    candidate.maximum_absolute_correlation
                ),
                active_features_before=active_features,
                candidate_features=candidate_features,
                decision=(
                    "ACCEPT"
                    if comparison.accepted
                    else "REJECT"
                ),
                selected_for_removal=False,
                run=candidate_run,
                comparison=comparison,
            )
        )

    accepted = [
        evaluation
        for evaluation in evaluations
        if evaluation.comparison.accepted
    ]

    if not accepted:
        return (
            tuple(evaluations),
            None,
        )

    selected = max(
        accepted,
        key=_candidate_order_key,
    )

    updated_evaluations = tuple(
        GreedyCandidateEvaluation(
            iteration=evaluation.iteration,
            feature=evaluation.feature,
            group_ids=evaluation.group_ids,
            source=evaluation.source,
            correlation_link_count=(
                evaluation.correlation_link_count
            ),
            cumulative_absolute_correlation=(
                evaluation.cumulative_absolute_correlation
            ),
            maximum_absolute_correlation=(
                evaluation.maximum_absolute_correlation
            ),
            active_features_before=(
                evaluation.active_features_before
            ),
            candidate_features=(
                evaluation.candidate_features
            ),
            decision=evaluation.decision,
            selected_for_removal=(
                evaluation.feature
                == selected.feature
            ),
            run=evaluation.run,
            comparison=evaluation.comparison,
        )
        for evaluation in evaluations
    )

    selected_updated = next(
        evaluation
        for evaluation in updated_evaluations
        if evaluation.selected_for_removal
    )

    return (
        updated_evaluations,
        selected_updated,
    )


def run_greedy_pruning(
    config: GreedyPruningConfig,
) -> GreedyPruningReport:
    """Execute the complete cumulative temporal pruning algorithm."""

    config.validated()
    pruning_config = (
        config.to_pruning_config()
    )

    try:
        plan, pairs = build_pruning_plan(
            pruning_config
        )

        initial_candidates = (
            build_evaluation_candidates(
                config=pruning_config,
                plan=plan,
                pairs=pairs,
            )
        )
    except FeaturePruningError:
        raise
    except Exception as exc:
        raise GreedyEvaluationError(
            "Unable to build the initial pruning plan"
        ) from exc

    if not initial_candidates:
        raise GreedyEvaluationError(
            "No feature was selected for cumulative evaluation"
        )

    try:
        (
            training_dataset,
            validation_dataset,
            dataset_summary,
        ) = prepare_temporal_datasets(
            pruning_config
        )
    except FeaturePruningError:
        raise
    except Exception as exc:
        raise GreedyEvaluationError(
            "Unable to prepare cumulative temporal datasets"
        ) from exc

    initial_features = _normalise_active_features(
        V7RankingDataset.feature_columns()
    )

    try:
        initial_baseline = run_feature_subset(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=build_subset_config(
                experiment_name="greedy_baseline",
                feature_columns=initial_features,
                top_k=config.top_k,
            ),
        )
    except Exception as exc:
        raise GreedyEvaluationError(
            "Unable to evaluate the greedy baseline"
        ) from exc

    active_features = initial_features
    current_baseline = initial_baseline
    remaining_candidates = tuple(
        initial_candidates
    )
    iterations: list[
        GreedyPruningIteration
    ] = []
    accepted_sequence: list[str] = []
    stop_reason = "maximum_iterations_reached"

    for iteration_number in range(
        1,
        config.maximum_iterations + 1,
    ):
        if len(active_features) <= config.minimum_features:
            stop_reason = "minimum_feature_count_reached"
            break

        eligible_candidates = tuple(
            candidate
            for candidate in remaining_candidates
            if candidate.feature in active_features
        )

        if not eligible_candidates:
            stop_reason = "no_candidates_remaining"
            break

        (
            candidate_evaluations,
            selected,
        ) = _evaluate_iteration_candidates(
            iteration=iteration_number,
            baseline=current_baseline,
            active_features=active_features,
            remaining_candidates=eligible_candidates,
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            config=config,
        )

        accepted_count = sum(
            evaluation.comparison.accepted
            for evaluation in candidate_evaluations
        )

        if selected is None:
            iterations.append(
                GreedyPruningIteration(
                    iteration=iteration_number,
                    baseline_experiment=(
                        current_baseline.experiment_name
                    ),
                    baseline_features=active_features,
                    baseline_mean_hits_at_k=(
                        current_baseline.mean_hits_at_k
                    ),
                    baseline_target_hit_rate=(
                        current_baseline.target_hit_rate
                    ),
                    evaluated_candidate_count=len(
                        candidate_evaluations
                    ),
                    accepted_candidate_count=accepted_count,
                    selected_feature=None,
                    selected_delta=None,
                    selected_mean_hits_at_k=None,
                    selected_target_hit_rate=None,
                    active_features_after=active_features,
                    candidate_evaluations=(
                        candidate_evaluations
                    ),
                )
            )

            stop_reason = "no_acceptable_removal"
            break

        active_features_after = (
            selected.candidate_features
        )

        iterations.append(
            GreedyPruningIteration(
                iteration=iteration_number,
                baseline_experiment=(
                    current_baseline.experiment_name
                ),
                baseline_features=active_features,
                baseline_mean_hits_at_k=(
                    current_baseline.mean_hits_at_k
                ),
                baseline_target_hit_rate=(
                    current_baseline.target_hit_rate
                ),
                evaluated_candidate_count=len(
                    candidate_evaluations
                ),
                accepted_candidate_count=accepted_count,
                selected_feature=selected.feature,
                selected_delta=(
                    selected.comparison.absolute_delta
                ),
                selected_mean_hits_at_k=(
                    selected.run.mean_hits_at_k
                ),
                selected_target_hit_rate=(
                    selected.run.target_hit_rate
                ),
                active_features_after=(
                    active_features_after
                ),
                candidate_evaluations=(
                    candidate_evaluations
                ),
            )
        )

        accepted_sequence.append(
            selected.feature
        )
        active_features = (
            active_features_after
        )
        current_baseline = selected.run
        remaining_candidates = tuple(
            candidate
            for candidate in remaining_candidates
            if candidate.feature != selected.feature
        )

        if len(active_features) <= config.minimum_features:
            stop_reason = "minimum_feature_count_reached"
            break

        if not remaining_candidates:
            stop_reason = "no_candidates_remaining"
            break

    total_delta = (
        current_baseline.mean_hits_at_k
        - initial_baseline.mean_hits_at_k
    )

    total_relative_delta = (
        total_delta
        / initial_baseline.mean_hits_at_k
        if initial_baseline.mean_hits_at_k != 0
        else None
    )

    return GreedyPruningReport(
        status="success",
        version=VERSION,
        protocol=(
            "greedy cumulative single-feature removal; "
            f"{config.purge_targets} target(s) purged "
            "immediately before one shared chronological "
            "validation window; every iteration evaluates "
            "all remaining candidates against the current "
            "cumulative baseline"
        ),
        stop_reason=stop_reason,
        correlation_threshold=(
            config.correlation_threshold
        ),
        candidate_scope=(
            "explicit"
            if config.explicit_features
            else config.candidate_scope
        ),
        tolerance=config.tolerance,
        window_size=config.window_size,
        max_training_targets=(
            config.max_training_targets
        ),
        validation_targets=(
            config.validation_targets
        ),
        top_k=config.top_k,
        purge_targets=config.purge_targets,
        maximum_iterations=(
            config.maximum_iterations
        ),
        minimum_features=(
            config.minimum_features
        ),
        dataset=dataset_summary,
        plan=plan,
        initial_candidates=tuple(
            initial_candidates
        ),
        initial_baseline=initial_baseline,
        iterations=tuple(iterations),
        accepted_removal_sequence=tuple(
            accepted_sequence
        ),
        final_features=active_features,
        final_feature_count=len(
            active_features
        ),
        final_run=current_baseline,
        initial_mean_hits_at_k=(
            initial_baseline.mean_hits_at_k
        ),
        final_mean_hits_at_k=(
            current_baseline.mean_hits_at_k
        ),
        total_absolute_delta=float(
            total_delta
        ),
        total_relative_delta=(
            float(total_relative_delta)
            if total_relative_delta is not None
            else None
        ),
    )


def _json_safe(
    value: Any,
) -> Any:
    """Recursively convert values to strict JSON-safe objects."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, list):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


def _write_json(
    payload: Mapping[str, Any],
    path: Path,
) -> Path:
    path.write_text(
        json.dumps(
            _json_safe(
                dict(payload)
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return path.resolve()


def _write_csv(
    *,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    fieldnames: Sequence[str],
) -> Path:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                fieldnames
            ),
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    field: _json_safe(
                        row.get(field)
                    )
                    for field in fieldnames
                }
            )

    return path.resolve()


def export_greedy_report(
    report: GreedyPruningReport,
    output_directory: Path,
) -> dict[str, Path]:
    """Export JSON, text, iteration CSV, and candidate CSV reports."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "feature_pruning_greedy.json"
    )
    text_path = (
        output_directory
        / "feature_pruning_greedy.txt"
    )
    iterations_csv_path = (
        output_directory
        / "feature_pruning_greedy_iterations.csv"
    )
    candidates_csv_path = (
        output_directory
        / "feature_pruning_greedy_candidates.csv"
    )
    final_features_path = (
        output_directory
        / "feature_pruning_greedy_final_features.txt"
    )

    _write_json(
        asdict(report),
        json_path,
    )

    iteration_rows: list[
        dict[str, Any]
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    for iteration in report.iterations:
        iteration_rows.append(
            {
                "iteration": iteration.iteration,
                "baseline_experiment": (
                    iteration.baseline_experiment
                ),
                "baseline_feature_count": len(
                    iteration.baseline_features
                ),
                "baseline_features": "|".join(
                    iteration.baseline_features
                ),
                "baseline_mean_hits_at_k": (
                    iteration.baseline_mean_hits_at_k
                ),
                "baseline_target_hit_rate": (
                    iteration.baseline_target_hit_rate
                ),
                "evaluated_candidate_count": (
                    iteration.evaluated_candidate_count
                ),
                "accepted_candidate_count": (
                    iteration.accepted_candidate_count
                ),
                "selected_feature": (
                    iteration.selected_feature
                ),
                "selected_delta": (
                    iteration.selected_delta
                ),
                "selected_mean_hits_at_k": (
                    iteration.selected_mean_hits_at_k
                ),
                "selected_target_hit_rate": (
                    iteration.selected_target_hit_rate
                ),
                "active_feature_count_after": len(
                    iteration.active_features_after
                ),
                "active_features_after": "|".join(
                    iteration.active_features_after
                ),
            }
        )

        for evaluation in (
            iteration.candidate_evaluations
        ):
            candidate_rows.append(
                {
                    "iteration": (
                        evaluation.iteration
                    ),
                    "feature": (
                        evaluation.feature
                    ),
                    "group_ids": "|".join(
                        str(group_id)
                        for group_id
                        in evaluation.group_ids
                    ),
                    "source": evaluation.source,
                    "correlation_link_count": (
                        evaluation
                        .correlation_link_count
                    ),
                    "cumulative_absolute_correlation": (
                        evaluation
                        .cumulative_absolute_correlation
                    ),
                    "maximum_absolute_correlation": (
                        evaluation
                        .maximum_absolute_correlation
                    ),
                    "active_feature_count_before": len(
                        evaluation.active_features_before
                    ),
                    "candidate_feature_count": len(
                        evaluation.candidate_features
                    ),
                    "baseline_mean_hits_at_k": (
                        evaluation
                        .comparison
                        .baseline_mean_hits_at_k
                    ),
                    "candidate_mean_hits_at_k": (
                        evaluation
                        .comparison
                        .candidate_mean_hits_at_k
                    ),
                    "absolute_delta": (
                        evaluation
                        .comparison
                        .absolute_delta
                    ),
                    "relative_delta": (
                        evaluation
                        .comparison
                        .relative_delta
                    ),
                    "candidate_target_hit_rate": (
                        evaluation
                        .run
                        .target_hit_rate
                    ),
                    "candidate_total_hits": (
                        evaluation
                        .run
                        .total_hits
                    ),
                    "candidate_total_seconds": (
                        evaluation
                        .run
                        .total_seconds
                    ),
                    "decision": (
                        evaluation.decision
                    ),
                    "selected_for_removal": (
                        evaluation
                        .selected_for_removal
                    ),
                }
            )

    _write_csv(
        rows=iteration_rows,
        path=iterations_csv_path,
        fieldnames=(
            "iteration",
            "baseline_experiment",
            "baseline_feature_count",
            "baseline_features",
            "baseline_mean_hits_at_k",
            "baseline_target_hit_rate",
            "evaluated_candidate_count",
            "accepted_candidate_count",
            "selected_feature",
            "selected_delta",
            "selected_mean_hits_at_k",
            "selected_target_hit_rate",
            "active_feature_count_after",
            "active_features_after",
        ),
    )

    _write_csv(
        rows=candidate_rows,
        path=candidates_csv_path,
        fieldnames=(
            "iteration",
            "feature",
            "group_ids",
            "source",
            "correlation_link_count",
            "cumulative_absolute_correlation",
            "maximum_absolute_correlation",
            "active_feature_count_before",
            "candidate_feature_count",
            "baseline_mean_hits_at_k",
            "candidate_mean_hits_at_k",
            "absolute_delta",
            "relative_delta",
            "candidate_target_hit_rate",
            "candidate_total_hits",
            "candidate_total_seconds",
            "decision",
            "selected_for_removal",
        ),
    )

    lines = [
        "=" * 136,
        "PREDIXA AI V7 GREEDY CUMULATIVE FEATURE PRUNING",
        "=" * 136,
        f"Status                         : {report.status}",
        f"Version                        : {report.version}",
        f"Protocol                       : {report.protocol}",
        f"Stop reason                    : {report.stop_reason}",
        (
            "Correlation threshold          : "
            f"{report.correlation_threshold:.6f}"
        ),
        (
            "Candidate scope                : "
            f"{report.candidate_scope}"
        ),
        (
            "Acceptance tolerance           : "
            f"{report.tolerance:.6f}"
        ),
        f"Window size                    : {report.window_size}",
        (
            "Max training targets           : "
            f"{report.max_training_targets}"
        ),
        (
            "Validation targets             : "
            f"{report.validation_targets}"
        ),
        f"Top K                          : {report.top_k}",
        f"Purged targets                 : {report.purge_targets}",
        (
            "Maximum iterations             : "
            f"{report.maximum_iterations}"
        ),
        (
            "Minimum features               : "
            f"{report.minimum_features}"
        ),
        "",
        "TEMPORAL DATASET",
        "-" * 136,
        (
            "Draw count                     : "
            f"{report.dataset.draw_count}"
        ),
        (
            "Dataset rows                   : "
            f"{report.dataset.dataset_rows}"
        ),
        (
            "Dataset targets                : "
            f"{report.dataset.dataset_targets}"
        ),
        (
            "Training targets               : "
            f"{report.dataset.training_targets}"
        ),
        (
            "Purged target indices          : "
            + (
                ", ".join(
                    str(value)
                    for value
                    in report.dataset.purged_target_indices
                )
                if report.dataset.purged_target_indices
                else "none"
            )
        ),
        (
            "Validation targets             : "
            f"{report.dataset.validation_targets}"
        ),
        "",
        "INITIAL BASELINE",
        "-" * 136,
        (
            "Initial feature count          : "
            f"{report.initial_baseline.feature_count}"
        ),
        (
            "Initial mean Hits@K            : "
            f"{report.initial_mean_hits_at_k:.6f}"
        ),
        (
            "Initial target hit rate        : "
            f"{report.initial_baseline.target_hit_rate:.6f}"
        ),
        "",
        "GREEDY ITERATIONS",
        "-" * 136,
        (
            f"{'Iter':>6}"
            f"{'Base Features':>16}"
            f"{'Base Hits':>14}"
            f"{'Evaluated':>12}"
            f"{'Accepted':>12}"
            f"{'Selected':>28}"
            f"{'Delta':>12}"
            f"{'New Hits':>14}"
            f"{'New Count':>12}"
        ),
        "-" * 136,
    ]

    if not report.iterations:
        lines.append(
            "No greedy iteration was executed."
        )
    else:
        for iteration in report.iterations:
            lines.append(
                f"{iteration.iteration:>6}"
                f"{len(iteration.baseline_features):>16}"
                f"{iteration.baseline_mean_hits_at_k:>14.6f}"
                f"{iteration.evaluated_candidate_count:>12}"
                f"{iteration.accepted_candidate_count:>12}"
                f"{str(iteration.selected_feature):>28}"
                f"{(
                    iteration.selected_delta
                    if iteration.selected_delta is not None
                    else 0.0
                ):>12.6f}"
                f"{(
                    iteration.selected_mean_hits_at_k
                    if iteration.selected_mean_hits_at_k is not None
                    else iteration.baseline_mean_hits_at_k
                ):>14.6f}"
                f"{len(iteration.active_features_after):>12}"
            )

    lines.extend(
        [
            "",
            "FINAL RECOMMENDATION",
            "-" * 136,
            (
                "Accepted removal sequence       : "
                + (
                    " -> ".join(
                        report.accepted_removal_sequence
                    )
                    if report.accepted_removal_sequence
                    else "none"
                )
            ),
            (
                "Final feature count              : "
                f"{report.final_feature_count}"
            ),
            (
                "Final features                   : "
                f"{', '.join(report.final_features)}"
            ),
            (
                "Final mean Hits@K                : "
                f"{report.final_mean_hits_at_k:.6f}"
            ),
            (
                "Total absolute delta             : "
                f"{report.total_absolute_delta:.6f}"
            ),
            (
                "Total relative delta             : "
                + (
                    f"{report.total_relative_delta:.6f}"
                    if report.total_relative_delta is not None
                    else "none"
                )
            ),
            "",
            (
                "The final feature list is a temporal validation "
                "recommendation. It is not applied automatically "
                "to the production model."
            ),
            "=" * 136,
        ]
    )

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    final_features_path.write_text(
        "\n".join(
            report.final_features
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "json": json_path.resolve(),
        "text": text_path.resolve(),
        "iterations_csv": (
            iterations_csv_path.resolve()
        ),
        "candidates_csv": (
            candidates_csv_path.resolve()
        ),
        "final_features": (
            final_features_path.resolve()
        ),
    }


def print_greedy_report(
    report: GreedyPruningReport,
    generated_files: Mapping[str, Path],
) -> None:
    """Print a compact cumulative-pruning summary."""

    print("=" * 132)
    print(
        "PREDIXA AI V7 GREEDY CUMULATIVE "
        "FEATURE PRUNING"
    )
    print("=" * 132)
    print(
        f"Status                  : "
        f"{report.status}"
    )
    print(
        f"Stop reason             : "
        f"{report.stop_reason}"
    )
    print(
        f"Training targets        : "
        f"{report.dataset.training_targets}"
    )
    print(
        f"Purged targets          : "
        f"{report.dataset.purged_targets}"
    )
    print(
        f"Validation targets      : "
        f"{report.dataset.validation_targets}"
    )
    print(
        f"Initial features        : "
        f"{report.initial_baseline.feature_count}"
    )
    print(
        f"Initial mean Hits@K     : "
        f"{report.initial_mean_hits_at_k:.6f}"
    )
    print()
    print(
        f"{'Iteration':>10}"
        f"{'Selected feature':>28}"
        f"{'Delta':>12}"
        f"{'Mean Hits@K':>16}"
        f"{'Features left':>16}"
    )
    print("-" * 132)

    for iteration in report.iterations:
        print(
            f"{iteration.iteration:>10}"
            f"{str(iteration.selected_feature):>28}"
            f"{(
                iteration.selected_delta
                if iteration.selected_delta is not None
                else 0.0
            ):>12.6f}"
            f"{(
                iteration.selected_mean_hits_at_k
                if iteration.selected_mean_hits_at_k is not None
                else iteration.baseline_mean_hits_at_k
            ):>16.6f}"
            f"{len(iteration.active_features_after):>16}"
        )

    print()
    print(
        "Accepted sequence       : "
        + (
            " -> ".join(
                report.accepted_removal_sequence
            )
            if report.accepted_removal_sequence
            else "none"
        )
    )
    print(
        f"Final features          : "
        f"{report.final_feature_count}"
    )
    print(
        f"Final mean Hits@K       : "
        f"{report.final_mean_hits_at_k:.6f}"
    )
    print(
        f"Total delta             : "
        f"{report.total_absolute_delta:.6f}"
    )
    print()
    print("GENERATED FILES")
    print("-" * 132)

    for name, path in (
        generated_files.items()
    ):
        print(
            f"{name.upper():24}: "
            f"{path}"
        )

    print()
    print("=" * 132)
    print("SUCCESS")
    print("=" * 132)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Sprint 3 command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Run cumulative greedy temporal feature pruning "
            "for PredixaAI V7."
        )
    )

    parser.add_argument(
        "--correlation-report",
        type=Path,
        default=DEFAULT_CORRELATION_REPORT,
        help=(
            "Path to feature_correlation_report.json"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            DEFAULT_GREEDY_OUTPUT_DIRECTORY
        ),
        help=(
            "Directory used for greedy pruning exports"
        ),
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=(
            DEFAULT_CORRELATION_THRESHOLD
        ),
        help=(
            "Minimum absolute correlation used "
            "to create candidate groups"
        ),
    )

    parser.add_argument(
        "--minimum-group-size",
        type=int,
        default=2,
        help=(
            "Minimum number of features in a "
            "correlation group"
        ),
    )

    parser.add_argument(
        "--maximum-candidates-per-group",
        type=int,
        default=1,
        help=(
            "Number of provisional Sprint 1 "
            "candidates selected per group"
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="V7 historical feature window",
    )

    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            DEFAULT_MAX_TRAINING_TARGETS
        ),
        help=(
            "Maximum target count built by "
            "V7RankingDataset; 0 disables the limit"
        ),
    )

    parser.add_argument(
        "--validation-targets",
        type=int,
        default=(
            DEFAULT_VALIDATION_TARGETS
        ),
        help=(
            "Number of final chronological targets "
            "used for validation"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Ranking cutoff used for Hits@K",
    )

    parser.add_argument(
        "--purge-targets",
        type=int,
        default=DEFAULT_PURGE_TARGETS,
        help=(
            "Number of targets removed immediately "
            "before validation"
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Maximum accepted decrease in mean Hits@K "
            "at each cumulative iteration"
        ),
    )

    parser.add_argument(
        "--candidate-scope",
        choices=CANDIDATE_SCOPES,
        default=DEFAULT_CANDIDATE_SCOPE,
        help=(
            "'selected' uses provisional Sprint 1 candidates; "
            "'all_group_features' uses every feature in a "
            "high-correlation group"
        ),
    )

    parser.add_argument(
        "--features",
        nargs="*",
        default=(),
        help=(
            "Optional explicit cumulative candidates. "
            "Overrides candidate-scope."
        ),
    )

    parser.add_argument(
        "--maximum-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=(
            "Maximum number of accepted cumulative removals"
        ),
    )

    parser.add_argument(
        "--minimum-features",
        type=int,
        default=DEFAULT_MINIMUM_FEATURES,
        help=(
            "Never reduce the active feature set below "
            "this count"
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    config = GreedyPruningConfig(
        correlation_report=(
            arguments
            .correlation_report
            .expanduser()
            .resolve()
        ),
        output_directory=(
            arguments
            .output_directory
            .expanduser()
            .resolve()
        ),
        correlation_threshold=(
            arguments.correlation_threshold
        ),
        minimum_group_size=(
            arguments.minimum_group_size
        ),
        maximum_candidates_per_group=(
            arguments
            .maximum_candidates_per_group
        ),
        window_size=(
            arguments.window_size
        ),
        max_training_targets=(
            arguments.max_training_targets
        ),
        validation_targets=(
            arguments.validation_targets
        ),
        top_k=arguments.top_k,
        purge_targets=(
            arguments.purge_targets
        ),
        tolerance=arguments.tolerance,
        candidate_scope=(
            arguments.candidate_scope
        ),
        explicit_features=tuple(
            arguments.features
        ),
        maximum_iterations=(
            arguments.maximum_iterations
        ),
        minimum_features=(
            arguments.minimum_features
        ),
    )

    try:
        report = run_greedy_pruning(
            config
        )

        generated_files = (
            export_greedy_report(
                report=report,
                output_directory=(
                    config.output_directory
                ),
            )
        )

    except FeaturePruningError as exc:
        print("=" * 104)
        print(
            "PREDIXA AI V7 GREEDY "
            "FEATURE PRUNING"
        )
        print("=" * 104)
        print(
            f"ERROR: {exc}"
        )

        cause = exc.__cause__

        if cause is not None:
            print(
                "CAUSE: "
                f"{type(cause).__name__}: "
                f"{cause}"
            )

        print("=" * 104)

        return 1

    print_greedy_report(
        report=report,
        generated_files=(
            generated_files
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
