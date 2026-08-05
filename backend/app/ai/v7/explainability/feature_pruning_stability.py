from __future__ import annotations

"""
PredixaAI V7 - Multi-window feature-pruning stability validation, Sprint 4.

This module validates the cumulative pruning sequence recommended by Sprint 3
across several chronological holdout windows.

Default sequence
----------------
1. Remove ``short_vs_long``.
2. Remove ``rate_10``.

Protocol
--------
- Build the V7 ranking dataset once.
- Construct several historical validation windows from newest to oldest.
- Purge the target(s) immediately preceding every validation window.
- Train only on targets strictly earlier than each purged gap.
- Evaluate the all-feature baseline and every cumulative pruning step on the
  exact same train/validation split for that window.
- Aggregate acceptance rates and mean deltas across windows.
- Recommend the cumulative sequence only when every removal step and the final
  pruned model satisfy the configured stability threshold.

The module produces a recommendation only. It never mutates production model
features.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from app.database import SessionLocal
from app.ai.v7.explainability.feature_ablation import (
    V7FeatureAblationReport,
)
from app.ai.v7.explainability.feature_ablation_runner import (
    FeatureAblationComparison,
    FeatureAblationRunResult,
    compare_feature_runs,
    run_feature_subset,
)
from app.ai.v7.explainability.feature_pruning_greedy import (
    BACKEND_DIRECTORY,
    build_subset_config,
)
from app.ai.v7.explainability.feature_pruning_optimizer import (
    DEFAULT_MAX_TRAINING_TARGETS,
    DEFAULT_PURGE_TARGETS,
    DEFAULT_TOLERANCE,
    DEFAULT_TOP_K,
    DEFAULT_VALIDATION_TARGETS,
    DEFAULT_WINDOW_SIZE,
    FeaturePruningError,
)
from app.ai.v7.ranking_dataset import V7RankingDataset


VERSION = "V7-FEATURE-PRUNING-STABILITY-MULTI-WINDOW-V1"

DEFAULT_OUTPUT_DIRECTORY = (
    BACKEND_DIRECTORY
    / "reports"
    / "v7"
    / "feature_pruning_stability"
)

DEFAULT_REMOVAL_SEQUENCE = (
    "short_vs_long",
    "rate_10",
)

DEFAULT_WINDOW_COUNT = 5
DEFAULT_WINDOW_STEP_TARGETS = 100
DEFAULT_MINIMUM_STABILITY_RATE = 0.80
DEFAULT_MINIMUM_TRAINING_TARGETS = 500


class FeaturePruningStabilityError(FeaturePruningError):
    """Base exception for Sprint 4 failures."""


class StabilityConfigurationError(
    FeaturePruningStabilityError
):
    """Raised when multi-window parameters are invalid."""


class StabilityDatasetError(
    FeaturePruningStabilityError
):
    """Raised when chronological windows cannot be constructed."""


class StabilityEvaluationError(
    FeaturePruningStabilityError
):
    """Raised when a model evaluation fails."""


@dataclass(frozen=True)
class StabilityConfig:
    """Complete multi-window validation configuration."""

    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    removal_sequence: tuple[str, ...] = (
        DEFAULT_REMOVAL_SEQUENCE
    )
    window_size: int = DEFAULT_WINDOW_SIZE
    max_training_targets: int = (
        DEFAULT_MAX_TRAINING_TARGETS
    )
    validation_targets: int = (
        DEFAULT_VALIDATION_TARGETS
    )
    top_k: int = DEFAULT_TOP_K
    purge_targets: int = DEFAULT_PURGE_TARGETS
    tolerance: float = DEFAULT_TOLERANCE
    window_count: int = DEFAULT_WINDOW_COUNT
    window_step_targets: int = (
        DEFAULT_WINDOW_STEP_TARGETS
    )
    minimum_stability_rate: float = (
        DEFAULT_MINIMUM_STABILITY_RATE
    )
    minimum_training_targets: int = (
        DEFAULT_MINIMUM_TRAINING_TARGETS
    )

    def validated(self) -> "StabilityConfig":
        """Validate and return this immutable configuration."""

        validate_config(self)
        return self


@dataclass(frozen=True)
class TemporalWindow:
    """One chronological training/purge/validation definition."""

    window_id: int
    training_target_indices: tuple[int, ...]
    purged_target_indices: tuple[int, ...]
    validation_target_indices: tuple[int, ...]

    @property
    def training_target_count(self) -> int:
        return len(
            self.training_target_indices
        )

    @property
    def validation_target_count(self) -> int:
        return len(
            self.validation_target_indices
        )

    @property
    def first_training_target(self) -> int:
        return self.training_target_indices[0]

    @property
    def last_training_target(self) -> int:
        return self.training_target_indices[-1]

    @property
    def first_validation_target(self) -> int:
        return self.validation_target_indices[0]

    @property
    def last_validation_target(self) -> int:
        return self.validation_target_indices[-1]


@dataclass(frozen=True)
class StabilityDatasetSummary:
    """Dataset and generated-window metadata."""

    draw_count: int
    dataset_rows: int
    dataset_targets: int
    first_dataset_target: int
    last_dataset_target: int
    requested_window_count: int
    generated_window_count: int
    validation_targets_per_window: int
    window_step_targets: int
    purge_targets_per_window: int
    minimum_training_targets: int


@dataclass(frozen=True)
class StabilityStepResult:
    """One cumulative removal step inside one window."""

    window_id: int
    step_number: int
    removed_feature: str
    baseline_features: tuple[str, ...]
    candidate_features: tuple[str, ...]
    baseline_run: FeatureAblationRunResult
    candidate_run: FeatureAblationRunResult
    comparison: FeatureAblationComparison
    decision: str


@dataclass(frozen=True)
class StabilityWindowResult:
    """Complete evaluation result for one temporal window."""

    window: TemporalWindow
    training_rows: int
    validation_rows: int
    baseline_run: FeatureAblationRunResult
    step_results: tuple[StabilityStepResult, ...]
    final_run: FeatureAblationRunResult
    final_features: tuple[str, ...]
    all_steps_accepted: bool
    total_absolute_delta: float
    total_relative_delta: float | None


@dataclass(frozen=True)
class StabilityStepAggregate:
    """Cross-window stability metrics for one removal step."""

    step_number: int
    removed_feature: str
    evaluated_window_count: int
    accepted_window_count: int
    acceptance_rate: float
    mean_baseline_hits_at_k: float
    mean_candidate_hits_at_k: float
    mean_absolute_delta: float
    minimum_absolute_delta: float
    maximum_absolute_delta: float
    stable: bool


@dataclass(frozen=True)
class FeaturePruningStabilityReport:
    """Complete Sprint 4 report."""

    status: str
    version: str
    protocol: str
    stable: bool
    recommendation: str
    removal_sequence: tuple[str, ...]
    final_features: tuple[str, ...]
    tolerance: float
    minimum_stability_rate: float
    window_size: int
    max_training_targets: int
    validation_targets: int
    top_k: int
    purge_targets: int
    window_count: int
    window_step_targets: int
    minimum_training_targets: int
    dataset: StabilityDatasetSummary
    windows: tuple[StabilityWindowResult, ...]
    step_aggregates: tuple[StabilityStepAggregate, ...]
    final_model_accepted_window_count: int
    final_model_acceptance_rate: float
    mean_baseline_hits_at_k: float
    mean_final_hits_at_k: float
    mean_total_absolute_delta: float
    minimum_total_absolute_delta: float
    maximum_total_absolute_delta: float


def _normalise_feature_sequence(
    values: Sequence[str],
    *,
    field_name: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Validate, strip, preserve order, and reject duplicates."""

    if isinstance(
        values,
        (str, bytes, bytearray),
    ):
        raise StabilityConfigurationError(
            f"{field_name} must be a sequence of feature names"
        )

    normalised: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(value, str):
            raise StabilityConfigurationError(
                f"{field_name} values must be strings"
            )

        feature = value.strip()

        if not feature:
            raise StabilityConfigurationError(
                f"{field_name} values cannot be empty"
            )

        if feature in seen:
            raise StabilityConfigurationError(
                f"duplicate feature in {field_name}: {feature}"
            )

        seen.add(feature)
        normalised.append(feature)

    if not normalised and not allow_empty:
        raise StabilityConfigurationError(
            f"{field_name} cannot be empty"
        )

    return tuple(normalised)


def validate_config(
    config: StabilityConfig,
) -> None:
    """Validate every Sprint 4 parameter."""

    if not isinstance(
        config.output_directory,
        Path,
    ):
        raise StabilityConfigurationError(
            "output_directory must be a pathlib.Path"
        )

    sequence = _normalise_feature_sequence(
        config.removal_sequence,
        field_name="removal_sequence",
        allow_empty=False,
    )

    model_features = tuple(
        V7RankingDataset.feature_columns()
    )
    model_feature_set = set(
        model_features
    )

    unknown = sorted(
        set(sequence)
        - model_feature_set
    )

    if unknown:
        raise StabilityConfigurationError(
            f"unknown removal features: {unknown}"
        )

    if len(sequence) >= len(model_features):
        raise StabilityConfigurationError(
            "removal_sequence must leave at least one model feature"
        )

    if config.window_size < 100:
        raise StabilityConfigurationError(
            "window_size must be at least 100"
        )

    if config.max_training_targets < 0:
        raise StabilityConfigurationError(
            "max_training_targets cannot be negative"
        )

    if config.validation_targets < 5:
        raise StabilityConfigurationError(
            "validation_targets must be at least 5"
        )

    if not 1 <= config.top_k <= 49:
        raise StabilityConfigurationError(
            "top_k must be between 1 and 49"
        )

    if config.purge_targets < 0:
        raise StabilityConfigurationError(
            "purge_targets cannot be negative"
        )

    if not math.isfinite(
        config.tolerance
    ):
        raise StabilityConfigurationError(
            "tolerance must be finite"
        )

    if config.tolerance < 0:
        raise StabilityConfigurationError(
            "tolerance cannot be negative"
        )

    if config.window_count < 2:
        raise StabilityConfigurationError(
            "window_count must be at least 2"
        )

    if config.window_step_targets < 1:
        raise StabilityConfigurationError(
            "window_step_targets must be at least 1"
        )

    if not math.isfinite(
        config.minimum_stability_rate
    ):
        raise StabilityConfigurationError(
            "minimum_stability_rate must be finite"
        )

    if not 0.0 < config.minimum_stability_rate <= 1.0:
        raise StabilityConfigurationError(
            "minimum_stability_rate must be greater than 0 and at most 1"
        )

    if config.minimum_training_targets < 1:
        raise StabilityConfigurationError(
            "minimum_training_targets must be at least 1"
        )


def build_temporal_windows(
    *,
    target_indices: Sequence[int],
    validation_targets: int,
    purge_targets: int,
    window_count: int,
    window_step_targets: int,
    minimum_training_targets: int,
) -> tuple[TemporalWindow, ...]:
    """Build newest-to-oldest chronological holdout windows."""

    if isinstance(
        target_indices,
        (str, bytes, bytearray),
    ):
        raise StabilityDatasetError(
            "target_indices must be a sequence"
        )

    ordered: list[int] = []
    seen: set[int] = set()

    for raw_value in target_indices:
        try:
            value = int(raw_value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise StabilityDatasetError(
                f"invalid target index: {raw_value!r}"
            ) from exc

        if value in seen:
            continue

        seen.add(value)
        ordered.append(value)

    ordered.sort()

    if not ordered:
        raise StabilityDatasetError(
            "no target indices are available"
        )

    windows: list[TemporalWindow] = []

    for offset in range(
        window_count
    ):
        validation_end = (
            len(ordered)
            - (
                offset
                * window_step_targets
            )
        )
        validation_start = (
            validation_end
            - validation_targets
        )
        purge_start = (
            validation_start
            - purge_targets
        )

        if validation_start < 0:
            break

        if purge_start < 0:
            break

        training_indices = tuple(
            ordered[:purge_start]
        )
        purged_indices = tuple(
            ordered[
                purge_start:
                validation_start
            ]
        )
        validation_indices = tuple(
            ordered[
                validation_start:
                validation_end
            ]
        )

        if (
            len(training_indices)
            < minimum_training_targets
        ):
            break

        if (
            len(validation_indices)
            != validation_targets
        ):
            break

        if len(purged_indices) != purge_targets:
            break

        if (
            training_indices
            and validation_indices
            and training_indices[-1]
            >= validation_indices[0]
        ):
            raise StabilityDatasetError(
                "training targets must precede validation targets"
            )

        windows.append(
            TemporalWindow(
                window_id=offset + 1,
                training_target_indices=(
                    training_indices
                ),
                purged_target_indices=(
                    purged_indices
                ),
                validation_target_indices=(
                    validation_indices
                ),
            )
        )

    if len(windows) != window_count:
        required = (
            minimum_training_targets
            + purge_targets
            + validation_targets
            + (
                (window_count - 1)
                * window_step_targets
            )
        )

        raise StabilityDatasetError(
            "Not enough chronological targets to build "
            f"{window_count} windows. Available={len(ordered)}, "
            f"approximately required={required}."
        )

    return tuple(windows)


def prepare_dataset_and_windows(
    config: StabilityConfig,
) -> tuple[
    pd.DataFrame,
    StabilityDatasetSummary,
    tuple[TemporalWindow, ...],
]:
    """Load draws, build the ranking dataset once, and define windows."""

    config.validated()
    database = SessionLocal()

    try:
        draws = (
            V7FeatureAblationReport
            ._load_draws(
                database
            )
        )

        dataset, metadata = (
            V7RankingDataset()
            .build_from_draws(
                draws=draws,
                window_size=(
                    config.window_size
                ),
                max_training_targets=(
                    config.max_training_targets
                ),
            )
        )
    except Exception as exc:
        raise StabilityDatasetError(
            "Unable to build the V7 ranking dataset"
        ) from exc
    finally:
        database.close()

    try:
        (
            V7FeatureAblationReport
            ._validate_dataset(
                dataset
            )
        )
    except Exception as exc:
        raise StabilityDatasetError(
            "The generated V7 ranking dataset is invalid"
        ) from exc

    if (
        "target_draw_index"
        not in dataset.columns
    ):
        raise StabilityDatasetError(
            "dataset is missing target_draw_index"
        )

    target_indices = tuple(
        sorted(
            int(value)
            for value
            in dataset[
                "target_draw_index"
            ].unique()
        )
    )

    windows = build_temporal_windows(
        target_indices=target_indices,
        validation_targets=(
            config.validation_targets
        ),
        purge_targets=(
            config.purge_targets
        ),
        window_count=(
            config.window_count
        ),
        window_step_targets=(
            config.window_step_targets
        ),
        minimum_training_targets=(
            config.minimum_training_targets
        ),
    )

    summary = StabilityDatasetSummary(
        draw_count=len(draws),
        dataset_rows=len(dataset),
        dataset_targets=len(metadata),
        first_dataset_target=(
            target_indices[0]
        ),
        last_dataset_target=(
            target_indices[-1]
        ),
        requested_window_count=(
            config.window_count
        ),
        generated_window_count=len(
            windows
        ),
        validation_targets_per_window=(
            config.validation_targets
        ),
        window_step_targets=(
            config.window_step_targets
        ),
        purge_targets_per_window=(
            config.purge_targets
        ),
        minimum_training_targets=(
            config.minimum_training_targets
        ),
    )

    return (
        dataset,
        summary,
        windows,
    )


def split_dataset_for_window(
    dataset: pd.DataFrame,
    window: TemporalWindow,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Materialise one window's chronological train and validation frames."""

    if not isinstance(
        dataset,
        pd.DataFrame,
    ):
        raise StabilityDatasetError(
            "dataset must be a pandas DataFrame"
        )

    if (
        "target_draw_index"
        not in dataset.columns
    ):
        raise StabilityDatasetError(
            "dataset is missing target_draw_index"
        )

    training = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(
                window.training_target_indices
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    validation = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(
                window.validation_target_indices
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    training_targets = tuple(
        sorted(
            int(value)
            for value
            in training[
                "target_draw_index"
            ].unique()
        )
    )
    validation_targets = tuple(
        sorted(
            int(value)
            for value
            in validation[
                "target_draw_index"
            ].unique()
        )
    )

    if training_targets != (
        window.training_target_indices
    ):
        raise StabilityDatasetError(
            f"window {window.window_id} training targets are incomplete"
        )

    if validation_targets != (
        window.validation_target_indices
    ):
        raise StabilityDatasetError(
            f"window {window.window_id} validation targets are incomplete"
        )

    if training.empty:
        raise StabilityDatasetError(
            f"window {window.window_id} training dataset is empty"
        )

    if validation.empty:
        raise StabilityDatasetError(
            f"window {window.window_id} validation dataset is empty"
        )

    return (
        training,
        validation,
    )


def evaluate_window(
    *,
    dataset: pd.DataFrame,
    window: TemporalWindow,
    config: StabilityConfig,
) -> StabilityWindowResult:
    """Evaluate baseline and cumulative removals on one window."""

    training, validation = (
        split_dataset_for_window(
            dataset,
            window,
        )
    )

    baseline_features = tuple(
        V7RankingDataset.feature_columns()
    )

    try:
        baseline_run = run_feature_subset(
            training_dataset=training,
            validation_dataset=validation,
            config=build_subset_config(
                experiment_name=(
                    f"stability_w{window.window_id}_baseline"
                ),
                feature_columns=(
                    baseline_features
                ),
                top_k=config.top_k,
            ),
        )
    except Exception as exc:
        raise StabilityEvaluationError(
            "Baseline evaluation failed for "
            f"window {window.window_id}"
        ) from exc

    current_run = baseline_run
    current_features = (
        baseline_features
    )
    step_results: list[
        StabilityStepResult
    ] = []

    for step_number, feature in enumerate(
        config.removal_sequence,
        start=1,
    ):
        if feature not in current_features:
            raise StabilityEvaluationError(
                f"feature {feature} is not active before "
                f"window {window.window_id} step {step_number}"
            )

        candidate_features = tuple(
            active
            for active in current_features
            if active != feature
        )

        try:
            candidate_run = run_feature_subset(
                training_dataset=training,
                validation_dataset=validation,
                config=build_subset_config(
                    experiment_name=(
                        f"stability_w{window.window_id}_"
                        f"s{step_number}_without_{feature}"
                    ),
                    feature_columns=(
                        candidate_features
                    ),
                    top_k=config.top_k,
                ),
            )

            comparison = compare_feature_runs(
                baseline=current_run,
                candidate=candidate_run,
                tolerance=config.tolerance,
            )
        except Exception as exc:
            raise StabilityEvaluationError(
                "Cumulative evaluation failed for "
                f"window {window.window_id}, step {step_number}, "
                f"feature {feature}"
            ) from exc

        step_results.append(
            StabilityStepResult(
                window_id=window.window_id,
                step_number=step_number,
                removed_feature=feature,
                baseline_features=(
                    current_features
                ),
                candidate_features=(
                    candidate_features
                ),
                baseline_run=current_run,
                candidate_run=(
                    candidate_run
                ),
                comparison=comparison,
                decision=(
                    "ACCEPT"
                    if comparison.accepted
                    else "REJECT"
                ),
            )
        )

        current_run = candidate_run
        current_features = (
            candidate_features
        )

    total_delta = (
        current_run.mean_hits_at_k
        - baseline_run.mean_hits_at_k
    )

    total_relative_delta = (
        total_delta
        / baseline_run.mean_hits_at_k
        if baseline_run.mean_hits_at_k != 0
        else None
    )

    return StabilityWindowResult(
        window=window,
        training_rows=len(training),
        validation_rows=len(validation),
        baseline_run=baseline_run,
        step_results=tuple(
            step_results
        ),
        final_run=current_run,
        final_features=current_features,
        all_steps_accepted=all(
            result.comparison.accepted
            for result in step_results
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


def _mean(
    values: Sequence[float],
) -> float:
    """Return an arithmetic mean with explicit empty validation."""

    if not values:
        raise StabilityEvaluationError(
            "cannot calculate a mean from an empty sequence"
        )

    return float(
        sum(values)
        / len(values)
    )


def aggregate_step_results(
    *,
    window_results: Sequence[
        StabilityWindowResult
    ],
    removal_sequence: Sequence[str],
    minimum_stability_rate: float,
) -> tuple[StabilityStepAggregate, ...]:
    """Aggregate every cumulative step across all windows."""

    if not window_results:
        raise StabilityEvaluationError(
            "window_results cannot be empty"
        )

    aggregates: list[
        StabilityStepAggregate
    ] = []

    for step_number, feature in enumerate(
        removal_sequence,
        start=1,
    ):
        matching: list[
            StabilityStepResult
        ] = []

        for window_result in (
            window_results
        ):
            found = next(
                (
                    result
                    for result
                    in window_result.step_results
                    if (
                        result.step_number
                        == step_number
                        and result.removed_feature
                        == feature
                    )
                ),
                None,
            )

            if found is None:
                raise StabilityEvaluationError(
                    f"missing step {step_number} ({feature}) "
                    f"for window {window_result.window.window_id}"
                )

            matching.append(found)

        accepted_count = sum(
            result.comparison.accepted
            for result in matching
        )
        acceptance_rate = (
            accepted_count
            / len(matching)
        )
        baseline_scores = [
            result.baseline_run.mean_hits_at_k
            for result in matching
        ]
        candidate_scores = [
            result.candidate_run.mean_hits_at_k
            for result in matching
        ]
        deltas = [
            result.comparison.absolute_delta
            for result in matching
        ]

        aggregates.append(
            StabilityStepAggregate(
                step_number=step_number,
                removed_feature=feature,
                evaluated_window_count=len(
                    matching
                ),
                accepted_window_count=(
                    accepted_count
                ),
                acceptance_rate=float(
                    acceptance_rate
                ),
                mean_baseline_hits_at_k=(
                    _mean(
                        baseline_scores
                    )
                ),
                mean_candidate_hits_at_k=(
                    _mean(
                        candidate_scores
                    )
                ),
                mean_absolute_delta=(
                    _mean(
                        deltas
                    )
                ),
                minimum_absolute_delta=float(
                    min(deltas)
                ),
                maximum_absolute_delta=float(
                    max(deltas)
                ),
                stable=(
                    acceptance_rate
                    >= minimum_stability_rate
                ),
            )
        )

    return tuple(aggregates)


def run_stability_validation(
    config: StabilityConfig,
) -> FeaturePruningStabilityReport:
    """Execute the complete multi-window stability protocol."""

    config.validated()

    (
        dataset,
        dataset_summary,
        windows,
    ) = prepare_dataset_and_windows(
        config
    )

    window_results = tuple(
        evaluate_window(
            dataset=dataset,
            window=window,
            config=config,
        )
        for window in windows
    )

    step_aggregates = (
        aggregate_step_results(
            window_results=(
                window_results
            ),
            removal_sequence=(
                config.removal_sequence
            ),
            minimum_stability_rate=(
                config.minimum_stability_rate
            ),
        )
    )

    final_accepted_count = sum(
        (
            result.final_run.mean_hits_at_k
            >= (
                result.baseline_run.mean_hits_at_k
                - config.tolerance
            )
        )
        for result in window_results
    )
    final_acceptance_rate = (
        final_accepted_count
        / len(window_results)
    )

    baseline_scores = [
        result.baseline_run.mean_hits_at_k
        for result in window_results
    ]
    final_scores = [
        result.final_run.mean_hits_at_k
        for result in window_results
    ]
    total_deltas = [
        result.total_absolute_delta
        for result in window_results
    ]

    stable = (
        all(
            aggregate.stable
            for aggregate
            in step_aggregates
        )
        and final_acceptance_rate
        >= config.minimum_stability_rate
        and _mean(total_deltas)
        >= -config.tolerance
    )

    initial_features = tuple(
        V7RankingDataset.feature_columns()
    )
    final_features = tuple(
        feature
        for feature in initial_features
        if feature
        not in set(
            config.removal_sequence
        )
    )

    recommendation = (
        "ACCEPT_CUMULATIVE_PRUNING"
        if stable
        else "REJECT_OR_REVIEW_CUMULATIVE_PRUNING"
    )

    return FeaturePruningStabilityReport(
        status="success",
        version=VERSION,
        protocol=(
            "multi-window chronological validation; "
            f"{config.purge_targets} target(s) purged before "
            "every validation window; cumulative sequence "
            "evaluated from the all-feature baseline on each "
            "window independently"
        ),
        stable=stable,
        recommendation=(
            recommendation
        ),
        removal_sequence=(
            config.removal_sequence
        ),
        final_features=(
            final_features
        ),
        tolerance=config.tolerance,
        minimum_stability_rate=(
            config.minimum_stability_rate
        ),
        window_size=config.window_size,
        max_training_targets=(
            config.max_training_targets
        ),
        validation_targets=(
            config.validation_targets
        ),
        top_k=config.top_k,
        purge_targets=(
            config.purge_targets
        ),
        window_count=(
            config.window_count
        ),
        window_step_targets=(
            config.window_step_targets
        ),
        minimum_training_targets=(
            config.minimum_training_targets
        ),
        dataset=dataset_summary,
        windows=window_results,
        step_aggregates=(
            step_aggregates
        ),
        final_model_accepted_window_count=(
            final_accepted_count
        ),
        final_model_acceptance_rate=float(
            final_acceptance_rate
        ),
        mean_baseline_hits_at_k=(
            _mean(
                baseline_scores
            )
        ),
        mean_final_hits_at_k=(
            _mean(
                final_scores
            )
        ),
        mean_total_absolute_delta=(
            _mean(
                total_deltas
            )
        ),
        minimum_total_absolute_delta=float(
            min(total_deltas)
        ),
        maximum_total_absolute_delta=float(
            max(total_deltas)
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

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


def _write_csv(
    *,
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write one deterministic UTF-8 CSV file."""

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


def export_stability_report(
    *,
    report: FeaturePruningStabilityReport,
    output_directory: Path,
) -> dict[str, Path]:
    """Export JSON, text, window CSV, and step CSV reports."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "feature_pruning_stability.json"
    )
    text_path = (
        output_directory
        / "feature_pruning_stability.txt"
    )
    windows_csv_path = (
        output_directory
        / "feature_pruning_stability_windows.csv"
    )
    steps_csv_path = (
        output_directory
        / "feature_pruning_stability_steps.csv"
    )
    recommendation_path = (
        output_directory
        / "feature_pruning_stability_recommendation.txt"
    )

    json_path.write_text(
        json.dumps(
            _json_safe(
                asdict(report)
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    window_rows: list[
        dict[str, Any]
    ] = []
    step_rows: list[
        dict[str, Any]
    ] = []

    for window_result in report.windows:
        window = (
            window_result.window
        )

        window_rows.append(
            {
                "window_id": window.window_id,
                "training_target_count": (
                    window.training_target_count
                ),
                "first_training_target": (
                    window.first_training_target
                ),
                "last_training_target": (
                    window.last_training_target
                ),
                "purged_target_indices": "|".join(
                    str(value)
                    for value
                    in window.purged_target_indices
                ),
                "validation_target_count": (
                    window.validation_target_count
                ),
                "first_validation_target": (
                    window.first_validation_target
                ),
                "last_validation_target": (
                    window.last_validation_target
                ),
                "baseline_mean_hits_at_k": (
                    window_result
                    .baseline_run
                    .mean_hits_at_k
                ),
                "final_mean_hits_at_k": (
                    window_result
                    .final_run
                    .mean_hits_at_k
                ),
                "total_absolute_delta": (
                    window_result
                    .total_absolute_delta
                ),
                "total_relative_delta": (
                    window_result
                    .total_relative_delta
                ),
                "all_steps_accepted": (
                    window_result
                    .all_steps_accepted
                ),
            }
        )

        for step in (
            window_result.step_results
        ):
            step_rows.append(
                {
                    "window_id": (
                        step.window_id
                    ),
                    "step_number": (
                        step.step_number
                    ),
                    "removed_feature": (
                        step.removed_feature
                    ),
                    "baseline_feature_count": len(
                        step.baseline_features
                    ),
                    "candidate_feature_count": len(
                        step.candidate_features
                    ),
                    "baseline_mean_hits_at_k": (
                        step
                        .baseline_run
                        .mean_hits_at_k
                    ),
                    "candidate_mean_hits_at_k": (
                        step
                        .candidate_run
                        .mean_hits_at_k
                    ),
                    "absolute_delta": (
                        step
                        .comparison
                        .absolute_delta
                    ),
                    "relative_delta": (
                        step
                        .comparison
                        .relative_delta
                    ),
                    "candidate_target_hit_rate": (
                        step
                        .candidate_run
                        .target_hit_rate
                    ),
                    "candidate_total_hits": (
                        step
                        .candidate_run
                        .total_hits
                    ),
                    "candidate_total_seconds": (
                        step
                        .candidate_run
                        .total_seconds
                    ),
                    "decision": (
                        step.decision
                    ),
                }
            )

    _write_csv(
        path=windows_csv_path,
        fieldnames=(
            "window_id",
            "training_target_count",
            "first_training_target",
            "last_training_target",
            "purged_target_indices",
            "validation_target_count",
            "first_validation_target",
            "last_validation_target",
            "baseline_mean_hits_at_k",
            "final_mean_hits_at_k",
            "total_absolute_delta",
            "total_relative_delta",
            "all_steps_accepted",
        ),
        rows=window_rows,
    )

    _write_csv(
        path=steps_csv_path,
        fieldnames=(
            "window_id",
            "step_number",
            "removed_feature",
            "baseline_feature_count",
            "candidate_feature_count",
            "baseline_mean_hits_at_k",
            "candidate_mean_hits_at_k",
            "absolute_delta",
            "relative_delta",
            "candidate_target_hit_rate",
            "candidate_total_hits",
            "candidate_total_seconds",
            "decision",
        ),
        rows=step_rows,
    )

    lines = [
        "=" * 144,
        "PREDIXA AI V7 FEATURE PRUNING MULTI-WINDOW STABILITY",
        "=" * 144,
        f"Status                         : {report.status}",
        f"Version                        : {report.version}",
        f"Protocol                       : {report.protocol}",
        f"Stable                         : {report.stable}",
        f"Recommendation                 : {report.recommendation}",
        (
            "Removal sequence               : "
            + " -> ".join(
                report.removal_sequence
            )
        ),
        (
            "Minimum stability rate         : "
            f"{report.minimum_stability_rate:.6f}"
        ),
        (
            "Acceptance tolerance           : "
            f"{report.tolerance:.6f}"
        ),
        (
            "Generated windows              : "
            f"{report.dataset.generated_window_count}"
        ),
        (
            "Validation targets/window      : "
            f"{report.validation_targets}"
        ),
        (
            "Window step targets            : "
            f"{report.window_step_targets}"
        ),
        (
            "Purged targets/window          : "
            f"{report.purge_targets}"
        ),
        "",
        "WINDOW RESULTS",
        "-" * 144,
        (
            f"{'Window':>8}"
            f"{'Train':>10}"
            f"{'Validation range':>24}"
            f"{'Baseline':>14}"
            f"{'Final':>14}"
            f"{'Delta':>14}"
            f"{'All steps':>14}"
        ),
        "-" * 144,
    ]

    for window_result in report.windows:
        window = (
            window_result.window
        )

        lines.append(
            f"{window.window_id:>8}"
            f"{window.training_target_count:>10}"
            f"{(
                str(window.first_validation_target)
                + ' -> '
                + str(window.last_validation_target)
            ):>24}"
            f"{window_result.baseline_run.mean_hits_at_k:>14.6f}"
            f"{window_result.final_run.mean_hits_at_k:>14.6f}"
            f"{window_result.total_absolute_delta:>14.6f}"
            f"{str(window_result.all_steps_accepted):>14}"
        )

    lines.extend(
        [
            "",
            "STEP STABILITY",
            "-" * 144,
            (
                f"{'Step':>8}"
                f"{'Removed feature':>28}"
                f"{'Accepted':>14}"
                f"{'Rate':>12}"
                f"{'Mean baseline':>16}"
                f"{'Mean candidate':>18}"
                f"{'Mean delta':>14}"
                f"{'Stable':>12}"
            ),
            "-" * 144,
        ]
    )

    for aggregate in (
        report.step_aggregates
    ):
        lines.append(
            f"{aggregate.step_number:>8}"
            f"{aggregate.removed_feature:>28}"
            f"{(
                str(aggregate.accepted_window_count)
                + '/'
                + str(aggregate.evaluated_window_count)
            ):>14}"
            f"{aggregate.acceptance_rate:>12.6f}"
            f"{aggregate.mean_baseline_hits_at_k:>16.6f}"
            f"{aggregate.mean_candidate_hits_at_k:>18.6f}"
            f"{aggregate.mean_absolute_delta:>14.6f}"
            f"{str(aggregate.stable):>12}"
        )

    lines.extend(
        [
            "",
            "FINAL MODEL STABILITY",
            "-" * 144,
            (
                "Accepted windows               : "
                f"{report.final_model_accepted_window_count}"
                f"/{report.window_count}"
            ),
            (
                "Acceptance rate                : "
                f"{report.final_model_acceptance_rate:.6f}"
            ),
            (
                "Mean baseline Hits@K           : "
                f"{report.mean_baseline_hits_at_k:.6f}"
            ),
            (
                "Mean final Hits@K              : "
                f"{report.mean_final_hits_at_k:.6f}"
            ),
            (
                "Mean total delta               : "
                f"{report.mean_total_absolute_delta:.6f}"
            ),
            (
                "Minimum window delta           : "
                f"{report.minimum_total_absolute_delta:.6f}"
            ),
            (
                "Maximum window delta           : "
                f"{report.maximum_total_absolute_delta:.6f}"
            ),
            (
                "Final features                 : "
                + ", ".join(
                    report.final_features
                )
            ),
            "",
            (
                "This stability report is a recommendation only. "
                "It does not modify V7RankingDataset.feature_columns()."
            ),
            "=" * 144,
        ]
    )

    text_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    recommendation_path.write_text(
        "\n".join(
            [
                f"recommendation={report.recommendation}",
                f"stable={str(report.stable).lower()}",
                (
                    "removal_sequence="
                    + ",".join(
                        report.removal_sequence
                    )
                ),
                (
                    "final_features="
                    + ",".join(
                        report.final_features
                    )
                ),
                (
                    "final_model_acceptance_rate="
                    f"{report.final_model_acceptance_rate:.6f}"
                ),
                (
                    "mean_total_absolute_delta="
                    f"{report.mean_total_absolute_delta:.6f}"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "json": json_path.resolve(),
        "text": text_path.resolve(),
        "windows_csv": (
            windows_csv_path.resolve()
        ),
        "steps_csv": (
            steps_csv_path.resolve()
        ),
        "recommendation": (
            recommendation_path.resolve()
        ),
    }


def print_stability_report(
    *,
    report: FeaturePruningStabilityReport,
    generated_files: Mapping[str, Path],
) -> None:
    """Print a compact console summary."""

    print("=" * 136)
    print(
        "PREDIXA AI V7 FEATURE PRUNING "
        "MULTI-WINDOW STABILITY"
    )
    print("=" * 136)
    print(
        f"Status                  : "
        f"{report.status}"
    )
    print(
        f"Stable                  : "
        f"{report.stable}"
    )
    print(
        f"Recommendation          : "
        f"{report.recommendation}"
    )
    print(
        "Removal sequence        : "
        + " -> ".join(
            report.removal_sequence
        )
    )
    print(
        f"Windows                 : "
        f"{report.window_count}"
    )
    print(
        f"Minimum stability rate  : "
        f"{report.minimum_stability_rate:.6f}"
    )
    print()

    print(
        f"{'Window':>8}"
        f"{'Train targets':>16}"
        f"{'Validation range':>24}"
        f"{'Baseline':>14}"
        f"{'Final':>14}"
        f"{'Delta':>14}"
    )
    print("-" * 136)

    for window_result in report.windows:
        window = (
            window_result.window
        )

        print(
            f"{window.window_id:>8}"
            f"{window.training_target_count:>16}"
            f"{(
                str(window.first_validation_target)
                + ' -> '
                + str(window.last_validation_target)
            ):>24}"
            f"{window_result.baseline_run.mean_hits_at_k:>14.6f}"
            f"{window_result.final_run.mean_hits_at_k:>14.6f}"
            f"{window_result.total_absolute_delta:>14.6f}"
        )

    print()
    print("STEP ACCEPTANCE")
    print("-" * 136)

    for aggregate in (
        report.step_aggregates
    ):
        print(
            f"{aggregate.step_number}. "
            f"{aggregate.removed_feature}: "
            f"{aggregate.accepted_window_count}/"
            f"{aggregate.evaluated_window_count} "
            f"({aggregate.acceptance_rate:.6f}), "
            f"mean delta={aggregate.mean_absolute_delta:.6f}, "
            f"stable={aggregate.stable}"
        )

    print()
    print(
        f"Final acceptance rate   : "
        f"{report.final_model_acceptance_rate:.6f}"
    )
    print(
        f"Mean baseline Hits@K    : "
        f"{report.mean_baseline_hits_at_k:.6f}"
    )
    print(
        f"Mean final Hits@K       : "
        f"{report.mean_final_hits_at_k:.6f}"
    )
    print(
        f"Mean total delta        : "
        f"{report.mean_total_absolute_delta:.6f}"
    )
    print()
    print("GENERATED FILES")
    print("-" * 136)

    for name, path in (
        generated_files.items()
    ):
        print(
            f"{name.upper():24}: "
            f"{path}"
        )

    print()
    print("=" * 136)
    print("SUCCESS")
    print("=" * 136)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the Sprint 4 command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate a cumulative PredixaAI V7 feature-pruning "
            "sequence across multiple chronological windows."
        )
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help=(
            "Directory used for stability exports"
        ),
    )

    parser.add_argument(
        "--removals",
        nargs="+",
        default=list(
            DEFAULT_REMOVAL_SEQUENCE
        ),
        help=(
            "Ordered cumulative feature-removal sequence"
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
            "Maximum target count built by V7RankingDataset; "
            "0 disables the limit"
        ),
    )

    parser.add_argument(
        "--validation-targets",
        type=int,
        default=(
            DEFAULT_VALIDATION_TARGETS
        ),
        help=(
            "Number of targets in each validation window"
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
            "Number of targets purged immediately before "
            "every validation window"
        ),
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Maximum accepted decrease in mean Hits@K"
        ),
    )

    parser.add_argument(
        "--window-count",
        type=int,
        default=DEFAULT_WINDOW_COUNT,
        help=(
            "Number of chronological validation windows"
        ),
    )

    parser.add_argument(
        "--window-step-targets",
        type=int,
        default=(
            DEFAULT_WINDOW_STEP_TARGETS
        ),
        help=(
            "Number of targets separating consecutive "
            "validation window endpoints"
        ),
    )

    parser.add_argument(
        "--minimum-stability-rate",
        type=float,
        default=(
            DEFAULT_MINIMUM_STABILITY_RATE
        ),
        help=(
            "Minimum fraction of windows that must accept "
            "each removal and the final model"
        ),
    )

    parser.add_argument(
        "--minimum-training-targets",
        type=int,
        default=(
            DEFAULT_MINIMUM_TRAINING_TARGETS
        ),
        help=(
            "Minimum chronological training targets required "
            "for the oldest window"
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    config = StabilityConfig(
        output_directory=(
            arguments
            .output_directory
            .expanduser()
            .resolve()
        ),
        removal_sequence=tuple(
            arguments.removals
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
        tolerance=(
            arguments.tolerance
        ),
        window_count=(
            arguments.window_count
        ),
        window_step_targets=(
            arguments.window_step_targets
        ),
        minimum_stability_rate=(
            arguments.minimum_stability_rate
        ),
        minimum_training_targets=(
            arguments.minimum_training_targets
        ),
    )

    try:
        report = (
            run_stability_validation(
                config
            )
        )

        generated_files = (
            export_stability_report(
                report=report,
                output_directory=(
                    config.output_directory
                ),
            )
        )

    except FeaturePruningError as exc:
        print("=" * 112)
        print(
            "PREDIXA AI V7 FEATURE PRUNING "
            "MULTI-WINDOW STABILITY"
        )
        print("=" * 112)
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

        print("=" * 112)

        return 1

    print_stability_report(
        report=report,
        generated_files=(
            generated_files
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
