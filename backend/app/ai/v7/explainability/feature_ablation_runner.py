from __future__ import annotations

"""
PredixaAI V7 - Reusable single-feature ablation runner.

This module evaluates an arbitrary feature subset on datasets that have
already been split chronologically by the validated V7 temporal pipeline.

Design goals
------------
- Do not modify V7RankingDataset.
- Do not modify V7FeatureAblationReport.
- Reuse the Random Forest factory from feature_ablation.py.
- Keep temporal slicing outside this module.
- Provide deterministic, serialisable evaluation results.
"""

from dataclasses import asdict, dataclass
from math import isfinite
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.ai.v7.explainability.feature_ablation import (
    V7FeatureAblationReport,
)
from app.ai.v7.ranking_dataset import V7RankingDataset


class FeatureAblationRunnerError(RuntimeError):
    """Base exception raised by the reusable ablation runner."""


class FeatureConfigurationError(FeatureAblationRunnerError):
    """Raised when a feature subset is invalid."""


class DatasetValidationError(FeatureAblationRunnerError):
    """Raised when an evaluation dataset is invalid."""


@dataclass(frozen=True)
class FeatureAblationRunConfig:
    """Configuration for one baseline or candidate evaluation."""

    experiment_name: str
    feature_columns: tuple[str, ...]
    top_k: int = 5

    def validated(self) -> "FeatureAblationRunConfig":
        validate_run_config(self)
        return self


@dataclass(frozen=True)
class TargetEvaluation:
    """Evaluation details for one validation target draw."""

    target_draw_index: int
    target_draw_date: str
    selected_numbers: tuple[int, ...]
    actual_numbers: tuple[int, ...]
    hits: int


@dataclass(frozen=True)
class FeatureAblationRunResult:
    """Serializable result of one feature-subset evaluation."""

    experiment_name: str
    feature_columns: tuple[str, ...]
    feature_count: int
    removed_features: tuple[str, ...]
    top_k: int
    training_rows: int
    validation_rows: int
    training_targets: int
    validation_targets: int
    fit_seconds: float
    prediction_seconds: float
    total_seconds: float
    total_hits: int
    mean_hits_at_k: float
    normalized_hits_at_k: float
    targets_with_at_least_one_hit: int
    target_hit_rate: float
    target_evaluations: tuple[TargetEvaluation, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return _json_safe(asdict(self))


@dataclass(frozen=True)
class FeatureAblationComparison:
    """Comparison between the baseline and one candidate run."""

    baseline_experiment: str
    candidate_experiment: str
    baseline_features: tuple[str, ...]
    candidate_features: tuple[str, ...]
    removed_features: tuple[str, ...]
    baseline_mean_hits_at_k: float
    candidate_mean_hits_at_k: float
    absolute_delta: float
    relative_delta: float | None
    accepted: bool
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary."""

        return _json_safe(asdict(self))


REQUIRED_DATASET_COLUMNS = (
    "candidate_number",
    "target",
    "target_draw_index",
    "target_draw_date",
)


def _normalise_feature_columns(
    feature_columns: Iterable[str],
) -> tuple[str, ...]:
    """Normalise a feature iterable while preserving its original order."""

    if isinstance(feature_columns, (str, bytes, bytearray)):
        raise FeatureConfigurationError(
            "feature_columns must be an iterable of feature names, "
            "not a single string."
        )

    normalised: list[str] = []
    seen: set[str] = set()

    for raw_feature in feature_columns:
        if not isinstance(raw_feature, str):
            raise FeatureConfigurationError(
                "Every feature name must be a string."
            )

        feature = raw_feature.strip()

        if not feature:
            raise FeatureConfigurationError(
                "Feature names cannot be empty."
            )

        if feature in seen:
            raise FeatureConfigurationError(
                f"Duplicate feature name: {feature}"
            )

        seen.add(feature)
        normalised.append(feature)

    return tuple(normalised)


def validate_run_config(config: FeatureAblationRunConfig) -> None:
    """Validate one evaluation configuration."""

    if not isinstance(config.experiment_name, str):
        raise FeatureConfigurationError(
            "experiment_name must be a string."
        )

    if not config.experiment_name.strip():
        raise FeatureConfigurationError(
            "experiment_name cannot be empty."
        )

    features = _normalise_feature_columns(config.feature_columns)

    if not features:
        raise FeatureConfigurationError(
            "At least one feature column is required."
        )

    model_features = tuple(V7RankingDataset.feature_columns())
    model_feature_set = set(model_features)
    unknown_features = sorted(set(features) - model_feature_set)

    if unknown_features:
        raise FeatureConfigurationError(
            "Unknown V7 model features: "
            f"{unknown_features}"
        )

    if not 1 <= config.top_k <= 49:
        raise FeatureConfigurationError(
            "top_k must be between 1 and 49."
        )


def validate_dataset(
    dataset: pd.DataFrame,
    feature_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Validate one candidate-level V7 dataset."""

    if not isinstance(dataset, pd.DataFrame):
        raise DatasetValidationError(
            f"{dataset_name} must be a pandas DataFrame."
        )

    if dataset.empty:
        raise DatasetValidationError(
            f"{dataset_name} cannot be empty."
        )

    required_columns = (
        *REQUIRED_DATASET_COLUMNS,
        *feature_columns,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:
        raise DatasetValidationError(
            f"{dataset_name} is missing columns: "
            f"{missing_columns}"
        )

    relevant_columns = list(dict.fromkeys(required_columns))

    if dataset[relevant_columns].isnull().any().any():
        null_columns = (
            dataset[relevant_columns]
            .columns[
                dataset[relevant_columns]
                .isnull()
                .any()
            ]
            .tolist()
        )

        raise DatasetValidationError(
            f"{dataset_name} contains missing values in: "
            f"{null_columns}"
        )

    numeric_columns = [
        "candidate_number",
        "target",
        "target_draw_index",
        *feature_columns,
    ]

    non_finite_columns: list[str] = []

    for column in numeric_columns:
        values = pd.to_numeric(
            dataset[column],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(values).all():
            non_finite_columns.append(column)

    if non_finite_columns:
        raise DatasetValidationError(
            f"{dataset_name} contains non-finite numeric values in: "
            f"{non_finite_columns}"
        )

    target_values = set(
        pd.to_numeric(
            dataset["target"],
            errors="raise",
        )
        .astype(int)
        .unique()
        .tolist()
    )

    if not target_values.issubset({0, 1}):
        raise DatasetValidationError(
            f"{dataset_name}.target must contain only 0 and 1."
        )

    candidate_numbers = pd.to_numeric(
        dataset["candidate_number"],
        errors="raise",
    ).astype(int)

    if not candidate_numbers.between(1, 49).all():
        raise DatasetValidationError(
            f"{dataset_name}.candidate_number must be between 1 and 49."
        )

    rows_per_target = (
        dataset
        .groupby("target_draw_index", sort=True)
        .size()
    )

    if not (rows_per_target == 49).all():
        invalid_targets = rows_per_target[
            rows_per_target != 49
        ].to_dict()

        raise DatasetValidationError(
            f"{dataset_name} must contain exactly 49 candidate rows "
            f"per target. Invalid targets: {invalid_targets}"
        )

    duplicate_candidates = (
        dataset
        .duplicated(
            subset=[
                "target_draw_index",
                "candidate_number",
            ],
            keep=False,
        )
    )

    if duplicate_candidates.any():
        raise DatasetValidationError(
            f"{dataset_name} contains duplicate candidate numbers "
            "inside at least one target."
        )


def validate_temporal_order(
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
) -> None:
    """
    Ensure validation targets strictly follow all training targets.

    The upstream splitter remains responsible for the T-2 / purge T-1
    protocol. This check prevents an accidental overlap or reversed split.
    """

    training_indices = (
        pd.to_numeric(
            training_dataset["target_draw_index"],
            errors="raise",
        )
        .astype(int)
        .unique()
    )

    validation_indices = (
        pd.to_numeric(
            validation_dataset["target_draw_index"],
            errors="raise",
        )
        .astype(int)
        .unique()
    )

    overlap = set(training_indices).intersection(validation_indices)

    if overlap:
        raise DatasetValidationError(
            "Training and validation target indices overlap: "
            f"{sorted(overlap)}"
        )

    if int(training_indices.max()) >= int(validation_indices.min()):
        raise DatasetValidationError(
            "Temporal order is invalid: every validation target must "
            "strictly follow every training target."
        )


def _positive_class_probabilities(
    model: RandomForestClassifier,
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    """Return probabilities for target class 1."""

    probabilities = model.predict_proba(feature_frame)
    classes = [int(value) for value in model.classes_.tolist()]

    if 1 in classes:
        positive_index = classes.index(1)
        return probabilities[:, positive_index].astype(float)

    if classes == [0]:
        return np.zeros(len(feature_frame), dtype=float)

    raise FeatureAblationRunnerError(
        f"Unsupported classifier classes: {classes}"
    )


def _evaluate_targets(
    validation_dataset: pd.DataFrame,
    probabilities: np.ndarray,
    top_k: int,
) -> tuple[TargetEvaluation, ...]:
    """Rank candidates and calculate Hits@K for each target draw."""

    scored = validation_dataset[
        [
            "candidate_number",
            "target",
            "target_draw_index",
            "target_draw_date",
        ]
    ].copy()

    scored["probability"] = probabilities

    evaluations: list[TargetEvaluation] = []

    for target_index, group in scored.groupby(
        "target_draw_index",
        sort=True,
    ):
        ranked = group.sort_values(
            by=[
                "probability",
                "candidate_number",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )

        selected_numbers = tuple(
            ranked
            .head(top_k)["candidate_number"]
            .astype(int)
            .tolist()
        )

        actual_numbers = tuple(
            sorted(
                group.loc[
                    group["target"].astype(int) == 1,
                    "candidate_number",
                ]
                .astype(int)
                .tolist()
            )
        )

        hits = len(
            set(selected_numbers)
            .intersection(actual_numbers)
        )

        target_date = str(
            group["target_draw_date"].iloc[0]
        )

        evaluations.append(
            TargetEvaluation(
                target_draw_index=int(target_index),
                target_draw_date=target_date,
                selected_numbers=selected_numbers,
                actual_numbers=actual_numbers,
                hits=hits,
            )
        )

    return tuple(evaluations)


def run_feature_subset(
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    config: FeatureAblationRunConfig,
) -> FeatureAblationRunResult:
    """
    Train and evaluate one arbitrary V7 feature subset.

    The datasets must already be produced and split by the validated
    chronological V7 pipeline.
    """

    config.validated()
    feature_columns = _normalise_feature_columns(
        config.feature_columns
    )

    validate_dataset(
        training_dataset,
        feature_columns,
        "training_dataset",
    )
    validate_dataset(
        validation_dataset,
        feature_columns,
        "validation_dataset",
    )
    validate_temporal_order(
        training_dataset,
        validation_dataset,
    )

    model_features = tuple(V7RankingDataset.feature_columns())
    removed_features = tuple(
        feature
        for feature in model_features
        if feature not in feature_columns
    )

    training_features = training_dataset[
        list(feature_columns)
    ]
    training_target = (
        training_dataset["target"]
        .astype(int)
    )
    validation_features = validation_dataset[
        list(feature_columns)
    ]

    model = V7FeatureAblationReport._build_model()

    started_at = perf_counter()

    fit_started_at = perf_counter()
    model.fit(
        training_features,
        training_target,
    )
    fit_seconds = perf_counter() - fit_started_at

    prediction_started_at = perf_counter()
    probabilities = _positive_class_probabilities(
        model,
        validation_features,
    )
    target_evaluations = _evaluate_targets(
        validation_dataset,
        probabilities,
        config.top_k,
    )
    prediction_seconds = (
        perf_counter()
        - prediction_started_at
    )

    total_seconds = perf_counter() - started_at
    total_hits = sum(
        evaluation.hits
        for evaluation in target_evaluations
    )
    validation_target_count = len(target_evaluations)

    mean_hits_at_k = (
        total_hits / validation_target_count
        if validation_target_count
        else 0.0
    )

    normalized_hits_at_k = (
        mean_hits_at_k / config.top_k
        if config.top_k
        else 0.0
    )

    targets_with_at_least_one_hit = sum(
        evaluation.hits > 0
        for evaluation in target_evaluations
    )

    target_hit_rate = (
        targets_with_at_least_one_hit
        / validation_target_count
        if validation_target_count
        else 0.0
    )

    return FeatureAblationRunResult(
        experiment_name=config.experiment_name.strip(),
        feature_columns=feature_columns,
        feature_count=len(feature_columns),
        removed_features=removed_features,
        top_k=config.top_k,
        training_rows=len(training_dataset),
        validation_rows=len(validation_dataset),
        training_targets=int(
            training_dataset[
                "target_draw_index"
            ].nunique()
        ),
        validation_targets=validation_target_count,
        fit_seconds=float(fit_seconds),
        prediction_seconds=float(
            prediction_seconds
        ),
        total_seconds=float(total_seconds),
        total_hits=int(total_hits),
        mean_hits_at_k=float(mean_hits_at_k),
        normalized_hits_at_k=float(
            normalized_hits_at_k
        ),
        targets_with_at_least_one_hit=int(
            targets_with_at_least_one_hit
        ),
        target_hit_rate=float(target_hit_rate),
        target_evaluations=target_evaluations,
    )


def compare_feature_runs(
    baseline: FeatureAblationRunResult,
    candidate: FeatureAblationRunResult,
    tolerance: float = 0.0,
) -> FeatureAblationComparison:
    """
    Compare one candidate against the baseline.

    A candidate is accepted when:
        candidate_mean_hits_at_k >= baseline_mean_hits_at_k - tolerance
    """

    if not isfinite(tolerance) or tolerance < 0:
        raise FeatureConfigurationError(
            "tolerance must be finite and non-negative."
        )

    if baseline.top_k != candidate.top_k:
        raise FeatureConfigurationError(
            "Baseline and candidate must use the same top_k."
        )

    if (
        baseline.validation_targets
        != candidate.validation_targets
    ):
        raise FeatureConfigurationError(
            "Baseline and candidate must evaluate the same number "
            "of validation targets."
        )

    baseline_features = tuple(
        baseline.feature_columns
    )
    candidate_features = tuple(
        candidate.feature_columns
    )

    removed_features = tuple(
        feature
        for feature in baseline_features
        if feature not in candidate_features
    )

    absolute_delta = (
        candidate.mean_hits_at_k
        - baseline.mean_hits_at_k
    )

    relative_delta = (
        absolute_delta
        / baseline.mean_hits_at_k
        if baseline.mean_hits_at_k != 0
        else None
    )

    accepted = (
        candidate.mean_hits_at_k
        >= baseline.mean_hits_at_k - tolerance
    )

    return FeatureAblationComparison(
        baseline_experiment=baseline.experiment_name,
        candidate_experiment=candidate.experiment_name,
        baseline_features=baseline_features,
        candidate_features=candidate_features,
        removed_features=removed_features,
        baseline_mean_hits_at_k=(
            baseline.mean_hits_at_k
        ),
        candidate_mean_hits_at_k=(
            candidate.mean_hits_at_k
        ),
        absolute_delta=float(absolute_delta),
        relative_delta=(
            float(relative_delta)
            if relative_delta is not None
            else None
        ),
        accepted=accepted,
        tolerance=float(tolerance),
    )


def build_baseline_config(
    top_k: int = 5,
) -> FeatureAblationRunConfig:
    """Build the canonical all-feature baseline configuration."""

    return FeatureAblationRunConfig(
        experiment_name="baseline",
        feature_columns=tuple(
            V7RankingDataset.feature_columns()
        ),
        top_k=top_k,
    )


def build_single_removal_config(
    feature_to_remove: str,
    top_k: int = 5,
) -> FeatureAblationRunConfig:
    """Build a configuration that removes exactly one V7 feature."""

    if not isinstance(feature_to_remove, str):
        raise FeatureConfigurationError(
            "feature_to_remove must be a string."
        )

    feature_to_remove = feature_to_remove.strip()
    model_features = tuple(
        V7RankingDataset.feature_columns()
    )

    if feature_to_remove not in model_features:
        raise FeatureConfigurationError(
            "Cannot remove an unknown V7 feature: "
            f"{feature_to_remove}"
        )

    remaining_features = tuple(
        feature
        for feature in model_features
        if feature != feature_to_remove
    )

    return FeatureAblationRunConfig(
        experiment_name=(
            f"without_{feature_to_remove}"
        ),
        feature_columns=remaining_features,
        top_k=top_k,
    )


def result_summary(
    result: FeatureAblationRunResult,
) -> str:
    """Render a compact human-readable result summary."""

    removed = (
        ", ".join(result.removed_features)
        if result.removed_features
        else "none"
    )

    return "\n".join(
        [
            "=" * 88,
            "PREDIXA AI V7 FEATURE ABLATION RUN",
            "=" * 88,
            f"Experiment              : {result.experiment_name}",
            f"Features                : {result.feature_count}",
            f"Removed                 : {removed}",
            f"Training targets        : {result.training_targets}",
            f"Validation targets      : {result.validation_targets}",
            f"Top K                   : {result.top_k}",
            f"Total hits              : {result.total_hits}",
            (
                "Mean Hits@K             : "
                f"{result.mean_hits_at_k:.6f}"
            ),
            (
                "Normalized Hits@K       : "
                f"{result.normalized_hits_at_k:.6f}"
            ),
            (
                "Targets with >= 1 hit   : "
                f"{result.targets_with_at_least_one_hit}"
            ),
            (
                "Target hit rate         : "
                f"{result.target_hit_rate:.6f}"
            ),
            (
                "Fit seconds             : "
                f"{result.fit_seconds:.6f}"
            ),
            (
                "Prediction seconds      : "
                f"{result.prediction_seconds:.6f}"
            ),
            (
                "Total seconds           : "
                f"{result.total_seconds:.6f}"
            ),
            "=" * 88,
        ]
    )


def _json_safe(value: Any) -> Any:
    """Recursively convert common objects to JSON-safe values."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

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

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, float) and not isfinite(value):
        return None

    return value


__all__ = [
    "DatasetValidationError",
    "FeatureAblationComparison",
    "FeatureAblationRunConfig",
    "FeatureAblationRunResult",
    "FeatureAblationRunnerError",
    "FeatureConfigurationError",
    "TargetEvaluation",
    "build_baseline_config",
    "build_single_removal_config",
    "compare_feature_runs",
    "result_summary",
    "run_feature_subset",
    "validate_dataset",
    "validate_run_config",
    "validate_temporal_order",
]
