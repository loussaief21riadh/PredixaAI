from __future__ import annotations

"""
PredixaAI V7 - Final production benchmark and smoke test, Sprint 6.

Purpose
-------
Validate the production application of the stable ``rate_10`` pruning by
comparing, on one shared purged chronological split:

1. the historical 12-feature reference contract;
2. the active 11-feature production contract.

The module also executes a public production smoke path:

    V7RankingModel.fit(...)
    V7RankingModel.predict_top_k(...)

The smoke input dictionary is reconstructed from one candidate-level
validation target. This verifies that:

- all engineered features, including ``rate_10``, are still available;
- only the 11 active production features enter the model matrix;
- the public ranking output contains 49 unique candidates;
- the public Top-K output is valid and deterministic.

No production source is modified by this module.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from app.ai.v7.explainability.feature_ablation import (
    V7FeatureAblationReport,
)
from app.ai.v7.explainability.feature_ablation_runner import (
    FeatureAblationRunResult,
    TargetEvaluation,
    validate_dataset,
    validate_temporal_order,
)
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
from app.database import SessionLocal


VERSION = "V7-PRODUCTION-BENCHMARK-SMOKE-V1"

BACKEND_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parents[4]
)

DEFAULT_OUTPUT_DIRECTORY = (
    BACKEND_DIRECTORY
    / "reports"
    / "v7"
    / "production_benchmark"
)

DEFAULT_WINDOW_SIZE = 100
DEFAULT_MAX_TRAINING_TARGETS = 1500
DEFAULT_VALIDATION_TARGETS = 100
DEFAULT_TOP_K = 5
DEFAULT_PURGE_TARGETS = 1
DEFAULT_ACCURACY_TOLERANCE = 0.0
DEFAULT_MAXIMUM_RUNTIME_RATIO = 2.0

PRUNED_FEATURE = "rate_10"
RETAINED_CORRELATED_FEATURE = "short_vs_long"

EXPECTED_PRODUCTION_FEATURE_COUNT = 11
EXPECTED_REFERENCE_FEATURE_COUNT = 12
EXPECTED_CANDIDATE_COUNT = 49
EXPECTED_POSITIVE_COUNT = 5


class ProductionBenchmarkError(RuntimeError):
    """Base exception for Sprint 6 failures."""


class BenchmarkConfigurationError(
    ProductionBenchmarkError
):
    """Raised when benchmark parameters are invalid."""


class BenchmarkContractError(
    ProductionBenchmarkError
):
    """Raised when the active production contract is invalid."""


class BenchmarkDatasetError(
    ProductionBenchmarkError
):
    """Raised when benchmark data cannot be prepared."""


class BenchmarkEvaluationError(
    ProductionBenchmarkError
):
    """Raised when one model benchmark fails."""


class ProductionSmokeError(
    ProductionBenchmarkError
):
    """Raised when the public production prediction path fails."""


@dataclass(frozen=True)
class ProductionBenchmarkConfig:
    """Complete Sprint 6 configuration."""

    output_directory: Path = (
        DEFAULT_OUTPUT_DIRECTORY
    )
    window_size: int = (
        DEFAULT_WINDOW_SIZE
    )
    max_training_targets: int = (
        DEFAULT_MAX_TRAINING_TARGETS
    )
    validation_targets: int = (
        DEFAULT_VALIDATION_TARGETS
    )
    top_k: int = DEFAULT_TOP_K
    purge_targets: int = (
        DEFAULT_PURGE_TARGETS
    )
    accuracy_tolerance: float = (
        DEFAULT_ACCURACY_TOLERANCE
    )
    maximum_runtime_ratio: float = (
        DEFAULT_MAXIMUM_RUNTIME_RATIO
    )

    def validated(
        self,
    ) -> "ProductionBenchmarkConfig":
        """Validate and return this immutable configuration."""

        validate_config(
            self
        )
        return self


@dataclass(frozen=True)
class ProductionContractCheck:
    """Runtime verification of the active 11-feature contract."""

    passed: bool
    production_features: tuple[str, ...]
    reference_features: tuple[str, ...]
    pruned_features: tuple[str, ...]
    feature_family_order: tuple[str, ...]
    configured_family_features: tuple[str, ...]
    production_feature_count: int
    reference_feature_count: int
    expected_model_feature_count: int
    full_engineered_feature_count: int
    rate_10_engineered: bool
    rate_10_active: bool
    short_vs_long_active: bool
    family_contract_matches: bool


@dataclass(frozen=True)
class BenchmarkDatasetSummary:
    """Chronological split and dataset metadata."""

    draw_count: int
    dataset_rows: int
    dataset_targets: int
    training_rows: int
    training_targets: int
    validation_rows: int
    validation_targets: int
    purge_targets: int
    first_training_target: int
    last_training_target: int
    purged_target_indices: tuple[int, ...]
    first_validation_target: int
    last_validation_target: int


@dataclass(frozen=True)
class ProductionSmokeResult:
    """Result of one public fit and prediction smoke test."""

    passed: bool
    target_draw_index: int
    target_draw_date: str
    feature_dictionary_count: int
    engineered_rate_10_present: bool
    model_feature_count: int
    candidate_count: int
    top_k: int
    predicted_numbers: tuple[int, ...]
    actual_numbers: tuple[int, ...]
    hits: int
    probability_count: int
    ranking_is_sorted: bool
    unique_ranking_numbers: bool
    finite_probabilities: bool
    probability_bounds_valid: bool


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison of production and historical reference runs."""

    reference_experiment: str
    production_experiment: str
    reference_feature_count: int
    production_feature_count: int
    removed_features: tuple[str, ...]
    reference_mean_hits_at_k: float
    production_mean_hits_at_k: float
    absolute_hits_delta: float
    relative_hits_delta: float | None
    reference_target_hit_rate: float
    production_target_hit_rate: float
    absolute_target_hit_rate_delta: float
    reference_total_seconds: float
    production_total_seconds: float
    runtime_ratio: float | None
    accuracy_tolerance: float
    maximum_runtime_ratio: float
    accuracy_accepted: bool
    runtime_accepted: bool


@dataclass(frozen=True)
class ProductionBenchmarkReport:
    """Complete Sprint 6 report."""

    status: str
    version: str
    protocol: str
    ready_for_production: bool
    recommendation: str
    window_size: int
    max_training_targets: int
    validation_targets: int
    top_k: int
    purge_targets: int
    accuracy_tolerance: float
    maximum_runtime_ratio: float
    contract: ProductionContractCheck
    dataset: BenchmarkDatasetSummary
    reference_run: FeatureAblationRunResult
    production_run: FeatureAblationRunResult
    comparison: BenchmarkComparison
    smoke: ProductionSmokeResult


def validate_config(
    config: ProductionBenchmarkConfig,
) -> None:
    """Validate every Sprint 6 parameter."""

    if not isinstance(
        config.output_directory,
        Path,
    ):
        raise BenchmarkConfigurationError(
            "output_directory must be a pathlib.Path"
        )

    if config.window_size < 100:
        raise BenchmarkConfigurationError(
            "window_size must be at least 100"
        )

    if config.max_training_targets < 0:
        raise BenchmarkConfigurationError(
            "max_training_targets cannot be negative"
        )

    if config.validation_targets < 5:
        raise BenchmarkConfigurationError(
            "validation_targets must be at least 5"
        )

    if not 1 <= config.top_k <= 49:
        raise BenchmarkConfigurationError(
            "top_k must be between 1 and 49"
        )

    if config.purge_targets < 0:
        raise BenchmarkConfigurationError(
            "purge_targets cannot be negative"
        )

    if not math.isfinite(
        config.accuracy_tolerance
    ):
        raise BenchmarkConfigurationError(
            "accuracy_tolerance must be finite"
        )

    if config.accuracy_tolerance < 0:
        raise BenchmarkConfigurationError(
            "accuracy_tolerance cannot be negative"
        )

    if not math.isfinite(
        config.maximum_runtime_ratio
    ):
        raise BenchmarkConfigurationError(
            "maximum_runtime_ratio must be finite"
        )

    if (
        config.maximum_runtime_ratio
        <= 0
    ):
        raise BenchmarkConfigurationError(
            "maximum_runtime_ratio must be positive"
        )


def _normalise_feature_sequence(
    values: Sequence[str],
    *,
    field_name: str,
) -> tuple[str, ...]:
    """Validate one ordered feature-name sequence."""

    if isinstance(
        values,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        raise BenchmarkContractError(
            f"{field_name} must be a sequence"
        )

    result: list[str] = []
    seen: set[str] = set()

    for value in values:
        if not isinstance(
            value,
            str,
        ):
            raise BenchmarkContractError(
                f"{field_name} must contain strings"
            )

        feature = value.strip()

        if not feature:
            raise BenchmarkContractError(
                f"{field_name} contains an empty feature"
            )

        if feature in seen:
            raise BenchmarkContractError(
                f"{field_name} contains duplicate feature: "
                f"{feature}"
            )

        seen.add(
            feature
        )
        result.append(
            feature
        )

    if not result:
        raise BenchmarkContractError(
            f"{field_name} cannot be empty"
        )

    return tuple(
        result
    )


def reference_feature_columns() -> tuple[str, ...]:
    """Return the historical pre-pruning 12-feature contract."""

    global_features = (
        _normalise_feature_sequence(
            tuple(
                V7RankingDataset
                .GLOBAL_FEATURES
            ),
            field_name="GLOBAL_FEATURES",
        )
    )
    candidate_features = (
        _normalise_feature_sequence(
            tuple(
                V7RankingDataset
                .CANDIDATE_FEATURES
            ),
            field_name="CANDIDATE_FEATURES",
        )
    )

    reference = (
        *global_features,
        *candidate_features,
    )

    if len(
        reference
    ) != len(
        set(
            reference
        )
    ):
        raise BenchmarkContractError(
            "Historical reference features are not unique"
        )

    return tuple(
        reference
    )


def validate_production_contract(
) -> ProductionContractCheck:
    """Validate the exact rate_10-pruned production contract."""

    production_features = (
        _normalise_feature_sequence(
            tuple(
                V7RankingDataset
                .feature_columns()
            ),
            field_name="production features",
        )
    )

    reference_features = (
        reference_feature_columns()
    )

    pruned_features = tuple(
        feature
        for feature
        in reference_features
        if feature
        not in set(
            production_features
        )
    )

    family_order = (
        _normalise_feature_sequence(
            tuple(
                FEATURE_FAMILY_ORDER
            ),
            field_name="FEATURE_FAMILY_ORDER",
        )
    )

    missing_families = [
        family
        for family in family_order
        if family
        not in FEATURE_FAMILIES
    ]

    if missing_families:
        raise BenchmarkContractError(
            "Feature families are missing configured names: "
            f"{missing_families}"
        )

    configured_family_features = tuple(
        feature
        for family_name
        in family_order
        for feature
        in FEATURE_FAMILIES[
            family_name
        ]
    )

    if len(
        configured_family_features
    ) != len(
        set(
            configured_family_features
        )
    ):
        raise BenchmarkContractError(
            "Feature-family configuration contains duplicates"
        )

    family_contract_matches = (
        set(
            configured_family_features
        )
        == set(
            production_features
        )
    )

    rate_10_engineered = (
        PRUNED_FEATURE
        in tuple(
            V7RankingDataset
            .CANDIDATE_FEATURES
        )
    )
    rate_10_active = (
        PRUNED_FEATURE
        in production_features
    )
    short_vs_long_active = (
        RETAINED_CORRELATED_FEATURE
        in production_features
    )

    expected_model_feature_count = int(
        V7RankingModel
        .EXPECTED_FEATURE_COUNT
    )

    full_engineered_feature_count = int(
        V7RankingDataset
        .EXPECTED_FULL_FEATURE_COUNT
    )

    marker = tuple(
        getattr(
            V7RankingDataset,
            "PRUNED_MODEL_FEATURES",
            (),
        )
    )

    failures: list[str] = []

    if len(
        production_features
    ) != EXPECTED_PRODUCTION_FEATURE_COUNT:
        failures.append(
            "production feature count is not 11"
        )

    if len(
        reference_features
    ) != EXPECTED_REFERENCE_FEATURE_COUNT:
        failures.append(
            "reference feature count is not 12"
        )

    if pruned_features != (
        PRUNED_FEATURE,
    ):
        failures.append(
            "the only pruned feature must be rate_10"
        )

    if marker != (
        PRUNED_FEATURE,
    ):
        failures.append(
            "PRUNED_MODEL_FEATURES must equal ('rate_10',)"
        )

    if not rate_10_engineered:
        failures.append(
            "rate_10 is no longer engineered"
        )

    if rate_10_active:
        failures.append(
            "rate_10 remains active"
        )

    if not short_vs_long_active:
        failures.append(
            "short_vs_long is not active"
        )

    if expected_model_feature_count != (
        EXPECTED_PRODUCTION_FEATURE_COUNT
    ):
        failures.append(
            "V7RankingModel expected feature count is not 11"
        )

    if not family_contract_matches:
        failures.append(
            "feature-family configuration does not match production"
        )

    if failures:
        raise BenchmarkContractError(
            "Invalid V7 production contract: "
            + "; ".join(
                failures
            )
        )

    return ProductionContractCheck(
        passed=True,
        production_features=(
            production_features
        ),
        reference_features=(
            reference_features
        ),
        pruned_features=(
            pruned_features
        ),
        feature_family_order=(
            family_order
        ),
        configured_family_features=(
            configured_family_features
        ),
        production_feature_count=len(
            production_features
        ),
        reference_feature_count=len(
            reference_features
        ),
        expected_model_feature_count=(
            expected_model_feature_count
        ),
        full_engineered_feature_count=(
            full_engineered_feature_count
        ),
        rate_10_engineered=(
            rate_10_engineered
        ),
        rate_10_active=(
            rate_10_active
        ),
        short_vs_long_active=(
            short_vs_long_active
        ),
        family_contract_matches=(
            family_contract_matches
        ),
    )


def _target_indices(
    dataset: pd.DataFrame,
) -> tuple[int, ...]:
    """Return sorted unique chronological target indices."""

    if not isinstance(
        dataset,
        pd.DataFrame,
    ):
        raise BenchmarkDatasetError(
            "dataset must be a pandas DataFrame"
        )

    if (
        "target_draw_index"
        not in dataset.columns
    ):
        raise BenchmarkDatasetError(
            "dataset is missing target_draw_index"
        )

    try:
        values = tuple(
            sorted(
                int(value)
                for value
                in dataset[
                    "target_draw_index"
                ].unique()
            )
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise BenchmarkDatasetError(
            "dataset contains invalid target indices"
        ) from exc

    if not values:
        raise BenchmarkDatasetError(
            "dataset has no target indices"
        )

    return values


def split_dataset_with_purge(
    *,
    dataset: pd.DataFrame,
    validation_targets: int,
    purge_targets: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Create one strict train, purge, and validation split."""

    indices = (
        _target_indices(
            dataset
        )
    )

    required = (
        validation_targets
        + purge_targets
        + 1
    )

    if len(
        indices
    ) < required:
        raise BenchmarkDatasetError(
            "Not enough targets for benchmark split. "
            f"Available={len(indices)}, required={required}"
        )

    validation_indices = tuple(
        indices[
            -validation_targets:
        ]
    )

    validation_start = (
        len(indices)
        - validation_targets
    )

    purge_start = (
        validation_start
        - purge_targets
    )

    purged_indices = tuple(
        indices[
            purge_start:
            validation_start
        ]
    )

    training_indices = tuple(
        indices[
            :purge_start
        ]
    )

    if not training_indices:
        raise BenchmarkDatasetError(
            "Temporal training target set is empty"
        )

    training = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(
                training_indices
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
                validation_indices
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    if training.empty:
        raise BenchmarkDatasetError(
            "Temporal training dataset is empty"
        )

    if validation.empty:
        raise BenchmarkDatasetError(
            "Temporal validation dataset is empty"
        )

    actual_training_indices = tuple(
        sorted(
            int(value)
            for value
            in training[
                "target_draw_index"
            ].unique()
        )
    )

    actual_validation_indices = tuple(
        sorted(
            int(value)
            for value
            in validation[
                "target_draw_index"
            ].unique()
        )
    )

    if (
        actual_training_indices
        != training_indices
    ):
        raise BenchmarkDatasetError(
            "Training targets are incomplete"
        )

    if (
        actual_validation_indices
        != validation_indices
    ):
        raise BenchmarkDatasetError(
            "Validation targets are incomplete"
        )

    return (
        training,
        validation,
        training_indices,
        purged_indices,
        validation_indices,
    )


def prepare_benchmark_datasets(
    config: ProductionBenchmarkConfig,
    contract: ProductionContractCheck,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    BenchmarkDatasetSummary,
]:
    """Load, build, validate, and split the shared benchmark dataset."""

    config.validated()

    database = (
        SessionLocal()
    )

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
        raise BenchmarkDatasetError(
            "Unable to build the V7 benchmark dataset"
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
        raise BenchmarkDatasetError(
            "Generated V7 benchmark dataset is invalid"
        ) from exc

    (
        training,
        validation,
        training_indices,
        purged_indices,
        validation_indices,
    ) = split_dataset_with_purge(
        dataset=dataset,
        validation_targets=(
            config.validation_targets
        ),
        purge_targets=(
            config.purge_targets
        ),
    )

    try:
        validate_dataset(
            training,
            contract.reference_features,
            "training_dataset",
        )
        validate_dataset(
            validation,
            contract.reference_features,
            "validation_dataset",
        )
        validate_temporal_order(
            training,
            validation,
        )
    except Exception as exc:
        raise BenchmarkDatasetError(
            "Shared benchmark split failed validation"
        ) from exc

    dataset_targets = (
        len(
            metadata
        )
        if metadata is not None
        else len(
            _target_indices(
                dataset
            )
        )
    )

    summary = (
        BenchmarkDatasetSummary(
            draw_count=len(
                draws
            ),
            dataset_rows=len(
                dataset
            ),
            dataset_targets=int(
                dataset_targets
            ),
            training_rows=len(
                training
            ),
            training_targets=len(
                training_indices
            ),
            validation_rows=len(
                validation
            ),
            validation_targets=len(
                validation_indices
            ),
            purge_targets=len(
                purged_indices
            ),
            first_training_target=(
                training_indices[0]
            ),
            last_training_target=(
                training_indices[-1]
            ),
            purged_target_indices=(
                purged_indices
            ),
            first_validation_target=(
                validation_indices[0]
            ),
            last_validation_target=(
                validation_indices[-1]
            ),
        )
    )

    return (
        training,
        validation,
        summary,
    )


def _positive_probabilities(
    model: Any,
    features: pd.DataFrame,
) -> np.ndarray:
    """Return deterministic positive-class probabilities."""

    probabilities = (
        model.predict_proba(
            features
        )
    )

    classes = [
        int(value)
        for value
        in model.classes_
    ]

    if 1 in classes:
        positive_index = (
            classes.index(
                1
            )
        )
        scores = (
            probabilities[
                :,
                positive_index
            ]
            .astype(
                float
            )
        )
    elif classes == [
        0,
    ]:
        scores = np.zeros(
            len(
                features
            ),
            dtype=float,
        )
    else:
        raise BenchmarkEvaluationError(
            "Unsupported classifier classes: "
            f"{classes}"
        )

    if len(
        scores
    ) != len(
        features
    ):
        raise BenchmarkEvaluationError(
            "Probability count does not match validation rows"
        )

    if not np.isfinite(
        scores
    ).all():
        raise BenchmarkEvaluationError(
            "Benchmark probabilities are not finite"
        )

    if (
        np.any(
            scores
            < 0.0
        )
        or np.any(
            scores
            > 1.0
        )
    ):
        raise BenchmarkEvaluationError(
            "Benchmark probabilities must be inside [0, 1]"
        )

    return scores


def _evaluate_target_groups(
    *,
    validation_dataset: pd.DataFrame,
    probabilities: np.ndarray,
    top_k: int,
) -> tuple[
    TargetEvaluation,
    ...,
]:
    """Build deterministic target-level Top-K evaluations."""

    scored = (
        validation_dataset[
            [
                "candidate_number",
                "target",
                "target_draw_index",
                "target_draw_date",
            ]
        ]
        .copy()
    )

    scored[
        "probability"
    ] = probabilities

    results: list[
        TargetEvaluation
    ] = []

    for (
        target_index,
        group,
    ) in scored.groupby(
        "target_draw_index",
        sort=True,
    ):
        if len(
            group
        ) != EXPECTED_CANDIDATE_COUNT:
            raise BenchmarkEvaluationError(
                f"Target {target_index} does not contain "
                "49 candidate rows"
            )

        ranked = (
            group.sort_values(
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
        )

        selected_numbers = tuple(
            ranked.head(
                top_k
            )[
                "candidate_number"
            ]
            .astype(
                int
            )
            .tolist()
        )

        actual_numbers = tuple(
            sorted(
                group.loc[
                    group[
                        "target"
                    ].astype(
                        int
                    )
                    == 1,
                    "candidate_number",
                ]
                .astype(
                    int
                )
                .tolist()
            )
        )

        if len(
            actual_numbers
        ) != EXPECTED_POSITIVE_COUNT:
            raise BenchmarkEvaluationError(
                f"Target {target_index} does not contain "
                "exactly 5 positive labels"
            )

        if len(
            set(
                selected_numbers
            )
        ) != top_k:
            raise BenchmarkEvaluationError(
                f"Target {target_index} Top-K contains duplicates"
            )

        target_dates = (
            group[
                "target_draw_date"
            ]
            .astype(
                str
            )
            .unique()
            .tolist()
        )

        if len(
            target_dates
        ) != 1:
            raise BenchmarkEvaluationError(
                f"Target {target_index} has inconsistent dates"
            )

        hits = len(
            set(
                selected_numbers
            )
            .intersection(
                actual_numbers
            )
        )

        results.append(
            TargetEvaluation(
                target_draw_index=int(
                    target_index
                ),
                target_draw_date=(
                    target_dates[0]
                ),
                selected_numbers=(
                    selected_numbers
                ),
                actual_numbers=(
                    actual_numbers
                ),
                hits=hits,
            )
        )

    if not results:
        raise BenchmarkEvaluationError(
            "No validation targets were evaluated"
        )

    return tuple(
        results
    )


def evaluate_feature_contract(
    *,
    experiment_name: str,
    feature_columns: Sequence[str],
    reference_features: Sequence[str],
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    top_k: int,
) -> FeatureAblationRunResult:
    """Train and evaluate one arbitrary engineered feature contract."""

    if not isinstance(
        experiment_name,
        str,
    ) or not experiment_name.strip():
        raise BenchmarkEvaluationError(
            "experiment_name must be a non-empty string"
        )

    features = (
        _normalise_feature_sequence(
            feature_columns,
            field_name=(
                f"{experiment_name} features"
            ),
        )
    )
    reference = (
        _normalise_feature_sequence(
            reference_features,
            field_name="reference features",
        )
    )

    unknown = sorted(
        set(
            features
        )
        - set(
            reference
        )
    )

    if unknown:
        raise BenchmarkEvaluationError(
            "Benchmark features are outside the engineered "
            f"reference contract: {unknown}"
        )

    if not 1 <= top_k <= 49:
        raise BenchmarkEvaluationError(
            "top_k must be between 1 and 49"
        )

    try:
        validate_dataset(
            training_dataset,
            features,
            "training_dataset",
        )
        validate_dataset(
            validation_dataset,
            features,
            "validation_dataset",
        )
        validate_temporal_order(
            training_dataset,
            validation_dataset,
        )
    except Exception as exc:
        raise BenchmarkEvaluationError(
            f"{experiment_name} dataset validation failed"
        ) from exc

    try:
        model = (
            V7FeatureAblationReport
            ._build_model()
        )

        training_features = (
            training_dataset[
                list(
                    features
                )
            ]
            .astype(
                float
            )
        )
        training_target = (
            training_dataset[
                "target"
            ]
            .astype(
                int
            )
        )

        fit_started = (
            perf_counter()
        )

        model.fit(
            training_features,
            training_target,
        )

        fit_seconds = (
            perf_counter()
            - fit_started
        )

        validation_features = (
            validation_dataset[
                list(
                    features
                )
            ]
            .astype(
                float
            )
        )

        prediction_started = (
            perf_counter()
        )

        probabilities = (
            _positive_probabilities(
                model,
                validation_features,
            )
        )

        evaluations = (
            _evaluate_target_groups(
                validation_dataset=(
                    validation_dataset
                ),
                probabilities=(
                    probabilities
                ),
                top_k=top_k,
            )
        )

        prediction_seconds = (
            perf_counter()
            - prediction_started
        )
    except ProductionBenchmarkError:
        raise
    except Exception as exc:
        raise BenchmarkEvaluationError(
            f"{experiment_name} model evaluation failed"
        ) from exc

    total_hits = sum(
        evaluation.hits
        for evaluation
        in evaluations
    )
    validation_target_count = len(
        evaluations
    )
    mean_hits = (
        total_hits
        / validation_target_count
    )
    targets_with_hit = sum(
        evaluation.hits
        >= 1
        for evaluation
        in evaluations
    )

    removed_features = tuple(
        feature
        for feature
        in reference
        if feature
        not in set(
            features
        )
    )

    return (
        FeatureAblationRunResult(
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
            top_k=top_k,
            training_rows=len(
                training_dataset
            ),
            validation_rows=len(
                validation_dataset
            ),
            training_targets=int(
                training_dataset[
                    "target_draw_index"
                ].nunique()
            ),
            validation_targets=(
                validation_target_count
            ),
            fit_seconds=float(
                fit_seconds
            ),
            prediction_seconds=float(
                prediction_seconds
            ),
            total_seconds=float(
                fit_seconds
                + prediction_seconds
            ),
            total_hits=int(
                total_hits
            ),
            mean_hits_at_k=float(
                mean_hits
            ),
            normalized_hits_at_k=float(
                mean_hits
                / top_k
            ),
            targets_with_at_least_one_hit=int(
                targets_with_hit
            ),
            target_hit_rate=float(
                targets_with_hit
                / validation_target_count
            ),
            target_evaluations=(
                evaluations
            ),
        )
    )


def build_full_feature_dictionary(
    target_rows: pd.DataFrame,
) -> dict[str, int | float]:
    """Reconstruct one complete assembler-compatible feature dictionary."""

    if not isinstance(
        target_rows,
        pd.DataFrame,
    ):
        raise ProductionSmokeError(
            "target_rows must be a pandas DataFrame"
        )

    if target_rows.empty:
        raise ProductionSmokeError(
            "target_rows cannot be empty"
        )

    required_columns = {
        "candidate_number",
        *tuple(
            V7RankingDataset
            .GLOBAL_FEATURES
        ),
        *tuple(
            V7RankingDataset
            .CANDIDATE_FEATURES
        ),
    }

    missing = sorted(
        required_columns
        - set(
            target_rows.columns
        )
    )

    if missing:
        raise ProductionSmokeError(
            "Target rows are missing engineered columns: "
            f"{missing}"
        )

    ordered = (
        target_rows
        .sort_values(
            by="candidate_number",
            ascending=True,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    expected_candidates = list(
        range(
            1,
            EXPECTED_CANDIDATE_COUNT
            + 1,
        )
    )
    actual_candidates = (
        ordered[
            "candidate_number"
        ]
        .astype(
            int
        )
        .tolist()
    )

    if (
        actual_candidates
        != expected_candidates
    ):
        raise ProductionSmokeError(
            "Target rows must contain candidates 1 through 49 "
            "in unique order"
        )

    features: dict[
        str,
        int | float
    ] = {}

    for global_feature in (
        V7RankingDataset
        .GLOBAL_FEATURES
    ):
        values = (
            ordered[
                global_feature
            ]
            .to_numpy()
        )

        numeric_values = (
            pd.to_numeric(
                pd.Series(
                    values
                ),
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            numeric_values
        ).all():
            raise ProductionSmokeError(
                f"Global feature {global_feature} is not finite"
            )

        if not np.allclose(
            numeric_values,
            numeric_values[0],
            rtol=0.0,
            atol=0.0,
        ):
            raise ProductionSmokeError(
                f"Global feature {global_feature} varies "
                "inside one target"
            )

        value = values[0]

        if isinstance(
            value,
            np.generic,
        ):
            value = (
                value.item()
            )

        features[
            global_feature
        ] = value

    for candidate_feature in (
        V7RankingDataset
        .CANDIDATE_FEATURES
    ):
        for row in ordered.itertuples(
            index=False
        ):
            candidate_number = int(
                getattr(
                    row,
                    "candidate_number",
                )
            )
            value = getattr(
                row,
                candidate_feature,
            )

            if isinstance(
                value,
                np.generic,
            ):
                value = (
                    value.item()
                )

            try:
                numeric_value = float(
                    value
                )
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ProductionSmokeError(
                    "Candidate feature is non-numeric: "
                    f"{candidate_feature}_{candidate_number}"
                ) from exc

            if not math.isfinite(
                numeric_value
            ):
                raise ProductionSmokeError(
                    "Candidate feature is non-finite: "
                    f"{candidate_feature}_{candidate_number}"
                )

            features[
                f"{candidate_feature}_{candidate_number}"
            ] = value

    expected_count = int(
        V7RankingDataset
        .EXPECTED_FULL_FEATURE_COUNT
    )

    if len(
        features
    ) != expected_count:
        raise ProductionSmokeError(
            "Unexpected reconstructed feature dictionary count. "
            f"Expected={expected_count}, received={len(features)}"
        )

    return features


def _ranking_sorted(
    ranking: Sequence[
        Mapping[str, Any]
    ],
) -> bool:
    """Check descending score and ascending-number tie order."""

    keys: list[
        tuple[
            float,
            int,
        ]
    ] = []

    for item in ranking:
        if not isinstance(
            item,
            Mapping,
        ):
            return False

        try:
            number = int(
                item[
                    "number"
                ]
            )
            score = float(
                item[
                    "score"
                ]
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return False

        keys.append(
            (
                -score,
                number,
            )
        )

    return keys == sorted(
        keys
    )


def run_production_smoke(
    *,
    training_dataset: pd.DataFrame,
    validation_dataset: pd.DataFrame,
    top_k: int,
) -> ProductionSmokeResult:
    """Execute the public production fit and Top-K prediction path."""

    if not isinstance(
        validation_dataset,
        pd.DataFrame,
    ) or validation_dataset.empty:
        raise ProductionSmokeError(
            "validation_dataset must be a non-empty DataFrame"
        )

    target_indices = tuple(
        sorted(
            int(value)
            for value
            in validation_dataset[
                "target_draw_index"
            ].unique()
        )
    )

    if not target_indices:
        raise ProductionSmokeError(
            "No validation targets are available for smoke test"
        )

    target_index = (
        target_indices[-1]
    )

    target_rows = (
        validation_dataset[
            validation_dataset[
                "target_draw_index"
            ].astype(
                int
            )
            == target_index
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    features = (
        build_full_feature_dictionary(
            target_rows
        )
    )

    try:
        model = (
            V7RankingModel()
        )

        model.fit(
            training_dataset
        )

        prediction = (
            model.predict_top_k(
                features=features,
                top_k=top_k,
            )
        )
    except Exception as exc:
        raise ProductionSmokeError(
            "Public production fit/predict smoke path failed"
        ) from exc

    try:
        predicted_numbers = tuple(
            int(value)
            for value
            in prediction[
                "predicted_numbers"
            ]
        )
        ranking = tuple(
            prediction[
                "ranking"
            ]
        )
        probabilities = dict(
            prediction[
                "probabilities"
            ]
        )
        model_feature_count = int(
            prediction[
                "feature_count"
            ]
        )
        candidate_count = int(
            prediction[
                "candidate_count"
            ]
        )
        returned_top_k = int(
            prediction[
                "top_k"
            ]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ProductionSmokeError(
            "Public production prediction has an invalid schema"
        ) from exc

    actual_numbers = tuple(
        sorted(
            target_rows.loc[
                target_rows[
                    "target"
                ].astype(
                    int
                )
                == 1,
                "candidate_number",
            ]
            .astype(
                int
            )
            .tolist()
        )
    )

    if len(
        actual_numbers
    ) != EXPECTED_POSITIVE_COUNT:
        raise ProductionSmokeError(
            "Smoke target does not contain exactly 5 positives"
        )

    target_dates = (
        target_rows[
            "target_draw_date"
        ]
        .astype(
            str
        )
        .unique()
        .tolist()
    )

    if len(
        target_dates
    ) != 1:
        raise ProductionSmokeError(
            "Smoke target has inconsistent dates"
        )

    ranking_numbers: list[int] = []
    ranking_scores: list[float] = []

    for item in ranking:
        try:
            ranking_numbers.append(
                int(
                    item[
                        "number"
                    ]
                )
            )
            ranking_scores.append(
                float(
                    item[
                        "score"
                    ]
                )
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ProductionSmokeError(
                "Smoke ranking item is invalid"
            ) from exc

    finite_probabilities = (
        np.isfinite(
            np.asarray(
                ranking_scores,
                dtype=float,
            )
        ).all()
    )
    probability_bounds_valid = (
        all(
            0.0
            <= score
            <= 1.0
            for score
            in ranking_scores
        )
    )
    unique_ranking_numbers = (
        len(
            ranking_numbers
        )
        == len(
            set(
                ranking_numbers
            )
        )
        == EXPECTED_CANDIDATE_COUNT
    )
    ranking_is_sorted = (
        _ranking_sorted(
            ranking
        )
    )

    checks = (
        len(
            features
        )
        == int(
            V7RankingDataset
            .EXPECTED_FULL_FEATURE_COUNT
        ),
        any(
            key.startswith(
                f"{PRUNED_FEATURE}_"
            )
            for key
            in features
        ),
        model_feature_count
        == EXPECTED_PRODUCTION_FEATURE_COUNT,
        candidate_count
        == EXPECTED_CANDIDATE_COUNT,
        returned_top_k
        == top_k,
        len(
            predicted_numbers
        )
        == top_k,
        len(
            set(
                predicted_numbers
            )
        )
        == top_k,
        all(
            1
            <= number
            <= 49
            for number
            in predicted_numbers
        ),
        len(
            ranking
        )
        == EXPECTED_CANDIDATE_COUNT,
        len(
            probabilities
        )
        == EXPECTED_CANDIDATE_COUNT,
        ranking_is_sorted,
        unique_ranking_numbers,
        bool(
            finite_probabilities
        ),
        probability_bounds_valid,
        predicted_numbers
        == tuple(
            ranking_numbers[
                :top_k
            ]
        ),
    )

    if not all(
        checks
    ):
        raise ProductionSmokeError(
            "Public production smoke invariants failed"
        )

    hits = len(
        set(
            predicted_numbers
        )
        .intersection(
            actual_numbers
        )
    )

    return ProductionSmokeResult(
        passed=True,
        target_draw_index=(
            target_index
        ),
        target_draw_date=(
            target_dates[0]
        ),
        feature_dictionary_count=len(
            features
        ),
        engineered_rate_10_present=True,
        model_feature_count=(
            model_feature_count
        ),
        candidate_count=(
            candidate_count
        ),
        top_k=(
            returned_top_k
        ),
        predicted_numbers=(
            predicted_numbers
        ),
        actual_numbers=(
            actual_numbers
        ),
        hits=hits,
        probability_count=len(
            probabilities
        ),
        ranking_is_sorted=(
            ranking_is_sorted
        ),
        unique_ranking_numbers=(
            unique_ranking_numbers
        ),
        finite_probabilities=bool(
            finite_probabilities
        ),
        probability_bounds_valid=(
            probability_bounds_valid
        ),
    )


def compare_benchmark_runs(
    *,
    reference_run: FeatureAblationRunResult,
    production_run: FeatureAblationRunResult,
    accuracy_tolerance: float,
    maximum_runtime_ratio: float,
) -> BenchmarkComparison:
    """Compare accuracy and runtime of the two feature contracts."""

    absolute_hits_delta = (
        production_run
        .mean_hits_at_k
        - reference_run
        .mean_hits_at_k
    )

    relative_hits_delta = (
        absolute_hits_delta
        / reference_run
        .mean_hits_at_k
        if reference_run
        .mean_hits_at_k
        != 0.0
        else None
    )

    target_hit_rate_delta = (
        production_run
        .target_hit_rate
        - reference_run
        .target_hit_rate
    )

    runtime_ratio = (
        production_run
        .total_seconds
        / reference_run
        .total_seconds
        if reference_run
        .total_seconds
        > 0.0
        else None
    )

    accuracy_accepted = (
        production_run
        .mean_hits_at_k
        >= (
            reference_run
            .mean_hits_at_k
            - accuracy_tolerance
        )
    )

    runtime_accepted = (
        runtime_ratio is None
        or runtime_ratio
        <= maximum_runtime_ratio
    )

    return BenchmarkComparison(
        reference_experiment=(
            reference_run
            .experiment_name
        ),
        production_experiment=(
            production_run
            .experiment_name
        ),
        reference_feature_count=(
            reference_run
            .feature_count
        ),
        production_feature_count=(
            production_run
            .feature_count
        ),
        removed_features=tuple(
            feature
            for feature
            in reference_run
            .feature_columns
            if feature
            not in set(
                production_run
                .feature_columns
            )
        ),
        reference_mean_hits_at_k=(
            reference_run
            .mean_hits_at_k
        ),
        production_mean_hits_at_k=(
            production_run
            .mean_hits_at_k
        ),
        absolute_hits_delta=(
            absolute_hits_delta
        ),
        relative_hits_delta=(
            relative_hits_delta
        ),
        reference_target_hit_rate=(
            reference_run
            .target_hit_rate
        ),
        production_target_hit_rate=(
            production_run
            .target_hit_rate
        ),
        absolute_target_hit_rate_delta=(
            target_hit_rate_delta
        ),
        reference_total_seconds=(
            reference_run
            .total_seconds
        ),
        production_total_seconds=(
            production_run
            .total_seconds
        ),
        runtime_ratio=(
            runtime_ratio
        ),
        accuracy_tolerance=(
            accuracy_tolerance
        ),
        maximum_runtime_ratio=(
            maximum_runtime_ratio
        ),
        accuracy_accepted=(
            accuracy_accepted
        ),
        runtime_accepted=(
            runtime_accepted
        ),
    )


def run_production_benchmark(
    config: ProductionBenchmarkConfig,
) -> ProductionBenchmarkReport:
    """Run the complete Sprint 6 benchmark and smoke test."""

    config.validated()

    contract = (
        validate_production_contract()
    )

    (
        training,
        validation,
        dataset_summary,
    ) = prepare_benchmark_datasets(
        config,
        contract,
    )

    try:
        reference_run = (
            evaluate_feature_contract(
                experiment_name=(
                    "historical_reference_12_features"
                ),
                feature_columns=(
                    contract
                    .reference_features
                ),
                reference_features=(
                    contract
                    .reference_features
                ),
                training_dataset=(
                    training
                ),
                validation_dataset=(
                    validation
                ),
                top_k=config.top_k,
            )
        )

        production_run = (
            evaluate_feature_contract(
                experiment_name=(
                    "production_11_features"
                ),
                feature_columns=(
                    contract
                    .production_features
                ),
                reference_features=(
                    contract
                    .reference_features
                ),
                training_dataset=(
                    training
                ),
                validation_dataset=(
                    validation
                ),
                top_k=config.top_k,
            )
        )
    except ProductionBenchmarkError:
        raise
    except Exception as exc:
        raise BenchmarkEvaluationError(
            "Unable to evaluate benchmark feature contracts"
        ) from exc

    comparison = (
        compare_benchmark_runs(
            reference_run=(
                reference_run
            ),
            production_run=(
                production_run
            ),
            accuracy_tolerance=(
                config
                .accuracy_tolerance
            ),
            maximum_runtime_ratio=(
                config
                .maximum_runtime_ratio
            ),
        )
    )

    smoke = (
        run_production_smoke(
            training_dataset=(
                training
            ),
            validation_dataset=(
                validation
            ),
            top_k=config.top_k,
        )
    )

    ready = (
        contract.passed
        and comparison
        .accuracy_accepted
        and comparison
        .runtime_accepted
        and smoke.passed
    )

    recommendation = (
        "READY_FOR_PRODUCTION"
        if ready
        else "REVIEW_PRODUCTION_PRUNING"
    )

    return ProductionBenchmarkReport(
        status="success",
        version=VERSION,
        protocol=(
            "Shared purged chronological holdout; historical "
            "12-feature reference versus active 11-feature "
            "production; public fit/predict Top-K smoke."
        ),
        ready_for_production=(
            ready
        ),
        recommendation=(
            recommendation
        ),
        window_size=(
            config.window_size
        ),
        max_training_targets=(
            config
            .max_training_targets
        ),
        validation_targets=(
            config
            .validation_targets
        ),
        top_k=config.top_k,
        purge_targets=(
            config.purge_targets
        ),
        accuracy_tolerance=(
            config
            .accuracy_tolerance
        ),
        maximum_runtime_ratio=(
            config
            .maximum_runtime_ratio
        ),
        contract=contract,
        dataset=(
            dataset_summary
        ),
        reference_run=(
            reference_run
        ),
        production_run=(
            production_run
        ),
        comparison=(
            comparison
        ),
        smoke=smoke,
    )


def _json_safe(
    value: Any,
) -> Any:
    """Recursively convert dataclass values into JSON-safe values."""

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(
                key
            ): _json_safe(
                item
            )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        tuple,
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        list,
    ):
        return [
            _json_safe(
                item
            )
            for item
            in value
        ]

    if isinstance(
        value,
        np.generic,
    ):
        return value.item()

    if isinstance(
        value,
        float,
    ) and not math.isfinite(
        value
    ):
        return None

    return value


def _write_csv(
    *,
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[
        Mapping[
            str,
            Any,
        ]
    ],
) -> Path:
    """Write one deterministic UTF-8 CSV."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
                        row.get(
                            field
                        )
                    )
                    for field
                    in fieldnames
                }
            )

    return path.resolve()


def _target_comparison_rows(
    report: ProductionBenchmarkReport,
) -> tuple[
    dict[str, Any],
    ...,
]:
    """Build aligned target-level benchmark rows."""

    reference_by_target = {
        evaluation
        .target_draw_index: evaluation
        for evaluation
        in report.reference_run
        .target_evaluations
    }
    production_by_target = {
        evaluation
        .target_draw_index: evaluation
        for evaluation
        in report.production_run
        .target_evaluations
    }

    if (
        set(
            reference_by_target
        )
        != set(
            production_by_target
        )
    ):
        raise BenchmarkEvaluationError(
            "Reference and production target sets differ"
        )

    rows: list[
        dict[str, Any]
    ] = []

    for target_index in sorted(
        reference_by_target
    ):
        reference = (
            reference_by_target[
                target_index
            ]
        )
        production = (
            production_by_target[
                target_index
            ]
        )

        if (
            reference
            .actual_numbers
            != production
            .actual_numbers
        ):
            raise BenchmarkEvaluationError(
                f"Actual numbers differ for target {target_index}"
            )

        rows.append(
            {
                "target_draw_index": (
                    target_index
                ),
                "target_draw_date": (
                    production
                    .target_draw_date
                ),
                "reference_selected": (
                    ",".join(
                        str(
                            value
                        )
                        for value
                        in reference
                        .selected_numbers
                    )
                ),
                "production_selected": (
                    ",".join(
                        str(
                            value
                        )
                        for value
                        in production
                        .selected_numbers
                    )
                ),
                "actual_numbers": (
                    ",".join(
                        str(
                            value
                        )
                        for value
                        in production
                        .actual_numbers
                    )
                ),
                "reference_hits": (
                    reference.hits
                ),
                "production_hits": (
                    production.hits
                ),
                "hits_delta": (
                    production.hits
                    - reference.hits
                ),
            }
        )

    return tuple(
        rows
    )


def _report_text(
    report: ProductionBenchmarkReport,
) -> str:
    """Render the complete human-readable report."""

    comparison = (
        report.comparison
    )
    smoke = report.smoke

    return "\n".join(
        [
            "=" * 120,
            (
                "PREDIXA AI V7 FINAL PRODUCTION "
                "BENCHMARK AND SMOKE TEST"
            ),
            "=" * 120,
            f"Status                  : {report.status}",
            (
                "Ready for production    : "
                f"{report.ready_for_production}"
            ),
            (
                "Recommendation          : "
                f"{report.recommendation}"
            ),
            (
                "Production features     : "
                f"{report.contract.production_feature_count}"
            ),
            (
                "Reference features      : "
                f"{report.contract.reference_feature_count}"
            ),
            (
                "Removed                 : "
                + ",".join(
                    comparison.removed_features
                )
            ),
            "",
            "TEMPORAL DATASET",
            "-" * 120,
            (
                "Training targets        : "
                f"{report.dataset.training_targets}"
            ),
            (
                "Purged targets          : "
                f"{report.dataset.purge_targets}"
            ),
            (
                "Validation targets      : "
                f"{report.dataset.validation_targets}"
            ),
            (
                "Validation range        : "
                f"{report.dataset.first_validation_target}"
                " -> "
                f"{report.dataset.last_validation_target}"
            ),
            "",
            "ACCURACY BENCHMARK",
            "-" * 120,
            (
                "Reference Mean Hits@K   : "
                f"{comparison.reference_mean_hits_at_k:.6f}"
            ),
            (
                "Production Mean Hits@K  : "
                f"{comparison.production_mean_hits_at_k:.6f}"
            ),
            (
                "Absolute delta          : "
                f"{comparison.absolute_hits_delta:.6f}"
            ),
            (
                "Reference target rate   : "
                f"{comparison.reference_target_hit_rate:.6f}"
            ),
            (
                "Production target rate  : "
                f"{comparison.production_target_hit_rate:.6f}"
            ),
            (
                "Target-rate delta       : "
                f"{comparison.absolute_target_hit_rate_delta:.6f}"
            ),
            (
                "Accuracy accepted       : "
                f"{comparison.accuracy_accepted}"
            ),
            "",
            "RUNTIME BENCHMARK",
            "-" * 120,
            (
                "Reference seconds       : "
                f"{comparison.reference_total_seconds:.6f}"
            ),
            (
                "Production seconds      : "
                f"{comparison.production_total_seconds:.6f}"
            ),
            (
                "Runtime ratio           : "
                + (
                    f"{comparison.runtime_ratio:.6f}"
                    if comparison.runtime_ratio
                    is not None
                    else "n/a"
                )
            ),
            (
                "Runtime accepted        : "
                f"{comparison.runtime_accepted}"
            ),
            "",
            "PUBLIC PRODUCTION SMOKE",
            "-" * 120,
            (
                "Passed                  : "
                f"{smoke.passed}"
            ),
            (
                "Target                  : "
                f"{smoke.target_draw_index}"
            ),
            (
                "Feature dictionary      : "
                f"{smoke.feature_dictionary_count}"
            ),
            (
                "Model features          : "
                f"{smoke.model_feature_count}"
            ),
            (
                "Candidates              : "
                f"{smoke.candidate_count}"
            ),
            (
                "Predicted               : "
                + ",".join(
                    str(
                        value
                    )
                    for value
                    in smoke.predicted_numbers
                )
            ),
            (
                "Actual                  : "
                + ",".join(
                    str(
                        value
                    )
                    for value
                    in smoke.actual_numbers
                )
            ),
            (
                "Hits                    : "
                f"{smoke.hits}"
            ),
            "=" * 120,
        ]
    )


def _recommendation_text(
    report: ProductionBenchmarkReport,
) -> str:
    """Render a compact machine-readable recommendation."""

    comparison = (
        report.comparison
    )

    return "\n".join(
        [
            (
                "recommendation="
                f"{report.recommendation}"
            ),
            (
                "ready_for_production="
                f"{str(report.ready_for_production).lower()}"
            ),
            (
                "production_features="
                + ",".join(
                    report.contract
                    .production_features
                )
            ),
            (
                "removed_features="
                + ",".join(
                    comparison
                    .removed_features
                )
            ),
            (
                "reference_mean_hits_at_k="
                f"{comparison.reference_mean_hits_at_k:.6f}"
            ),
            (
                "production_mean_hits_at_k="
                f"{comparison.production_mean_hits_at_k:.6f}"
            ),
            (
                "absolute_hits_delta="
                f"{comparison.absolute_hits_delta:.6f}"
            ),
            (
                "accuracy_accepted="
                f"{str(comparison.accuracy_accepted).lower()}"
            ),
            (
                "runtime_accepted="
                f"{str(comparison.runtime_accepted).lower()}"
            ),
            (
                "smoke_passed="
                f"{str(report.smoke.passed).lower()}"
            ),
            "",
        ]
    )


def export_production_benchmark(
    *,
    report: ProductionBenchmarkReport,
    output_directory: Path,
) -> dict[str, Path]:
    """Export JSON, text, target CSV, and recommendation files."""

    if not isinstance(
        output_directory,
        Path,
    ):
        raise BenchmarkConfigurationError(
            "output_directory must be a pathlib.Path"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "production_benchmark.json"
    )
    text_path = (
        output_directory
        / "production_benchmark.txt"
    )
    targets_path = (
        output_directory
        / "production_benchmark_targets.csv"
    )
    recommendation_path = (
        output_directory
        / "production_benchmark_recommendation.txt"
    )

    json_path.write_text(
        json.dumps(
            _json_safe(
                asdict(
                    report
                )
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    text_path.write_text(
        _report_text(
            report
        )
        + "\n",
        encoding="utf-8",
    )

    _write_csv(
        path=targets_path,
        fieldnames=(
            "target_draw_index",
            "target_draw_date",
            "reference_selected",
            "production_selected",
            "actual_numbers",
            "reference_hits",
            "production_hits",
            "hits_delta",
        ),
        rows=(
            _target_comparison_rows(
                report
            )
        ),
    )

    recommendation_path.write_text(
        _recommendation_text(
            report
        ),
        encoding="utf-8",
    )

    return {
        "json": (
            json_path.resolve()
        ),
        "text": (
            text_path.resolve()
        ),
        "targets_csv": (
            targets_path.resolve()
        ),
        "recommendation": (
            recommendation_path
            .resolve()
        ),
    }


def print_production_benchmark(
    *,
    report: ProductionBenchmarkReport,
    generated_files: Mapping[
        str,
        Path,
    ],
) -> None:
    """Print the report and generated paths."""

    print(
        _report_text(
            report
        )
    )
    print()
    print("GENERATED FILES")
    print("-" * 120)

    for name, path in (
        generated_files.items()
    ):
        print(
            f"{name.upper():<24}: {path}"
        )

    print("=" * 120)

    if (
        report
        .ready_for_production
    ):
        print("SUCCESS")
    else:
        print("REVIEW REQUIRED")

    print("=" * 120)


def build_argument_parser(
) -> argparse.ArgumentParser:
    """Build the Sprint 6 CLI."""

    parser = (
        argparse.ArgumentParser(
            description=(
                "Benchmark PredixaAI V7 production 11-feature "
                "model against the historical 12-feature "
                "reference and run a public prediction smoke test."
            )
        )
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIRECTORY
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            DEFAULT_WINDOW_SIZE
        ),
    )
    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            DEFAULT_MAX_TRAINING_TARGETS
        ),
    )
    parser.add_argument(
        "--validation-targets",
        type=int,
        default=(
            DEFAULT_VALIDATION_TARGETS
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--purge-targets",
        type=int,
        default=(
            DEFAULT_PURGE_TARGETS
        ),
    )
    parser.add_argument(
        "--accuracy-tolerance",
        type=float,
        default=(
            DEFAULT_ACCURACY_TOLERANCE
        ),
    )
    parser.add_argument(
        "--maximum-runtime-ratio",
        type=float,
        default=(
            DEFAULT_MAXIMUM_RUNTIME_RATIO
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    arguments = (
        build_argument_parser()
        .parse_args()
    )

    config = (
        ProductionBenchmarkConfig(
            output_directory=(
                arguments
                .output_directory
            ),
            window_size=(
                arguments
                .window_size
            ),
            max_training_targets=(
                arguments
                .max_training_targets
            ),
            validation_targets=(
                arguments
                .validation_targets
            ),
            top_k=(
                arguments.top_k
            ),
            purge_targets=(
                arguments
                .purge_targets
            ),
            accuracy_tolerance=(
                arguments
                .accuracy_tolerance
            ),
            maximum_runtime_ratio=(
                arguments
                .maximum_runtime_ratio
            ),
        )
    )

    try:
        report = (
            run_production_benchmark(
                config
            )
        )

        generated_files = (
            export_production_benchmark(
                report=report,
                output_directory=(
                    config
                    .output_directory
                ),
            )
        )
    except ProductionBenchmarkError as exc:
        print("=" * 120)
        print(
            "PREDIXA AI V7 FINAL PRODUCTION "
            "BENCHMARK AND SMOKE TEST"
        )
        print("=" * 120)
        print(
            f"ERROR: {exc}"
        )

        if (
            exc.__cause__
            is not None
        ):
            cause = (
                exc.__cause__
            )
            print(
                "CAUSE: "
                f"{type(cause).__name__}: "
                f"{cause}"
            )

        print("=" * 120)

        return 1

    print_production_benchmark(
        report=report,
        generated_files=(
            generated_files
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
