from __future__ import annotations

"""
PredixaAI V7 - Feature Pruning Optimizer, Sprint 2.

Responsibilities
----------------
1. Load the V7 feature-correlation report.
2. Extract highly correlated feature pairs.
3. Build connected correlation groups.
4. Produce the deterministic Sprint 1 pruning plan.
5. Build one chronological candidate-level dataset.
6. Create a purged temporal training/validation split.
7. Evaluate the all-feature baseline.
8. Evaluate each requested single-feature removal.
9. Accept or reject each removal from temporal Hits@K.
10. Export JSON, CSV, and text reports.

Important
---------
- This module never modifies V7RankingDataset or the production model.
- Every candidate is evaluated independently against the same baseline.
- Accepted removals are recommendations only; they are not applied
  cumulatively by this Sprint 2 implementation.
"""

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from app.ai.v7.explainability.feature_ablation import (
    V7FeatureAblationReport,
)
from app.ai.v7.explainability.feature_ablation_runner import (
    FeatureAblationComparison,
    FeatureAblationRunResult,
    build_baseline_config,
    build_single_removal_config,
    compare_feature_runs,
    run_feature_subset,
)
from app.ai.v7.ranking_dataset import V7RankingDataset
from app.database import SessionLocal


VERSION = "V7-FEATURE-PRUNING-OPTIMIZER-TEMPORAL-V2"

BACKEND_DIRECTORY = Path(__file__).resolve().parents[4]

DEFAULT_CORRELATION_REPORT = (
    BACKEND_DIRECTORY
    / "reports"
    / "v7"
    / "feature_correlation"
    / "feature_correlation_report.json"
)

DEFAULT_OUTPUT_DIRECTORY = (
    BACKEND_DIRECTORY
    / "reports"
    / "v7"
    / "feature_pruning"
)

DEFAULT_CORRELATION_THRESHOLD = 0.80
DEFAULT_WINDOW_SIZE = 100
DEFAULT_MAX_TRAINING_TARGETS = 1500
DEFAULT_VALIDATION_TARGETS = 100
DEFAULT_TOP_K = 5
DEFAULT_PURGE_TARGETS = 1
DEFAULT_TOLERANCE = 0.0
DEFAULT_CANDIDATE_SCOPE = "all_group_features"

CANDIDATE_SCOPES = (
    "selected",
    "all_group_features",
)


class FeaturePruningError(RuntimeError):
    """Base exception for feature-pruning failures."""


class ConfigurationError(FeaturePruningError):
    """Raised when pruning configuration is invalid."""


class CorrelationReportError(FeaturePruningError):
    """Raised when the correlation report is missing or malformed."""


class DatasetPreparationError(FeaturePruningError):
    """Raised when temporal evaluation datasets cannot be prepared."""


class EvaluationError(FeaturePruningError):
    """Raised when baseline or candidate evaluation fails."""


@dataclass(frozen=True)
class PruningConfig:
    """Configuration for planning and temporal pruning evaluation."""

    correlation_report: Path = DEFAULT_CORRELATION_REPORT
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
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
    plan_only: bool = False

    def validated(self) -> "PruningConfig":
        """Validate and return this immutable configuration."""

        validate_config(self)
        return self


@dataclass(frozen=True)
class CorrelationPair:
    """One relationship between two V7 model features."""

    feature_a: str
    feature_b: str
    pearson: float | None
    spearman: float | None
    maximum_absolute_correlation: float
    high_correlation: bool


@dataclass(frozen=True)
class FeatureGroup:
    """Connected component of correlated model features."""

    group_id: int
    features: tuple[str, ...]
    pair_count: int
    maximum_absolute_correlation: float


@dataclass(frozen=True)
class PruningCandidate:
    """Provisional Sprint 1 removal candidate."""

    group_id: int
    feature: str
    retained_features: tuple[str, ...]
    reason: str
    correlation_link_count: int
    cumulative_absolute_correlation: float
    maximum_absolute_correlation: float


@dataclass(frozen=True)
class PruningPlan:
    """Correlation-based plan built before temporal evaluation."""

    status: str
    correlation_threshold: float
    pair_count: int
    reported_total_pair_count: int | None
    high_pair_count: int
    reported_high_pair_count: int | None
    constant_features: tuple[str, ...]
    group_count: int
    candidate_count: int
    feature_groups: tuple[FeatureGroup, ...]
    pruning_candidates: tuple[PruningCandidate, ...]


@dataclass(frozen=True)
class EvaluationCandidateSpec:
    """One feature scheduled for independent temporal removal testing."""

    feature: str
    group_ids: tuple[int, ...]
    correlation_link_count: int
    cumulative_absolute_correlation: float
    maximum_absolute_correlation: float
    source: str


@dataclass(frozen=True)
class CandidateEvaluation:
    """Temporal result for one independently removed feature."""

    feature: str
    group_ids: tuple[int, ...]
    correlation_link_count: int
    cumulative_absolute_correlation: float
    maximum_absolute_correlation: float
    source: str
    decision: str
    run: FeatureAblationRunResult
    comparison: FeatureAblationComparison


@dataclass(frozen=True)
class TemporalDatasetSummary:
    """Metadata describing the shared temporal split."""

    draw_count: int
    dataset_rows: int
    dataset_targets: int
    training_rows: int
    training_targets: int
    validation_rows: int
    validation_targets: int
    purged_targets: int
    first_training_target: int
    last_training_target: int
    purged_target_indices: tuple[int, ...]
    first_validation_target: int
    last_validation_target: int


@dataclass(frozen=True)
class PruningEvaluationReport:
    """Complete Sprint 2 result."""

    status: str
    version: str
    protocol: str
    correlation_threshold: float
    candidate_scope: str
    tolerance: float
    window_size: int
    max_training_targets: int
    validation_targets: int
    top_k: int
    purge_targets: int
    dataset: TemporalDatasetSummary
    plan: PruningPlan
    baseline: FeatureAblationRunResult
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    accepted_features: tuple[str, ...]
    rejected_features: tuple[str, ...]
    best_single_removal: str | None
    best_single_removal_delta: float | None


def validate_config(config: PruningConfig) -> None:
    """Validate every planning and evaluation parameter."""

    if not isinstance(config.correlation_report, Path):
        raise ConfigurationError(
            "correlation_report must be a pathlib.Path"
        )

    if not isinstance(config.output_directory, Path):
        raise ConfigurationError(
            "output_directory must be a pathlib.Path"
        )

    if not math.isfinite(config.correlation_threshold):
        raise ConfigurationError(
            "correlation_threshold must be finite"
        )

    if not 0.0 < config.correlation_threshold <= 1.0:
        raise ConfigurationError(
            "correlation_threshold must be greater than 0 and at most 1"
        )

    if config.minimum_group_size < 2:
        raise ConfigurationError(
            "minimum_group_size must be at least 2"
        )

    if config.maximum_candidates_per_group < 1:
        raise ConfigurationError(
            "maximum_candidates_per_group must be at least 1"
        )

    if config.window_size < 100:
        raise ConfigurationError(
            "window_size must be at least 100"
        )

    if config.max_training_targets < 0:
        raise ConfigurationError(
            "max_training_targets cannot be negative"
        )

    if config.validation_targets < 5:
        raise ConfigurationError(
            "validation_targets must be at least 5"
        )

    if not 1 <= config.top_k <= 49:
        raise ConfigurationError(
            "top_k must be between 1 and 49"
        )

    if config.purge_targets < 0:
        raise ConfigurationError(
            "purge_targets cannot be negative"
        )

    if not math.isfinite(config.tolerance):
        raise ConfigurationError(
            "tolerance must be finite"
        )

    if config.tolerance < 0:
        raise ConfigurationError(
            "tolerance cannot be negative"
        )

    if config.candidate_scope not in CANDIDATE_SCOPES:
        raise ConfigurationError(
            "candidate_scope must be one of: "
            f"{', '.join(CANDIDATE_SCOPES)}"
        )

    seen: set[str] = set()

    for raw_feature in config.explicit_features:
        if not isinstance(raw_feature, str):
            raise ConfigurationError(
                "explicit feature names must be strings"
            )

        feature = raw_feature.strip()

        if not feature:
            raise ConfigurationError(
                "explicit feature names cannot be empty"
            )

        if feature in seen:
            raise ConfigurationError(
                f"duplicate explicit feature: {feature}"
            )

        seen.add(feature)


def _require_mapping(
    value: Any,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrelationReportError(
            f"{name} must be a JSON object"
        )

    return value


def _normalise_feature_name(
    value: Any,
    field_name: str,
) -> str:
    if not isinstance(value, str):
        raise CorrelationReportError(
            f"{field_name} must be a string"
        )

    feature = value.strip()

    if not feature:
        raise CorrelationReportError(
            f"{field_name} cannot be empty"
        )

    return feature


def _optional_float(
    value: Any,
    field_name: str,
) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise CorrelationReportError(
            f"{field_name} must be numeric or null"
        )

    result = float(value)

    if not math.isfinite(result):
        raise CorrelationReportError(
            f"{field_name} must be finite"
        )

    return result


def _optional_int(
    value: Any,
    field_name: str,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise CorrelationReportError(
            f"{field_name} must be numeric or null"
        )

    result = int(value)

    if result < 0:
        raise CorrelationReportError(
            f"{field_name} cannot be negative"
        )

    return result


def _first_present(
    item: Mapping[str, Any],
    keys: Sequence[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in item:
            return item[key]

    return default


def load_correlation_report(
    path: Path,
) -> dict[str, Any]:
    """Load and minimally validate the correlation JSON."""

    if not path.exists():
        raise CorrelationReportError(
            f"Correlation report does not exist: {path}"
        )

    if not path.is_file():
        raise CorrelationReportError(
            f"Correlation report is not a file: {path}"
        )

    try:
        raw_text = path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise CorrelationReportError(
            f"Unable to read correlation report: {path}"
        ) from exc

    try:
        report = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CorrelationReportError(
            f"Invalid JSON correlation report: {path}"
        ) from exc

    return dict(
        _require_mapping(
            report,
            "correlation report",
        )
    )


def _candidate_pair_collections(
    report: Mapping[str, Any],
) -> Iterable[Sequence[Any]]:
    """Yield supported pair-list locations."""

    direct_keys = (
        "feature_pairs",
        "correlation_pairs",
        "pairs",
        "top_pairs",
        "high_pairs",
        "high_correlation_pairs",
    )

    for key in direct_keys:
        value = report.get(key)

        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            yield value

    nested_keys = (
        "results",
        "analysis",
        "correlations",
        "report",
        "summary",
    )

    for nested_key in nested_keys:
        nested = report.get(nested_key)

        if not isinstance(nested, Mapping):
            continue

        for key in direct_keys:
            value = nested.get(key)

            if isinstance(value, Sequence) and not isinstance(
                value,
                (str, bytes, bytearray),
            ):
                yield value


def _parse_pair(
    item: Any,
    threshold: float,
) -> CorrelationPair:
    mapping = _require_mapping(
        item,
        "correlation pair",
    )

    feature_a = _normalise_feature_name(
        _first_present(
            mapping,
            (
                "feature_a",
                "feature_1",
                "first_feature",
                "left_feature",
                "source",
            ),
        ),
        "feature_a",
    )

    feature_b = _normalise_feature_name(
        _first_present(
            mapping,
            (
                "feature_b",
                "feature_2",
                "second_feature",
                "right_feature",
                "target",
            ),
        ),
        "feature_b",
    )

    if feature_a == feature_b:
        raise CorrelationReportError(
            "Correlation pair contains the same "
            f"feature twice: {feature_a}"
        )

    pearson = _optional_float(
        _first_present(
            mapping,
            (
                "pearson",
                "pearson_correlation",
                "pearson_value",
            ),
        ),
        "pearson",
    )

    spearman = _optional_float(
        _first_present(
            mapping,
            (
                "spearman",
                "spearman_correlation",
                "spearman_value",
            ),
        ),
        "spearman",
    )

    maximum_value = _first_present(
        mapping,
        (
            "maximum_absolute_correlation",
            "maximum_correlation",
            "max_absolute_correlation",
            "max_correlation",
            "maximum",
            "strongest_correlation",
        ),
    )

    maximum = _optional_float(
        maximum_value,
        "maximum_absolute_correlation",
    )

    if maximum is None:
        available = [
            abs(value)
            for value in (
                pearson,
                spearman,
            )
            if value is not None
        ]

        if not available:
            raise CorrelationReportError(
                "No correlation value found for "
                f"{feature_a}/{feature_b}"
            )

        maximum = max(available)
    else:
        maximum = abs(maximum)

    high_value = _first_present(
        mapping,
        (
            "high_correlation",
            "is_high_correlation",
            "high",
        ),
    )

    if high_value is None:
        high_correlation = (
            maximum >= threshold
        )
    elif isinstance(high_value, bool):
        high_correlation = high_value
    else:
        raise CorrelationReportError(
            "high_correlation must be a boolean when present"
        )

    return CorrelationPair(
        feature_a=feature_a,
        feature_b=feature_b,
        pearson=pearson,
        spearman=spearman,
        maximum_absolute_correlation=maximum,
        high_correlation=high_correlation,
    )


def extract_correlation_pairs(
    report: Mapping[str, Any],
    threshold: float,
) -> tuple[CorrelationPair, ...]:
    """Extract and deduplicate correlation-pair records."""

    collections = list(
        _candidate_pair_collections(
            report
        )
    )

    if not collections:
        raise CorrelationReportError(
            "No feature-pair collection was found "
            "in the correlation report"
        )

    parsed: dict[
        tuple[str, str],
        CorrelationPair,
    ] = {}

    for collection in collections:
        for item in collection:
            pair = _parse_pair(
                item,
                threshold,
            )

            key = tuple(
                sorted(
                    (
                        pair.feature_a,
                        pair.feature_b,
                    )
                )
            )

            previous = parsed.get(key)

            if (
                previous is None
                or pair.maximum_absolute_correlation
                > previous.maximum_absolute_correlation
            ):
                parsed[key] = pair

    if not parsed:
        raise CorrelationReportError(
            "The correlation report contains "
            "no usable feature pairs"
        )

    return tuple(
        sorted(
            parsed.values(),
            key=lambda pair: (
                -pair.maximum_absolute_correlation,
                pair.feature_a,
                pair.feature_b,
            ),
        )
    )


def build_feature_groups(
    pairs: Sequence[CorrelationPair],
    threshold: float,
    minimum_group_size: int = 2,
) -> tuple[FeatureGroup, ...]:
    """Build connected components from qualifying pairs."""

    adjacency: dict[
        str,
        set[str],
    ] = {}

    qualifying_pairs: list[
        CorrelationPair
    ] = []

    for pair in pairs:
        if (
            pair.maximum_absolute_correlation
            < threshold
        ):
            continue

        qualifying_pairs.append(pair)

        adjacency.setdefault(
            pair.feature_a,
            set(),
        ).add(pair.feature_b)

        adjacency.setdefault(
            pair.feature_b,
            set(),
        ).add(pair.feature_a)

    visited: set[str] = set()
    components: list[set[str]] = []

    for feature in sorted(adjacency):
        if feature in visited:
            continue

        stack = [feature]
        component: set[str] = set()

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            component.add(current)

            stack.extend(
                sorted(
                    adjacency.get(
                        current,
                        set(),
                    )
                    - visited
                )
            )

        if len(component) >= minimum_group_size:
            components.append(component)

    components.sort(
        key=lambda component: tuple(
            sorted(component)
        )
    )

    groups: list[FeatureGroup] = []

    for group_id, component in enumerate(
        components,
        start=1,
    ):
        internal_pairs = [
            pair
            for pair in qualifying_pairs
            if (
                pair.feature_a in component
                and pair.feature_b in component
            )
        ]

        maximum = max(
            pair.maximum_absolute_correlation
            for pair in internal_pairs
        )

        groups.append(
            FeatureGroup(
                group_id=group_id,
                features=tuple(
                    sorted(component)
                ),
                pair_count=len(
                    internal_pairs
                ),
                maximum_absolute_correlation=maximum,
            )
        )

    return tuple(groups)


def _feature_redundancy_scores(
    group: FeatureGroup,
    pairs: Sequence[CorrelationPair],
) -> dict[str, tuple[int, float, float]]:
    """
    Return link count, cumulative correlation, and maximum correlation.
    """

    scores: dict[
        str,
        tuple[int, float, float],
    ] = {
        feature: (
            0,
            0.0,
            0.0,
        )
        for feature in group.features
    }

    group_features = set(
        group.features
    )

    for pair in pairs:
        if (
            pair.feature_a not in group_features
            or pair.feature_b not in group_features
        ):
            continue

        for feature in (
            pair.feature_a,
            pair.feature_b,
        ):
            count, total, maximum = (
                scores[feature]
            )

            correlation = (
                pair.maximum_absolute_correlation
            )

            scores[feature] = (
                count + 1,
                total + correlation,
                max(
                    maximum,
                    correlation,
                ),
            )

    return scores


def select_pruning_candidates(
    groups: Sequence[FeatureGroup],
    pairs: Sequence[CorrelationPair],
    maximum_candidates_per_group: int = 1,
) -> tuple[PruningCandidate, ...]:
    """Select deterministic provisional candidates."""

    candidates: list[
        PruningCandidate
    ] = []

    for group in groups:
        scores = _feature_redundancy_scores(
            group,
            pairs,
        )

        ordered_features = sorted(
            group.features,
            key=lambda feature: (
                scores[feature][0],
                scores[feature][1],
                scores[feature][2],
                feature,
            ),
            reverse=True,
        )

        number_to_select = min(
            maximum_candidates_per_group,
            max(
                1,
                len(group.features) - 1,
            ),
        )

        for feature in ordered_features[
            :number_to_select
        ]:
            retained = tuple(
                item
                for item in group.features
                if item != feature
            )

            degree, total, maximum = (
                scores[feature]
            )

            candidates.append(
                PruningCandidate(
                    group_id=group.group_id,
                    feature=feature,
                    retained_features=retained,
                    reason=(
                        "Highest redundancy score in "
                        "correlation group: "
                        f"{degree} correlated links, "
                        "cumulative absolute correlation "
                        f"{total:.6f}. Temporal evaluation "
                        "is required before removal."
                    ),
                    correlation_link_count=degree,
                    cumulative_absolute_correlation=total,
                    maximum_absolute_correlation=maximum,
                )
            )

    return tuple(candidates)


def build_pruning_plan(
    config: PruningConfig,
) -> tuple[
    PruningPlan,
    tuple[CorrelationPair, ...],
]:
    """Build the complete correlation-based pruning plan."""

    config.validated()

    report = load_correlation_report(
        config.correlation_report
    )

    pairs = extract_correlation_pairs(
        report=report,
        threshold=(
            config.correlation_threshold
        ),
    )

    high_pairs = tuple(
        pair
        for pair in pairs
        if (
            pair.maximum_absolute_correlation
            >= config.correlation_threshold
        )
    )

    groups = build_feature_groups(
        pairs=pairs,
        threshold=(
            config.correlation_threshold
        ),
        minimum_group_size=(
            config.minimum_group_size
        ),
    )

    candidates = select_pruning_candidates(
        groups=groups,
        pairs=high_pairs,
        maximum_candidates_per_group=(
            config.maximum_candidates_per_group
        ),
    )

    constant_features_value = report.get(
        "constant_features",
        [],
    )

    if not isinstance(
        constant_features_value,
        Sequence,
    ) or isinstance(
        constant_features_value,
        (str, bytes, bytearray),
    ):
        raise CorrelationReportError(
            "constant_features must be a list"
        )

    constant_features = tuple(
        _normalise_feature_name(
            feature,
            "constant feature",
        )
        for feature in constant_features_value
    )

    plan = PruningPlan(
        status="success",
        correlation_threshold=(
            config.correlation_threshold
        ),
        pair_count=len(pairs),
        reported_total_pair_count=(
            _optional_int(
                report.get(
                    "total_feature_pairs"
                ),
                "total_feature_pairs",
            )
        ),
        high_pair_count=len(high_pairs),
        reported_high_pair_count=(
            _optional_int(
                report.get(
                    "high_correlation_pair_count"
                ),
                "high_correlation_pair_count",
            )
        ),
        constant_features=constant_features,
        group_count=len(groups),
        candidate_count=len(candidates),
        feature_groups=groups,
        pruning_candidates=candidates,
    )

    return (
        plan,
        pairs,
    )


def _normalise_explicit_features(
    features: Sequence[str],
) -> tuple[str, ...]:
    normalised: list[str] = []
    seen: set[str] = set()

    for raw_feature in features:
        feature = raw_feature.strip()

        if feature in seen:
            continue

        seen.add(feature)
        normalised.append(feature)

    return tuple(normalised)


def build_evaluation_candidates(
    config: PruningConfig,
    plan: PruningPlan,
    pairs: Sequence[CorrelationPair],
) -> tuple[EvaluationCandidateSpec, ...]:
    """Choose the single-feature removals to evaluate."""

    model_features = tuple(
        V7RankingDataset.feature_columns()
    )
    model_feature_set = set(
        model_features
    )

    if config.explicit_features:
        explicit_features = (
            _normalise_explicit_features(
                config.explicit_features
            )
        )

        unknown = sorted(
            set(explicit_features)
            - model_feature_set
        )

        if unknown:
            raise ConfigurationError(
                "Unknown explicit V7 features: "
                f"{unknown}"
            )

        specs: list[
            EvaluationCandidateSpec
        ] = []

        for feature in explicit_features:
            group_ids = tuple(
                group.group_id
                for group in plan.feature_groups
                if feature in group.features
            )

            links = [
                pair.maximum_absolute_correlation
                for pair in pairs
                if (
                    pair.maximum_absolute_correlation
                    >= config.correlation_threshold
                    and feature
                    in (
                        pair.feature_a,
                        pair.feature_b,
                    )
                )
            ]

            specs.append(
                EvaluationCandidateSpec(
                    feature=feature,
                    group_ids=group_ids,
                    correlation_link_count=len(
                        links
                    ),
                    cumulative_absolute_correlation=sum(
                        links
                    ),
                    maximum_absolute_correlation=(
                        max(links)
                        if links
                        else 0.0
                    ),
                    source="explicit",
                )
            )

        return tuple(specs)

    if config.candidate_scope == "selected":
        return tuple(
            EvaluationCandidateSpec(
                feature=candidate.feature,
                group_ids=(
                    candidate.group_id,
                ),
                correlation_link_count=(
                    candidate.correlation_link_count
                ),
                cumulative_absolute_correlation=(
                    candidate.cumulative_absolute_correlation
                ),
                maximum_absolute_correlation=(
                    candidate.maximum_absolute_correlation
                ),
                source="selected",
            )
            for candidate
            in plan.pruning_candidates
        )

    high_pairs = tuple(
        pair
        for pair in pairs
        if (
            pair.maximum_absolute_correlation
            >= config.correlation_threshold
        )
    )

    aggregate: dict[
        str,
        dict[str, Any],
    ] = {}

    for group in plan.feature_groups:
        scores = _feature_redundancy_scores(
            group,
            high_pairs,
        )

        for feature in group.features:
            if feature not in model_feature_set:
                raise ConfigurationError(
                    "Correlation report references "
                    "an unknown V7 feature: "
                    f"{feature}"
                )

            count, total, maximum = (
                scores[feature]
            )

            item = aggregate.setdefault(
                feature,
                {
                    "group_ids": set(),
                    "count": 0,
                    "total": 0.0,
                    "maximum": 0.0,
                },
            )

            item["group_ids"].add(
                group.group_id
            )

            item["count"] = max(
                int(item["count"]),
                count,
            )

            item["total"] = max(
                float(item["total"]),
                total,
            )

            item["maximum"] = max(
                float(item["maximum"]),
                maximum,
            )

    specs = [
        EvaluationCandidateSpec(
            feature=feature,
            group_ids=tuple(
                sorted(
                    data["group_ids"]
                )
            ),
            correlation_link_count=int(
                data["count"]
            ),
            cumulative_absolute_correlation=float(
                data["total"]
            ),
            maximum_absolute_correlation=float(
                data["maximum"]
            ),
            source="all_group_features",
        )
        for feature, data in aggregate.items()
    ]

    specs.sort(
        key=lambda spec: (
            -spec.correlation_link_count,
            -spec.cumulative_absolute_correlation,
            -spec.maximum_absolute_correlation,
            spec.feature,
        )
    )

    return tuple(specs)


def _split_dataset_with_purge(
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
    """Create a chronological split with an optional pre-validation purge."""

    target_indices = tuple(
        sorted(
            dataset[
                "target_draw_index"
            ]
            .astype(int)
            .unique()
            .tolist()
        )
    )

    minimum_required = (
        validation_targets
        + purge_targets
        + 1
    )

    if len(target_indices) < minimum_required:
        raise DatasetPreparationError(
            "Not enough targets for the requested "
            "temporal split. "
            f"Available={len(target_indices)}, "
            f"validation={validation_targets}, "
            f"purge={purge_targets}."
        )

    validation_start = (
        len(target_indices)
        - validation_targets
    )

    training_end = (
        validation_start
        - purge_targets
    )

    training_indices = target_indices[
        :training_end
    ]

    purged_indices = target_indices[
        training_end:validation_start
    ]

    validation_indices = target_indices[
        validation_start:
    ]

    if not training_indices:
        raise DatasetPreparationError(
            "Temporal training target set is empty"
        )

    training_dataset = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(training_indices)
        ]
        .copy()
        .reset_index(drop=True)
    )

    validation_dataset = (
        dataset[
            dataset[
                "target_draw_index"
            ].isin(validation_indices)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if training_dataset.empty:
        raise DatasetPreparationError(
            "Temporal training dataset is empty"
        )

    if validation_dataset.empty:
        raise DatasetPreparationError(
            "Temporal validation dataset is empty"
        )

    if set(training_indices).intersection(
        validation_indices
    ):
        raise DatasetPreparationError(
            "Training and validation targets overlap"
        )

    if (
        max(training_indices)
        >= min(validation_indices)
    ):
        raise DatasetPreparationError(
            "Temporal target ordering is invalid"
        )

    return (
        training_dataset,
        validation_dataset,
        training_indices,
        purged_indices,
        validation_indices,
    )


def prepare_temporal_datasets(
    config: PruningConfig,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    TemporalDatasetSummary,
]:
    """Build the shared V7 dataset and purged chronological split."""

    config.validated()

    db = SessionLocal()

    try:
        draws = (
            V7FeatureAblationReport
            ._load_draws(db)
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
        raise DatasetPreparationError(
            "Unable to build the V7 ranking dataset"
        ) from exc
    finally:
        db.close()

    try:
        V7FeatureAblationReport._validate_dataset(
            dataset
        )
    except Exception as exc:
        raise DatasetPreparationError(
            "The generated V7 ranking dataset is invalid"
        ) from exc

    (
        training_dataset,
        validation_dataset,
        training_indices,
        purged_indices,
        validation_indices,
    ) = _split_dataset_with_purge(
        dataset=dataset,
        validation_targets=(
            config.validation_targets
        ),
        purge_targets=(
            config.purge_targets
        ),
    )

    summary = TemporalDatasetSummary(
        draw_count=len(draws),
        dataset_rows=len(dataset),
        dataset_targets=len(metadata),
        training_rows=len(
            training_dataset
        ),
        training_targets=len(
            training_indices
        ),
        validation_rows=len(
            validation_dataset
        ),
        validation_targets=len(
            validation_indices
        ),
        purged_targets=len(
            purged_indices
        ),
        first_training_target=int(
            training_indices[0]
        ),
        last_training_target=int(
            training_indices[-1]
        ),
        purged_target_indices=tuple(
            int(value)
            for value in purged_indices
        ),
        first_validation_target=int(
            validation_indices[0]
        ),
        last_validation_target=int(
            validation_indices[-1]
        ),
    )

    return (
        training_dataset,
        validation_dataset,
        summary,
    )


def evaluate_pruning_plan(
    config: PruningConfig,
    plan: PruningPlan,
    pairs: Sequence[CorrelationPair],
) -> PruningEvaluationReport:
    """Evaluate baseline and all requested single-feature removals."""

    candidates = build_evaluation_candidates(
        config=config,
        plan=plan,
        pairs=pairs,
    )

    if not candidates:
        raise EvaluationError(
            "No feature was selected for temporal evaluation"
        )

    (
        training_dataset,
        validation_dataset,
        dataset_summary,
    ) = prepare_temporal_datasets(
        config
    )

    try:
        baseline = run_feature_subset(
            training_dataset=(
                training_dataset
            ),
            validation_dataset=(
                validation_dataset
            ),
            config=build_baseline_config(
                top_k=config.top_k
            ),
        )
    except Exception as exc:
        raise EvaluationError(
            "Baseline evaluation failed"
        ) from exc

    evaluations: list[
        CandidateEvaluation
    ] = []

    for candidate in candidates:
        try:
            candidate_run = run_feature_subset(
                training_dataset=(
                    training_dataset
                ),
                validation_dataset=(
                    validation_dataset
                ),
                config=(
                    build_single_removal_config(
                        feature_to_remove=(
                            candidate.feature
                        ),
                        top_k=config.top_k,
                    )
                ),
            )

            comparison = compare_feature_runs(
                baseline=baseline,
                candidate=candidate_run,
                tolerance=config.tolerance,
            )
        except Exception as exc:
            raise EvaluationError(
                "Candidate evaluation failed for "
                f"feature: {candidate.feature}"
            ) from exc

        evaluations.append(
            CandidateEvaluation(
                feature=candidate.feature,
                group_ids=(
                    candidate.group_ids
                ),
                correlation_link_count=(
                    candidate
                    .correlation_link_count
                ),
                cumulative_absolute_correlation=(
                    candidate
                    .cumulative_absolute_correlation
                ),
                maximum_absolute_correlation=(
                    candidate
                    .maximum_absolute_correlation
                ),
                source=candidate.source,
                decision=(
                    "ACCEPT"
                    if comparison.accepted
                    else "REJECT"
                ),
                run=candidate_run,
                comparison=comparison,
            )
        )

    accepted = tuple(
        evaluation.feature
        for evaluation in evaluations
        if evaluation.comparison.accepted
    )

    rejected = tuple(
        evaluation.feature
        for evaluation in evaluations
        if not evaluation.comparison.accepted
    )

    accepted_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.comparison.accepted
    ]

    if accepted_evaluations:
        best = max(
            accepted_evaluations,
            key=lambda evaluation: (
                evaluation
                .comparison
                .absolute_delta,
                evaluation
                .run
                .target_hit_rate,
                -evaluation
                .run
                .total_seconds,
                evaluation.feature,
            ),
        )

        best_feature: str | None = (
            best.feature
        )

        best_delta: float | None = (
            best.comparison.absolute_delta
        )
    else:
        best_feature = None
        best_delta = None

    return PruningEvaluationReport(
        status="success",
        version=VERSION,
        protocol=(
            "single chronological holdout; "
            f"{config.purge_targets} target(s) purged "
            "immediately before validation; every "
            "single-feature removal evaluated independently "
            "against one shared all-feature baseline"
        ),
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
        purge_targets=(
            config.purge_targets
        ),
        dataset=dataset_summary,
        plan=plan,
        baseline=baseline,
        candidate_evaluations=tuple(
            evaluations
        ),
        accepted_features=accepted,
        rejected_features=rejected,
        best_single_removal=best_feature,
        best_single_removal_delta=(
            best_delta
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

    if isinstance(value, float):
        if not math.isfinite(value):
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


def _write_csv_rows(
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


def export_pruning_plan(
    plan: PruningPlan,
    output_directory: Path,
) -> dict[str, Path]:
    """Export the Sprint 1 plan for compatibility and auditability."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "feature_pruning_plan.json"
    )

    text_path = (
        output_directory
        / "feature_pruning_plan.txt"
    )

    _write_json(
        asdict(plan),
        json_path,
    )

    lines = [
        "=" * 104,
        "PREDIXA AI V7 FEATURE PRUNING PLAN",
        "=" * 104,
        f"Status                    : {plan.status}",
        (
            "Correlation threshold     : "
            f"{plan.correlation_threshold:.6f}"
        ),
        (
            "Loaded feature pairs      : "
            f"{plan.pair_count}"
        ),
        (
            "Reported feature pairs    : "
            f"{plan.reported_total_pair_count}"
        ),
        (
            "Loaded high pairs         : "
            f"{plan.high_pair_count}"
        ),
        (
            "Reported high pairs       : "
            f"{plan.reported_high_pair_count}"
        ),
        (
            "Constant features         : "
            + (
                ", ".join(
                    plan.constant_features
                )
                if plan.constant_features
                else "none"
            )
        ),
        f"Feature groups            : {plan.group_count}",
        (
            "Provisional candidates    : "
            f"{plan.candidate_count}"
        ),
        "",
        "FEATURE GROUPS",
        "-" * 104,
    ]

    if not plan.feature_groups:
        lines.append(
            "No group exceeded the selected threshold."
        )
    else:
        for group in plan.feature_groups:
            lines.extend(
                [
                    f"Group {group.group_id}",
                    (
                        "Features                  : "
                        f"{', '.join(group.features)}"
                    ),
                    (
                        "Pair count                : "
                        f"{group.pair_count}"
                    ),
                    (
                        "Maximum correlation       : "
                        f"{group.maximum_absolute_correlation:.6f}"
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "PROVISIONAL CANDIDATES",
            "-" * 104,
        ]
    )

    if not plan.pruning_candidates:
        lines.append(
            "No provisional candidate was selected."
        )
    else:
        for candidate in plan.pruning_candidates:
            lines.extend(
                [
                    f"Group {candidate.group_id}",
                    (
                        "Candidate                 : "
                        f"{candidate.feature}"
                    ),
                    (
                        "Retained group features   : "
                        f"{', '.join(candidate.retained_features)}"
                    ),
                    (
                        "Correlation links         : "
                        f"{candidate.correlation_link_count}"
                    ),
                    (
                        "Cumulative correlation    : "
                        f"{candidate.cumulative_absolute_correlation:.6f}"
                    ),
                    (
                        "Maximum correlation       : "
                        f"{candidate.maximum_absolute_correlation:.6f}"
                    ),
                    (
                        "Reason                    : "
                        f"{candidate.reason}"
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "=" * 104,
            (
                "This plan does not remove features. "
                "Sprint 2 temporal evaluation is required."
            ),
            "=" * 104,
        ]
    )

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return {
        "plan_json": json_path.resolve(),
        "plan_text": text_path.resolve(),
    }


def export_evaluation_report(
    report: PruningEvaluationReport,
    output_directory: Path,
) -> dict[str, Path]:
    """Export complete Sprint 2 reports."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_directory
        / "feature_pruning_evaluation.json"
    )

    text_path = (
        output_directory
        / "feature_pruning_evaluation.txt"
    )

    summary_csv_path = (
        output_directory
        / "feature_pruning_evaluation.csv"
    )

    accepted_csv_path = (
        output_directory
        / "accepted_removals.csv"
    )

    rejected_csv_path = (
        output_directory
        / "rejected_removals.csv"
    )

    _write_json(
        asdict(report),
        json_path,
    )

    summary_rows: list[
        dict[str, Any]
    ] = []

    for evaluation in (
        report.candidate_evaluations
    ):
        summary_rows.append(
            {
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
                "candidate_fit_seconds": (
                    evaluation
                    .run
                    .fit_seconds
                ),
                "candidate_total_seconds": (
                    evaluation
                    .run
                    .total_seconds
                ),
                "tolerance": (
                    evaluation
                    .comparison
                    .tolerance
                ),
                "decision": (
                    evaluation.decision
                ),
            }
        )

    fieldnames = (
        "feature",
        "group_ids",
        "source",
        "correlation_link_count",
        "cumulative_absolute_correlation",
        "maximum_absolute_correlation",
        "baseline_mean_hits_at_k",
        "candidate_mean_hits_at_k",
        "absolute_delta",
        "relative_delta",
        "candidate_target_hit_rate",
        "candidate_total_hits",
        "candidate_fit_seconds",
        "candidate_total_seconds",
        "tolerance",
        "decision",
    )

    _write_csv_rows(
        summary_rows,
        summary_csv_path,
        fieldnames,
    )

    _write_csv_rows(
        [
            row
            for row in summary_rows
            if row["decision"] == "ACCEPT"
        ],
        accepted_csv_path,
        fieldnames,
    )

    _write_csv_rows(
        [
            row
            for row in summary_rows
            if row["decision"] == "REJECT"
        ],
        rejected_csv_path,
        fieldnames,
    )

    lines = [
        "=" * 132,
        "PREDIXA AI V7 FEATURE PRUNING TEMPORAL EVALUATION",
        "=" * 132,
        f"Status                         : {report.status}",
        f"Version                        : {report.version}",
        f"Protocol                       : {report.protocol}",
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
            "Validation targets requested   : "
            f"{report.validation_targets}"
        ),
        f"Top K                          : {report.top_k}",
        f"Purged targets                 : {report.purge_targets}",
        "",
        "TEMPORAL DATASET",
        "-" * 132,
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
            "Training rows                  : "
            f"{report.dataset.training_rows}"
        ),
        (
            "Training targets               : "
            f"{report.dataset.training_targets}"
        ),
        (
            "Training target range          : "
            f"{report.dataset.first_training_target}"
            " -> "
            f"{report.dataset.last_training_target}"
        ),
        (
            "Purged target indices          : "
            + (
                ", ".join(
                    str(value)
                    for value
                    in report
                    .dataset
                    .purged_target_indices
                )
                if (
                    report
                    .dataset
                    .purged_target_indices
                )
                else "none"
            )
        ),
        (
            "Validation rows                : "
            f"{report.dataset.validation_rows}"
        ),
        (
            "Validation targets             : "
            f"{report.dataset.validation_targets}"
        ),
        (
            "Validation target range        : "
            f"{report.dataset.first_validation_target}"
            " -> "
            f"{report.dataset.last_validation_target}"
        ),
        "",
        "BASELINE",
        "-" * 132,
        (
            "Features                       : "
            f"{report.baseline.feature_count}"
        ),
        (
            "Mean Hits@K                    : "
            f"{report.baseline.mean_hits_at_k:.6f}"
        ),
        (
            "Normalized Hits@K              : "
            f"{report.baseline.normalized_hits_at_k:.6f}"
        ),
        (
            "Target hit rate                : "
            f"{report.baseline.target_hit_rate:.6f}"
        ),
        (
            "Total hits                     : "
            f"{report.baseline.total_hits}"
        ),
        (
            "Fit seconds                    : "
            f"{report.baseline.fit_seconds:.6f}"
        ),
        (
            "Total seconds                  : "
            f"{report.baseline.total_seconds:.6f}"
        ),
        "",
        "SINGLE-FEATURE REMOVAL EXPERIMENTS",
        "-" * 132,
        (
            f"{'Feature':<26}"
            f"{'Links':>8}"
            f"{'MaxCorr':>12}"
            f"{'Base Hits':>14}"
            f"{'Candidate':>14}"
            f"{'Delta':>12}"
            f"{'Hit Rate':>12}"
            f"{'Seconds':>12}"
            f"{'Decision':>12}"
        ),
        "-" * 132,
    ]

    for evaluation in (
        report.candidate_evaluations
    ):
        lines.append(
            f"{evaluation.feature:<26}"
            f"{evaluation.correlation_link_count:>8}"
            f"{evaluation.maximum_absolute_correlation:>12.6f}"
            f"{evaluation.comparison.baseline_mean_hits_at_k:>14.6f}"
            f"{evaluation.comparison.candidate_mean_hits_at_k:>14.6f}"
            f"{evaluation.comparison.absolute_delta:>12.6f}"
            f"{evaluation.run.target_hit_rate:>12.6f}"
            f"{evaluation.run.total_seconds:>12.3f}"
            f"{evaluation.decision:>12}"
        )

    lines.extend(
        [
            "",
            "RECOMMENDATION",
            "-" * 132,
            (
                "Accepted independent removals  : "
                + (
                    ", ".join(
                        report.accepted_features
                    )
                    if report.accepted_features
                    else "none"
                )
            ),
            (
                "Rejected independent removals  : "
                + (
                    ", ".join(
                        report.rejected_features
                    )
                    if report.rejected_features
                    else "none"
                )
            ),
            (
                "Best single removal            : "
                f"{report.best_single_removal}"
            ),
            (
                "Best single-removal delta      : "
                + (
                    f"{report.best_single_removal_delta:.6f}"
                    if (
                        report
                        .best_single_removal_delta
                        is not None
                    )
                    else "none"
                )
            ),
            "",
            (
                "Important: accepted features were tested "
                "independently. This report does not prove that "
                "removing several accepted features together will "
                "preserve performance. A cumulative greedy Sprint 3 "
                "must validate combined removals."
            ),
            "=" * 132,
        ]
    )

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return {
        "evaluation_json": (
            json_path.resolve()
        ),
        "evaluation_text": (
            text_path.resolve()
        ),
        "evaluation_csv": (
            summary_csv_path.resolve()
        ),
        "accepted_csv": (
            accepted_csv_path.resolve()
        ),
        "rejected_csv": (
            rejected_csv_path.resolve()
        ),
    }


def print_pruning_plan(
    plan: PruningPlan,
    generated_files: Mapping[str, Path],
) -> None:
    """Print a compact Sprint 1 console summary."""

    print("=" * 104)
    print(
        "PREDIXA AI V7 FEATURE PRUNING PLAN"
    )
    print("=" * 104)
    print(
        f"Status                    : "
        f"{plan.status}"
    )
    print(
        "Correlation threshold     : "
        f"{plan.correlation_threshold:.6f}"
    )
    print(
        f"Loaded feature pairs      : "
        f"{plan.pair_count}"
    )
    print(
        f"Reported feature pairs    : "
        f"{plan.reported_total_pair_count}"
    )
    print(
        f"Loaded high pairs         : "
        f"{plan.high_pair_count}"
    )
    print(
        f"Feature groups            : "
        f"{plan.group_count}"
    )
    print(
        f"Provisional candidates    : "
        f"{plan.candidate_count}"
    )
    print()
    print("FEATURE GROUPS")
    print("-" * 104)

    if not plan.feature_groups:
        print(
            "No group exceeded the selected threshold."
        )
    else:
        for group in plan.feature_groups:
            print(
                f"Group {group.group_id}: "
                f"{', '.join(group.features)} "
                f"(max="
                f"{group.maximum_absolute_correlation:.6f})"
            )

    print()
    print("PROVISIONAL CANDIDATES")
    print("-" * 104)

    if not plan.pruning_candidates:
        print("No candidate selected.")
    else:
        for candidate in (
            plan.pruning_candidates
        ):
            print(
                f"Group {candidate.group_id}: "
                f"test removal of "
                f"{candidate.feature}"
            )

    print()
    print("GENERATED FILES")
    print("-" * 104)

    for name, path in (
        generated_files.items()
    ):
        print(
            f"{name.upper():28}: "
            f"{path}"
        )

    print()
    print("=" * 104)
    print("PLAN SUCCESS")
    print("=" * 104)


def print_evaluation_report(
    report: PruningEvaluationReport,
    generated_files: Mapping[str, Path],
) -> None:
    """Print the essential Sprint 2 result."""

    print("=" * 132)
    print(
        "PREDIXA AI V7 FEATURE PRUNING "
        "TEMPORAL EVALUATION"
    )
    print("=" * 132)
    print(
        f"Status                  : "
        f"{report.status}"
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
        f"Baseline features       : "
        f"{report.baseline.feature_count}"
    )
    print(
        f"Baseline mean Hits@K    : "
        f"{report.baseline.mean_hits_at_k:.6f}"
    )
    print(
        f"Acceptance tolerance    : "
        f"{report.tolerance:.6f}"
    )
    print()
    print(
        f"{'Feature':<26}"
        f"{'Candidate Hits':>16}"
        f"{'Delta':>12}"
        f"{'Hit Rate':>12}"
        f"{'Decision':>12}"
    )
    print("-" * 132)

    for evaluation in (
        report.candidate_evaluations
    ):
        print(
            f"{evaluation.feature:<26}"
            f"{evaluation.comparison.candidate_mean_hits_at_k:>16.6f}"
            f"{evaluation.comparison.absolute_delta:>12.6f}"
            f"{evaluation.run.target_hit_rate:>12.6f}"
            f"{evaluation.decision:>12}"
        )

    print()
    print(
        "Accepted independent removals : "
        + (
            ", ".join(
                report.accepted_features
            )
            if report.accepted_features
            else "none"
        )
    )
    print(
        "Rejected independent removals : "
        + (
            ", ".join(
                report.rejected_features
            )
            if report.rejected_features
            else "none"
        )
    )
    print(
        "Best single removal           : "
        f"{report.best_single_removal}"
    )
    print()
    print("GENERATED FILES")
    print("-" * 132)

    for name, path in (
        generated_files.items()
    ):
        print(
            f"{name.upper():28}: "
            f"{path}"
        )

    print()
    print("=" * 132)
    print("SUCCESS")
    print("=" * 132)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Plan and temporally evaluate PredixaAI V7 "
            "single-feature pruning candidates."
        )
    )

    parser.add_argument(
        "--correlation-report",
        type=Path,
        default=(
            DEFAULT_CORRELATION_REPORT
        ),
        help=(
            "Path to "
            "feature_correlation_report.json"
        ),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=(
            DEFAULT_OUTPUT_DIRECTORY
        ),
        help=(
            "Directory used for pruning exports"
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
            "to construct feature groups"
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
        help=(
            "V7 historical feature window"
        ),
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
        help=(
            "Ranking cutoff used for Hits@K"
        ),
    )

    parser.add_argument(
        "--purge-targets",
        type=int,
        default=(
            DEFAULT_PURGE_TARGETS
        ),
        help=(
            "Number of targets removed immediately "
            "before the validation window"
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
        "--candidate-scope",
        choices=CANDIDATE_SCOPES,
        default=(
            DEFAULT_CANDIDATE_SCOPE
        ),
        help=(
            "'selected' tests only provisional Sprint 1 "
            "candidates; 'all_group_features' tests every "
            "feature in a high-correlation group"
        ),
    )

    parser.add_argument(
        "--features",
        nargs="*",
        default=(),
        help=(
            "Optional explicit features to test. "
            "Overrides candidate-scope."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Build correlation plan without "
            "training models"
        ),
    )

    return parser


def main() -> int:
    """CLI entry point."""

    parser = build_argument_parser()
    arguments = parser.parse_args()

    config = PruningConfig(
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
            arguments
            .correlation_threshold
        ),
        minimum_group_size=(
            arguments
            .minimum_group_size
        ),
        maximum_candidates_per_group=(
            arguments
            .maximum_candidates_per_group
        ),
        window_size=(
            arguments.window_size
        ),
        max_training_targets=(
            arguments
            .max_training_targets
        ),
        validation_targets=(
            arguments
            .validation_targets
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
        plan_only=arguments.plan_only,
    )

    try:
        plan, pairs = build_pruning_plan(
            config
        )

        plan_files = export_pruning_plan(
            plan=plan,
            output_directory=(
                config.output_directory
            ),
        )

        if config.plan_only:
            print_pruning_plan(
                plan=plan,
                generated_files=(
                    plan_files
                ),
            )

            return 0

        evaluation = evaluate_pruning_plan(
            config=config,
            plan=plan,
            pairs=pairs,
        )

        evaluation_files = (
            export_evaluation_report(
                report=evaluation,
                output_directory=(
                    config.output_directory
                ),
            )
        )

        generated_files = {
            **plan_files,
            **evaluation_files,
        }

    except FeaturePruningError as exc:
        print("=" * 104)
        print(
            "PREDIXA AI V7 FEATURE PRUNING"
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

    print_evaluation_report(
        report=evaluation,
        generated_files=(
            generated_files
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
