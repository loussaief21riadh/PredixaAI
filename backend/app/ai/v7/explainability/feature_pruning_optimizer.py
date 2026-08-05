"""
PredixaAI V7 - Feature Pruning Optimizer.

Sprint 1 responsibilities:
- Validate pruning configuration.
- Load the existing feature-correlation JSON report.
- Extract highly correlated feature pairs.
- Build connected feature groups.
- Select deterministic pruning candidates.
- Export a preliminary pruning plan.

This version does not retrain the machine-learning model yet.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_CORRELATION_REPORT = Path(
    "reports/v7/feature_correlation/feature_correlation_report.json"
)

DEFAULT_OUTPUT_DIRECTORY = Path("reports/v7/feature_pruning")

DEFAULT_CORRELATION_THRESHOLD = 0.80


class FeaturePruningError(RuntimeError):
    """Base exception for feature-pruning failures."""


class ConfigurationError(FeaturePruningError):
    """Raised when the pruning configuration is invalid."""


class CorrelationReportError(FeaturePruningError):
    """Raised when the correlation report is missing or malformed."""


@dataclass(frozen=True)
class PruningConfig:
    """Configuration used to construct the preliminary pruning plan."""

    correlation_report: Path = DEFAULT_CORRELATION_REPORT
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
    minimum_group_size: int = 2
    maximum_candidates_per_group: int = 1

    def validated(self) -> "PruningConfig":
        """Validate this configuration and return it unchanged."""

        validate_config(self)
        return self


@dataclass(frozen=True)
class CorrelationPair:
    """One correlation relationship between two model features."""

    feature_a: str
    feature_b: str
    pearson: float | None
    spearman: float | None
    maximum_absolute_correlation: float
    high_correlation: bool


@dataclass(frozen=True)
class FeatureGroup:
    """Connected group of mutually related features."""

    group_id: int
    features: tuple[str, ...]
    pair_count: int
    maximum_absolute_correlation: float


@dataclass(frozen=True)
class PruningCandidate:
    """Feature proposed for removal during a future model evaluation."""

    group_id: int
    feature: str
    retained_features: tuple[str, ...]
    reason: str
    maximum_absolute_correlation: float


@dataclass(frozen=True)
class PruningPlan:
    """Complete result of the Sprint 1 pruning analysis."""

    status: str
    correlation_threshold: float
    pair_count: int
    high_pair_count: int
    group_count: int
    candidate_count: int
    feature_groups: tuple[FeatureGroup, ...]
    pruning_candidates: tuple[PruningCandidate, ...]


def validate_config(config: PruningConfig) -> None:
    """Validate all pruning parameters."""

    if not isinstance(config.correlation_report, Path):
        raise ConfigurationError("correlation_report must be a pathlib.Path")

    if not isinstance(config.output_directory, Path):
        raise ConfigurationError("output_directory must be a pathlib.Path")

    if not math.isfinite(config.correlation_threshold):
        raise ConfigurationError("correlation_threshold must be finite")

    if not 0.0 < config.correlation_threshold <= 1.0:
        raise ConfigurationError(
            "correlation_threshold must be greater than 0 and at most 1"
        )

    if config.minimum_group_size < 2:
        raise ConfigurationError("minimum_group_size must be at least 2")

    if config.maximum_candidates_per_group < 1:
        raise ConfigurationError(
            "maximum_candidates_per_group must be at least 1"
        )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrelationReportError(f"{name} must be a JSON object")
    return value


def _normalise_feature_name(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise CorrelationReportError(f"{field_name} must be a string")

    feature = value.strip()

    if not feature:
        raise CorrelationReportError(f"{field_name} cannot be empty")

    return feature


def _optional_float(value: Any, field_name: str) -> float | None:
    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CorrelationReportError(
            f"{field_name} must be numeric or null"
        )

    result = float(value)

    if not math.isfinite(result):
        raise CorrelationReportError(f"{field_name} must be finite")

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


def load_correlation_report(path: Path) -> dict[str, Any]:
    """Load and minimally validate the feature-correlation report."""

    if not path.exists():
        raise CorrelationReportError(
            f"Correlation report does not exist: {path}"
        )

    if not path.is_file():
        raise CorrelationReportError(
            f"Correlation report is not a file: {path}"
        )

    try:
        raw_text = path.read_text(encoding="utf-8")
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

    return dict(_require_mapping(report, "correlation report"))


def _candidate_pair_collections(
    report: Mapping[str, Any],
) -> Iterable[Sequence[Any]]:
    """Yield supported locations for correlation-pair records."""

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
            value, (str, bytes, bytearray)
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
                value, (str, bytes, bytearray)
            ):
                yield value


def _parse_pair(item: Any, threshold: float) -> CorrelationPair:
    mapping = _require_mapping(item, "correlation pair")

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
            f"Correlation pair contains the same feature twice: {feature_a}"
        )

    pearson = _optional_float(
        _first_present(
            mapping,
            ("pearson", "pearson_correlation", "pearson_value"),
        ),
        "pearson",
    )

    spearman = _optional_float(
        _first_present(
            mapping,
            ("spearman", "spearman_correlation", "spearman_value"),
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
            for value in (pearson, spearman)
            if value is not None
        ]

        if not available:
            raise CorrelationReportError(
                f"No correlation value found for {feature_a}/{feature_b}"
            )

        maximum = max(available)
    else:
        maximum = abs(maximum)

    high_value = _first_present(
        mapping,
        ("high_correlation", "is_high_correlation", "high"),
    )

    if high_value is None:
        high_correlation = maximum >= threshold
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
    """Extract supported correlation-pair records from a JSON report."""

    collections = list(_candidate_pair_collections(report))

    if not collections:
        raise CorrelationReportError(
            "No feature-pair collection was found in the correlation report"
        )

    parsed: dict[tuple[str, str], CorrelationPair] = {}

    for collection in collections:
        for item in collection:
            pair = _parse_pair(item, threshold)

            key = tuple(sorted((pair.feature_a, pair.feature_b)))

            previous = parsed.get(key)

            if (
                previous is None
                or pair.maximum_absolute_correlation
                > previous.maximum_absolute_correlation
            ):
                parsed[key] = pair

    if not parsed:
        raise CorrelationReportError(
            "The correlation report contains no usable feature pairs"
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
    """Build connected components from highly correlated feature pairs."""

    adjacency: dict[str, set[str]] = {}
    qualifying_pairs: list[CorrelationPair] = []

    for pair in pairs:
        if pair.maximum_absolute_correlation < threshold:
            continue

        qualifying_pairs.append(pair)
        adjacency.setdefault(pair.feature_a, set()).add(pair.feature_b)
        adjacency.setdefault(pair.feature_b, set()).add(pair.feature_a)

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
            stack.extend(sorted(adjacency.get(current, ()) - visited))

        if len(component) >= minimum_group_size:
            components.append(component)

    components.sort(key=lambda component: tuple(sorted(component)))

    groups: list[FeatureGroup] = []

    for group_id, component in enumerate(components, start=1):
        internal_pairs = [
            pair
            for pair in qualifying_pairs
            if pair.feature_a in component and pair.feature_b in component
        ]

        maximum = max(
            pair.maximum_absolute_correlation
            for pair in internal_pairs
        )

        groups.append(
            FeatureGroup(
                group_id=group_id,
                features=tuple(sorted(component)),
                pair_count=len(internal_pairs),
                maximum_absolute_correlation=maximum,
            )
        )

    return tuple(groups)


def _feature_redundancy_scores(
    group: FeatureGroup,
    pairs: Sequence[CorrelationPair],
) -> dict[str, tuple[int, float]]:
    """Return degree and summed correlation for every feature."""

    scores: dict[str, tuple[int, float]] = {
        feature: (0, 0.0)
        for feature in group.features
    }

    group_features = set(group.features)

    for pair in pairs:
        if (
            pair.feature_a not in group_features
            or pair.feature_b not in group_features
        ):
            continue

        for feature in (pair.feature_a, pair.feature_b):
            count, total = scores[feature]
            scores[feature] = (
                count + 1,
                total + pair.maximum_absolute_correlation,
            )

    return scores


def select_pruning_candidates(
    groups: Sequence[FeatureGroup],
    pairs: Sequence[CorrelationPair],
    maximum_candidates_per_group: int = 1,
) -> tuple[PruningCandidate, ...]:
    """
    Select deterministic candidates.

    Sprint 1 heuristic:
    - Prefer the feature participating in the largest number of correlations.
    - Then prefer the largest cumulative absolute correlation.
    - Then use reverse lexical order for stable tie-breaking.

    Actual removal will only be accepted in Sprint 2 after temporal evaluation.
    """

    candidates: list[PruningCandidate] = []

    for group in groups:
        scores = _feature_redundancy_scores(group, pairs)

        ordered_features = sorted(
            group.features,
            key=lambda feature: (
                scores[feature][0],
                scores[feature][1],
                feature,
            ),
            reverse=True,
        )

        number_to_select = min(
            maximum_candidates_per_group,
            max(1, len(group.features) - 1),
        )

        for feature in ordered_features[:number_to_select]:
            retained = tuple(
                item
                for item in group.features
                if item != feature
            )

            degree, total = scores[feature]

            candidates.append(
                PruningCandidate(
                    group_id=group.group_id,
                    feature=feature,
                    retained_features=retained,
                    reason=(
                        "Highest redundancy score in correlation group: "
                        f"{degree} correlated links, "
                        f"cumulative absolute correlation {total:.6f}. "
                        "Removal remains provisional until temporal "
                        "model evaluation."
                    ),
                    maximum_absolute_correlation=(
                        group.maximum_absolute_correlation
                    ),
                )
            )

    return tuple(candidates)


def build_pruning_plan(config: PruningConfig) -> PruningPlan:
    """Build the complete preliminary pruning plan."""

    config.validated()

    report = load_correlation_report(config.correlation_report)

    pairs = extract_correlation_pairs(
        report=report,
        threshold=config.correlation_threshold,
    )

    high_pairs = tuple(
        pair
        for pair in pairs
        if pair.maximum_absolute_correlation
        >= config.correlation_threshold
    )

    groups = build_feature_groups(
        pairs=pairs,
        threshold=config.correlation_threshold,
        minimum_group_size=config.minimum_group_size,
    )

    candidates = select_pruning_candidates(
        groups=groups,
        pairs=high_pairs,
        maximum_candidates_per_group=(
            config.maximum_candidates_per_group
        ),
    )

    return PruningPlan(
        status="success",
        correlation_threshold=config.correlation_threshold,
        pair_count=len(pairs),
        high_pair_count=len(high_pairs),
        group_count=len(groups),
        candidate_count=len(candidates),
        feature_groups=groups,
        pruning_candidates=candidates,
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, float):
        if not math.isfinite(value):
            return None

    return value


def export_pruning_plan(
    plan: PruningPlan,
    output_directory: Path,
) -> dict[str, Path]:
    """Export the preliminary plan as JSON and text."""

    output_directory.mkdir(parents=True, exist_ok=True)

    json_path = output_directory / "feature_pruning_plan.json"
    text_path = output_directory / "feature_pruning_plan.txt"

    payload = _json_safe(asdict(plan))

    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    lines = [
        "=" * 100,
        "PREDIXA AI V7 FEATURE PRUNING PLAN",
        "=" * 100,
        f"Status                  : {plan.status}",
        (
            "Correlation threshold   : "
            f"{plan.correlation_threshold:.6f}"
        ),
        f"Total feature pairs     : {plan.pair_count}",
        f"High-correlation pairs  : {plan.high_pair_count}",
        f"Feature groups          : {plan.group_count}",
        f"Pruning candidates      : {plan.candidate_count}",
        "",
        "FEATURE GROUPS",
        "-" * 100,
    ]

    if not plan.feature_groups:
        lines.append("No feature group exceeded the selected threshold.")
    else:
        for group in plan.feature_groups:
            lines.extend(
                [
                    f"Group {group.group_id}",
                    f"Features                : {', '.join(group.features)}",
                    f"Pair count              : {group.pair_count}",
                    (
                        "Maximum correlation      : "
                        f"{group.maximum_absolute_correlation:.6f}"
                    ),
                    "",
                ]
            )

    lines.extend(
        [
            "PRUNING CANDIDATES",
            "-" * 100,
        ]
    )

    if not plan.pruning_candidates:
        lines.append("No provisional pruning candidate was selected.")
    else:
        for candidate in plan.pruning_candidates:
            lines.extend(
                [
                    f"Group {candidate.group_id}",
                    f"Candidate               : {candidate.feature}",
                    (
                        "Retained features        : "
                        f"{', '.join(candidate.retained_features)}"
                    ),
                    (
                        "Maximum correlation      : "
                        f"{candidate.maximum_absolute_correlation:.6f}"
                    ),
                    f"Reason                  : {candidate.reason}",
                    "",
                ]
            )

    lines.extend(
        [
            "=" * 100,
            "SPRINT 1 COMPLETE",
            (
                "No feature has been removed. Candidates must be evaluated "
                "against the temporal Hits@5 baseline during Sprint 2."
            ),
            "=" * 100,
        ]
    )

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "text": text_path,
    }


def print_pruning_plan(
    plan: PruningPlan,
    generated_files: Mapping[str, Path],
) -> None:
    """Print a readable console summary."""

    print("=" * 100)
    print("PREDIXA AI V7 FEATURE PRUNING PLAN")
    print("=" * 100)
    print(f"Status                  : {plan.status}")
    print(
        "Correlation threshold   : "
        f"{plan.correlation_threshold:.6f}"
    )
    print(f"Total feature pairs     : {plan.pair_count}")
    print(f"High-correlation pairs  : {plan.high_pair_count}")
    print(f"Feature groups          : {plan.group_count}")
    print(f"Pruning candidates      : {plan.candidate_count}")
    print()

    print("FEATURE GROUPS")
    print("-" * 100)

    if not plan.feature_groups:
        print("No feature group exceeded the selected threshold.")
    else:
        for group in plan.feature_groups:
            print(
                f"Group {group.group_id}: "
                f"{', '.join(group.features)} "
                f"(max={group.maximum_absolute_correlation:.6f})"
            )

    print()
    print("PROVISIONAL CANDIDATES")
    print("-" * 100)

    if not plan.pruning_candidates:
        print("No candidate selected.")
    else:
        for candidate in plan.pruning_candidates:
            print(
                f"Group {candidate.group_id}: "
                f"test removal of {candidate.feature}"
            )

    print()
    print("GENERATED FILES")
    print("-" * 100)

    for name, path in generated_files.items():
        print(f"{name.upper():24}: {path.resolve()}")

    print()
    print("=" * 100)
    print("SUCCESS")
    print("=" * 100)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a preliminary feature-pruning plan from the PredixaAI "
            "V7 feature-correlation report."
        )
    )

    parser.add_argument(
        "--correlation-report",
        type=Path,
        default=DEFAULT_CORRELATION_REPORT,
        help="Path to feature_correlation_report.json",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory used for pruning-plan exports",
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=DEFAULT_CORRELATION_THRESHOLD,
        help="Minimum absolute correlation used to create groups",
    )

    parser.add_argument(
        "--minimum-group-size",
        type=int,
        default=2,
        help="Minimum number of features required in a group",
    )

    parser.add_argument(
        "--maximum-candidates-per-group",
        type=int,
        default=1,
        help="Maximum provisional removals selected per group",
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args()

    config = PruningConfig(
        correlation_report=arguments.correlation_report,
        output_directory=arguments.output_directory,
        correlation_threshold=arguments.correlation_threshold,
        minimum_group_size=arguments.minimum_group_size,
        maximum_candidates_per_group=(
            arguments.maximum_candidates_per_group
        ),
    )

    try:
        plan = build_pruning_plan(config)
        generated_files = export_pruning_plan(
            plan=plan,
            output_directory=config.output_directory,
        )
    except FeaturePruningError as exc:
        print("=" * 100)
        print("PREDIXA AI V7 FEATURE PRUNING PLAN")
        print("=" * 100)
        print(f"ERROR: {exc}")
        print("=" * 100)
        return 1

    print_pruning_plan(
        plan=plan,
        generated_files=generated_files,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
