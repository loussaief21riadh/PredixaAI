from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.ai.v7.explainability import feature_ablation_runner as runner


MODEL_FEATURES = (
    "feature_a",
    "feature_b",
    "feature_c",
)


class DeterministicModel:
    """
    Minimal deterministic classifier used by the unit tests.

    Lower feature_a values receive larger positive-class probabilities.
    With feature_a equal to candidate_number, candidates 1..5 rank first.
    """

    def __init__(self) -> None:
        self.classes_ = np.array([0, 1])
        self.fit_columns: tuple[str, ...] | None = None
        self.fit_rows = 0

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "DeterministicModel":
        self.fit_columns = tuple(features.columns)
        self.fit_rows = len(features)

        if len(features) != len(target):
            raise AssertionError(
                "Feature and target row counts differ."
            )

        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        values = (
            pd.to_numeric(
                features.iloc[:, 0],
                errors="raise",
            )
            .astype(float)
            .to_numpy()
        )

        positive = 1.0 - (values / 100.0)
        positive = np.clip(positive, 0.0, 1.0)
        negative = 1.0 - positive

        return np.column_stack(
            [
                negative,
                positive,
            ]
        )


class ZeroOnlyModel:
    """Classifier exposing only class 0."""

    def __init__(self) -> None:
        self.classes_ = np.array([0])

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "ZeroOnlyModel":
        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.ones(
            (len(features), 1),
            dtype=float,
        )


class UnsupportedClassModel:
    """Classifier exposing an unsupported class set."""

    def __init__(self) -> None:
        self.classes_ = np.array([2])

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> "UnsupportedClassModel":
        return self

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> np.ndarray:
        return np.ones(
            (len(features), 1),
            dtype=float,
        )


@pytest.fixture(autouse=True)
def patch_model_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.V7RankingDataset,
        "feature_columns",
        classmethod(
            lambda cls: MODEL_FEATURES
        ),
    )


def build_dataset(
    target_indices: list[int],
    *,
    positive_numbers: tuple[int, ...] = (
        1,
        2,
        3,
        4,
        5,
    ),
) -> pd.DataFrame:
    """Build a valid candidate-level dataset with 49 rows per target."""

    rows: list[dict[str, Any]] = []

    for target_index in target_indices:
        target_date = (
            pd.Timestamp("2025-01-01")
            + pd.Timedelta(days=target_index)
        )

        for candidate_number in range(1, 50):
            rows.append(
                {
                    "candidate_number": candidate_number,
                    "target": int(
                        candidate_number
                        in positive_numbers
                    ),
                    "target_draw_index": target_index,
                    "target_draw_date": (
                        target_date.date().isoformat()
                    ),
                    "feature_a": float(
                        candidate_number
                    ),
                    "feature_b": float(
                        candidate_number % 7
                    ),
                    "feature_c": float(
                        target_index
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_result(
    *,
    name: str,
    features: tuple[str, ...],
    mean_hits: float,
    validation_targets: int = 2,
    top_k: int = 5,
) -> runner.FeatureAblationRunResult:
    """Build a compact result object for comparison tests."""

    return runner.FeatureAblationRunResult(
        experiment_name=name,
        feature_columns=features,
        feature_count=len(features),
        removed_features=tuple(
            feature
            for feature in MODEL_FEATURES
            if feature not in features
        ),
        top_k=top_k,
        training_rows=98,
        validation_rows=98,
        training_targets=2,
        validation_targets=validation_targets,
        fit_seconds=0.1,
        prediction_seconds=0.1,
        total_seconds=0.2,
        total_hits=int(
            mean_hits * validation_targets
        ),
        mean_hits_at_k=mean_hits,
        normalized_hits_at_k=(
            mean_hits / top_k
        ),
        targets_with_at_least_one_hit=(
            validation_targets
        ),
        target_hit_rate=1.0,
        target_evaluations=(),
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            runner.FeatureAblationRunConfig(
                experiment_name="",
                feature_columns=MODEL_FEATURES,
            ),
            "experiment_name cannot be empty",
        ),
        (
            runner.FeatureAblationRunConfig(
                experiment_name="baseline",
                feature_columns=(),
            ),
            "At least one feature column",
        ),
        (
            runner.FeatureAblationRunConfig(
                experiment_name="baseline",
                feature_columns=(
                    "feature_a",
                    "feature_a",
                ),
            ),
            "Duplicate feature name",
        ),
        (
            runner.FeatureAblationRunConfig(
                experiment_name="baseline",
                feature_columns=(
                    "unknown_feature",
                ),
            ),
            "Unknown V7 model features",
        ),
        (
            runner.FeatureAblationRunConfig(
                experiment_name="baseline",
                feature_columns=MODEL_FEATURES,
                top_k=0,
            ),
            "top_k must be between 1 and 49",
        ),
        (
            runner.FeatureAblationRunConfig(
                experiment_name="baseline",
                feature_columns=MODEL_FEATURES,
                top_k=50,
            ),
            "top_k must be between 1 and 49",
        ),
    ],
)
def test_validate_run_config_rejects_invalid_values(
    config: runner.FeatureAblationRunConfig,
    message: str,
) -> None:
    with pytest.raises(
        runner.FeatureConfigurationError,
        match=message,
    ):
        runner.validate_run_config(config)


def test_validate_run_config_accepts_valid_values() -> None:
    config = runner.FeatureAblationRunConfig(
        experiment_name="baseline",
        feature_columns=MODEL_FEATURES,
        top_k=5,
    )

    assert config.validated() is config


def test_build_baseline_config() -> None:
    config = runner.build_baseline_config(
        top_k=7
    )

    assert config.experiment_name == "baseline"
    assert config.feature_columns == MODEL_FEATURES
    assert config.top_k == 7


def test_build_single_removal_config() -> None:
    config = (
        runner.build_single_removal_config(
            feature_to_remove="feature_b",
            top_k=5,
        )
    )

    assert (
        config.experiment_name
        == "without_feature_b"
    )
    assert config.feature_columns == (
        "feature_a",
        "feature_c",
    )
    assert config.top_k == 5


def test_build_single_removal_config_rejects_unknown_feature() -> None:
    with pytest.raises(
        runner.FeatureConfigurationError,
        match="unknown V7 feature",
    ):
        runner.build_single_removal_config(
            "does_not_exist"
        )


def test_validate_dataset_accepts_valid_dataset() -> None:
    dataset = build_dataset([1, 2])

    runner.validate_dataset(
        dataset,
        MODEL_FEATURES,
        "dataset",
    )


def test_validate_dataset_rejects_non_dataframe() -> None:
    with pytest.raises(
        runner.DatasetValidationError,
        match="must be a pandas DataFrame",
    ):
        runner.validate_dataset(
            [],
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_empty_dataset() -> None:
    with pytest.raises(
        runner.DatasetValidationError,
        match="cannot be empty",
    ):
        runner.validate_dataset(
            pd.DataFrame(),
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_missing_column() -> None:
    dataset = build_dataset([1])
    dataset = dataset.drop(
        columns=["feature_b"]
    )

    with pytest.raises(
        runner.DatasetValidationError,
        match="is missing columns",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_missing_values() -> None:
    dataset = build_dataset([1])
    dataset.loc[
        dataset.index[0],
        "feature_a",
    ] = np.nan

    with pytest.raises(
        runner.DatasetValidationError,
        match="contains missing values",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_non_finite_values() -> None:
    dataset = build_dataset([1])
    dataset.loc[
        dataset.index[0],
        "feature_a",
    ] = np.inf

    with pytest.raises(
        runner.DatasetValidationError,
        match="non-finite numeric values",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_invalid_target_value() -> None:
    dataset = build_dataset([1])
    dataset.loc[
        dataset.index[0],
        "target",
    ] = 2

    with pytest.raises(
        runner.DatasetValidationError,
        match="target must contain only 0 and 1",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_invalid_candidate_number() -> None:
    dataset = build_dataset([1])
    dataset.loc[
        dataset.index[0],
        "candidate_number",
    ] = 50

    with pytest.raises(
        runner.DatasetValidationError,
        match="candidate_number must be between 1 and 49",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_invalid_rows_per_target() -> None:
    dataset = build_dataset([1])
    dataset = dataset.iloc[:-1].copy()

    with pytest.raises(
        runner.DatasetValidationError,
        match="exactly 49 candidate rows",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_dataset_rejects_duplicate_candidate() -> None:
    dataset = build_dataset([1])
    duplicate = dataset.iloc[[0]].copy()
    dataset = pd.concat(
        [
            dataset.iloc[:-1],
            duplicate,
        ],
        ignore_index=True,
    )

    with pytest.raises(
        runner.DatasetValidationError,
        match="duplicate candidate numbers",
    ):
        runner.validate_dataset(
            dataset,
            MODEL_FEATURES,
            "dataset",
        )


def test_validate_temporal_order_accepts_strict_order() -> None:
    training = build_dataset([1, 2])
    validation = build_dataset([4, 5])

    runner.validate_temporal_order(
        training,
        validation,
    )


def test_validate_temporal_order_rejects_overlap() -> None:
    training = build_dataset([1, 2])
    validation = build_dataset([2, 3])

    with pytest.raises(
        runner.DatasetValidationError,
        match="overlap",
    ):
        runner.validate_temporal_order(
            training,
            validation,
        )


def test_validate_temporal_order_rejects_reversed_split() -> None:
    training = build_dataset([4, 5])
    validation = build_dataset([1, 2])

    with pytest.raises(
        runner.DatasetValidationError,
        match="Temporal order is invalid",
    ):
        runner.validate_temporal_order(
            training,
            validation,
        )


def test_positive_class_probabilities() -> None:
    model = DeterministicModel()
    frame = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
            ],
        }
    )

    probabilities = (
        runner._positive_class_probabilities(
            model,
            frame,
        )
    )

    assert probabilities.tolist() == pytest.approx(
        [
            0.99,
            0.98,
        ]
    )


def test_positive_class_probabilities_handles_zero_only_model() -> None:
    model = ZeroOnlyModel()
    frame = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
            ],
        }
    )

    probabilities = (
        runner._positive_class_probabilities(
            model,
            frame,
        )
    )

    assert probabilities.tolist() == [
        0.0,
        0.0,
    ]


def test_positive_class_probabilities_rejects_unsupported_classes() -> None:
    model = UnsupportedClassModel()
    frame = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                2.0,
            ],
        }
    )

    with pytest.raises(
        runner.FeatureAblationRunnerError,
        match="Unsupported classifier classes",
    ):
        runner._positive_class_probabilities(
            model,
            frame,
        )


def test_evaluate_targets_ranks_candidates_deterministically() -> None:
    dataset = build_dataset([4])
    probabilities = (
        1.0
        - (
            dataset["candidate_number"]
            .astype(float)
            .to_numpy()
            / 100.0
        )
    )

    evaluations = runner._evaluate_targets(
        dataset,
        probabilities,
        top_k=5,
    )

    assert len(evaluations) == 1
    assert evaluations[0].selected_numbers == (
        1,
        2,
        3,
        4,
        5,
    )
    assert evaluations[0].actual_numbers == (
        1,
        2,
        3,
        4,
        5,
    )
    assert evaluations[0].hits == 5


def test_run_feature_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = DeterministicModel()

    monkeypatch.setattr(
        runner.V7FeatureAblationReport,
        "_build_model",
        staticmethod(lambda: model),
    )

    training = build_dataset([1, 2])
    validation = build_dataset([4, 5])

    config = runner.FeatureAblationRunConfig(
        experiment_name="without_feature_c",
        feature_columns=(
            "feature_a",
            "feature_b",
        ),
        top_k=5,
    )

    result = runner.run_feature_subset(
        training,
        validation,
        config,
    )

    assert (
        result.experiment_name
        == "without_feature_c"
    )
    assert result.feature_columns == (
        "feature_a",
        "feature_b",
    )
    assert result.feature_count == 2
    assert result.removed_features == (
        "feature_c",
    )
    assert result.training_rows == 98
    assert result.validation_rows == 98
    assert result.training_targets == 2
    assert result.validation_targets == 2
    assert result.total_hits == 10
    assert result.mean_hits_at_k == pytest.approx(
        5.0
    )
    assert (
        result.normalized_hits_at_k
        == pytest.approx(1.0)
    )
    assert (
        result.targets_with_at_least_one_hit
        == 2
    )
    assert result.target_hit_rate == pytest.approx(
        1.0
    )
    assert len(result.target_evaluations) == 2
    assert model.fit_columns == (
        "feature_a",
        "feature_b",
    )
    assert model.fit_rows == 98


def test_run_feature_subset_to_dict_is_json_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.V7FeatureAblationReport,
        "_build_model",
        staticmethod(
            lambda: DeterministicModel()
        ),
    )

    result = runner.run_feature_subset(
        build_dataset([1, 2]),
        build_dataset([4]),
        runner.FeatureAblationRunConfig(
            experiment_name="baseline",
            feature_columns=MODEL_FEATURES,
            top_k=5,
        ),
    )

    payload = result.to_dict()

    assert payload["experiment_name"] == "baseline"
    assert payload["feature_columns"] == list(
        MODEL_FEATURES
    )
    assert isinstance(
        payload["target_evaluations"],
        list,
    )


def test_compare_feature_runs_accepts_equal_performance() -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=2.5,
    )
    candidate = make_result(
        name="without_feature_c",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.5,
    )

    comparison = runner.compare_feature_runs(
        baseline,
        candidate,
    )

    assert comparison.accepted is True
    assert comparison.removed_features == (
        "feature_c",
    )
    assert comparison.absolute_delta == pytest.approx(
        0.0
    )
    assert comparison.relative_delta == pytest.approx(
        0.0
    )


def test_compare_feature_runs_accepts_within_tolerance() -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=2.5,
    )
    candidate = make_result(
        name="without_feature_c",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.45,
    )

    comparison = runner.compare_feature_runs(
        baseline,
        candidate,
        tolerance=0.05,
    )

    assert comparison.accepted is True
    assert comparison.absolute_delta == pytest.approx(
        -0.05
    )


def test_compare_feature_runs_rejects_beyond_tolerance() -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=2.5,
    )
    candidate = make_result(
        name="without_feature_c",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.4,
    )

    comparison = runner.compare_feature_runs(
        baseline,
        candidate,
        tolerance=0.05,
    )

    assert comparison.accepted is False


def test_compare_feature_runs_handles_zero_baseline() -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=0.0,
    )
    candidate = make_result(
        name="without_feature_c",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=0.0,
    )

    comparison = runner.compare_feature_runs(
        baseline,
        candidate,
    )

    assert comparison.relative_delta is None
    assert comparison.accepted is True


@pytest.mark.parametrize(
    "tolerance",
    [
        -0.01,
        float("nan"),
        float("inf"),
    ],
)
def test_compare_feature_runs_rejects_invalid_tolerance(
    tolerance: float,
) -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=2.5,
    )
    candidate = make_result(
        name="candidate",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.5,
    )

    with pytest.raises(
        runner.FeatureConfigurationError,
        match="tolerance must be finite and non-negative",
    ):
        runner.compare_feature_runs(
            baseline,
            candidate,
            tolerance=tolerance,
        )


def test_compare_feature_runs_rejects_different_top_k() -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=2.5,
        top_k=5,
    )
    candidate = make_result(
        name="candidate",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.5,
        top_k=6,
    )

    with pytest.raises(
        runner.FeatureConfigurationError,
        match="same top_k",
    ):
        runner.compare_feature_runs(
            baseline,
            candidate,
        )


def test_compare_feature_runs_rejects_different_validation_counts() -> None:
    baseline = make_result(
        name="baseline",
        features=MODEL_FEATURES,
        mean_hits=2.5,
        validation_targets=2,
    )
    candidate = make_result(
        name="candidate",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.5,
        validation_targets=3,
    )

    with pytest.raises(
        runner.FeatureConfigurationError,
        match="same number of validation targets",
    ):
        runner.compare_feature_runs(
            baseline,
            candidate,
        )


def test_result_summary() -> None:
    result = make_result(
        name="without_feature_c",
        features=(
            "feature_a",
            "feature_b",
        ),
        mean_hits=2.5,
    )

    text = runner.result_summary(result)

    assert (
        "PREDIXA AI V7 FEATURE ABLATION RUN"
        in text
    )
    assert "without_feature_c" in text
    assert "feature_c" in text
    assert "Mean Hits@K" in text


def test_json_safe_converts_numpy_values() -> None:
    payload = runner._json_safe(
        {
            "integer": np.int64(3),
            "float": np.float64(1.5),
            "tuple": (
                np.int64(1),
                np.int64(2),
            ),
            "nan": float("nan"),
        }
    )

    assert payload == {
        "integer": 3,
        "float": 1.5,
        "tuple": [
            1,
            2,
        ],
        "nan": None,
    }
