from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
)
from app.ai.v7.ranking_dataset import (
    V7RankingDataset,
)
from app.core.settings import (
    MAX_DEPTH,
    N_ESTIMATORS,
    RANDOM_STATE,
)
from app.database import SessionLocal
from app.models.draw import Draw


class V7FeatureAblationReport:
    """
    Run temporal feature-family ablation experiments for PredixaAI V7.

    Experiments:
        - baseline: all model features;
        - without_global;
        - without_frequency;
        - without_recency;
        - without_trend;
        - without_volatility.

    Evaluation protocol:
        - build the candidate-level dataset chronologically;
        - split target draws chronologically into training and validation;
        - train one RandomForestClassifier per experiment;
        - rank the 49 candidates for every validation target;
        - evaluate Hits@5.

    This module does not modify the validated V7 model or dataset classes.
    """

    VERSION = "V7-FEATURE-FAMILY-ABLATION-TEMPORAL-V1"

    MODERN_LOTO_START_DATE = "2008-10-06"

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_MAX_TRAINING_TARGETS = 1500
    DEFAULT_VALIDATION_TARGETS = 100
    DEFAULT_TOP_K = 5

    REQUIRED_DATASET_COLUMNS = (
        "candidate_number",
        "target",
        "target_draw_index",
        "target_draw_date",
    )

    BACKEND_DIR = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    DEFAULT_OUTPUT_DIR = (
        BACKEND_DIR
        / "reports"
        / "v7"
        / "feature_ablation"
    )

    @classmethod
    def _validate_parameters(
        cls,
        window_size: int,
        max_training_targets: int,
        validation_targets: int,
        top_k: int,
    ) -> None:
        if window_size < 100:
            raise ValueError(
                "window_size must be at least 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        if validation_targets < 5:
            raise ValueError(
                "validation_targets must be at least 5."
            )

        if not (
            1
            <= top_k
            <= 49
        ):
            raise ValueError(
                "top_k must be between 1 and 49."
            )

    @classmethod
    def _load_draws(
        cls,
        db: Any,
    ) -> list[Draw]:
        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date
                >= cls.MODERN_LOTO_START_DATE
            )
            .order_by(
                Draw.draw_date.asc(),
                Draw.id.asc(),
            )
            .all()
        )

        if not draws:
            raise ValueError(
                "No modern draws were found."
            )

        return list(
            draws
        )

    @classmethod
    def _validate_dataset(
        cls,
        dataset: pd.DataFrame,
    ) -> None:
        if not isinstance(
            dataset,
            pd.DataFrame,
        ):
            raise ValueError(
                "dataset must be a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                "dataset cannot be empty."
            )

        missing_columns = [
            column
            for column in (
                cls.REQUIRED_DATASET_COLUMNS
            )
            if column not in dataset.columns
        ]

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing_columns}"
            )

        model_features = (
            V7RankingDataset
            .feature_columns()
        )

        missing_features = [
            feature
            for feature in model_features
            if feature not in dataset.columns
        ]

        if missing_features:
            raise ValueError(
                "Dataset is missing model features: "
                f"{missing_features}"
            )

        if dataset.isnull().any().any():
            null_columns = (
                dataset.columns[
                    dataset.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Dataset contains missing values: "
                f"{null_columns}"
            )

    @staticmethod
    def _validate_feature_configuration() -> None:
        model_features = set(
            V7RankingDataset
            .feature_columns()
        )

        configured_features = {
            feature_name
            for family_name in (
                FEATURE_FAMILY_ORDER
            )
            for feature_name in (
                FEATURE_FAMILIES[
                    family_name
                ]
            )
        }

        missing_configuration = sorted(
            model_features
            - configured_features
        )

        unknown_configuration = sorted(
            configured_features
            - model_features
        )

        if missing_configuration:
            raise ValueError(
                "Model features missing from family "
                f"configuration: {missing_configuration}"
            )

        if unknown_configuration:
            raise ValueError(
                "Configured features not used by the "
                f"model: {unknown_configuration}"
            )

    @staticmethod
    def _split_dataset(
        dataset: pd.DataFrame,
        validation_targets: int,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        list[int],
        list[int],
    ]:
        target_indices = sorted(
            dataset[
                "target_draw_index"
            ]
            .astype(int)
            .unique()
            .tolist()
        )

        if len(
            target_indices
        ) <= validation_targets:
            raise ValueError(
                "Not enough target draws for the requested "
                "temporal validation split. "
                f"Available={len(target_indices)}, "
                f"validation={validation_targets}."
            )

        training_target_indices = (
            target_indices[
                :-validation_targets
            ]
        )

        validation_target_indices = (
            target_indices[
                -validation_targets:
            ]
        )

        training_dataset = (
            dataset[
                dataset[
                    "target_draw_index"
                ].isin(
                    training_target_indices
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        validation_dataset = (
            dataset[
                dataset[
                    "target_draw_index"
                ].isin(
                    validation_target_indices
                )
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        if training_dataset.empty:
            raise ValueError(
                "Temporal training dataset is empty."
            )

        if validation_dataset.empty:
            raise ValueError(
                "Temporal validation dataset is empty."
            )

        return (
            training_dataset,
            validation_dataset,
            training_target_indices,
            validation_target_indices,
        )

    @staticmethod
    def _feature_sets() -> dict[
        str,
        list[str],
    ]:
        baseline_features = (
            V7RankingDataset
            .feature_columns()
        )

        feature_sets = {
            "baseline": list(
                baseline_features
            ),
        }

        for family_name in (
            FEATURE_FAMILY_ORDER
        ):
            removed_features = set(
                FEATURE_FAMILIES[
                    family_name
                ]
            )

            retained_features = [
                feature_name
                for feature_name in (
                    baseline_features
                )
                if feature_name
                not in removed_features
            ]

            if not retained_features:
                raise ValueError(
                    "Ablation removed all model features "
                    f"for family: {family_name}"
                )

            feature_sets[
                f"without_{family_name}"
            ] = retained_features

        return feature_sets

    @staticmethod
    def _build_model() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

    @classmethod
    def _evaluate_experiment(
        cls,
        experiment_name: str,
        feature_columns: list[str],
        training_dataset: pd.DataFrame,
        validation_dataset: pd.DataFrame,
        top_k: int,
    ) -> dict[str, Any]:
        if not feature_columns:
            raise ValueError(
                f"{experiment_name}: feature list is empty."
            )

        model = cls._build_model()

        training_X = (
            training_dataset[
                feature_columns
            ]
            .astype(float)
        )

        training_y = (
            training_dataset[
                "target"
            ]
            .astype(int)
        )

        training_started = perf_counter()

        model.fit(
            training_X,
            training_y,
        )

        training_seconds = (
            perf_counter()
            - training_started
        )

        classes = list(
            model.classes_
        )

        if 1 not in classes:
            raise ValueError(
                f"{experiment_name}: positive class "
                "is unavailable."
            )

        positive_class_index = (
            classes.index(1)
        )

        evaluation_started = perf_counter()

        total_hits = 0
        evaluated_targets = 0
        at_least_one_hit = 0
        at_least_two_hits = 0

        hit_distribution = {
            hit_count: 0
            for hit_count in range(
                0,
                top_k + 1,
            )
        }

        target_details: list[
            dict[str, Any]
        ] = []

        grouped_targets = (
            validation_dataset
            .groupby(
                "target_draw_index",
                sort=True,
            )
        )

        for (
            target_draw_index,
            target_rows,
        ) in grouped_targets:
            ordered_rows = (
                target_rows
                .sort_values(
                    by="candidate_number",
                    ascending=True,
                )
                .reset_index(
                    drop=True
                )
            )

            if len(
                ordered_rows
            ) != 49:
                raise ValueError(
                    f"{experiment_name}: target "
                    f"{target_draw_index} does not contain "
                    "49 candidate rows."
                )

            candidate_numbers = (
                ordered_rows[
                    "candidate_number"
                ]
                .astype(int)
                .to_numpy()
            )

            actual_numbers = set(
                ordered_rows.loc[
                    ordered_rows[
                        "target"
                    ].astype(int)
                    == 1,
                    "candidate_number",
                ]
                .astype(int)
                .tolist()
            )

            if len(
                actual_numbers
            ) != 5:
                raise ValueError(
                    f"{experiment_name}: target "
                    f"{target_draw_index} does not contain "
                    "exactly 5 positive labels."
                )

            probabilities = (
                model.predict_proba(
                    ordered_rows[
                        feature_columns
                    ].astype(float)
                )[
                    :,
                    positive_class_index
                ]
            )

            ranking_indices = sorted(
                range(
                    len(
                        candidate_numbers
                    )
                ),
                key=lambda index: (
                    -float(
                        probabilities[
                            index
                        ]
                    ),
                    int(
                        candidate_numbers[
                            index
                        ]
                    ),
                ),
            )

            predicted_numbers = [
                int(
                    candidate_numbers[
                        index
                    ]
                )
                for index in (
                    ranking_indices[
                        :top_k
                    ]
                )
            ]

            hits = len(
                set(
                    predicted_numbers
                )
                & actual_numbers
            )

            total_hits += hits
            evaluated_targets += 1

            hit_distribution[
                hits
            ] += 1

            if hits >= 1:
                at_least_one_hit += 1

            if hits >= 2:
                at_least_two_hits += 1

            target_details.append(
                {
                    "target_draw_index": int(
                        target_draw_index
                    ),
                    "target_draw_date": str(
                        ordered_rows[
                            "target_draw_date"
                        ].iloc[0]
                    ),
                    "predicted_top_k": (
                        predicted_numbers
                    ),
                    "actual_numbers": sorted(
                        actual_numbers
                    ),
                    "hits": hits,
                }
            )

        evaluation_seconds = (
            perf_counter()
            - evaluation_started
        )

        if evaluated_targets == 0:
            raise ValueError(
                f"{experiment_name}: no validation "
                "targets were evaluated."
            )

        average_hits = (
            total_hits
            / evaluated_targets
        )

        return {
            "experiment": experiment_name,
            "removed_family": (
                None
                if experiment_name
                == "baseline"
                else experiment_name.replace(
                    "without_",
                    "",
                    1,
                )
            ),
            "feature_count": len(
                feature_columns
            ),
            "features": list(
                feature_columns
            ),
            "training_rows": len(
                training_dataset
            ),
            "validation_rows": len(
                validation_dataset
            ),
            "evaluated_targets": (
                evaluated_targets
            ),
            "total_hits": total_hits,
            "average_hits_at_5": round(
                average_hits,
                6,
            ),
            "precision_at_5": round(
                total_hits
                / (
                    evaluated_targets
                    * top_k
                ),
                6,
            ),
            "at_least_1_hit_rate": round(
                at_least_one_hit
                / evaluated_targets,
                6,
            ),
            "at_least_2_hit_rate": round(
                at_least_two_hits
                / evaluated_targets,
                6,
            ),
            "hit_distribution": (
                hit_distribution
            ),
            "training_seconds": round(
                training_seconds,
                6,
            ),
            "evaluation_seconds": round(
                evaluation_seconds,
                6,
            ),
            "details": target_details,
        }

    @staticmethod
    def _add_baseline_comparison(
        experiments: list[
            dict[str, Any]
        ],
    ) -> None:
        baseline = next(
            (
                experiment
                for experiment in experiments
                if experiment[
                    "experiment"
                ]
                == "baseline"
            ),
            None,
        )

        if baseline is None:
            raise ValueError(
                "Baseline experiment is missing."
            )

        baseline_hits = float(
            baseline[
                "average_hits_at_5"
            ]
        )

        for experiment in experiments:
            experiment_hits = float(
                experiment[
                    "average_hits_at_5"
                ]
            )

            delta = (
                experiment_hits
                - baseline_hits
            )

            experiment[
                "delta_vs_baseline"
            ] = round(
                delta,
                6,
            )

            if (
                experiment[
                    "experiment"
                ]
                == "baseline"
            ):
                conclusion = "reference"
            elif delta > 0:
                conclusion = (
                    "performance_improved_without_family"
                )
            elif delta < 0:
                conclusion = (
                    "performance_degraded_without_family"
                )
            else:
                conclusion = (
                    "no_observed_change"
                )

            experiment[
                "conclusion"
            ] = conclusion

    @staticmethod
    def _summary_dataframe(
        experiments: list[
            dict[str, Any]
        ],
    ) -> pd.DataFrame:
        rows = []

        for experiment in experiments:
            rows.append(
                {
                    "experiment": experiment[
                        "experiment"
                    ],
                    "removed_family": experiment[
                        "removed_family"
                    ],
                    "feature_count": experiment[
                        "feature_count"
                    ],
                    "evaluated_targets": experiment[
                        "evaluated_targets"
                    ],
                    "total_hits": experiment[
                        "total_hits"
                    ],
                    "average_hits_at_5": experiment[
                        "average_hits_at_5"
                    ],
                    "delta_vs_baseline": experiment[
                        "delta_vs_baseline"
                    ],
                    "precision_at_5": experiment[
                        "precision_at_5"
                    ],
                    "at_least_1_hit_rate": experiment[
                        "at_least_1_hit_rate"
                    ],
                    "at_least_2_hit_rate": experiment[
                        "at_least_2_hit_rate"
                    ],
                    "training_seconds": experiment[
                        "training_seconds"
                    ],
                    "evaluation_seconds": experiment[
                        "evaluation_seconds"
                    ],
                    "conclusion": experiment[
                        "conclusion"
                    ],
                }
            )

        return pd.DataFrame(
            rows
        )

    @staticmethod
    def _write_csv(
        dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        dataframe.to_csv(
            output_path,
            index=False,
        )

        if not output_path.exists():
            raise ValueError(
                "CSV ablation export failed."
            )

        return output_path.resolve()

    @staticmethod
    def _write_json(
        result: dict[str, Any],
        output_path: Path,
    ) -> Path:
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                indent=4,
            )

        if not output_path.exists():
            raise ValueError(
                "JSON ablation export failed."
            )

        return output_path.resolve()

    @staticmethod
    def _write_text(
        result: dict[str, Any],
        output_path: Path,
    ) -> Path:
        lines = [
            "=" * 100,
            "PREDIXA AI V7 FEATURE FAMILY ABLATION REPORT",
            "=" * 100,
            "",
            (
                "Version              : "
                f"{result['version']}"
            ),
            (
                "Training targets     : "
                f"{result['training_targets']}"
            ),
            (
                "Validation targets   : "
                f"{result['validation_targets']}"
            ),
            (
                "Window size          : "
                f"{result['window_size']}"
            ),
            (
                "Top-K                : "
                f"{result['top_k']}"
            ),
            "",
            "EXPERIMENTS",
            "-" * 100,
            (
                f"{'Experiment':<24}"
                f"{'Features':>10}"
                f"{'Hits@5':>12}"
                f"{'Delta':>12}"
                f"{'Hit>=1':>12}"
                f"{'Train(s)':>12}"
                f"{'Conclusion':>18}"
            ),
            "-" * 100,
        ]

        for experiment in result[
            "experiments"
        ]:
            lines.append(
                f"{experiment['experiment']:<24}"
                f"{int(experiment['feature_count']):>10}"
                f"{float(experiment['average_hits_at_5']):>12.6f}"
                f"{float(experiment['delta_vs_baseline']):>12.6f}"
                f"{float(experiment['at_least_1_hit_rate']):>12.6f}"
                f"{float(experiment['training_seconds']):>12.3f}"
                f"{str(experiment['conclusion']):>18}"
            )

        lines.extend(
            [
                "",
                "=" * 100,
                "INTERPRETATION",
                "=" * 100,
                "",
                (
                    "Negative delta: removing the family "
                    "reduced Hits@5; the family may be useful."
                ),
                (
                    "Positive delta: removing the family "
                    "improved Hits@5; the family may add noise."
                ),
                (
                    "Zero delta: no difference was observed "
                    "on this validation sample."
                ),
                "",
            ]
        )

        output_path.write_text(
            "\n".join(
                lines
            ),
            encoding="utf-8",
        )

        if not output_path.exists():
            raise ValueError(
                "Text ablation export failed."
            )

        return output_path.resolve()

    @classmethod
    def run(
        cls,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_training_targets: int = (
            DEFAULT_MAX_TRAINING_TARGETS
        ),
        validation_targets: int = (
            DEFAULT_VALIDATION_TARGETS
        ),
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        cls._validate_parameters(
            window_size=window_size,
            max_training_targets=(
                max_training_targets
            ),
            validation_targets=(
                validation_targets
            ),
            top_k=top_k,
        )

        cls._validate_feature_configuration()

        destination = Path(
            output_dir
        ).expanduser().resolve()

        destination.mkdir(
            parents=True,
            exist_ok=True,
        )

        db = SessionLocal()

        try:
            draws = cls._load_draws(
                db
            )

            dataset, metadata = (
                V7RankingDataset()
                .build_from_draws(
                    draws=draws,
                    window_size=window_size,
                    max_training_targets=(
                        max_training_targets
                    ),
                )
            )

        finally:
            db.close()

        cls._validate_dataset(
            dataset
        )

        (
            training_dataset,
            validation_dataset,
            training_target_indices,
            validation_target_indices,
        ) = cls._split_dataset(
            dataset=dataset,
            validation_targets=(
                validation_targets
            ),
        )

        experiments = []

        for (
            experiment_name,
            feature_columns,
        ) in cls._feature_sets().items():
            experiment_result = (
                cls._evaluate_experiment(
                    experiment_name=(
                        experiment_name
                    ),
                    feature_columns=(
                        feature_columns
                    ),
                    training_dataset=(
                        training_dataset
                    ),
                    validation_dataset=(
                        validation_dataset
                    ),
                    top_k=top_k,
                )
            )

            experiments.append(
                experiment_result
            )

        cls._add_baseline_comparison(
            experiments
        )

        result: dict[str, Any] = {
            "status": "success",
            "version": cls.VERSION,
            "model_type": (
                "RandomForestClassifier"
            ),
            "window_size": window_size,
            "max_training_targets": (
                max_training_targets
            ),
            "top_k": top_k,
            "draw_count": len(
                draws
            ),
            "dataset_rows": len(
                dataset
            ),
            "dataset_targets": len(
                metadata
            ),
            "training_rows": len(
                training_dataset
            ),
            "training_targets": len(
                training_target_indices
            ),
            "validation_rows": len(
                validation_dataset
            ),
            "validation_targets": len(
                validation_target_indices
            ),
            "experiments": experiments,
        }

        summary_dataframe = (
            cls._summary_dataframe(
                experiments
            )
        )

        csv_path = cls._write_csv(
            summary_dataframe,
            destination
            / "feature_ablation.csv",
        )

        text_path = cls._write_text(
            result=result,
            output_path=(
                destination
                / "feature_ablation.txt"
            ),
        )

        result[
            "files"
        ] = {
            "csv": str(
                csv_path
            ),
            "json": str(
                destination
                / "feature_ablation.json"
            ),
            "text": str(
                text_path
            ),
        }

        json_path = cls._write_json(
            result=result,
            output_path=(
                destination
                / "feature_ablation.json"
            ),
        )

        result[
            "files"
        ][
            "json"
        ] = str(
            json_path
        )

        return result

    @staticmethod
    def print_result(
        result: dict[str, Any],
    ) -> None:
        print("=" * 100)
        print(
            "PREDIXA AI V7 FEATURE FAMILY ABLATION"
        )
        print("=" * 100)

        print(
            f"Status             : "
            f"{result['status']}"
        )

        print(
            f"Training targets   : "
            f"{result['training_targets']}"
        )

        print(
            f"Validation targets : "
            f"{result['validation_targets']}"
        )

        print(
            f"Training rows      : "
            f"{result['training_rows']}"
        )

        print(
            f"Validation rows    : "
            f"{result['validation_rows']}"
        )

        print()

        print(
            f"{'Experiment':<24}"
            f"{'Features':>10}"
            f"{'Hits@5':>12}"
            f"{'Delta':>12}"
            f"{'Hit>=1':>12}"
            f"{'Train(s)':>12}"
        )

        print("-" * 100)

        for experiment in result[
            "experiments"
        ]:
            print(
                f"{experiment['experiment']:<24}"
                f"{int(experiment['feature_count']):>10}"
                f"{float(experiment['average_hits_at_5']):>12.6f}"
                f"{float(experiment['delta_vs_baseline']):>12.6f}"
                f"{float(experiment['at_least_1_hit_rate']):>12.6f}"
                f"{float(experiment['training_seconds']):>12.3f}"
            )

        print()

        print("GENERATED FILES")
        print("-" * 100)

        for file_type, file_path in (
            result[
                "files"
            ].items()
        ):
            print(
                f"{file_type.upper():<6}: "
                f"{file_path}"
            )

        print()
        print("=" * 100)
        print("SUCCESS")
        print("=" * 100)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run temporal PredixaAI V7 "
            "feature-family ablation."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            V7FeatureAblationReport
            .DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            V7FeatureAblationReport
            .DEFAULT_WINDOW_SIZE
        ),
    )

    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            V7FeatureAblationReport
            .DEFAULT_MAX_TRAINING_TARGETS
        ),
    )

    parser.add_argument(
        "--validation-targets",
        type=int,
        default=(
            V7FeatureAblationReport
            .DEFAULT_VALIDATION_TARGETS
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=(
            V7FeatureAblationReport
            .DEFAULT_TOP_K
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()

    arguments = parser.parse_args()

    result = V7FeatureAblationReport.run(
        output_dir=arguments.output_dir,
        window_size=arguments.window_size,
        max_training_targets=(
            arguments.max_training_targets
        ),
        validation_targets=(
            arguments.validation_targets
        ),
        top_k=arguments.top_k,
    )

    V7FeatureAblationReport.print_result(
        result
    )


if __name__ == "__main__":
    main()