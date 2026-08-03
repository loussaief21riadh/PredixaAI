from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from app.ai.v7.explainability.feature_importance import (
    FeatureImportanceAnalyzer,
)
from app.ai.v7.ranking_dataset import (
    V7RankingDataset,
)
from app.ai.v7.ranking_model import (
    V7RankingModel,
)
from app.database import SessionLocal
from app.models.draw import Draw


class V7FeatureImportanceReport:
    """
    Train the PredixaAI V7 ranking model on real historical draws
    and generate feature-importance reports.

    Generated files:
        - feature_importance.csv
        - feature_importance.json
        - feature_importance.txt
    """

    VERSION = "V7-FEATURE-IMPORTANCE-REPORT"

    MODERN_LOTO_START_DATE = "2008-10-06"

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_MAX_TRAINING_TARGETS = 1500
    DEFAULT_TOP_FEATURES = 20

    BACKEND_DIR = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    DEFAULT_OUTPUT_DIR = (
        BACKEND_DIR
        / "reports"
        / "v7"
        / "feature_importance"
    )

    @classmethod
    def _validate_parameters(
        cls,
        window_size: int,
        max_training_targets: int,
        top_features: int,
    ) -> None:
        if window_size < 100:
            raise ValueError(
                "window_size must be at least 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        if top_features <= 0:
            raise ValueError(
                "top_features must be positive."
            )

    @classmethod
    def _load_draws(
        cls,
        db: Any,
    ) -> list[Draw]:
        """
        Load modern draws in strict chronological order.
        """

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
                "No modern draws were found in the database."
            )

        return list(
            draws
        )

    @classmethod
    def _build_training_dataset(
        cls,
        draws: list[Draw],
        window_size: int,
        max_training_targets: int,
    ):
        """
        Build the candidate-level V7 training dataset.
        """

        dataset_builder = (
            V7RankingDataset()
        )

        dataset, metadata = (
            dataset_builder.build_from_draws(
                draws=draws,
                window_size=window_size,
                max_training_targets=(
                    max_training_targets
                ),
            )
        )

        if dataset.empty:
            raise ValueError(
                "The generated training dataset is empty."
            )

        if not metadata:
            raise ValueError(
                "The generated training metadata is empty."
            )

        return (
            dataset,
            metadata,
        )

    @staticmethod
    def _write_text_report(
        output_path: Path,
        analyzer: FeatureImportanceAnalyzer,
        model: V7RankingModel,
        dataset_rows: int,
        training_targets: int,
        window_size: int,
        max_training_targets: int,
    ) -> Path:
        """
        Write a complete human-readable report.
        """

        importance_report = (
            analyzer.to_text()
        )

        header_lines = [
            "=" * 80,
            "PREDIXA AI V7 FEATURE IMPORTANCE REPORT",
            "=" * 80,
            "",
            "MODEL",
            "-" * 80,
            f"Version              : {model.VERSION}",
            (
                "Estimator            : "
                f"{type(model.model).__name__}"
            ),
            (
                "Estimators           : "
                f"{model.n_estimators}"
            ),
            (
                "Maximum depth        : "
                f"{model.max_depth}"
            ),
            (
                "Random state         : "
                f"{model.random_state}"
            ),
            (
                "Model feature count  : "
                f"{len(model.feature_columns)}"
            ),
            "",
            "DATASET",
            "-" * 80,
            (
                "Training rows        : "
                f"{dataset_rows}"
            ),
            (
                "Training targets     : "
                f"{training_targets}"
            ),
            (
                "Window size          : "
                f"{window_size}"
            ),
            (
                "Max training targets : "
                f"{max_training_targets}"
            ),
            "",
        ]

        complete_report = (
            "\n".join(
                header_lines
            )
            + importance_report
            + "\n"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            complete_report,
            encoding="utf-8",
        )

        if not output_path.exists():
            raise ValueError(
                "Text report export failed."
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                "Generated text report is empty."
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
        top_features: int = DEFAULT_TOP_FEATURES,
    ) -> dict[str, Any]:
        """
        Train V7 on real draws and generate importance reports.
        """

        cls._validate_parameters(
            window_size=window_size,
            max_training_targets=(
                max_training_targets
            ),
            top_features=top_features,
        )

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
                cls._build_training_dataset(
                    draws=draws,
                    window_size=window_size,
                    max_training_targets=(
                        max_training_targets
                    ),
                )
            )

            model = V7RankingModel()

            model.fit(
                dataset
            )

            analyzer = (
                FeatureImportanceAnalyzer(
                    model
                )
            )

            dataframe = (
                analyzer.to_dataframe()
            )

            csv_path = analyzer.to_csv(
                destination
                / "feature_importance.csv"
            )

            json_path = analyzer.to_json(
                destination
                / "feature_importance.json"
            )

            text_path = (
                cls._write_text_report(
                    output_path=(
                        destination
                        / "feature_importance.txt"
                    ),
                    analyzer=analyzer,
                    model=model,
                    dataset_rows=len(
                        dataset
                    ),
                    training_targets=len(
                        metadata
                    ),
                    window_size=window_size,
                    max_training_targets=(
                        max_training_targets
                    ),
                )
            )

        finally:
            db.close()

        displayed_top_features = min(
            top_features,
            len(
                dataframe
            ),
        )

        top_dataframe = dataframe.head(
            displayed_top_features
        )

        result = {
            "status": "success",
            "version": cls.VERSION,
            "model_version": model.VERSION,
            "draw_count": len(draws),
            "training_rows": len(dataset),
            "training_targets": len(metadata),
            "window_size": window_size,
            "max_training_targets": (
                max_training_targets
            ),
            "feature_count": len(
                model.feature_columns
            ),
            "top_features": (
                top_dataframe.to_dict(
                    orient="records"
                )
            ),
            "files": {
                "csv": str(
                    csv_path
                ),
                "json": str(
                    json_path
                ),
                "text": str(
                    text_path
                ),
            },
        }

        return result

    @staticmethod
    def print_result(
        result: dict[str, Any],
    ) -> None:
        """
        Print a concise console summary.
        """

        print(
            "=" * 80
        )

        print(
            "PREDIXA AI V7 FEATURE IMPORTANCE"
        )

        print(
            "=" * 80
        )

        print(
            f"Status           : "
            f"{result['status']}"
        )

        print(
            f"Version          : "
            f"{result['version']}"
        )

        print(
            f"Model            : "
            f"{result['model_version']}"
        )

        print(
            f"Draws            : "
            f"{result['draw_count']}"
        )

        print(
            f"Training targets : "
            f"{result['training_targets']}"
        )

        print(
            f"Training rows    : "
            f"{result['training_rows']}"
        )

        print(
            f"Features         : "
            f"{result['feature_count']}"
        )

        print()

        print(
            "TOP FEATURES"
        )

        print(
            "-" * 80
        )

        for item in result[
            "top_features"
        ]:
            print(
                f"{int(item['rank']):>2}. "
                f"{str(item['feature']):<35}"
                f"{float(item['importance']):.8f}"
            )

        print()

        print(
            "GENERATED FILES"
        )

        print(
            "-" * 80
        )

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

        print(
            "=" * 80
        )

        print(
            "SUCCESS"
        )

        print(
            "=" * 80
        )


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate the PredixaAI V7 "
            "feature-importance report."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            V7FeatureImportanceReport
            .DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory where CSV, JSON and text "
            "reports will be generated."
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            V7FeatureImportanceReport
            .DEFAULT_WINDOW_SIZE
        ),
        help=(
            "Number of historical draws used "
            "for feature generation."
        ),
    )

    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            V7FeatureImportanceReport
            .DEFAULT_MAX_TRAINING_TARGETS
        ),
        help=(
            "Maximum number of historical targets. "
            "Use 0 for no limit."
        ),
    )

    parser.add_argument(
        "--top-features",
        type=int,
        default=(
            V7FeatureImportanceReport
            .DEFAULT_TOP_FEATURES
        ),
        help=(
            "Number of top features displayed "
            "in the terminal."
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()

    arguments = parser.parse_args()

    result = (
        V7FeatureImportanceReport.run(
            output_dir=arguments.output_dir,
            window_size=arguments.window_size,
            max_training_targets=(
                arguments.max_training_targets
            ),
            top_features=arguments.top_features,
        )
    )

    V7FeatureImportanceReport.print_result(
        result
    )


if __name__ == "__main__":
    main()