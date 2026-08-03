from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
)
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


class V7FeatureFamilyReport:
    """
    Train the V7 ranking model and aggregate feature importance
    values by configured feature family.
    """

    VERSION = "V7-FEATURE-FAMILY-REPORT"

    MODERN_LOTO_START_DATE = "2008-10-06"

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_MAX_TRAINING_TARGETS = 1500

    BACKEND_DIR = (
        Path(__file__)
        .resolve()
        .parents[4]
    )

    DEFAULT_OUTPUT_DIR = (
        BACKEND_DIR
        / "reports"
        / "v7"
        / "feature_family"
    )

    @classmethod
    def _validate_parameters(
        cls,
        window_size: int,
        max_training_targets: int,
    ) -> None:
        if window_size < 100:
            raise ValueError(
                "window_size must be at least 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
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

        return list(draws)

    @staticmethod
    def _build_family_dataframe(
        feature_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        required_columns = {
            "feature",
            "importance",
        }

        missing_columns = sorted(
            required_columns
            - set(feature_dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Feature importance DataFrame is missing "
                f"columns: {missing_columns}"
            )

        importance_by_feature = {
            str(row["feature"]): float(
                row["importance"]
            )
            for _, row in feature_dataframe.iterrows()
        }

        configured_features = {
            feature_name
            for family_name in FEATURE_FAMILY_ORDER
            for feature_name in FEATURE_FAMILIES[
                family_name
            ]
        }

        model_features = set(
            importance_by_feature
        )

        missing_features = sorted(
            configured_features
            - model_features
        )

        unknown_features = sorted(
            model_features
            - configured_features
        )

        if missing_features:
            raise ValueError(
                "Configured features missing from model: "
                f"{missing_features}"
            )

        if unknown_features:
            raise ValueError(
                "Model features are not assigned to a family: "
                f"{unknown_features}"
            )

        rows: list[dict[str, object]] = []

        for family_name in FEATURE_FAMILY_ORDER:
            family_features = FEATURE_FAMILIES[
                family_name
            ]

            family_total = sum(
                importance_by_feature[
                    feature_name
                ]
                for feature_name in family_features
            )

            rows.append(
                {
                    "family": family_name,
                    "feature_count": len(
                        family_features
                    ),
                    "importance": float(
                        family_total
                    ),
                    "features": list(
                        family_features
                    ),
                }
            )

        dataframe = pd.DataFrame(
            rows
        )

        dataframe = dataframe.sort_values(
            by=[
                "importance",
                "family",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

        dataframe.insert(
            0,
            "rank",
            range(
                1,
                len(dataframe) + 1,
            ),
        )

        total_importance = float(
            dataframe[
                "importance"
            ].sum()
        )

        if not (
            0.999999
            <= total_importance
            <= 1.000001
        ):
            raise ValueError(
                "Unexpected total family importance. "
                f"Received {total_importance:.12f}."
            )

        return dataframe

    @staticmethod
    def _write_csv(
        dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        csv_dataframe = dataframe.drop(
            columns=[
                "features",
            ]
        )

        csv_dataframe.to_csv(
            output_path,
            index=False,
        )

        return output_path.resolve()

    @staticmethod
    def _write_json(
        dataframe: pd.DataFrame,
        output_path: Path,
    ) -> Path:
        records = dataframe.to_dict(
            orient="records"
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                records,
                file,
                indent=4,
            )

        return output_path.resolve()

    @staticmethod
    def _write_text(
        family_dataframe: pd.DataFrame,
        feature_dataframe: pd.DataFrame,
        output_path: Path,
        model: V7RankingModel,
        training_rows: int,
        training_targets: int,
    ) -> Path:
        importance_by_feature = {
            str(row["feature"]): float(
                row["importance"]
            )
            for _, row in feature_dataframe.iterrows()
        }

        lines = [
            "=" * 80,
            "PREDIXA AI V7 FEATURE FAMILY REPORT",
            "=" * 80,
            "",
            f"Model version    : {model.VERSION}",
            f"Training rows    : {training_rows}",
            f"Training targets : {training_targets}",
            f"Feature count    : {len(model.feature_columns)}",
            "",
        ]

        for _, family_row in (
            family_dataframe.iterrows()
        ):
            family_name = str(
                family_row["family"]
            )

            lines.extend(
                [
                    family_name.upper(),
                    "-" * 80,
                    (
                        "Feature count    : "
                        f"{int(family_row['feature_count'])}"
                    ),
                    (
                        "Total importance : "
                        f"{float(family_row['importance']):.8f}"
                    ),
                    "",
                ]
            )

            family_features = sorted(
                FEATURE_FAMILIES[
                    family_name
                ],
                key=lambda feature_name: (
                    -importance_by_feature[
                        feature_name
                    ],
                    feature_name,
                ),
            )

            for feature_name in family_features:
                lines.append(
                    f"  {feature_name:<35}"
                    f"{importance_by_feature[feature_name]:.8f}"
                )

            lines.append("")

        lines.extend(
            [
                "=" * 80,
                (
                    "TOTAL IMPORTANCE : "
                    f"{family_dataframe['importance'].sum():.8f}"
                ),
                "=" * 80,
                "",
            ]
        )

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
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
    ) -> dict[str, Any]:
        cls._validate_parameters(
            window_size=window_size,
            max_training_targets=(
                max_training_targets
            ),
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
                V7RankingDataset()
                .build_from_draws(
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

            analyzer = FeatureImportanceAnalyzer(
                model
            )

            feature_dataframe = (
                analyzer.to_dataframe()
            )

            family_dataframe = (
                cls._build_family_dataframe(
                    feature_dataframe
                )
            )

            csv_path = cls._write_csv(
                family_dataframe,
                destination
                / "feature_family_report.csv",
            )

            json_path = cls._write_json(
                family_dataframe,
                destination
                / "feature_family_report.json",
            )

            text_path = cls._write_text(
                family_dataframe=family_dataframe,
                feature_dataframe=feature_dataframe,
                output_path=(
                    destination
                    / "feature_family_report.txt"
                ),
                model=model,
                training_rows=len(dataset),
                training_targets=len(metadata),
            )

        finally:
            db.close()

        return {
            "status": "success",
            "version": cls.VERSION,
            "model_version": model.VERSION,
            "draw_count": len(draws),
            "training_rows": len(dataset),
            "training_targets": len(metadata),
            "feature_count": len(
                model.feature_columns
            ),
            "total_importance": float(
                family_dataframe[
                    "importance"
                ].sum()
            ),
            "families": (
                family_dataframe.to_dict(
                    orient="records"
                )
            ),
            "files": {
                "csv": str(csv_path),
                "json": str(json_path),
                "text": str(text_path),
            },
        }

    @staticmethod
    def print_result(
        result: dict[str, Any],
    ) -> None:
        print("=" * 80)
        print("PREDIXA AI V7 FEATURE FAMILY REPORT")
        print("=" * 80)

        print(
            f"Status           : {result['status']}"
        )
        print(
            f"Model            : {result['model_version']}"
        )
        print(
            f"Training rows    : {result['training_rows']}"
        )
        print(
            f"Training targets : {result['training_targets']}"
        )
        print(
            f"Features         : {result['feature_count']}"
        )

        print()
        print("FEATURE FAMILIES")
        print("-" * 80)

        for family in result["families"]:
            print(
                f"{int(family['rank']):>2}. "
                f"{str(family['family']):<20}"
                f"{float(family['importance']):.8f}"
            )

        print()
        print(
            "Total importance : "
            f"{float(result['total_importance']):.8f}"
        )

        print()
        print("GENERATED FILES")
        print("-" * 80)

        for file_type, file_path in (
            result["files"].items()
        ):
            print(
                f"{file_type.upper():<6}: "
                f"{file_path}"
            )

        print()
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the PredixaAI V7 "
            "feature-family importance report."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            V7FeatureFamilyReport
            .DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            V7FeatureFamilyReport
            .DEFAULT_WINDOW_SIZE
        ),
    )

    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            V7FeatureFamilyReport
            .DEFAULT_MAX_TRAINING_TARGETS
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()

    arguments = parser.parse_args()

    result = V7FeatureFamilyReport.run(
        output_dir=arguments.output_dir,
        window_size=arguments.window_size,
        max_training_targets=(
            arguments.max_training_targets
        ),
    )

    V7FeatureFamilyReport.print_result(
        result
    )


if __name__ == "__main__":
    main()