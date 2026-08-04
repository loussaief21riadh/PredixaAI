from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.ai.v7.explainability.feature_families import (
    FEATURE_FAMILIES,
    FEATURE_FAMILY_ORDER,
    family_for_feature,
)
from app.ai.v7.ranking_dataset import V7RankingDataset
from app.database import SessionLocal
from app.models.draw import Draw


class V7FeatureCorrelationReport:
    """
    Generate correlation diagnostics for the PredixaAI V7 model features.

    The report measures:

    - Pearson correlations;
    - Spearman correlations;
    - strongly correlated feature pairs;
    - strongest correlation partner for every feature;
    - correlation summaries by feature family;
    - descriptive associations between each feature and the binary target.

    Correlation is descriptive. It does not measure causal importance,
    out-of-sample predictive utility, or feature-ablation performance.
    """

    VERSION = "V7-FEATURE-CORRELATION-REPORT-V1"

    MODERN_LOTO_START_DATE = "2008-10-06"

    DEFAULT_WINDOW_SIZE = 100
    DEFAULT_MAX_TRAINING_TARGETS = 1500
    DEFAULT_CORRELATION_THRESHOLD = 0.80
    DEFAULT_TOP_PAIRS = 20

    REQUIRED_DATASET_COLUMNS = (
        "target_draw_index",
        "target_draw_date",
        "candidate_number",
        "target",
    )

    BACKEND_DIR = Path(__file__).resolve().parents[4]

    DEFAULT_OUTPUT_DIR = (
        BACKEND_DIR
        / "reports"
        / "v7"
        / "feature_correlation"
    )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    @classmethod
    def _validate_parameters(
        cls,
        window_size: int,
        max_training_targets: int,
        correlation_threshold: float,
        top_pairs: int,
    ) -> None:
        if window_size < 100:
            raise ValueError(
                "window_size must be at least 100."
            )

        if max_training_targets < 0:
            raise ValueError(
                "max_training_targets cannot be negative."
            )

        if not 0.0 < correlation_threshold <= 1.0:
            raise ValueError(
                "correlation_threshold must be greater than 0 "
                "and less than or equal to 1."
            )

        if top_pairs < 1:
            raise ValueError(
                "top_pairs must be at least 1."
            )

    @staticmethod
    def _validate_feature_configuration() -> None:
        model_features = set(
            V7RankingDataset.feature_columns()
        )

        configured_features = {
            feature_name
            for family_name in FEATURE_FAMILY_ORDER
            for feature_name in FEATURE_FAMILIES[family_name]
        }

        missing_configuration = sorted(
            model_features - configured_features
        )

        unknown_configuration = sorted(
            configured_features - model_features
        )

        if missing_configuration:
            raise ValueError(
                "Model features missing from the feature-family "
                f"configuration: {missing_configuration}"
            )

        if unknown_configuration:
            raise ValueError(
                "Configured features not used by the V7 model: "
                f"{unknown_configuration}"
            )

    @classmethod
    def _validate_dataset(
        cls,
        dataset: pd.DataFrame,
    ) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(
                "dataset must be a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                "dataset cannot be empty."
            )

        missing_required_columns = [
            column
            for column in cls.REQUIRED_DATASET_COLUMNS
            if column not in dataset.columns
        ]

        if missing_required_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{missing_required_columns}"
            )

        feature_columns = V7RankingDataset.feature_columns()

        missing_features = [
            feature_name
            for feature_name in feature_columns
            if feature_name not in dataset.columns
        ]

        if missing_features:
            raise ValueError(
                "Dataset is missing V7 model features: "
                f"{missing_features}"
            )

        if dataset[
            list(cls.REQUIRED_DATASET_COLUMNS)
            + list(feature_columns)
        ].isnull().any().any():
            null_columns = (
                dataset[
                    list(cls.REQUIRED_DATASET_COLUMNS)
                    + list(feature_columns)
                ]
                .columns[
                    dataset[
                        list(cls.REQUIRED_DATASET_COLUMNS)
                        + list(feature_columns)
                    ]
                    .isnull()
                    .any()
                ]
                .tolist()
            )

            raise ValueError(
                "Dataset contains missing values in columns: "
                f"{null_columns}"
            )

        feature_values = (
            dataset[feature_columns]
            .astype(float)
            .to_numpy()
        )

        if not np.isfinite(feature_values).all():
            raise ValueError(
                "Dataset model features contain non-finite values."
            )

        target_values = set(
            dataset["target"]
            .astype(int)
            .unique()
            .tolist()
        )

        if target_values != {0, 1}:
            raise ValueError(
                "Dataset target must contain both binary classes 0 and 1."
            )

        rows_per_target = (
            dataset
            .groupby("target_draw_index")
            .size()
        )

        if not (rows_per_target == 49).all():
            raise ValueError(
                "Every target must contain exactly 49 candidate rows."
            )

        positives_per_target = (
            dataset
            .groupby("target_draw_index")["target"]
            .sum()
        )

        if not (positives_per_target == 5).all():
            raise ValueError(
                "Every target must contain exactly 5 positive labels."
            )

    # ==========================================================
    # DATABASE AND DATASET
    # ==========================================================

    @classmethod
    def _load_draws(
        cls,
        db: Any,
    ) -> list[Draw]:
        draws = (
            db.query(Draw)
            .filter(
                Draw.draw_date >= cls.MODERN_LOTO_START_DATE
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
    def _feature_dataframe(
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        feature_columns = V7RankingDataset.feature_columns()

        return (
            dataset[feature_columns]
            .astype(float)
            .copy()
            .reset_index(drop=True)
        )

    # ==========================================================
    # CORRELATION MATRICES
    # ==========================================================

    @staticmethod
    def _correlation_matrix(
        feature_dataframe: pd.DataFrame,
        method: str,
    ) -> pd.DataFrame:
        if method not in {
            "pearson",
            "spearman",
        }:
            raise ValueError(
                "Correlation method must be pearson or spearman."
            )

        return feature_dataframe.corr(
            method=method
        )

    @staticmethod
    def _clean_float(
        value: Any,
    ) -> float | None:
        if value is None:
            return None

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            return None

        return numeric_value

    @classmethod
    def _matrix_to_dictionary(
        cls,
        matrix: pd.DataFrame,
    ) -> dict[str, dict[str, float | None]]:
        return {
            str(row_name): {
                str(column_name): cls._clean_float(
                    matrix.loc[
                        row_name,
                        column_name,
                    ]
                )
                for column_name in matrix.columns
            }
            for row_name in matrix.index
        }

    # ==========================================================
    # FEATURE PAIRS
    # ==========================================================

    @classmethod
    def _build_pair_dataframe(
        cls,
        pearson_matrix: pd.DataFrame,
        spearman_matrix: pd.DataFrame,
        correlation_threshold: float,
    ) -> pd.DataFrame:
        feature_columns = V7RankingDataset.feature_columns()

        rows: list[dict[str, Any]] = []

        for feature_index, feature_a in enumerate(
            feature_columns
        ):
            for feature_b in feature_columns[
                feature_index + 1:
            ]:
                pearson_value = cls._clean_float(
                    pearson_matrix.loc[
                        feature_a,
                        feature_b,
                    ]
                )

                spearman_value = cls._clean_float(
                    spearman_matrix.loc[
                        feature_a,
                        feature_b,
                    ]
                )

                absolute_pearson = (
                    abs(pearson_value)
                    if pearson_value is not None
                    else None
                )

                absolute_spearman = (
                    abs(spearman_value)
                    if spearman_value is not None
                    else None
                )

                available_correlations = [
                    (
                        "pearson",
                        pearson_value,
                        absolute_pearson,
                    ),
                    (
                        "spearman",
                        spearman_value,
                        absolute_spearman,
                    ),
                ]

                available_correlations = [
                    correlation
                    for correlation in available_correlations
                    if correlation[2] is not None
                ]

                if available_correlations:
                    strongest_method, strongest_value, maximum_absolute = max(
                        available_correlations,
                        key=lambda correlation: float(
                            correlation[2]
                        ),
                    )

                    direction = (
                        "positive"
                        if float(strongest_value) > 0
                        else "negative"
                        if float(strongest_value) < 0
                        else "zero"
                    )
                else:
                    strongest_method = None
                    strongest_value = None
                    maximum_absolute = None
                    direction = "undefined"

                rows.append(
                    {
                        "feature_a": feature_a,
                        "family_a": family_for_feature(
                            feature_a
                        ),
                        "feature_b": feature_b,
                        "family_b": family_for_feature(
                            feature_b
                        ),
                        "pearson": pearson_value,
                        "absolute_pearson": absolute_pearson,
                        "spearman": spearman_value,
                        "absolute_spearman": absolute_spearman,
                        "strongest_method": strongest_method,
                        "strongest_correlation": strongest_value,
                        "max_absolute_correlation": maximum_absolute,
                        "direction": direction,
                        "high_correlation": (
                            maximum_absolute is not None
                            and maximum_absolute
                            >= correlation_threshold
                        ),
                    }
                )

        dataframe = pd.DataFrame(rows)

        return (
            dataframe
            .sort_values(
                by=[
                    "max_absolute_correlation",
                    "feature_a",
                    "feature_b",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    # ==========================================================
    # FEATURE SUMMARY
    # ==========================================================

    @classmethod
    def _build_feature_summary(
        cls,
        feature_dataframe: pd.DataFrame,
        pair_dataframe: pd.DataFrame,
        correlation_threshold: float,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for feature_name in V7RankingDataset.feature_columns():
            feature_pairs = pair_dataframe[
                (pair_dataframe["feature_a"] == feature_name)
                | (pair_dataframe["feature_b"] == feature_name)
            ]

            valid_pearson_pairs = feature_pairs[
                feature_pairs["absolute_pearson"].notna()
            ]

            valid_spearman_pairs = feature_pairs[
                feature_pairs["absolute_spearman"].notna()
            ]

            if valid_pearson_pairs.empty:
                strongest_pearson_partner = None
                strongest_pearson = None
            else:
                strongest_pearson_row = (
                    valid_pearson_pairs
                    .sort_values(
                        by="absolute_pearson",
                        ascending=False,
                    )
                    .iloc[0]
                )

                strongest_pearson_partner = (
                    strongest_pearson_row["feature_b"]
                    if strongest_pearson_row["feature_a"]
                    == feature_name
                    else strongest_pearson_row["feature_a"]
                )

                strongest_pearson = float(
                    strongest_pearson_row["pearson"]
                )

            if valid_spearman_pairs.empty:
                strongest_spearman_partner = None
                strongest_spearman = None
            else:
                strongest_spearman_row = (
                    valid_spearman_pairs
                    .sort_values(
                        by="absolute_spearman",
                        ascending=False,
                    )
                    .iloc[0]
                )

                strongest_spearman_partner = (
                    strongest_spearman_row["feature_b"]
                    if strongest_spearman_row["feature_a"]
                    == feature_name
                    else strongest_spearman_row["feature_a"]
                )

                strongest_spearman = float(
                    strongest_spearman_row["spearman"]
                )

            feature_series = feature_dataframe[
                feature_name
            ]

            rows.append(
                {
                    "feature": feature_name,
                    "family": family_for_feature(
                        feature_name
                    ),
                    "row_count": int(
                        feature_series.shape[0]
                    ),
                    "unique_values": int(
                        feature_series.nunique(
                            dropna=False
                        )
                    ),
                    "mean": float(
                        feature_series.mean()
                    ),
                    "standard_deviation": float(
                        feature_series.std(
                            ddof=0
                        )
                    ),
                    "minimum": float(
                        feature_series.min()
                    ),
                    "maximum": float(
                        feature_series.max()
                    ),
                    "is_constant": bool(
                        feature_series.nunique(
                            dropna=False
                        )
                        <= 1
                    ),
                    "strongest_pearson_partner": (
                        strongest_pearson_partner
                    ),
                    "strongest_pearson": (
                        strongest_pearson
                    ),
                    "strongest_spearman_partner": (
                        strongest_spearman_partner
                    ),
                    "strongest_spearman": (
                        strongest_spearman
                    ),
                    "high_correlation_pair_count": int(
                        (
                            feature_pairs[
                                "max_absolute_correlation"
                            ]
                            >= correlation_threshold
                        ).sum()
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ==========================================================
    # FAMILY SUMMARY
    # ==========================================================

    @classmethod
    def _build_family_summary(
        cls,
        pair_dataframe: pd.DataFrame,
        correlation_threshold: float,
    ) -> pd.DataFrame:
        family_positions = {
            family_name: position
            for position, family_name in enumerate(
                FEATURE_FAMILY_ORDER
            )
        }

        grouped_pairs: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = {}

        for pair in pair_dataframe.to_dict(
            orient="records"
        ):
            family_a = str(
                pair["family_a"]
            )

            family_b = str(
                pair["family_b"]
            )

            if (
                family_positions[family_a]
                <= family_positions[family_b]
            ):
                family_key = (
                    family_a,
                    family_b,
                )
            else:
                family_key = (
                    family_b,
                    family_a,
                )

            grouped_pairs.setdefault(
                family_key,
                [],
            ).append(pair)

        rows: list[dict[str, Any]] = []

        for (
            family_a,
            family_b,
        ), pairs in grouped_pairs.items():
            pearson_values = [
                float(pair["absolute_pearson"])
                for pair in pairs
                if pair["absolute_pearson"] is not None
                and pd.notna(pair["absolute_pearson"])
            ]

            spearman_values = [
                float(pair["absolute_spearman"])
                for pair in pairs
                if pair["absolute_spearman"] is not None
                and pd.notna(pair["absolute_spearman"])
            ]

            maximum_values = [
                float(pair["max_absolute_correlation"])
                for pair in pairs
                if pair["max_absolute_correlation"] is not None
                and pd.notna(
                    pair["max_absolute_correlation"]
                )
            ]

            rows.append(
                {
                    "family_a": family_a,
                    "family_b": family_b,
                    "relationship": (
                        "within_family"
                        if family_a == family_b
                        else "between_families"
                    ),
                    "pair_count": len(pairs),
                    "mean_absolute_pearson": (
                        float(np.mean(pearson_values))
                        if pearson_values
                        else None
                    ),
                    "maximum_absolute_pearson": (
                        float(np.max(pearson_values))
                        if pearson_values
                        else None
                    ),
                    "mean_absolute_spearman": (
                        float(np.mean(spearman_values))
                        if spearman_values
                        else None
                    ),
                    "maximum_absolute_spearman": (
                        float(np.max(spearman_values))
                        if spearman_values
                        else None
                    ),
                    "maximum_absolute_correlation": (
                        float(np.max(maximum_values))
                        if maximum_values
                        else None
                    ),
                    "high_correlation_pairs": sum(
                        1
                        for pair in pairs
                        if (
                            pair[
                                "max_absolute_correlation"
                            ]
                            is not None
                            and pd.notna(
                                pair[
                                    "max_absolute_correlation"
                                ]
                            )
                            and float(
                                pair[
                                    "max_absolute_correlation"
                                ]
                            )
                            >= correlation_threshold
                        )
                    ),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                by=[
                    "maximum_absolute_correlation",
                    "family_a",
                    "family_b",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    # ==========================================================
    # TARGET ASSOCIATION
    # ==========================================================

    @classmethod
    def _build_target_association(
        cls,
        dataset: pd.DataFrame,
        feature_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        target_series = (
            dataset["target"]
            .astype(float)
            .reset_index(drop=True)
        )

        rows: list[dict[str, Any]] = []

        for feature_name in V7RankingDataset.feature_columns():
            feature_series = feature_dataframe[
                feature_name
            ]

            pearson_value = cls._clean_float(
                feature_series.corr(
                    target_series,
                    method="pearson",
                )
            )

            spearman_value = cls._clean_float(
                feature_series.corr(
                    target_series,
                    method="spearman",
                )
            )

            absolute_pearson = (
                abs(pearson_value)
                if pearson_value is not None
                else None
            )

            absolute_spearman = (
                abs(spearman_value)
                if spearman_value is not None
                else None
            )

            available_values = [
                value
                for value in (
                    absolute_pearson,
                    absolute_spearman,
                )
                if value is not None
            ]

            maximum_absolute = (
                max(available_values)
                if available_values
                else None
            )

            rows.append(
                {
                    "feature": feature_name,
                    "family": family_for_feature(
                        feature_name
                    ),
                    "pearson_with_target": (
                        pearson_value
                    ),
                    "absolute_pearson_with_target": (
                        absolute_pearson
                    ),
                    "spearman_with_target": (
                        spearman_value
                    ),
                    "absolute_spearman_with_target": (
                        absolute_spearman
                    ),
                    "maximum_absolute_target_association": (
                        maximum_absolute
                    ),
                }
            )

        return (
            pd.DataFrame(rows)
            .sort_values(
                by=[
                    "maximum_absolute_target_association",
                    "feature",
                ],
                ascending=[
                    False,
                    True,
                ],
                na_position="last",
            )
            .reset_index(drop=True)
        )

    # ==========================================================
    # EXPORTS
    # ==========================================================

    @staticmethod
    def _write_dataframe(
        dataframe: pd.DataFrame,
        output_path: Path,
        include_index: bool = False,
        index_label: str | None = None,
    ) -> Path:
        dataframe.to_csv(
            output_path,
            index=include_index,
            index_label=index_label,
        )

        if not output_path.exists():
            raise ValueError(
                f"CSV export failed: {output_path}"
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                f"Generated CSV file is empty: {output_path}"
            )

        return output_path.resolve()

    @classmethod
    def _json_safe(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, dict):
            return {
                str(key): cls._json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                cls._json_safe(item)
                for item in value
            ]

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            numeric_value = float(value)

            return (
                numeric_value
                if math.isfinite(numeric_value)
                else None
            )

        if isinstance(value, float):
            return (
                value
                if math.isfinite(value)
                else None
            )

        if pd.isna(value):
            return None

        return value

    @classmethod
    def _write_json(
        cls,
        result: dict[str, Any],
        output_path: Path,
    ) -> Path:
        safe_result = cls._json_safe(
            result
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                safe_result,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False,
            )

        if not output_path.exists():
            raise ValueError(
                "JSON correlation export failed."
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                "Generated JSON report is empty."
            )

        return output_path.resolve()

    @staticmethod
    def _write_text(
        result: dict[str, Any],
        output_path: Path,
    ) -> Path:
        lines = [
            "=" * 120,
            "PREDIXA AI V7 FEATURE CORRELATION REPORT",
            "=" * 120,
            "",
            f"Version                 : {result['version']}",
            f"Dataset targets         : {result['dataset_targets']}",
            f"Dataset rows            : {result['dataset_rows']}",
            f"Model features          : {result['feature_count']}",
            (
                "Correlation threshold   : "
                f"{result['correlation_threshold']:.6f}"
            ),
            (
                "High-correlation pairs  : "
                f"{result['high_correlation_pair_count']}"
            ),
            (
                "Constant features       : "
                f"{len(result['constant_features'])}"
            ),
            "",
            "STRONGEST FEATURE PAIRS",
            "-" * 120,
            (
                f"{'Feature A':<28}"
                f"{'Feature B':<28}"
                f"{'Pearson':>12}"
                f"{'Spearman':>12}"
                f"{'Maximum':>12}"
                f"{'High':>8}"
            ),
            "-" * 120,
        ]

        for pair in result["top_pairs"]:
            pearson_value = pair["pearson"]
            spearman_value = pair["spearman"]
            maximum_value = pair[
                "max_absolute_correlation"
            ]

            pearson_text = (
                f"{pearson_value:.6f}"
                if pearson_value is not None
                else "N/A"
            )

            spearman_text = (
                f"{spearman_value:.6f}"
                if spearman_value is not None
                else "N/A"
            )

            maximum_text = (
                f"{maximum_value:.6f}"
                if maximum_value is not None
                else "N/A"
            )

            lines.append(
                f"{pair['feature_a']:<28}"
                f"{pair['feature_b']:<28}"
                f"{pearson_text:>12}"
                f"{spearman_text:>12}"
                f"{maximum_text:>12}"
                f"{str(pair['high_correlation']):>8}"
            )

        lines.extend(
            [
                "",
                "FEATURE FAMILY SUMMARY",
                "-" * 120,
                (
                    f"{'Family A':<18}"
                    f"{'Family B':<18}"
                    f"{'Pairs':>10}"
                    f"{'Mean |P|':>14}"
                    f"{'Max |P|':>14}"
                    f"{'Mean |S|':>14}"
                    f"{'Max |S|':>14}"
                    f"{'High':>10}"
                ),
                "-" * 120,
            ]
        )

        for family in result[
            "family_summary"
        ]:
            mean_pearson = family[
                "mean_absolute_pearson"
            ]

            maximum_pearson = family[
                "maximum_absolute_pearson"
            ]

            mean_spearman = family[
                "mean_absolute_spearman"
            ]

            maximum_spearman = family[
                "maximum_absolute_spearman"
            ]

            lines.append(
                f"{family['family_a']:<18}"
                f"{family['family_b']:<18}"
                f"{int(family['pair_count']):>10}"
                f"{float(mean_pearson or 0.0):>14.6f}"
                f"{float(maximum_pearson or 0.0):>14.6f}"
                f"{float(mean_spearman or 0.0):>14.6f}"
                f"{float(maximum_spearman or 0.0):>14.6f}"
                f"{int(family['high_correlation_pairs']):>10}"
            )

        lines.extend(
            [
                "",
                "CONSTANT FEATURES",
                "-" * 120,
            ]
        )

        if result["constant_features"]:
            lines.extend(
                result["constant_features"]
            )
        else:
            lines.append(
                "No constant model features detected."
            )

        lines.extend(
            [
                "",
                "=" * 120,
                "INTERPRETATION",
                "=" * 120,
                "",
                (
                    "High absolute correlation can indicate feature "
                    "redundancy, but it does not prove that a feature "
                    "should be removed."
                ),
                (
                    "Pearson measures linear association. Spearman "
                    "measures monotonic rank association."
                ),
                (
                    "Target associations are descriptive in-sample "
                    "statistics and are not out-of-sample performance "
                    "estimates."
                ),
                (
                    "Feature removal decisions must be confirmed with "
                    "temporal ablation and walk-forward evaluation."
                ),
                "",
            ]
        )

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        if not output_path.exists():
            raise ValueError(
                "Text correlation export failed."
            )

        if output_path.stat().st_size == 0:
            raise ValueError(
                "Generated text report is empty."
            )

        return output_path.resolve()

    # ==========================================================
    # MAIN RUNNER
    # ==========================================================

    @classmethod
    def run(
        cls,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        window_size: int = DEFAULT_WINDOW_SIZE,
        max_training_targets: int = (
            DEFAULT_MAX_TRAINING_TARGETS
        ),
        correlation_threshold: float = (
            DEFAULT_CORRELATION_THRESHOLD
        ),
        top_pairs: int = DEFAULT_TOP_PAIRS,
    ) -> dict[str, Any]:
        cls._validate_parameters(
            window_size=window_size,
            max_training_targets=max_training_targets,
            correlation_threshold=(
                correlation_threshold
            ),
            top_pairs=top_pairs,
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
            draws = cls._load_draws(db)

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

        cls._validate_dataset(dataset)

        feature_dataframe = cls._feature_dataframe(
            dataset
        )

        pearson_matrix = cls._correlation_matrix(
            feature_dataframe=feature_dataframe,
            method="pearson",
        )

        spearman_matrix = cls._correlation_matrix(
            feature_dataframe=feature_dataframe,
            method="spearman",
        )

        pair_dataframe = cls._build_pair_dataframe(
            pearson_matrix=pearson_matrix,
            spearman_matrix=spearman_matrix,
            correlation_threshold=(
                correlation_threshold
            ),
        )

        feature_summary_dataframe = (
            cls._build_feature_summary(
                feature_dataframe=feature_dataframe,
                pair_dataframe=pair_dataframe,
                correlation_threshold=(
                    correlation_threshold
                ),
            )
        )

        family_summary_dataframe = (
            cls._build_family_summary(
                pair_dataframe=pair_dataframe,
                correlation_threshold=(
                    correlation_threshold
                ),
            )
        )

        target_association_dataframe = (
            cls._build_target_association(
                dataset=dataset,
                feature_dataframe=(
                    feature_dataframe
                ),
            )
        )

        high_correlation_dataframe = (
            pair_dataframe[
                pair_dataframe[
                    "high_correlation"
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )

        constant_features = (
            feature_summary_dataframe.loc[
                feature_summary_dataframe[
                    "is_constant"
                ],
                "feature",
            ]
            .astype(str)
            .tolist()
        )

        pearson_path = cls._write_dataframe(
            dataframe=pearson_matrix,
            output_path=(
                destination
                / "correlation_pearson.csv"
            ),
            include_index=True,
            index_label="feature",
        )

        spearman_path = cls._write_dataframe(
            dataframe=spearman_matrix,
            output_path=(
                destination
                / "correlation_spearman.csv"
            ),
            include_index=True,
            index_label="feature",
        )

        pair_path = cls._write_dataframe(
            dataframe=pair_dataframe,
            output_path=(
                destination
                / "correlation_feature_pairs.csv"
            ),
        )

        high_pair_path = cls._write_dataframe(
            dataframe=high_correlation_dataframe,
            output_path=(
                destination
                / "correlation_high_pairs.csv"
            ),
        )

        feature_summary_path = cls._write_dataframe(
            dataframe=feature_summary_dataframe,
            output_path=(
                destination
                / "correlation_feature_summary.csv"
            ),
        )

        family_summary_path = cls._write_dataframe(
            dataframe=family_summary_dataframe,
            output_path=(
                destination
                / "correlation_family_summary.csv"
            ),
        )

        target_association_path = cls._write_dataframe(
            dataframe=target_association_dataframe,
            output_path=(
                destination
                / "correlation_target_association.csv"
            ),
        )

        result: dict[str, Any] = {
            "status": "success",
            "version": cls.VERSION,
            "scope": (
                "candidate-level V7 model design matrix; "
                "one row per target-candidate pair"
            ),
            "draw_count": len(draws),
            "dataset_rows": len(dataset),
            "dataset_targets": len(metadata),
            "window_size": window_size,
            "max_training_targets": (
                max_training_targets
            ),
            "feature_count": len(
                V7RankingDataset.feature_columns()
            ),
            "correlation_threshold": (
                correlation_threshold
            ),
            "total_feature_pairs": len(
                pair_dataframe
            ),
            "high_correlation_pair_count": len(
                high_correlation_dataframe
            ),
            "constant_features": (
                constant_features
            ),
            "top_pairs": (
                pair_dataframe
                .head(top_pairs)
                .to_dict(orient="records")
            ),
            "feature_summary": (
                feature_summary_dataframe
                .to_dict(orient="records")
            ),
            "family_summary": (
                family_summary_dataframe
                .to_dict(orient="records")
            ),
            "target_association": (
                target_association_dataframe
                .to_dict(orient="records")
            ),
            "pearson_matrix": (
                cls._matrix_to_dictionary(
                    pearson_matrix
                )
            ),
            "spearman_matrix": (
                cls._matrix_to_dictionary(
                    spearman_matrix
                )
            ),
        }

        text_path = cls._write_text(
            result=result,
            output_path=(
                destination
                / "feature_correlation_report.txt"
            ),
        )

        result["files"] = {
            "pearson_csv": str(
                pearson_path
            ),
            "spearman_csv": str(
                spearman_path
            ),
            "feature_pairs_csv": str(
                pair_path
            ),
            "high_pairs_csv": str(
                high_pair_path
            ),
            "feature_summary_csv": str(
                feature_summary_path
            ),
            "family_summary_csv": str(
                family_summary_path
            ),
            "target_association_csv": str(
                target_association_path
            ),
            "json": str(
                destination
                / "feature_correlation_report.json"
            ),
            "text": str(
                text_path
            ),
        }

        json_path = cls._write_json(
            result=result,
            output_path=(
                destination
                / "feature_correlation_report.json"
            ),
        )

        result["files"]["json"] = str(
            json_path
        )

        return result

    # ==========================================================
    # CONSOLE OUTPUT
    # ==========================================================

    @staticmethod
    def print_result(
        result: dict[str, Any],
    ) -> None:
        print("=" * 120)
        print(
            "PREDIXA AI V7 FEATURE CORRELATION REPORT"
        )
        print("=" * 120)

        print(
            f"Status                  : "
            f"{result['status']}"
        )

        print(
            f"Dataset targets         : "
            f"{result['dataset_targets']}"
        )

        print(
            f"Dataset rows            : "
            f"{result['dataset_rows']}"
        )

        print(
            f"Model features          : "
            f"{result['feature_count']}"
        )

        print(
            f"Total feature pairs     : "
            f"{result['total_feature_pairs']}"
        )

        print(
            f"Correlation threshold   : "
            f"{result['correlation_threshold']:.6f}"
        )

        print(
            f"High-correlation pairs  : "
            f"{result['high_correlation_pair_count']}"
        )

        print(
            f"Constant features       : "
            f"{len(result['constant_features'])}"
        )

        print()
        print("STRONGEST FEATURE PAIRS")
        print("-" * 120)

        print(
            f"{'Feature A':<28}"
            f"{'Feature B':<28}"
            f"{'Pearson':>12}"
            f"{'Spearman':>12}"
            f"{'Maximum':>12}"
            f"{'High':>8}"
        )

        print("-" * 120)

        for pair in result["top_pairs"]:
            pearson_value = pair["pearson"]
            spearman_value = pair["spearman"]
            maximum_value = pair[
                "max_absolute_correlation"
            ]

            pearson_text = (
                f"{pearson_value:.6f}"
                if pearson_value is not None
                and pd.notna(pearson_value)
                else "N/A"
            )

            spearman_text = (
                f"{spearman_value:.6f}"
                if spearman_value is not None
                and pd.notna(spearman_value)
                else "N/A"
            )

            maximum_text = (
                f"{maximum_value:.6f}"
                if maximum_value is not None
                and pd.notna(maximum_value)
                else "N/A"
            )

            print(
                f"{pair['feature_a']:<28}"
                f"{pair['feature_b']:<28}"
                f"{pearson_text:>12}"
                f"{spearman_text:>12}"
                f"{maximum_text:>12}"
                f"{str(pair['high_correlation']):>8}"
            )

        print()
        print("GENERATED FILES")
        print("-" * 120)

        for file_type, file_path in result[
            "files"
        ].items():
            print(
                f"{file_type.upper():<24}: "
                f"{file_path}"
            )

        print()
        print("=" * 120)
        print("SUCCESS")
        print("=" * 120)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Pearson and Spearman correlation "
            "diagnostics for PredixaAI V7 features."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            V7FeatureCorrelationReport
            .DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=(
            V7FeatureCorrelationReport
            .DEFAULT_WINDOW_SIZE
        ),
    )

    parser.add_argument(
        "--max-training-targets",
        type=int,
        default=(
            V7FeatureCorrelationReport
            .DEFAULT_MAX_TRAINING_TARGETS
        ),
    )

    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=(
            V7FeatureCorrelationReport
            .DEFAULT_CORRELATION_THRESHOLD
        ),
    )

    parser.add_argument(
        "--top-pairs",
        type=int,
        default=(
            V7FeatureCorrelationReport
            .DEFAULT_TOP_PAIRS
        ),
    )

    return parser


def main() -> None:
    parser = build_argument_parser()

    arguments = parser.parse_args()

    result = V7FeatureCorrelationReport.run(
        output_dir=arguments.output_dir,
        window_size=arguments.window_size,
        max_training_targets=(
            arguments.max_training_targets
        ),
        correlation_threshold=(
            arguments.correlation_threshold
        ),
        top_pairs=arguments.top_pairs,
    )

    V7FeatureCorrelationReport.print_result(
        result
    )


if __name__ == "__main__":
    main()
