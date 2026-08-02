from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from app.ai.v6b_clean.ranking_dataset import (
    V6BCleanRankingDataset,
)
from app.core.settings import (
    MAX_DEPTH,
    N_ESTIMATORS,
    RANDOM_STATE,
)


class V6BCleanRankingModel:
    """
    Predixa AI V6B-CLEAN - Global Candidate Ranking Model.

    Architecture:
        - one global binary Random Forest;
        - one training row per candidate number;
        - 49 candidate rows per historical target draw;
        - 12 model features per candidate;
        - candidate_number excluded from model features;
        - Top-K selected from the 49 candidate scores.

    Temporal slicing is handled by the ranking dataset and
    walk-forward backtester. Prediction features must come from
    a historical window ending at T-2.
    """

    VERSION = "V6B-CLEAN-GLOBAL-RANKING-RF"

    NUMBER_MIN = 1
    NUMBER_MAX = 49

    CANDIDATE_COUNT = 49
    EXPECTED_FEATURE_COUNT = 12

    TOP_K = 5

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth: int | None = MAX_DEPTH,
        random_state: int = RANDOM_STATE,
    ) -> None:
        if n_estimators <= 0:
            raise ValueError(
                "n_estimators must be positive."
            )

        self.n_estimators = int(
            n_estimators
        )

        self.max_depth = max_depth

        self.random_state = int(
            random_state
        )

        self.feature_columns = (
            V6BCleanRankingDataset
            .feature_columns()
        )

        if (
            len(self.feature_columns)
            != self.EXPECTED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected V6B-CLEAN model feature count. "
                f"Expected {self.EXPECTED_FEATURE_COUNT}, "
                f"received {len(self.feature_columns)}."
            )

        if (
            len(set(self.feature_columns))
            != len(self.feature_columns)
        ):
            raise ValueError(
                "V6B-CLEAN model feature columns "
                "must be unique."
            )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.is_fitted = False

    # ==========================================================
    # TRAINING VALIDATION
    # ==========================================================

    def _validate_training_dataset(
        self,
        dataset: pd.DataFrame,
    ) -> None:
        """
        Validate a candidate-level training dataset.
        """

        if not isinstance(
            dataset,
            pd.DataFrame,
        ):
            raise ValueError(
                "Training dataset must be "
                "a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                "Training dataset is empty."
            )

        required_columns = {
            *self.feature_columns,
            "target",
        }

        missing_columns = sorted(
            required_columns
            - set(dataset.columns)
        )

        if missing_columns:
            raise ValueError(
                "Training dataset is missing columns: "
                f"{missing_columns}"
            )

        X = dataset[
            self.feature_columns
        ]

        y = dataset[
            "target"
        ]

        if X.isnull().any().any():
            null_columns = (
                X.columns[
                    X.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Training features contain missing "
                f"values in columns: {null_columns}"
            )

        if y.isnull().any():
            raise ValueError(
                "Training targets contain "
                "missing values."
            )

        numeric_X = X.apply(
            pd.to_numeric,
            errors="coerce",
        )

        if numeric_X.isnull().any().any():
            invalid_columns = (
                numeric_X.columns[
                    numeric_X.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "Training features contain "
                "non-numeric values in columns: "
                f"{invalid_columns}"
            )

        X_values = numeric_X.to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(X_values)
        ):
            raise ValueError(
                "Training features contain "
                "non-finite values."
            )

        unique_targets = set(
            y.astype(int)
            .unique()
            .tolist()
        )

        if not unique_targets.issubset(
            {0, 1}
        ):
            raise ValueError(
                "Training target must be binary."
            )

        if unique_targets != {0, 1}:
            raise ValueError(
                "Training dataset must contain "
                "both positive and negative targets."
            )

        if (
            len(self.feature_columns)
            != self.EXPECTED_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected V6B-CLEAN feature count. "
                f"Expected {self.EXPECTED_FEATURE_COUNT}, "
                f"received {len(self.feature_columns)}."
            )

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(
        self,
        dataset: pd.DataFrame,
    ) -> "V6BCleanRankingModel":
        """
        Fit the global candidate-ranking Random Forest.
        """

        self._validate_training_dataset(
            dataset
        )

        X = dataset[
            self.feature_columns
        ].astype(float)

        y = dataset[
            "target"
        ].astype(int)

        self.model.fit(
            X,
            y,
        )

        self.is_fitted = True

        return self

    # ==========================================================
    # POSITIVE-CLASS PROBABILITY
    # ==========================================================

    def _positive_probabilities(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return the probability of target class 1.
        """

        if not self.is_fitted:
            raise ValueError(
                "V6B-CLEAN ranking model "
                "has not been fitted."
            )

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise ValueError(
                "Prediction features must be "
                "a pandas DataFrame."
            )

        if X.empty:
            raise ValueError(
                "Prediction feature dataset is empty."
            )

        probabilities = (
            self.model.predict_proba(
                X
            )
        )

        classes = list(
            self.model.classes_
        )

        if 1 not in classes:
            return np.zeros(
                len(X),
                dtype=float,
            )

        positive_index = classes.index(
            1
        )

        scores = probabilities[
            :,
            positive_index
        ].astype(float)

        if len(scores) != len(X):
            raise ValueError(
                "Unexpected V6B-CLEAN "
                "prediction size."
            )

        if not np.all(
            np.isfinite(scores)
        ):
            raise ValueError(
                "V6B-CLEAN prediction contains "
                "non-finite scores."
            )

        if (
            np.any(scores < 0.0)
            or np.any(scores > 1.0)
        ):
            raise ValueError(
                "V6B-CLEAN scores must be "
                "inside [0, 1]."
            )

        return scores

    # ==========================================================
    # BUILD 49 PREDICTION ROWS
    # ==========================================================

    def build_prediction_dataset(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:
        """
        Convert one complete 396-feature dictionary into
        49 candidate rows.

        Target information is not used.
        """

        if not isinstance(
            features,
            dict,
        ):
            raise ValueError(
                "features must be a dictionary."
            )

        if (
            len(features)
            != V6BCleanRankingDataset
            .EXPECTED_FULL_FEATURE_COUNT
        ):
            raise ValueError(
                "Unexpected full prediction "
                "feature count. "
                f"Expected "
                f"{V6BCleanRankingDataset.EXPECTED_FULL_FEATURE_COUNT}, "
                f"received {len(features)}."
            )

        rows: list[
            dict[str, int | float]
        ] = []

        for candidate_number in range(
            self.NUMBER_MIN,
            self.NUMBER_MAX + 1,
        ):
            row: dict[
                str,
                int | float
            ] = {
                "candidate_number": (
                    candidate_number
                ),
            }

            for global_feature in (
                V6BCleanRankingDataset
                .GLOBAL_FEATURES
            ):
                if global_feature not in features:
                    raise ValueError(
                        "Missing global prediction "
                        f"feature: {global_feature}"
                    )

                row[
                    global_feature
                ] = features[
                    global_feature
                ]

            for family in (
                V6BCleanRankingDataset
                .CANDIDATE_FEATURES
            ):
                source_name = (
                    f"{family}_{candidate_number}"
                )

                if source_name not in features:
                    raise ValueError(
                        "Missing candidate prediction "
                        f"feature: {source_name}"
                    )

                row[
                    family
                ] = features[
                    source_name
                ]

            rows.append(
                row
            )

        prediction_dataset = (
            pd.DataFrame(
                rows
            )
        )

        if (
            len(prediction_dataset)
            != self.CANDIDATE_COUNT
        ):
            raise ValueError(
                "V6B-CLEAN prediction dataset "
                "must contain exactly 49 rows."
            )

        if (
            prediction_dataset[
                "candidate_number"
            ].nunique()
            != self.CANDIDATE_COUNT
        ):
            raise ValueError(
                "V6B-CLEAN prediction candidates "
                "must be unique."
            )

        expected_candidates = list(
            range(
                self.NUMBER_MIN,
                self.NUMBER_MAX + 1,
            )
        )

        actual_candidates = (
            prediction_dataset[
                "candidate_number"
            ]
            .astype(int)
            .tolist()
        )

        if (
            actual_candidates
            != expected_candidates
        ):
            raise ValueError(
                "V6B-CLEAN prediction candidate "
                "order is invalid."
            )

        missing_model_columns = [
            column
            for column in self.feature_columns
            if column
            not in prediction_dataset.columns
        ]

        if missing_model_columns:
            raise ValueError(
                "Prediction dataset is missing "
                "model feature columns: "
                f"{missing_model_columns}"
            )

        X = prediction_dataset[
            self.feature_columns
        ]

        if X.isnull().any().any():
            null_columns = (
                X.columns[
                    X.isnull().any()
                ]
                .tolist()
            )

            raise ValueError(
                "V6B-CLEAN prediction features "
                "contain missing values in columns: "
                f"{null_columns}"
            )

        X_values = X.to_numpy(
            dtype=float
        )

        if not np.all(
            np.isfinite(X_values)
        ):
            raise ValueError(
                "V6B-CLEAN prediction features "
                "contain non-finite values."
            )

        return prediction_dataset

    # ==========================================================
    # SCORE 49 CANDIDATES
    # ==========================================================

    def score_candidates(
        self,
        features: dict[str, Any],
    ) -> list[dict[str, float | int]]:
        """
        Score and rank all 49 candidate numbers.
        """

        prediction_dataset = (
            self.build_prediction_dataset(
                features
            )
        )

        X = prediction_dataset[
            self.feature_columns
        ].astype(float)

        scores = (
            self._positive_probabilities(
                X
            )
        )

        ranked: list[
            dict[str, float | int]
        ] = []

        for candidate_number, score in zip(
            prediction_dataset[
                "candidate_number"
            ].tolist(),
            scores.tolist(),
        ):
            ranked.append(
                {
                    "number": int(
                        candidate_number
                    ),
                    "score": float(
                        score
                    ),
                }
            )

        ranked.sort(
            key=lambda item: (
                -float(
                    item["score"]
                ),
                int(
                    item["number"]
                ),
            )
        )

        if (
            len(ranked)
            != self.CANDIDATE_COUNT
        ):
            raise ValueError(
                "V6B-CLEAN ranking must "
                "contain 49 candidates."
            )

        ranked_numbers = {
            int(
                item["number"]
            )
            for item in ranked
        }

        if (
            len(ranked_numbers)
            != self.CANDIDATE_COUNT
        ):
            raise ValueError(
                "V6B-CLEAN ranking contains "
                "duplicate candidates."
            )

        return ranked

    # ==========================================================
    # TOP-K
    # ==========================================================

    def predict_top_k(
        self,
        features: dict[str, Any],
        top_k: int = TOP_K,
    ) -> dict[str, Any]:
        """
        Rank all candidates and return the selected Top-K.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if top_k > self.CANDIDATE_COUNT:
            raise ValueError(
                "top_k cannot exceed 49."
            )

        ranking = self.score_candidates(
            features
        )

        predicted_numbers = [
            int(
                item["number"]
            )
            for item in ranking[
                :top_k
            ]
        ]

        if (
            len(set(predicted_numbers))
            != top_k
        ):
            raise ValueError(
                "V6B-CLEAN Top-K contains "
                "duplicate numbers."
            )

        probabilities = {
            int(
                item["number"]
            ): float(
                item["score"]
            )
            for item in ranking
        }

        if (
            len(probabilities)
            != self.CANDIDATE_COUNT
        ):
            raise ValueError(
                "V6B-CLEAN probability vector "
                "must contain 49 values."
            )

        return {
            "version": (
                self.VERSION
            ),

            "top_k": (
                top_k
            ),

            "predicted_numbers": (
                predicted_numbers
            ),

            "ranking": (
                ranking
            ),

            "probabilities": (
                probabilities
            ),

            "feature_count": len(
                self.feature_columns
            ),

            "candidate_count": len(
                ranking
            ),
        }