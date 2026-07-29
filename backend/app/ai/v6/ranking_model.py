from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from app.ai.v6.ranking_dataset import V6RankingDataset
from app.core.settings import (
    RANDOM_STATE,
    N_ESTIMATORS,
    MAX_DEPTH,
)


class V6RankingModel:
    """
    Predixa AI V6 - Global Candidate Ranking Model.

    V5:
        49 independent binary Random Forest models.

    V6:
        ONE global binary Random Forest trained on candidate rows.

    Each training row represents one candidate number for one
    historical target draw.

    Input:
        12 candidate/history features.

    Target:
        1 = candidate appeared in the target draw.
        0 = candidate did not appear.

    Prediction:
        Build 49 candidate rows from one T-2 feature history,
        score all candidates with the same model,
        rank candidates by score,
        return Top-K.

    candidate_number is intentionally NOT used as a model feature.
    """

    VERSION = "V6-GLOBAL-RANKING-RF"

    NUMBER_MIN = 1
    NUMBER_MAX = 49
    TOP_K = 5

    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        max_depth: int | None = MAX_DEPTH,
        random_state: int = RANDOM_STATE,
    ):
        if n_estimators <= 0:
            raise ValueError(
                "n_estimators must be positive."
            )

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.feature_columns = (
            V6RankingDataset.feature_columns()
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
        if not isinstance(
            dataset,
            pd.DataFrame,
        ):
            raise ValueError(
                "Training dataset must be a pandas DataFrame."
            )

        if dataset.empty:
            raise ValueError(
                "Training dataset is empty."
            )

        required_columns = {
            *self.feature_columns,
            "target",
        }

        missing = (
            required_columns
            - set(dataset.columns)
        )

        if missing:
            raise ValueError(
                "Training dataset is missing columns: "
                f"{sorted(missing)}"
            )

        X = dataset[
            self.feature_columns
        ]

        y = dataset[
            "target"
        ]

        if X.isnull().any().any():
            raise ValueError(
                "Training features contain missing values."
            )

        if y.isnull().any():
            raise ValueError(
                "Training targets contain missing values."
            )

        unique_targets = set(
            y.astype(int).unique().tolist()
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

        if len(self.feature_columns) != 12:
            raise ValueError(
                "Unexpected V6 model feature count. "
                f"Expected 12, received "
                f"{len(self.feature_columns)}."
            )

    # ==========================================================
    # FIT
    # ==========================================================

    def fit(
        self,
        dataset: pd.DataFrame,
    ) -> "V6RankingModel":
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
    # POSITIVE CLASS PROBABILITY
    # ==========================================================

    def _positive_probabilities(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(
                "V6 ranking model has not been fitted."
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

        positive_index = (
            classes.index(1)
        )

        scores = probabilities[
            :,
            positive_index
        ].astype(float)

        if len(scores) != len(X):
            raise ValueError(
                "Unexpected V6 prediction size."
            )

        if not np.all(
            np.isfinite(scores)
        ):
            raise ValueError(
                "V6 prediction contains "
                "non-finite scores."
            )

        if np.any(scores < 0.0) or np.any(
            scores > 1.0
        ):
            raise ValueError(
                "V6 scores must be inside [0, 1]."
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
        Convert one V5 full feature dictionary into
        49 V6 candidate rows.

        No target information is used.
        """

        if not isinstance(
            features,
            dict,
        ):
            raise ValueError(
                "features must be a dictionary."
            )

        rows = []

        for candidate_number in range(
            self.NUMBER_MIN,
            self.NUMBER_MAX + 1,
        ):
            row = {
                "candidate_number": (
                    candidate_number
                )
            }

            for global_feature in (
                V6RankingDataset.GLOBAL_FEATURES
            ):
                if global_feature not in features:
                    raise ValueError(
                        "Missing global prediction feature: "
                        f"{global_feature}"
                    )

                row[
                    global_feature
                ] = features[
                    global_feature
                ]

            for family in (
                V6RankingDataset.CANDIDATE_FAMILIES
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
            pd.DataFrame(rows)
        )

        if len(prediction_dataset) != 49:
            raise ValueError(
                "V6 prediction dataset must "
                "contain exactly 49 rows."
            )

        if prediction_dataset[
            "candidate_number"
        ].nunique() != 49:
            raise ValueError(
                "V6 prediction candidates "
                "must be unique."
            )

        X = prediction_dataset[
            self.feature_columns
        ]

        if X.isnull().any().any():
            raise ValueError(
                "V6 prediction features "
                "contain missing values."
            )

        return prediction_dataset

    # ==========================================================
    # SCORE 49 CANDIDATES
    # ==========================================================

    def score_candidates(
        self,
        features: dict[str, Any],
    ) -> list[dict[str, float | int]]:
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

        ranked = []

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
                -item["score"],
                item["number"],
            )
        )

        if len(ranked) != 49:
            raise ValueError(
                "V6 ranking must contain "
                "49 candidates."
            )

        if len(
            {
                item["number"]
                for item in ranked
            }
        ) != 49:
            raise ValueError(
                "V6 ranking contains "
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
        if top_k <= 0:
            raise ValueError(
                "top_k must be positive."
            )

        if top_k > 49:
            raise ValueError(
                "top_k cannot exceed 49."
            )

        ranking = self.score_candidates(
            features
        )

        predicted = [
            item["number"]
            for item in ranking[:top_k]
        ]

        probabilities = {
            item["number"]: (
                item["score"]
            )
            for item in ranking
        }

        return {
            "version": self.VERSION,

            "top_k": top_k,

            "predicted_numbers": (
                predicted
            ),

            "ranking": ranking,

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
