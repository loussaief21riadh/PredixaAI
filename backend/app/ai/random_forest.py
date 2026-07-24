import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from app.ai.evaluation import EvaluationEngine
from app.registry.model_registry import ModelRegistry

from app.core.settings import (
    RANDOM_STATE,
    TEST_SIZE,
    N_ESTIMATORS,
    MAX_DEPTH,
)

from app.core.logger import logger


class RandomForestEngine:
    """
    Predixa AI V4 binary Random Forest engine.

    Architecture:
        One independent binary model per lottery number.

    Trainer creates:
        random_forest_target_1
        random_forest_target_2
        ...
        random_forest_target_49
    """

    def __init__(
        self,
        model_name: str,
    ):
        self.model_name = model_name

        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.decision_threshold = 0.5

    @staticmethod
    def _positive_probability(
        model,
        X,
    ) -> np.ndarray:
        """
        Return P(class=1) for a binary model.

        Handles the edge case where the fitted model
        contains only one class.
        """

        probabilities = model.predict_proba(X)

        classes = list(model.classes_)

        if 1 not in classes:
            return np.zeros(
                len(X),
                dtype=float,
            )

        positive_index = classes.index(1)

        return probabilities[
            :,
            positive_index
        ]

    def _find_best_threshold(
        self,
        y_true,
        probabilities,
    ) -> float:
        """
        Find the threshold maximizing binary F1
        on the chronological validation period.
        """

        best_threshold = 0.5
        best_f1 = -1.0

        thresholds = np.arange(
            0.05,
            0.51,
            0.01,
        )

        for threshold in thresholds:
            predictions = (
                probabilities >= threshold
            ).astype(int)

            score = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(
                    threshold
                )

        return round(
            best_threshold,
            4,
        )

    def train(
        self,
        X,
        y,
    ):
        """
        Train one binary number model.
        """

        logger.info(
            f"Training started : {self.model_name}"
        )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if len(X) < 20:
            raise ValueError(
                "Not enough samples to train the model."
            )

        if getattr(y, "ndim", 1) != 1:
            raise ValueError(
                "RandomForestEngine expects one binary target."
            )

        # --------------------------------------------------
        # Chronological train/test split
        # --------------------------------------------------

        split_index = int(
            len(X) * (1 - TEST_SIZE)
        )

        if (
            split_index <= 0
            or split_index >= len(X)
        ):
            raise ValueError(
                "Invalid TEST_SIZE configuration."
            )

        X_train_full = X.iloc[
            :split_index
        ]

        X_test = X.iloc[
            split_index:
        ]

        y_train_full = y.iloc[
            :split_index
        ]

        y_test = y.iloc[
            split_index:
        ]

        # --------------------------------------------------
        # Chronological validation split
        # --------------------------------------------------

        validation_size = max(
            int(
                len(X_train_full)
                * 0.15
            ),
            1,
        )

        train_end = (
            len(X_train_full)
            - validation_size
        )

        if train_end <= 0:
            raise ValueError(
                "Training period is too small "
                "for validation splitting."
            )

        X_train = X_train_full.iloc[
            :train_end
        ]

        X_validation = X_train_full.iloc[
            train_end:
        ]

        y_train = y_train_full.iloc[
            :train_end
        ]

        y_validation = y_train_full.iloc[
            train_end:
        ]

        logger.info(
            f"Train size      : {len(X_train)}"
        )

        logger.info(
            f"Validation size : {len(X_validation)}"
        )

        logger.info(
            f"Test size       : {len(X_test)}"
        )

        logger.info(
            f"Feature count   : {X.shape[1]}"
        )

        # --------------------------------------------------
        # Validation model
        # --------------------------------------------------

        self.model.fit(
            X_train,
            y_train,
        )

        validation_probabilities = (
            self._positive_probability(
                self.model,
                X_validation,
            )
        )

        self.decision_threshold = (
            self._find_best_threshold(
                y_validation,
                validation_probabilities,
            )
        )

        logger.info(
            f"Decision threshold : "
            f"{self.decision_threshold}"
        )

        # --------------------------------------------------
        # Refit on complete training period
        # --------------------------------------------------

        self.model.fit(
            X_train_full,
            y_train_full,
        )

        # --------------------------------------------------
        # Test evaluation
        # --------------------------------------------------

        probabilities = (
            self._positive_probability(
                self.model,
                X_test,
            )
        )

        predictions = (
            probabilities
            >= self.decision_threshold
        ).astype(int)

        metrics = EvaluationEngine.evaluate(
            y_test,
            predictions,
            probabilities,
        )

        metrics[
            "decision_threshold"
        ] = self.decision_threshold

        # --------------------------------------------------
        # Feature importance
        # --------------------------------------------------

        feature_importance = {
            feature: round(
                float(score),
                6,
            )
            for feature, score in zip(
                X.columns,
                self.model.feature_importances_,
            )
        }

        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # --------------------------------------------------
        # Configuration
        # --------------------------------------------------

        training_config = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "split_type": "chronological",
            "class_weight": "balanced",
            "validation_ratio": 0.15,
            "decision_threshold": (
                self.decision_threshold
            ),
            "feature_count": int(
                X.shape[1]
            ),
            "architecture": (
                "independent_binary_random_forest"
            ),
            "version": "V4",
        }

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        ModelRegistry.save_model(
            model=self.model,
            model_name=self.model_name,
            metrics=metrics,
            feature_importance=feature_importance,
            training_config=training_config,
            dataset_size=len(X),
        )

        logger.info(
            f"Training completed : "
            f"{self.model_name}"
        )

        return {
            "model": self.model_name,
            "dataset_size": len(X),
            "feature_count": int(
                X.shape[1]
            ),
            "train_size": len(
                X_train_full
            ),
            "validation_size": (
                validation_size
            ),
            "test_size": len(
                X_test
            ),
            "decision_threshold": (
                self.decision_threshold
            ),
            "metrics": metrics,
            "feature_importance": (
                feature_importance
            ),
        }

    def predict(
        self,
        X,
    ):
        probabilities = (
            self._positive_probability(
                self.model,
                X,
            )
        )

        return (
            probabilities
            >= self.decision_threshold
        ).astype(int)

    def predict_proba(
        self,
        X,
    ):
        """
        Preserve sklearn-style predict_proba output.

        This is important for compatibility with the
        existing prediction router.
        """

        return self.model.predict_proba(X)

    def load(
        self,
    ):
        self.model = (
            ModelRegistry.load_model(
                self.model_name
            )
        )

        logger.info(
            f"Model loaded : {self.model_name}"
        )

        return self.model