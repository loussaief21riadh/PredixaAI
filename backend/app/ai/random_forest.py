from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

    def __init__(self, model_name: str):

        self.model_name = model_name

        self.model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    def train(self, X, y):

        logger.info(f"Training started : {self.model_name}")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        metrics = EvaluationEngine.evaluate(
            y_test,
            predictions,
        )

        feature_importance = {}

        for feature, score in zip(
            X.columns,
            self.model.feature_importances_,
        ):

            feature_importance[feature] = round(
                float(score),
                6,
            )

        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        training_config = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
        }

        ModelRegistry.save_model(
            model=self.model,
            model_name=self.model_name,
            metrics=metrics,
            feature_importance=feature_importance,
            training_config=training_config,
            dataset_size=len(X),
        )

        logger.info(f"Training completed : {self.model_name}")

        return {

            "model": self.model_name,

            "train_size": len(X_train),

            "test_size": len(X_test),

            "metrics": metrics,

            "feature_importance": feature_importance,
        }

    def predict(self, X):

        return self.model.predict(X)

    def predict_proba(self, X):

        return self.model.predict_proba(X)

    def load(self):

        self.model = ModelRegistry.load_model(
            self.model_name
        )

        logger.info(f"Model loaded : {self.model_name}")

        return self.model