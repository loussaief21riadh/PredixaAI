from pathlib import Path
import json
import joblib
from datetime import datetime

from app.core.logger import logger
from app.core.settings import MODELS_DIR, VERSION, AUTHOR


class ModelRegistry:
    """
    Predixa AI Model Registry

    Responsible for:

    - Saving trained models
    - Saving metadata
    - Saving metrics
    - Saving feature importance
    - Loading models
    """

    @staticmethod
    def _model_directory(model_name: str) -> Path:

        directory = MODELS_DIR / model_name

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory

    @staticmethod
    def save_model(
        model,
        model_name: str,
        metrics: dict,
        feature_importance: dict,
        training_config: dict,
        dataset_size: int,
    ):

        model_dir = ModelRegistry._model_directory(model_name)

        # --------------------------------------------------
        # Save sklearn model
        # --------------------------------------------------

        joblib.dump(
            model,
            model_dir / "model.pkl",
        )

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        metadata = {

            "model_name": model_name,

            "algorithm": "Random Forest",

            "version": VERSION,

            "author": AUTHOR,

            "created_at": datetime.now().isoformat(),

            "dataset_size": dataset_size,
        }

        with open(
            model_dir / "metadata.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        with open(
            model_dir / "metrics.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metrics,
                f,
                indent=4,
            )

        # --------------------------------------------------
        # Feature Importance
        # --------------------------------------------------

        with open(
            model_dir / "feature_importance.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                feature_importance,
                f,
                indent=4,
            )

        # --------------------------------------------------
        # Training configuration
        # --------------------------------------------------

        with open(
            model_dir / "training.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                training_config,
                f,
                indent=4,
            )

        # --------------------------------------------------
        # Version
        # --------------------------------------------------

        version = {

            "version": VERSION,

            "saved_at": datetime.now().isoformat(),
        }

        with open(
            model_dir / "version.json",
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                version,
                f,
                indent=4,
            )

        logger.info(f"Model saved : {model_name}")

    @staticmethod
    def load_model(model_name: str):

        model_dir = MODELS_DIR / model_name

        model = joblib.load(
            model_dir / "model.pkl"
        )

        logger.info(f"Model loaded : {model_name}")

        return model