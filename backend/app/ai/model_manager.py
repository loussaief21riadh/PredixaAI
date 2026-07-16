from pathlib import Path

import joblib


class ModelManager:
    """
    Gestionnaire des modèles Machine Learning.

    Sauvegarde, charge et supprime les modèles.
    """

    MODEL_DIR = Path("trained_models")

    @classmethod
    def initialize(cls):
        cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def save(cls, model, model_name: str):

        cls.initialize()

        file_path = cls.MODEL_DIR / f"{model_name}.pkl"

        joblib.dump(model, file_path)

        return str(file_path)

    @classmethod
    def load(cls, model_name: str):

        file_path = cls.MODEL_DIR / f"{model_name}.pkl"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Model '{model_name}' not found."
            )

        return joblib.load(file_path)

    @classmethod
    def exists(cls, model_name: str):

        file_path = cls.MODEL_DIR / f"{model_name}.pkl"

        return file_path.exists()

    @classmethod
    def delete(cls, model_name: str):

        file_path = cls.MODEL_DIR / f"{model_name}.pkl"

        if file_path.exists():
            file_path.unlink()

    @classmethod
    def list_models(cls):

        cls.initialize()

        return sorted(
            file.stem
            for file in cls.MODEL_DIR.glob("*.pkl")
        )