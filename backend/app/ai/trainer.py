from app.ai.dataset_builder import DatasetBuilder
from app.ai.random_forest import RandomForestEngine
from app.core.logger import logger


class Trainer:
    """
    Predixa AI Trainer

    - Build dataset
    - Train all Random Forest models
    - Save models
    - Return training report
    """

    @staticmethod
    def train_random_forest(db):

        logger.info("=" * 60)
        logger.info("PREDIXA AI - RANDOM FOREST TRAINING")
        logger.info("=" * 60)

        X, y = DatasetBuilder.build(db)

        logger.info(f"Dataset Size : {len(X)}")
        logger.info(f"Features     : {len(X.columns)}")
        logger.info(f"Targets      : {len(y.columns)}")

        results = {}

        average_accuracy = []

        for target in y.columns:

            logger.info(f"Training {target}...")

            engine = RandomForestEngine(
                model_name=f"random_forest_{target}"
            )

            report = engine.train(
                X,
                y[target],
            )

            results[target] = report

            accuracy = report["metrics"]["accuracy"]

            average_accuracy.append(accuracy)

            logger.info(
                f"✓ {target} | Accuracy : {accuracy:.4f}"
            )

        overall_accuracy = (
            round(
                sum(average_accuracy) / len(average_accuracy),
                4,
            )
            if average_accuracy
            else 0.0
        )

        logger.info("=" * 60)
        logger.info("TRAINING FINISHED SUCCESSFULLY")
        logger.info("=" * 60)

        return {

            "status": "success",

            "models_trained": len(results),

            "dataset_size": len(X),

            "features": len(X.columns),

            "average_accuracy": overall_accuracy,

            "results": results,
        }