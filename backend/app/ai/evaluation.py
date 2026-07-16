from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class EvaluationEngine:
    """
    Evaluation des modèles Machine Learning.
    """

    @staticmethod
    def evaluate(y_true, y_pred):

        return {

            "accuracy": round(
                accuracy_score(
                    y_true,
                    y_pred,
                ),
                4,
            ),

            "precision": round(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                4,
            ),

            "recall": round(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                4,
            ),

            "f1_score": round(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                4,
            ),

            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),
        }