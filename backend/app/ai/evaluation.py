from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


class EvaluationEngine:
    """
    Predixa AI V4 binary model evaluation.

    Each Random Forest model predicts one number:

        0 = number absent
        1 = number present

    Predixa trains 49 independent binary models.
    """

    @staticmethod
    def evaluate(
        y_true,
        y_pred,
        y_proba=None,
    ) -> dict:

        metrics = {
            "accuracy": round(
                float(
                    accuracy_score(
                        y_true,
                        y_pred,
                    )
                ),
                4,
            ),

            "precision": round(
                float(
                    precision_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),
                4,
            ),

            "recall": round(
                float(
                    recall_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),
                4,
            ),

            "f1_score": round(
                float(
                    f1_score(
                        y_true,
                        y_pred,
                        zero_division=0,
                    )
                ),
                4,
            ),

            "confusion_matrix": (
                confusion_matrix(
                    y_true,
                    y_pred,
                    labels=[0, 1],
                ).tolist()
            ),
        }

        if y_proba is not None:

            try:
                metrics["roc_auc"] = round(
                    float(
                        roc_auc_score(
                            y_true,
                            y_proba,
                        )
                    ),
                    4,
                )

            except ValueError:
                metrics["roc_auc"] = None

            try:
                metrics[
                    "average_precision"
                ] = round(
                    float(
                        average_precision_score(
                            y_true,
                            y_proba,
                        )
                    ),
                    4,
                )

            except ValueError:
                metrics[
                    "average_precision"
                ] = None

        return metrics