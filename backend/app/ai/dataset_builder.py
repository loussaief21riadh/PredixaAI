import pandas as pd
from sqlalchemy.orm import Session

from app.ai.feature_engineering import FeatureEngineering
from app.models.draw import Draw


class DatasetBuilder:
    """
    Construit le dataset Machine Learning.

    X : variables explicatives (features)

    y : dictionnaire contenant 49 cibles binaires
        (une cible par numéro de loto)
    """

    @staticmethod
    def build(db: Session):

        draws = (
            db.query(Draw)
            .order_by(Draw.draw_date.asc())
            .all()
        )

        rows = []

        for draw in draws:

            numbers = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ]

            features = FeatureEngineering.build(numbers)

            # Création de 49 colonnes cibles
            for number in range(1, 50):

                features[f"target_{number}"] = (
                    1 if number in numbers else 0
                )

            rows.append(features)

        df = pd.DataFrame(rows)

        feature_columns = [
            c for c in df.columns
            if not c.startswith("target_")
        ]

        target_columns = [
            c for c in df.columns
            if c.startswith("target_")
        ]

        X = df[feature_columns]

        y = df[target_columns]

        return X, y