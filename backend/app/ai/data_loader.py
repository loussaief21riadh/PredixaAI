import pandas as pd
from sqlalchemy.orm import Session

from app.ai.feature_engineering import FeatureEngineering
from app.models.draw import Draw


class DataLoader:
    """
    Charge les tirages depuis la base de données
    et construit le DataFrame Machine Learning.
    """

    @staticmethod
    def load(db: Session) -> pd.DataFrame:

        draws = (
            db.query(Draw)
            .order_by(Draw.draw_date.asc())
            .all()
        )

        dataset = []

        for draw in draws:

            numbers = [
                draw.n1,
                draw.n2,
                draw.n3,
                draw.n4,
                draw.n5,
            ]

            features = FeatureEngineering.build(numbers)

            features["draw_date"] = draw.draw_date

            if draw.chance is not None:
                features["chance"] = draw.chance

            dataset.append(features)

        return pd.DataFrame(dataset)