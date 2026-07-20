from pathlib import Path

from sqlalchemy.orm import Session

from app.auth.models import User
from app.models.draw import Draw


class DashboardService:

    @staticmethod
    def get_dashboard(db: Session):

        users = db.query(User).count()

        draws = db.query(Draw).count()

        models_path = Path("trained_models")

        if models_path.exists():
            models = len(
                list(models_path.glob("*.pkl"))
            )
        else:
            models = 0

        return {
            "users": users,
            "draws": draws,
            "models": models,
            "predictions": 0,
            "accuracy": 0.0,
            "last_training": "Not available",
        }