from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.ai.trainer import Trainer
from app.database import get_db

router = APIRouter(
    prefix="/ai",
    tags=["Artificial Intelligence"],
)


@router.post("/train")
def train_random_forest(
    db: Session = Depends(get_db),
):
    """
    Entraîne tous les modèles Random Forest.
    """

    return Trainer.train_random_forest(db)