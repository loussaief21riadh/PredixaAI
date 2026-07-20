from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.ai.trainer import Trainer
from app.auth.dependencies import get_current_admin
from app.auth.models import User
from app.database import get_db

router = APIRouter(
    prefix="/ai",
    tags=["Artificial Intelligence"],
)


@router.post("/train")
def train_random_forest(
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Train all Random Forest models.

    Access:
        Administrator only.
    """

    return Trainer.train_random_forest(db)