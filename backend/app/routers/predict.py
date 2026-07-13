from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"],
)


@router.get("/")
def predict(db: Session = Depends(get_db)):
    return {
        "prediction": "Not implemented yet",
        "success": True,
    }