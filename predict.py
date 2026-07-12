from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.database import get_db

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)

@router.get("/")
def predict(db: Session = Depends(get_db)):
    """Predict endpoint."""
    return {
        "prediction": "Not implemented yet",
        "success": True
    }
