from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.statistics.frequency import FrequencyAnalyzer

router = APIRouter(
    prefix="/statistics",
    tags=["Statistics"],
)


@router.get("/frequency")
def frequency(db: Session = Depends(get_db)):
    return FrequencyAnalyzer.calculate(db)