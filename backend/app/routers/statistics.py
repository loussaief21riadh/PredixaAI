from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db

from app.statistics.frequency import FrequencyAnalyzer
from app.statistics.hot_numbers import HotNumbersAnalyzer
from app.statistics.cold_numbers import ColdNumbersAnalyzer
from app.statistics.overdue import OverdueAnalyzer
from app.statistics.pair_analyzer import PairAnalyzer
from app.statistics.triplet_analyzer import TripletAnalyzer
from app.statistics.even_odd import EvenOddAnalyzer
from app.statistics.sum_analyzer import SumAnalyzer
from app.statistics.decade_analyzer import DecadeAnalyzer
from app.statistics.consecutive_analyzer import ConsecutiveAnalyzer
from app.statistics.dashboard import DashboardAnalyzer

router = APIRouter(
    prefix="/statistics",
    tags=["Statistics"],
)


@router.get("/dashboard")
def get_dashboard(
    db: Session = Depends(get_db),
):
    return DashboardAnalyzer.calculate(db)


@router.get("/frequency")
def get_frequency(db: Session = Depends(get_db)):
    return FrequencyAnalyzer.calculate(db)


@router.get("/hot")
def get_hot_numbers(
    limit: int = 10,
    db: Session = Depends(get_db),
):
    return HotNumbersAnalyzer.calculate(
        db=db,
        limit=limit,
    )


@router.get("/cold")
def get_cold_numbers(
    limit: int = 10,
    db: Session = Depends(get_db),
):
    return ColdNumbersAnalyzer.calculate(
        db=db,
        limit=limit,
    )


@router.get("/overdue")
def get_overdue_numbers(
    db: Session = Depends(get_db),
):
    return OverdueAnalyzer.calculate(db)


@router.get("/pairs")
def get_pairs(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    return PairAnalyzer.calculate(
        db=db,
        limit=limit,
    )


@router.get("/triplets")
def get_triplets(
    limit: int = 20,
    db: Session = Depends(get_db),
):
    return TripletAnalyzer.calculate(
        db=db,
        limit=limit,
    )


@router.get("/even-odd")
def get_even_odd(
    db: Session = Depends(get_db),
):
    return EvenOddAnalyzer.calculate(db)


@router.get("/sums")
def get_sums(
    db: Session = Depends(get_db),
):
    return SumAnalyzer.calculate(db)


@router.get("/decades")
def get_decades(
    db: Session = Depends(get_db),
):
    return DecadeAnalyzer.calculate(db)


@router.get("/consecutive")
def get_consecutive(
    db: Session = Depends(get_db),
):
    return ConsecutiveAnalyzer.calculate(db)