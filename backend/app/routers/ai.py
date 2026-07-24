from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.ai.backtester import Backtester
from app.ai.trainer import Trainer
from app.ai.walk_forward_backtester import WalkForwardBacktester

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


@router.post("/backtest")
def backtest_random_forest(
    test_draws: int = Query(
        default=100,
        ge=10,
        le=1000,
    ),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Run the V2 backtest.

    Access:
        Administrator only.
    """

    try:
        return Backtester.run(
            db=db,
            test_draws=test_draws,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )


@router.post("/walk-forward")
def walk_forward_backtest(
    test_draws: int = Query(
        default=5,
        ge=5,
        le=100,
    ),
    window_size: int = Query(
        default=100,
        ge=20,
        le=500,
    ),
    max_training_samples: int = Query(
        default=1500,
        ge=100,
        le=5000,
    ),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Predixa AI V3 strict walk-forward backtest.

    For each evaluated draw:

    - Uses only historical draws before the target.
    - Rebuilds the training dataset.
    - Trains temporary Random Forest models.
    - Generates a Top-5 prediction.
    - Compares against the actual draw.

    Access:
        Administrator only.

    Warning:
        Computationally expensive.
    """

    try:
        return WalkForwardBacktester.run(
            db=db,
            test_draws=test_draws,
            window_size=window_size,
            max_training_samples=max_training_samples,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Walk-forward backtest failed: "
                f"{str(exc)}"
            ),
        )