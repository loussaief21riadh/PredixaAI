from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.ai.backtester import Backtester
from app.ai.trainer import Trainer
from app.ai.walk_forward_backtester import WalkForwardBacktester

from app.ai.v5.walk_forward_backtester import (
    V5WalkForwardBacktester,
)

from app.ai.v5.walk_forward_backtester_v5b import (
    V5BWalkForwardBacktester,
)

from app.auth.dependencies import get_current_admin
from app.auth.models import User
from app.database import get_db


router = APIRouter(
    prefix="/ai",
    tags=["Artificial Intelligence"],
)


# ==========================================================
# TRAIN
# ==========================================================

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


# ==========================================================
# LEGACY BACKTEST
# ==========================================================

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
    Run the legacy Predixa backtest.

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


# ==========================================================
# V4-F WALK-FORWARD
# ==========================================================

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
    Predixa AI V4-F purged T-2 walk-forward benchmark.

    Temporal safeguards:
        - T-1 excluded from prediction features.
        - T-1 target purged from training.
        - Strict chronological evaluation.

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
                "V4-F walk-forward backtest failed: "
                f"{str(exc)}"
            ),
        )


# ==========================================================
# V5-A WALK-FORWARD
# ==========================================================

@router.post("/v5/walk-forward")
def v5_walk_forward_backtest(
    test_draws: int = Query(
        default=100,
        ge=5,
        le=1000,
        description=(
            "Number of historical draws evaluated "
            "by the V5-A walk-forward benchmark."
        ),
    ),
    window_size: int = Query(
        default=100,
        ge=100,
        le=500,
        description=(
            "Historical feature window. "
            "V5-A requires at least 100 draws."
        ),
    ),
    max_training_samples: int = Query(
        default=1500,
        ge=100,
        le=5000,
        description=(
            "Maximum number of chronological "
            "training samples per walk-forward step."
        ),
    ),
    monte_carlo_simulations: int = Query(
        default=10000,
        ge=100,
        le=100000,
        description=(
            "Number of Monte-Carlo simulations "
            "used for the random baseline."
        ),
    ),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Predixa AI V5-A purged T-2 walk-forward benchmark.

    V5-A features:
        - 396 features.
        - normalized frequency rates.
        - windows 10 / 20 / 50 / 100.
        - recency.
        - recency ratio.
        - short-vs-long frequency signal.
        - frequency volatility.
        - global structural statistics.

    Temporal safeguards:
        - prediction features stop at T-2.
        - T-1 excluded from prediction features.
        - T-1 training target purged.
        - strict chronological walk-forward.

    Access:
        Administrator only.
    """

    try:
        return V5WalkForwardBacktester.run(
            db=db,
            test_draws=test_draws,
            window_size=window_size,
            max_training_samples=max_training_samples,
            monte_carlo_simulations=monte_carlo_simulations,
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
                "V5-A walk-forward backtest failed: "
                f"{str(exc)}"
            ),
        )


# ==========================================================
# V5-B ABLATION WALK-FORWARD
# ==========================================================

@router.post("/v5b/walk-forward")
def v5b_walk_forward_backtest(
    variant: str = Query(
        default="full",
        description=(
            "V5-B ablation variant: "
            "full, "
            "no_recency, "
            "no_recency_ratio, "
            "no_short_vs_long, "
            "no_frequency_volatility, "
            "rates_only"
        ),
    ),
    test_draws: int = Query(
        default=100,
        ge=5,
        le=1000,
    ),
    window_size: int = Query(
        default=100,
        ge=100,
        le=500,
    ),
    max_training_samples: int = Query(
        default=1500,
        ge=100,
        le=5000,
    ),
    monte_carlo_simulations: int = Query(
        default=10000,
        ge=100,
        le=100000,
    ),
    db: Session = Depends(get_db),
    admin: User = Depends(get_current_admin),
):
    """
    Predixa AI V5-B Ablation Walk-Forward.

    Same evaluation protocol as V5-A.

    The only experimental variable is the selected
    feature family configuration.

    Supported variants:
        full
        no_recency
        no_recency_ratio
        no_short_vs_long
        no_frequency_volatility
        rates_only

    Temporal safeguards:
        - prediction features stop at T-2.
        - T-1 excluded from prediction features.
        - T-1 target purged from training.
        - strict chronological walk-forward.

    Access:
        Administrator only.
    """

    valid_variants = (
        V5BWalkForwardBacktester
        .EXPECTED_FEATURE_COUNTS
    )

    if variant not in valid_variants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown V5-B variant: {variant}. "
                f"Valid variants: "
                f"{sorted(valid_variants)}"
            ),
        )

    try:
        return V5BWalkForwardBacktester.run(
            db=db,
            variant=variant,
            test_draws=test_draws,
            window_size=window_size,
            max_training_samples=max_training_samples,
            monte_carlo_simulations=monte_carlo_simulations,
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
                "V5-B walk-forward backtest failed: "
                f"{str(exc)}"
            ),
        )