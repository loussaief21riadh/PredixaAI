from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.auth.models import User
from app.dashboard.schemas import DashboardResponse
from app.dashboard.service import DashboardService
from app.database import get_db

router = APIRouter(
    prefix="/dashboard",
    tags=["Dashboard"],
)


@router.get(
    "",
    response_model=DashboardResponse,
)
def get_dashboard(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Enterprise Dashboard

    Access:
        Authenticated users only.
    """

    return DashboardService.get_dashboard(db)