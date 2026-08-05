from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.settings import APP_ENV, VERSION
from app.database import get_db


router = APIRouter(
    tags=[
        "health",
    ]
)


@router.get(
    "/health",
    summary="Application liveness check",
)
def health_check() -> dict[str, str]:
    """Confirm that the FastAPI process is alive."""

    return {
        "status": "healthy",
        "environment": APP_ENV,
        "version": VERSION,
    }


@router.get(
    "/ready",
    summary="Application readiness check",
)
def readiness_check(
    database: Annotated[
        Session,
        Depends(get_db),
    ],
) -> dict[str, str]:
    """Confirm that the database can be reached."""

    try:
        database.execute(
            text("SELECT 1")
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail="Database is not ready.",
        ) from exc

    return {
        "status": "ready",
        "database": "reachable",
        "environment": APP_ENV,
        "version": VERSION,
    }
