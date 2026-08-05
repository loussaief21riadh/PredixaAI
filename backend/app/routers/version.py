from __future__ import annotations

from fastapi import APIRouter

from app.core.settings import (
    PROJECT_NAME,
    VERSION,
)


router = APIRouter(
    tags=[
        "system",
    ]
)


@router.get(
    "/version",
    summary="Application version",
)
def get_version() -> dict[str, str]:
    """Return the public application name and version."""

    return {
        "app": PROJECT_NAME,
        "version": VERSION,
    }
