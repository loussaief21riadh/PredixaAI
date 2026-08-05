from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.auth.router import router as auth_router
from app.core.settings import (
    APP_ENV,
    CORS_ALLOWED_ORIGINS,
    PROJECT_NAME,
    TRUSTED_HOSTS,
    VERSION,
)
from app.dashboard.router import router as dashboard_router
from app.routers import (
    ai,
    draws,
    health,
    import_csv,
    predict,
    statistics,
    version,
)


# =====================================================
# FastAPI application
# =====================================================

app = FastAPI(
    title=f"{PROJECT_NAME} Enterprise",
    version=VERSION,
    description=(
        "Enterprise AI Lottery Prediction Platform"
    ),
)


# =====================================================
# HTTP middleware
# =====================================================

if CORS_ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(
            CORS_ALLOWED_ORIGINS
        ),
        allow_credentials=True,
        allow_methods=[
            "*",
        ],
        allow_headers=[
            "*",
        ],
    )


if APP_ENV in {
    "staging",
    "production",
}:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(
            TRUSTED_HOSTS
        ),
    )


# =====================================================
# Root
# =====================================================

@app.get(
    "/",
    tags=[
        "Root",
    ],
)
def root() -> dict[str, str]:
    """Return basic application information."""

    return {
        "message": (
            "Welcome to Predixa AI Enterprise"
        ),
        "version": app.version,
        "environment": APP_ENV,
        "status": "running",
    }


# =====================================================
# Authentication
# =====================================================

app.include_router(
    auth_router
)


# =====================================================
# Dashboard
# =====================================================

app.include_router(
    dashboard_router
)


# =====================================================
# System
# =====================================================

app.include_router(
    health.router
)

app.include_router(
    version.router
)


# =====================================================
# Draws
# =====================================================

app.include_router(
    draws.router
)


# =====================================================
# Prediction
# =====================================================

app.include_router(
    predict.router
)


# =====================================================
# CSV import
# =====================================================

app.include_router(
    import_csv.router
)


# =====================================================
# Statistics
# =====================================================

app.include_router(
    statistics.router
)


# =====================================================
# Artificial intelligence
# =====================================================

app.include_router(
    ai.router
)
