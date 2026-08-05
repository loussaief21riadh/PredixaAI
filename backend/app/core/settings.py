from __future__ import annotations

import os
from pathlib import Path
from typing import Final


# =====================================================
# Helpers
# =====================================================

def _read_bool(
    name: str,
    default: bool,
) -> bool:
    """Read a strict boolean environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    value = raw_value.strip().lower()

    if value in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if value in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    raise RuntimeError(
        f"{name} must be one of: "
        "1, true, yes, on, 0, false, no, off."
    )


def _read_int(
    name: str,
    default: int,
    *,
    minimum: int | None = None,
) -> int:
    """Read and validate an integer environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        value = default
    else:
        try:
            value = int(
                raw_value.strip()
            )
        except ValueError as exc:
            raise RuntimeError(
                f"{name} must be an integer."
            ) from exc

    if (
        minimum is not None
        and value < minimum
    ):
        raise RuntimeError(
            f"{name} must be at least {minimum}."
        )

    return value


def _read_csv(
    name: str,
    default: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Read a comma-separated environment variable."""

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    return tuple(
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    )


# =====================================================
# Project paths
# =====================================================

BASE_DIR: Final[Path] = (
    Path(__file__)
    .resolve()
    .parent
    .parent
    .parent
)

APP_DIR: Final[Path] = (
    BASE_DIR / "app"
)

DATASETS_DIR: Final[Path] = (
    BASE_DIR / "datasets"
)

MODELS_DIR: Final[Path] = (
    BASE_DIR / "trained_models"
)

REPORTS_DIR: Final[Path] = (
    BASE_DIR / "reports"
)

LOGS_DIR: Final[Path] = (
    BASE_DIR / "logs"
)

DATABASE_PATH: Final[Path] = (
    BASE_DIR / "sql_app.db"
)


# =====================================================
# Runtime environment
# =====================================================

APP_ENV: Final[str] = (
    os.getenv(
        "APP_ENV",
        "development",
    )
    .strip()
    .lower()
)

SUPPORTED_ENVIRONMENTS: Final[
    tuple[str, ...]
] = (
    "development",
    "test",
    "staging",
    "production",
)

if APP_ENV not in SUPPORTED_ENVIRONMENTS:
    raise RuntimeError(
        "APP_ENV must be one of: "
        + ", ".join(
            SUPPORTED_ENVIRONMENTS
        )
    )

DEBUG: Final[bool] = _read_bool(
    "DEBUG",
    APP_ENV == "development",
)

SQL_ECHO: Final[bool] = _read_bool(
    "SQL_ECHO",
    False,
)

PORT: Final[int] = _read_int(
    "PORT",
    8000,
    minimum=1,
)


# =====================================================
# Project information
# =====================================================

PROJECT_NAME: Final[str] = os.getenv(
    "PROJECT_NAME",
    "Predixa AI",
)

VERSION: Final[str] = os.getenv(
    "APP_VERSION",
    "7.0.0",
)

AUTHOR: Final[str] = (
    "Loussaief Riadh"
)

OWNER: Final[str] = (
    "Loussaief Riadh"
)

LICENSE: Final[str] = (
    "Proprietary"
)


# =====================================================
# Database
# =====================================================

DEFAULT_DATABASE_URL: Final[str] = (
    f"sqlite:///{DATABASE_PATH}"
)

DATABASE_URL: Final[str] = (
    os.getenv(
        "DATABASE_URL",
        DEFAULT_DATABASE_URL,
    )
    .strip()
)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL cannot be empty."
    )

DB_POOL_PRE_PING: Final[bool] = (
    _read_bool(
        "DB_POOL_PRE_PING",
        True,
    )
)

DB_POOL_RECYCLE_SECONDS: Final[int] = (
    _read_int(
        "DB_POOL_RECYCLE_SECONDS",
        1800,
        minimum=0,
    )
)


# =====================================================
# Authentication
# =====================================================

_DEVELOPMENT_SECRET: Final[str] = (
    "development-only-secret-change-me"
)

SECRET_KEY: Final[str] = os.getenv(
    "SECRET_KEY",
    _DEVELOPMENT_SECRET,
)

ALGORITHM: Final[str] = os.getenv(
    "JWT_ALGORITHM",
    "HS256",
)

ACCESS_TOKEN_EXPIRE_MINUTES: Final[int] = (
    _read_int(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        60 * 24 * 7,
        minimum=1,
    )
)


# =====================================================
# HTTP
# =====================================================

CORS_ALLOWED_ORIGINS: Final[
    tuple[str, ...]
] = _read_csv(
    "CORS_ALLOWED_ORIGINS",
    (
        "http://localhost:3000",
        "http://localhost:5173",
    )
    if APP_ENV == "development"
    else (),
)

TRUSTED_HOSTS: Final[
    tuple[str, ...]
] = _read_csv(
    "TRUSTED_HOSTS",
    (
        "localhost",
        "127.0.0.1",
    )
    if APP_ENV == "development"
    else (),
)


# =====================================================
# Machine learning
# =====================================================

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
N_ESTIMATORS: Final[int] = 300
MAX_DEPTH: Final[int] = 12


# =====================================================
# Runtime validation
# =====================================================

def validate_runtime_settings() -> None:
    """
    Reject unsafe settings in staging and production.

    Development keeps SQLite and a development-only JWT key
    so the existing local workflow continues to operate.
    """

    errors: list[str] = []

    if APP_ENV in {
        "staging",
        "production",
    }:
        if (
            SECRET_KEY
            == _DEVELOPMENT_SECRET
        ):
            errors.append(
                "SECRET_KEY must be supplied "
                "through the environment."
            )

        if len(SECRET_KEY) < 32:
            errors.append(
                "SECRET_KEY must contain "
                "at least 32 characters."
            )

        if DATABASE_URL.startswith(
            "sqlite"
        ):
            errors.append(
                "DATABASE_URL must use PostgreSQL "
                "in staging/production."
            )

        if not TRUSTED_HOSTS:
            errors.append(
                "TRUSTED_HOSTS must contain "
                "at least one hostname."
            )

    if errors:
        raise RuntimeError(
            "Unsafe runtime configuration: "
            + " ".join(errors)
        )
