from __future__ import annotations

from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.settings import (
    DATABASE_URL,
    DB_POOL_PRE_PING,
    DB_POOL_RECYCLE_SECONDS,
    SQL_ECHO,
    validate_runtime_settings,
)


validate_runtime_settings()


def _engine_options(
    database_url: str,
) -> dict[str, Any]:
    """Return SQLAlchemy options for SQLite or PostgreSQL."""

    options: dict[str, Any] = {
        "echo": SQL_ECHO,
        "pool_pre_ping": DB_POOL_PRE_PING,
    }

    if DB_POOL_RECYCLE_SECONDS > 0:
        options["pool_recycle"] = (
            DB_POOL_RECYCLE_SECONDS
        )

    if database_url.startswith("sqlite"):
        options["connect_args"] = {
            "check_same_thread": False,
        }

    return options


engine: Engine = create_engine(
    DATABASE_URL,
    **_engine_options(DATABASE_URL),
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI database dependency."""

    database = SessionLocal()

    try:
        yield database
    finally:
        database.close()
