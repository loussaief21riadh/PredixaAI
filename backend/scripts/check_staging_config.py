from __future__ import annotations

from sqlalchemy import text

from app.core.settings import (
    APP_ENV,
    DATABASE_URL,
    SECRET_KEY,
    TRUSTED_HOSTS,
    validate_runtime_settings,
)
from app.database import engine


def _safe_database_label() -> str:
    """Return the database dialect without credentials."""

    return DATABASE_URL.split(
        ":",
        maxsplit=1,
    )[0]


def main() -> int:
    """Validate staging settings and connectivity."""

    validate_runtime_settings()

    if APP_ENV not in {
        "staging",
        "production",
    }:
        raise RuntimeError(
            "APP_ENV must be staging or production."
        )

    if len(SECRET_KEY) < 32:
        raise RuntimeError(
            "SECRET_KEY is too short."
        )

    if not TRUSTED_HOSTS:
        raise RuntimeError(
            "TRUSTED_HOSTS cannot be empty."
        )

    with engine.connect() as connection:
        connection.execute(
            text("SELECT 1")
        )

    print("PredixaAI runtime configuration is valid.")
    print(f"Environment: {APP_ENV}")
    print(
        "Database dialect:",
        _safe_database_label(),
    )
    print(
        "Trusted hosts:",
        len(TRUSTED_HOSTS),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
