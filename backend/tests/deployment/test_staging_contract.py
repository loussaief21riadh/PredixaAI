from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.core import settings
from app.database import _engine_options
from app.routers.health import (
    health_check,
    readiness_check,
)
from app.routers.version import (
    get_version,
)


class SuccessfulDatabase:
    def execute(
        self,
        statement: object,
    ) -> object:
        return statement


class FailingDatabase:
    def execute(
        self,
        statement: object,
    ) -> object:
        raise SQLAlchemyError(
            "database unavailable"
        )


def test_sqlite_engine_options_include_thread_setting() -> None:
    options = _engine_options(
        "sqlite:///./test.db"
    )

    assert options[
        "connect_args"
    ] == {
        "check_same_thread": False,
    }


def test_postgresql_engine_options_do_not_use_sqlite_setting() -> None:
    options = _engine_options(
        "postgresql+psycopg://"
        "user:password@db/predixa"
    )

    assert (
        "connect_args"
        not in options
    )


def test_health_check() -> None:
    response = health_check()

    assert (
        response["status"]
        == "healthy"
    )

    assert (
        response["environment"]
        == settings.APP_ENV
    )

    assert (
        response["version"]
        == settings.VERSION
    )


def test_readiness_check_success() -> None:
    response = readiness_check(
        SuccessfulDatabase()
    )

    assert response == {
        "status": "ready",
        "database": "reachable",
        "environment": settings.APP_ENV,
        "version": settings.VERSION,
    }


def test_readiness_check_returns_503() -> None:
    with pytest.raises(
        HTTPException,
    ) as captured:
        readiness_check(
            FailingDatabase()
        )

    assert (
        captured.value.status_code
        == 503
    )

    assert (
        captured.value.detail
        == "Database is not ready."
    )


def test_version_endpoint_contract() -> None:
    response = get_version()

    assert response == {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
    }


def test_password_hash_round_trip() -> None:
    password = (
        "Strong-test-password-123!"
    )

    hashed = hash_password(
        password
    )

    assert hashed != password

    assert verify_password(
        password,
        hashed,
    )

    assert not verify_password(
        "wrong-password",
        hashed,
    )


def test_access_token_round_trip() -> None:
    token = create_access_token(
        {
            "sub": (
                "deployment-test-user"
            ),
        }
    )

    payload = decode_access_token(
        token
    )

    assert payload is not None

    assert (
        payload["sub"]
        == "deployment-test-user"
    )


def test_decode_invalid_token_returns_none() -> None:
    assert (
        decode_access_token(
            "invalid-token"
        )
        is None
    )
