#!/bin/sh
set -eu

echo "Starting PredixaAI ${APP_VERSION:-unknown} in ${APP_ENV:-unknown} mode."

python scripts/check_staging_config.py

if [ -d "alembic/versions" ] \
   && find "alembic/versions" \
      -maxdepth 1 \
      -type f \
      -name "*.py" \
      | grep -q .; then

    echo "Applying Alembic migrations."
    alembic upgrade head
else
    echo "ERROR: no Alembic revision exists." >&2
    echo "Create and review the initial migration first." >&2
    exit 1
fi

exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --proxy-headers \
    --forwarded-allow-ips="*"
