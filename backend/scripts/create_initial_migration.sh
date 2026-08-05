#!/bin/sh
set -eu

if [ -d "alembic/versions" ] \
   && find "alembic/versions" \
      -maxdepth 1 \
      -type f \
      -name "*.py" \
      | grep -q .; then

    echo "ERROR: an Alembic revision already exists." >&2
    exit 1
fi

mkdir -p alembic/versions

TEMP_DATABASE="alembic_bootstrap.db"

rm -f "$TEMP_DATABASE"

DATABASE_URL="sqlite:///./${TEMP_DATABASE}" \
alembic revision \
    --autogenerate \
    -m "Create users and draws tables"

rm -f "$TEMP_DATABASE"

echo
echo "Initial migration generated."
echo "Review the new file in alembic/versions before applying it."
