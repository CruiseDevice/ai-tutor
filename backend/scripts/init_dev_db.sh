#!/usr/bin/env bash
# One-time dev database setup: ensure extensions and apply Alembic migrations.
# Usage: ./scripts/init_dev_db.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Ensuring Postgres extensions..."
psql "${DATABASE_URL}" -f scripts/ensure_extensions.sql 2>/dev/null || \
  echo "Could not ensure extensions automatically; run scripts/ensure_extensions.sql as an admin role."

echo "Running Alembic migrations..."
alembic upgrade head

echo "Dev database initialized."
