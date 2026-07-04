#!/usr/bin/env bash
# Run Alembic migrations against the database configured in the app environment.
# Usage: ./scripts/run_migrations.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running Alembic migrations..."
alembic upgrade head
echo "Migrations complete."
