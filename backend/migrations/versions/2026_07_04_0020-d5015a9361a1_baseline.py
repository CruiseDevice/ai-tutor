"""baseline

Baseline migration representing the schema state after the legacy hand-rolled
migrations in app/database_migrations.py. This revision intentionally performs
no DDL so that existing databases can be stamped as already-at-baseline
without applying destructive changes.

Known model/schema drift detected by autogenerate (to be reconciled in later
migrations):
- Several indexes created by database_migrations.py are not reflected in the
  SQLAlchemy models (e.g. HNSW vector index, FTS GIN index, composite FK
  indexes). Phase 2.4 will add these indexes to the models and create proper
  Alembic migrations.
- users.role uses a Postgres enum in the model but a VARCHAR in the current
  DB schema. Phase 2.3 will reconcile this.

Revision ID: d5015a9361a1
Revises:
Create Date: 2026-07-04 00:20:26.421364

"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5015a9361a1"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """No-op baseline.

    The schema already exists in databases that ran the legacy migration layer.
    Future migrations will add/drop objects as needed.
    """
    pass


def downgrade() -> None:
    """No-op baseline downgrade."""
    pass
