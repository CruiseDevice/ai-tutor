"""reconcile schema drift

Phase 2.3 schema reconciliation:

- conversations.user_id column name was camelCase ("userId") in the model but the
  SQLAlchemy model used `name="userId"`. The model is updated to use snake_case
  and the FK is renamed to `conversations_user_id_fkey` for consistency.

- users.role used a Postgres ENUM ("userrole") in the SQLAlchemy model but the
  existing database uses a VARCHAR. The model is switched to String with a
  CHECK constraint enforcing the allowed Python enum values.

- document_chunks.chunk_type and chunk_level were nullable in the model but the
  migration layer backfilled them to NOT NULL. The model is updated to match.

- The model does not yet reflect several indexes created by the legacy migration
  layer (HNSW, FTS, composite FK indexes). Those indexes are intentionally left
  untouched here and will be added via Alembic in Phase 2.4 after the model is
  extended to include them. This migration therefore does NOT drop them.

Revision ID: 5ea9a480db18
Revises: d5015a9361a1
Create Date: 2026-07-04 00:32:50.118000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "5ea9a480db18"
down_revision: Union[str, Sequence[str], None] = "d5015a9361a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to match the canonical model definitions."""
    # conversations: rename userId -> user_id and rename FK constraint.
    # Add as nullable first, backfill, then set NOT NULL.
    op.add_column("conversations", sa.Column("user_id", sa.String(), nullable=True))
    op.execute("UPDATE conversations SET user_id = \"userId\"")
    op.alter_column("conversations", "user_id", nullable=False)
    op.drop_constraint("conversations_userId_fkey", "conversations", type_="foreignkey")
    op.create_foreign_key(
        "conversations_user_id_fkey",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_column("conversations", "userId")

    # document_chunks: align NOT NULL constraints with model defaults.
    op.alter_column("document_chunks", "chunk_type", existing_type=sa.VARCHAR(), nullable=False)
    op.alter_column("document_chunks", "chunk_level", existing_type=sa.VARCHAR(), nullable=False)

    # users.role: switch from Postgres ENUM to VARCHAR + CHECK constraint.
    # First remove the default that depends on the enum type.
    op.execute("ALTER TABLE users ALTER COLUMN role DROP DEFAULT")
    op.alter_column(
        "users",
        "role",
        existing_type=postgresql.ENUM("user", "admin", "super_admin", name="userrole"),
        type_=sa.String(),
        existing_nullable=False,
        postgresql_using="role::text",
    )
    op.execute("DROP TYPE IF EXISTS userrole")

    # Restore a sensible default and add CHECK constraint.
    op.execute("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'user'")
    op.create_check_constraint(
        "ck_users_role_allowed",
        "users",
        sa.text("role IN ('user', 'admin', 'super_admin')"),
    )

    # NOTE: Indexes created by the legacy migration layer are intentionally left
    # in place here. They will be re-created through the models in Phase 2.4 so
    # that autogenerate can manage them going forward.


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_users_role_allowed", "users", type_="check")
    op.execute("ALTER TABLE users ALTER COLUMN role DROP DEFAULT")

    op.create_type(postgresql.ENUM("user", "admin", "super_admin", name="userrole"))
    op.alter_column(
        "users",
        "role",
        existing_type=sa.String(),
        type_=postgresql.ENUM("user", "admin", "super_admin", name="userrole"),
        existing_nullable=False,
        postgresql_using="role::userrole",
    )
    op.execute("ALTER TABLE users ALTER COLUMN role SET DEFAULT 'user'::userrole")

    op.alter_column("document_chunks", "chunk_level", existing_type=sa.VARCHAR(), nullable=True)
    op.alter_column("document_chunks", "chunk_type", existing_type=sa.VARCHAR(), nullable=True)

    op.add_column("conversations", sa.Column("userId", sa.VARCHAR(), autoincrement=False, nullable=True))
    op.execute('UPDATE conversations SET "userId" = user_id')
    op.alter_column("conversations", "userId", nullable=False)
    op.drop_constraint("conversations_user_id_fkey", "conversations", type_="foreignkey")
    op.create_foreign_key(
        "conversations_userId_fkey",
        "conversations",
        "users",
        ["userId"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_column("conversations", "user_id")
