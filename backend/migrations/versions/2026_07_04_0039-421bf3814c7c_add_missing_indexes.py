"""add missing indexes

Phase 2.4: add the missing indexes that the legacy migration layer created but
that were not reflected in the SQLAlchemy models.

- Foreign-key indexes on hot paths:
  - documents.user_id
  - conversations.user_id
  - conversations.document_id
  - messages.conversation_id

- DocumentChunks search indexes (re-created concurrently because CREATE INDEX
  CONCURRENTLY cannot run inside a transaction):
  - idx_document_chunks_embedding_hnsw (HNSW on embedding, cosine distance)
  - idx_document_chunks_content_fts (GIN tsvector on content)
  - idx_document_chunks_document_id (single-column FK)
  - idx_document_chunks_document_id_page_number (composite FK + page)

Note on naming: the model already defines ix_document_chunks_chunk_level, which
matches the legacy idx_document_chunks_level index on the same column. Alembic
sees one named index from the model side and one from the DB side and would
emit a drop/create. We avoid that collision by keeping only the model-side
index (ix_document_chunks_chunk_level) and dropping the legacy duplicate
(idx_document_chunks_level). The functional coverage is identical.

Revision ID: 421bf3814c7c
Revises: 5ea9a480db18
Create Date: 2026-07-04 00:39:26.666371

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = "421bf3814c7c"
down_revision: Union[str, Sequence[str], None] = "5ea9a480db18"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _create_concurrent_index(statement: str) -> None:
    """Execute a CREATE INDEX CONCURRENTLY statement outside a transaction."""
    with op.get_context().autocommit_block():
        op.execute(text(statement))


def _drop_concurrent_index(index_name: str) -> None:
    """Drop an index concurrently outside a transaction."""
    with op.get_context().autocommit_block():
        op.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {index_name}"))


def upgrade() -> None:
    """Upgrade schema."""
    # Standard B-tree indexes inside the migration transaction.
    op.create_index("ix_documents_user_id", "documents", ["user_id"], unique=False)
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"], unique=False)
    op.create_index("ix_conversations_document_id", "conversations", ["document_id"], unique=False)
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"], unique=False)

    # Drop duplicate/legacy indexes that the model does not reflect.
    # They may already be gone from a previous partial run, so use IF EXISTS.
    op.execute("DROP INDEX IF EXISTS idx_documents_status")
    op.execute("DROP INDEX IF EXISTS idx_document_chunks_level")
    op.execute("DROP INDEX IF EXISTS idx_parent_child_parent")
    op.execute("DROP INDEX IF EXISTS idx_parent_child_child")

    # Drop legacy special indexes so we can re-create them as model-managed.
    # Use CONCURRENTLY to avoid heavy locks; autocommit is required.
    _drop_concurrent_index("idx_document_chunks_embedding_hnsw")
    _drop_concurrent_index("idx_document_chunks_content_fts")
    _drop_concurrent_index("idx_document_chunks_document_id")
    _drop_concurrent_index("idx_document_chunks_document_id_page_number")

    # Re-create the special indexes concurrently.
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_fts "
        "ON document_chunks USING gin (to_tsvector('english'::regconfig, content))"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id "
        "ON document_chunks (document_id)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id_page_number "
        "ON document_chunks (document_id, page_number)"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Drop model-managed FK indexes.
    op.drop_index("ix_messages_conversation_id", table_name="messages")
    op.drop_index("ix_conversations_document_id", table_name="conversations")
    op.drop_index("ix_conversations_user_id", table_name="conversations")
    op.drop_index("ix_documents_user_id", table_name="documents")

    # Re-create legacy special indexes.
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_parent_child_parent "
        "ON parent_child_relationships (parent_chunk_id)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_parent_child_child "
        "ON parent_child_relationships (child_chunk_id)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_status "
        "ON documents (status)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_level "
        "ON document_chunks (chunk_level)"
    )

    # Drop model-managed special indexes.
    _drop_concurrent_index("idx_document_chunks_document_id_page_number")
    _drop_concurrent_index("idx_document_chunks_document_id")
    _drop_concurrent_index("idx_document_chunks_content_fts")
    _drop_concurrent_index("idx_document_chunks_embedding_hnsw")

    # Re-create the original legacy special indexes.
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_fts "
        "ON document_chunks USING gin (to_tsvector('english'::regconfig, content))"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id "
        "ON document_chunks (document_id)"
    )
    _create_concurrent_index(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id_page_number "
        "ON document_chunks (document_id, page_number)"
    )
