"""Tests for Phase 2.4 — missing indexes added via Alembic."""
import os
import pytest


def test_models_declare_expected_indexes():
    """SQLAlchemy models must declare the FK and search indexes."""
    from app.database import Base
    from app.models import document, conversation

    tables = {t.name: t for t in Base.metadata.sorted_tables}

    # FK indexes on hot paths
    documents_indexes = {idx.name for idx in tables["documents"].indexes}
    assert "ix_documents_user_id" in documents_indexes

    conversations_indexes = {idx.name for idx in tables["conversations"].indexes}
    assert "ix_conversations_user_id" in conversations_indexes
    assert "ix_conversations_document_id" in conversations_indexes

    messages_indexes = {idx.name for idx in tables["messages"].indexes}
    assert "ix_messages_conversation_id" in messages_indexes

    # DocumentChunks search/indexing indexes
    chunks_indexes = {idx.name for idx in tables["document_chunks"].indexes}
    assert "idx_document_chunks_document_id" in chunks_indexes
    assert "idx_document_chunks_document_id_page_number" in chunks_indexes
    assert "idx_document_chunks_embedding_hnsw" in chunks_indexes
    assert "idx_document_chunks_content_fts" in chunks_indexes


def test_migration_adds_indexes():
    """The index migration must exist and reference the expected index names."""
    versions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations", "versions")
    candidates = [f for f in os.listdir(versions_dir) if f.endswith(".py") and "missing" in f]
    assert len(candidates) == 1, f"Expected one 'missing indexes' migration, found {candidates}"

    migration_path = os.path.join(versions_dir, candidates[0])
    with open(migration_path) as f:
        source = f.read()

    for expected in [
        "ix_documents_user_id",
        "ix_conversations_user_id",
        "ix_conversations_document_id",
        "ix_messages_conversation_id",
        "idx_document_chunks_embedding_hnsw",
        "idx_document_chunks_content_fts",
        "idx_document_chunks_document_id",
        "idx_document_chunks_document_id_page_number",
        "CREATE INDEX CONCURRENTLY",
        "autocommit_block",
    ]:
        assert expected in source, f"Migration does not reference {expected}"


def test_document_status_has_server_default():
    """Document.status model must be NOT NULL with a server default."""
    from app.database import Base
    from sqlalchemy import Column

    docs = next(t for t in Base.metadata.sorted_tables if t.name == "documents")
    status_col = docs.columns["status"]
    assert status_col.nullable is False
    assert status_col.server_default is not None
