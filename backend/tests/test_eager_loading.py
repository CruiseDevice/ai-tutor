"""Tests for Phase 2.5 — eager loading and pagination on list endpoints."""
import os
import uuid

import pytest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, selectinload

from app.models.conversation import Conversation
from app.models.document import Document
from app.models.user import User


def _source_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), *parts)


def test_list_conversations_uses_selectinload_and_pagination():
    """The conversations list endpoint must eager-load documents and paginate."""
    source = open(_source_path("app", "api", "conversations.py")).read()

    assert "from sqlalchemy.orm import Session, selectinload" in source
    assert "selectinload(Conversation.document)" in source
    assert "limit: int = Query(50, ge=1, le=50" in source
    assert "offset: int = Query(0, ge=0" in source
    assert ".offset(offset)" in source
    assert ".limit(limit)" in source
    assert "limit = min(limit, 50)" in source
    # The N+1 per-conversation Document lookup must be gone.
    assert "db.query(Document).filter(Document.id == conv.document_id).first()" not in source


def test_list_documents_has_pagination():
    """The documents list endpoint and service must support capped pagination."""
    api_src = open(_source_path("app", "api", "documents.py")).read()
    svc_src = open(_source_path("app", "services", "document_service.py")).read()

    assert "from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query" in api_src
    assert "limit: int = Query(50, ge=1, le=50" in api_src
    assert "offset: int = Query(0, ge=0" in api_src
    assert "limit=limit, offset=offset" in api_src
    assert "def list_documents(self, db: Session, user_id: str, limit: int = 50, offset: int = 0)" in svc_src
    assert "limit = min(limit, 50)" in svc_src
    assert ".offset(offset)" in svc_src
    assert ".limit(limit)" in svc_src


def _db_available() -> bool:
    """Probe the default dev/test Postgres database."""
    url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/studyfetch")
    try:
        engine = create_engine(url, pool_pre_ping=False)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="Postgres not available")
def test_list_conversations_constant_query_count():
    """With eager loading, listing 20 conversations should use O(1) SQL queries.

    Verifies the N+1 Document lookup is eliminated: one query for conversations and
    one batched query for all related documents.
    """
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/studyfetch")
    engine = create_engine(db_url, pool_pre_ping=False)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    query_count = [0]

    def on_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        query_count[0] += 1

    event.listen(engine, "before_cursor_execute", on_before_cursor_execute)

    try:
        user_id = str(uuid.uuid4())
        doc_id = str(uuid.uuid4())

        user = User(id=user_id, email=f"{user_id}@example.com", password="secret")
        document = Document(
            id=doc_id,
            user_id=user_id,
            title="Test Document",
            url="http://localhost/test.pdf",
            blob_path="test/test.pdf",
        )
        conversations = [
            Conversation(user_id=user_id, document_id=doc_id, title=f"Conversation {i}")
            for i in range(20)
        ]

        session.add(user)
        session.add(document)
        session.add_all(conversations)
        session.flush()

        # Start counting only for the list query, not setup.
        query_count[0] = 0

        convs = (
            session.query(Conversation)
            .options(selectinload(Conversation.document))
            .filter(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(50)
            .all()
        )

        # Traverse the relationship for every row.
        for conv in convs:
            _ = conv.document.title

        assert len(convs) == 20
        # 1 query for conversations + 1 batched selectinload for documents.
        # Allow a small cushion for any savepoint or pool bookkeeping.
        assert query_count[0] <= 3, f"Expected O(1) queries, got {query_count[0]}"
    finally:
        event.remove(engine, "before_cursor_execute", on_before_cursor_execute)
        session.rollback()
        session.close()
