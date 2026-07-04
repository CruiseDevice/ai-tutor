"""Tests for Phase 2.3 — schema drift reconciliation."""
import os


def test_conversations_model_uses_snake_case_user_id():
    """Conversation.user_id must be snake_case in the model source."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "models", "conversation.py")
    with open(model_path) as f:
        source = f.read()

    assert "user_id = Column" in source
    assert "ForeignKey(\"users.id\"" in source
    assert 'name="userId"' not in source


def test_users_role_model_uses_string_not_enum():
    """User.role should be a String column with a CHECK constraint, not Postgres ENUM."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "models", "user.py")
    with open(model_path) as f:
        source = f.read()

    assert "role = Column(\n        String," in source
    assert "CheckConstraint" in source
    assert "ck_users_role_allowed" in source


def test_document_chunk_columns_are_not_nullable_in_model():
    """DocumentChunk.chunk_type and chunk_level must be NOT NULL with server defaults."""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "models", "document.py")
    with open(model_path) as f:
        source = f.read()

    assert "chunk_type = Column(String, nullable=False, default='text', server_default='text', index=True)" in source
    assert "chunk_level = Column(String, nullable=False, default='flat', server_default='flat', index=True)" in source


def test_reconcile_migration_exists_and_targets_drift():
    """A reconciliation migration must exist and address the drift items."""
    versions_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations", "versions")
    candidates = [f for f in os.listdir(versions_dir) if f.endswith(".py") and "reconcile" in f]
    assert len(candidates) == 1, f"Expected one reconcile migration, found {candidates}"

    migration_path = os.path.join(versions_dir, candidates[0])
    with open(migration_path) as f:
        source = f.read()

    assert "conversations_user_id_fkey" in source
    assert "ck_users_role_allowed" in source
    assert "chunk_type" in source
    assert "chunk_level" in source
