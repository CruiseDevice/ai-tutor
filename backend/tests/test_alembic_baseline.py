"""Tests for Phase 2.1 — Alembic baseline initialization."""
import pytest
import os
import sys

# Ensure backend is on path for Alembic env.py imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_alembic_ini_exists():
    """Alembic configuration file must exist."""
    ini_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini")
    assert os.path.isfile(ini_path)


def test_migrations_directory_exists():
    """Migrations directory and env.py must exist."""
    migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations")
    assert os.path.isdir(migrations_dir)
    assert os.path.isfile(os.path.join(migrations_dir, "env.py"))


def test_baseline_migration_is_no_op():
    """The baseline migration must perform no DDL."""
    migrations_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations", "versions")
    baseline_files = [f for f in os.listdir(migrations_dir) if f.endswith(".py") and "baseline" in f]
    assert len(baseline_files) == 1, f"Expected exactly one baseline migration, found {baseline_files}"

    baseline_path = os.path.join(migrations_dir, baseline_files[0])
    with open(baseline_path) as f:
        source = f.read()

    # Should not contain op.create_index, op.drop_index, op.create_table, etc.
    forbidden = ["op.create_index", "op.drop_index", "op.create_table", "op.drop_table", "op.add_column"]
    for token in forbidden:
        assert token not in source, f"Baseline migration contains DDL: {token}"


def test_env_py_imports_app_settings_and_metadata():
    """env.py must wire DATABASE_URL and Base.metadata for autogenerate."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "migrations", "env.py")
    with open(env_path) as f:
        source = f.read()

    assert "from app.config import settings" in source
    assert "from app.database import Base" in source
    assert "target_metadata = Base.metadata" in source
    assert "def get_url()" in source


@pytest.mark.skipif(
    os.environ.get("SKIP_ALEMBIC_DB_TEST") == "1",
    reason="Database-dependent Alembic test skipped via env var"
)
def test_alembic_current_matches_head():
    """If a database is available, it should be stamped at the baseline head."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from app.config import settings
    from app.database import engine

    if not settings.DATABASE_URL:
        pytest.skip("DATABASE_URL not set")

    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        pytest.skip(f"Database not reachable: {e}")

    config = Config(os.path.join(os.path.dirname(os.path.dirname(__file__)), "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    head = script.get_current_head()

    from alembic.migration import MigrationContext
    with engine.connect() as conn:
        context = MigrationContext.configure(conn)
        current_rev = context.get_current_revision()

    assert current_rev == head, f"DB is at {current_rev}, expected head {head}"
