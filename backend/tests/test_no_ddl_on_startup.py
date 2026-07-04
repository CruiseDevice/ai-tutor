"""Tests for Phase 2.2 — no DDL runs on application startup."""
import pytest
import ast
import os
import sys


def _load_main_ast():
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "main.py")
    with open(main_path) as f:
        return ast.parse(f.read())


def test_main_does_not_import_database_migrations():
    """The legacy hand-rolled migration module must not be imported by main.py."""
    tree = _load_main_ast()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = getattr(node, "module", "")
            names = {alias.name for alias in node.names}
            assert not (
                module and "database_migrations" in module
            ), f"main.py imports database_migrations via {module}"
            assert "database_migrations" not in names, "main.py imports database_migrations"


def test_main_does_not_call_create_all():
    """Base.metadata.create_all must not be invoked on startup."""
    tree = _load_main_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "create_all":
                assert False, "main.py calls Base.metadata.create_all() on startup"


def test_main_does_not_use_on_event_for_startup():
    """Legacy @app.on_event('startup') decorators must be gone."""
    tree = _load_main_ast()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "on_event":
                assert False, "main.py still uses deprecated @app.on_event"


@pytest.mark.asyncio
async def test_lifespan_does_not_run_create_all(monkeypatch):
    """The lifespan context manager must not call Base.metadata.create_all."""
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "main.py")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    import app.database as database_module
    from app.database import app_lifespan

    # Replace the real Base.metadata.create_all with a spy
    called = {"create_all": False}

    def fake_create_all(*args, **kwargs):
        called["create_all"] = True

    original_create_all = database_module.Base.metadata.create_all
    database_module.Base.metadata.create_all = fake_create_all

    # Also spy on database_migrations functions if they were somehow reachable
    from unittest.mock import MagicMock
    fake_migrations = MagicMock()
    monkeypatch.setitem(sys.modules, "app.database_migrations", fake_migrations)

    try:
        async with app_lifespan(MagicMock()):
            pass
    finally:
        database_module.Base.metadata.create_all = original_create_all

    assert not called["create_all"], "app_lifespan called Base.metadata.create_all()"


def test_lifespan_source_does_not_mention_create_all_or_migrations():
    """Read the lifespan source and verify no actual DDL/migration calls exist."""
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "database.py")
    with open(db_path) as f:
        source = f.read()

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "create_all":
                assert False, "database.py calls Base.metadata.create_all()"

    # SQL DDL strings should not be emitted by the lifespan code.
    assert "CREATE EXTENSION" not in source
    assert "CREATE INDEX" not in source
    # Legacy migration module should not be imported.
    assert "database_migrations" not in source
