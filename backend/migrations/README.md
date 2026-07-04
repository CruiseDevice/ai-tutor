# Alembic Migrations

This directory contains Alembic migrations for the StudyFetch backend.

## Workflow

1. **Generate a migration** after changing SQLAlchemy models:
   ```bash
   cd backend
   alembic revision --autogenerate -m "description"
   ```

2. **Review the generated migration** before committing. Autogenerate is a
   best-effort helper — always inspect the upgrade/downgrade functions.

3. **Apply migrations**:
   ```bash
   alembic upgrade head
   ```

4. **Downgrade (careful in production)**:
   ```bash
   alembic downgrade -1
   ```

## Baseline

The `baseline` migration is intentionally a no-op. It represents the schema state
after the legacy `app/database_migrations.py` hand-rolled migration layer. The
existing development database was stamped with this revision so that future
migrations apply cleanly from this point.

## Important patterns

### `CREATE INDEX CONCURRENTLY`

Postgres cannot create indexes concurrently inside a transaction. For index
migrations that must use `CONCURRENTLY`, wrap the operation in an autocommit
execution context:

```python
from alembic import op
from sqlalchemy import text

def upgrade():
    with op.get_context().autocommit_block():
        op.execute(text(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_name ON table(column)"
        ))
```

### Model/schema drift

The baseline intentionally does not reconcile drift between the SQLAlchemy
models and the existing schema (e.g. indexes created by the legacy migration
layer that are not reflected in the models). Such drift is addressed in
follow-up migrations per `BACKEND_QUALITY_PLAN.md` Phase 2.3 and 2.4.
