"""
Database migration utilities.
Handles schema migrations for existing databases.
"""
from sqlalchemy import text, inspect
from sqlalchemy.exc import ProgrammingError
import logging

logger = logging.getLogger(__name__)


def add_title_column_if_missing(engine):
    """
    Add the 'title' column to the 'conversations' table if it doesn't exist.
    This is a migration for existing databases that were created before the title column was added.
    """
    try:
        inspector = inspect(engine)

        # Check if conversations table exists
        table_names = inspector.get_table_names()
        if 'conversations' not in table_names:
            logger.info("conversations table does not exist, will be created by create_all()")
            return

        # Check if title column exists
        columns = [col['name'] for col in inspector.get_columns('conversations')]

        if 'title' not in columns:
            logger.info("Adding 'title' column to 'conversations' table...")
            with engine.begin() as conn:
                # Add the title column (nullable String)
                conn.execute(text("ALTER TABLE conversations ADD COLUMN title VARCHAR"))
                logger.info("Successfully added 'title' column to 'conversations' table")
        else:
            logger.debug("'title' column already exists in 'conversations' table")
    except Exception as e:
        # Log error but don't crash the backend - migration failures shouldn't prevent startup
        logger.error(f"Migration error (non-fatal): {e}", exc_info=True)
        logger.warning("Backend will continue to start, but some features may not work until migration is fixed")


def remove_unique_constraint_from_document_id(engine):
    """
    Remove the unique constraint from 'document_id' column in 'conversations' table if it exists.
    This allows multiple conversations per document.
    """
    try:
        inspector = inspect(engine)

        # Check if conversations table exists
        table_names = inspector.get_table_names()
        if 'conversations' not in table_names:
            logger.info("conversations table does not exist, will be created by create_all()")
            return

        with engine.begin() as conn:
            # Try multiple approaches to remove the unique constraint
            # PostgreSQL auto-generates constraint names, so we try common patterns

            constraint_names_to_try = [
                'conversations_document_id_key',  # PostgreSQL auto-generated name
                'uq_conversations_document_id',   # SQLAlchemy naming convention
                'conversations_document_id_unique',
            ]

            # Also try to find constraint dynamically
            try:
                unique_constraints = inspector.get_unique_constraints('conversations')
                for constraint in unique_constraints:
                    if 'document_id' in constraint['column_names']:
                        constraint_names_to_try.insert(0, constraint['name'])
                        break
            except Exception as e:
                logger.debug(f"Could not inspect unique constraints: {e}")

            # Try dropping each possible constraint name
            for constraint_name in constraint_names_to_try:
                try:
                    conn.execute(text(f"ALTER TABLE conversations DROP CONSTRAINT IF EXISTS {constraint_name}"))
                    logger.info(f"Attempted to drop constraint '{constraint_name}'")
                except Exception as e:
                    logger.debug(f"Could not drop constraint '{constraint_name}': {e}")

            # Also try dropping any unique index on document_id (PostgreSQL creates these for unique constraints)
            try:
                conn.execute(text("DROP INDEX IF EXISTS conversations_document_id_key"))
                logger.info("Attempted to drop index 'conversations_document_id_key'")
            except Exception as e:
                logger.debug(f"Could not drop index: {e}")

            logger.info("Unique constraint removal migration completed")

    except Exception as e:
        # Log error but don't crash the backend - migration failures shouldn't prevent startup
        logger.error(f"Migration error (non-fatal): {e}", exc_info=True)
        logger.warning("Backend will continue to start, but some features may not work until migration is fixed")

