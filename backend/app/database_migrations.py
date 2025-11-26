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

        # Get all unique constraints on the conversations table
        unique_constraints = inspector.get_unique_constraints('conversations')

        # Find the unique constraint on document_id
        document_id_unique_constraint = None
        for constraint in unique_constraints:
            if 'document_id' in constraint['column_names']:
                document_id_unique_constraint = constraint
                break

        if document_id_unique_constraint:
            constraint_name = document_id_unique_constraint['name']
            logger.info(f"Removing unique constraint '{constraint_name}' from 'conversations.document_id'...")
            with engine.begin() as conn:
                # Drop the unique constraint
                # PostgreSQL uses DROP CONSTRAINT, SQLite uses different syntax
                # Try PostgreSQL syntax first (most common)
                try:
                    conn.execute(text(f"ALTER TABLE conversations DROP CONSTRAINT IF EXISTS {constraint_name}"))
                    logger.info(f"Successfully removed unique constraint '{constraint_name}' from 'conversations.document_id'")
                except ProgrammingError:
                    # If PostgreSQL syntax fails, try SQLite syntax
                    try:
                        # SQLite doesn't support DROP CONSTRAINT directly, need to recreate table
                        # But this is complex, so we'll just log a warning
                        logger.warning("SQLite detected - unique constraint removal may require manual intervention")
                    except Exception as e:
                        logger.error(f"Failed to remove unique constraint: {e}")
        else:
            logger.debug("No unique constraint found on 'conversations.document_id' - already migrated or never existed")
    except Exception as e:
        # Log error but don't crash the backend - migration failures shouldn't prevent startup
        logger.error(f"Migration error (non-fatal): {e}", exc_info=True)
        logger.warning("Backend will continue to start, but some features may not work until migration is fixed")

