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

