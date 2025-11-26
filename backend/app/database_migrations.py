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


def add_document_chunks_indexes(engine):
    """
    Add performance indexes to the 'document_chunks' table:
    - Index on document_id for faster filtering
    - Composite index on (document_id, page_number) for common queries

    These indexes improve query performance when searching chunks by document.
    """
    try:
        inspector = inspect(engine)

        # Check if document_chunks table exists
        table_names = inspector.get_table_names()
        if 'document_chunks' not in table_names:
            logger.info("document_chunks table does not exist, will be created by create_all()")
            return

        # Get existing indexes
        existing_indexes = {idx['name'] for idx in inspector.get_indexes('document_chunks')}

        with engine.begin() as conn:
            # Index 1: document_id (for filtering by document)
            idx_name_document = 'idx_document_chunks_document_id'
            if idx_name_document not in existing_indexes:
                logger.info(f"Creating index '{idx_name_document}' on document_chunks.document_id...")
                conn.execute(text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id "
                    "ON document_chunks (document_id)"
                ))
                logger.info(f"Successfully created index '{idx_name_document}'")
            else:
                logger.debug(f"Index '{idx_name_document}' already exists")

            # Index 2: Composite (document_id, page_number) for ordered page queries
            idx_name_composite = 'idx_document_chunks_document_id_page_number'
            if idx_name_composite not in existing_indexes:
                logger.info(f"Creating composite index '{idx_name_composite}' on (document_id, page_number)...")
                conn.execute(text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_document_id_page_number "
                    "ON document_chunks (document_id, page_number)"
                ))
                logger.info(f"Successfully created composite index '{idx_name_composite}'")
            else:
                logger.debug(f"Index '{idx_name_composite}' already exists")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_document_chunks_indexes: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but query performance may be suboptimal")


def add_pgvector_hnsw_index(engine):
    """
    Add HNSW (Hierarchical Navigable Small World) index to the embedding column
    in document_chunks table for fast approximate nearest neighbor search.

    HNSW parameters:
    - m=16: Number of connections per layer (good balance of speed vs recall)
    - ef_construction=64: Size of dynamic candidate list (higher = better index quality)

    Uses cosine distance for semantic similarity (best for normalized embeddings).
    """
    try:
        inspector = inspect(engine)

        # Check if document_chunks table exists
        table_names = inspector.get_table_names()
        if 'document_chunks' not in table_names:
            logger.info("document_chunks table does not exist, will be created by create_all()")
            return

        # Get existing indexes
        existing_indexes = {idx['name'] for idx in inspector.get_indexes('document_chunks')}

        idx_name = 'idx_document_chunks_embedding_hnsw'
        if idx_name not in existing_indexes:
            logger.info(f"Creating HNSW index '{idx_name}' on document_chunks.embedding...")
            logger.info("This may take a few minutes for large datasets...")

            with engine.begin() as conn:
                # Create HNSW index with cosine distance operator
                # Using CONCURRENTLY to avoid table locks during creation
                conn.execute(text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_hnsw "
                    "ON document_chunks USING hnsw (embedding vector_cosine_ops) "
                    "WITH (m = 16, ef_construction = 64)"
                ))
                logger.info(f"Successfully created HNSW index '{idx_name}'")
                logger.info("Vector searches will now use approximate nearest neighbor (ANN) search")
        else:
            logger.debug(f"HNSW index '{idx_name}' already exists")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_pgvector_hnsw_index: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but vector search performance may be suboptimal")
        logger.warning("If pgvector extension is not installed, run: CREATE EXTENSION vector;")


def add_document_status_fields(engine):
    """
    Add status tracking fields to the 'documents' table for background job processing:
    - status: Track processing state (pending, queued, processing, completed, failed)
    - error_message: Store error details if processing fails
    - job_id: Store Arq job ID for tracking background jobs

    These fields enable non-blocking document processing with proper status tracking.
    """
    try:
        inspector = inspect(engine)

        # Check if documents table exists
        table_names = inspector.get_table_names()
        if 'documents' not in table_names:
            logger.info("documents table does not exist, will be created by create_all()")
            return

        # Check existing columns
        existing_columns = {col['name'] for col in inspector.get_columns('documents')}

        with engine.begin() as conn:
            # Add status column (default: pending)
            if 'status' not in existing_columns:
                logger.info("Adding 'status' column to 'documents' table...")
                conn.execute(text("""
                    ALTER TABLE documents
                    ADD COLUMN status VARCHAR(50) DEFAULT 'pending' NOT NULL
                """))
                logger.info("Successfully added 'status' column")
            else:
                logger.debug("'status' column already exists in 'documents' table")

            # Add error_message column
            if 'error_message' not in existing_columns:
                logger.info("Adding 'error_message' column to 'documents' table...")
                conn.execute(text("""
                    ALTER TABLE documents
                    ADD COLUMN error_message TEXT
                """))
                logger.info("Successfully added 'error_message' column")
            else:
                logger.debug("'error_message' column already exists in 'documents' table")

            # Add job_id column
            if 'job_id' not in existing_columns:
                logger.info("Adding 'job_id' column to 'documents' table...")
                conn.execute(text("""
                    ALTER TABLE documents
                    ADD COLUMN job_id VARCHAR(255)
                """))
                logger.info("Successfully added 'job_id' column")
            else:
                logger.debug("'job_id' column already exists in 'documents' table")

        # Add index on status column for filtering (separate transaction)
        with engine.begin() as conn:
            existing_indexes = {idx['name'] for idx in inspector.get_indexes('documents')}
            idx_name = 'idx_documents_status'

            if idx_name not in existing_indexes:
                logger.info(f"Creating index '{idx_name}' on documents.status...")
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status)
                """))
                logger.info(f"Successfully created index '{idx_name}'")
            else:
                logger.debug(f"Index '{idx_name}' already exists")

        logger.info("Document status tracking fields migration completed successfully")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_document_status_fields: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but background job tracking may not work")

