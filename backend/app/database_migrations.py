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


def add_fulltext_search_index(engine):
    """
    Add GIN (Generalized Inverted Index) index for full-text search on document_chunks.content.

    This enables hybrid search combining semantic (pgvector) and keyword (PostgreSQL FTS) search
    for improved retrieval accuracy, especially for exact matches and technical terms.

    Uses PostgreSQL's native full-text search with:
    - GIN index for fast text search (better for static data)
    - English language configuration for stemming and stop words
    - CONCURRENTLY option to avoid table locks during index creation
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

        idx_name = 'idx_document_chunks_content_fts'
        if idx_name not in existing_indexes:
            logger.info(f"Creating GIN full-text search index '{idx_name}' on document_chunks.content...")
            logger.info("This may take a few minutes for large datasets...")

            with engine.begin() as conn:
                # Create GIN index for full-text search
                # Using to_tsvector with 'english' configuration for proper stemming
                # CONCURRENTLY avoids blocking other operations
                conn.execute(text(
                    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_content_fts "
                    "ON document_chunks USING gin(to_tsvector('english', content))"
                ))
                logger.info(f"Successfully created GIN index '{idx_name}'")
                logger.info("Full-text search is now enabled for hybrid search")
        else:
            logger.debug(f"GIN index '{idx_name}' already exists")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_fulltext_search_index: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but keyword search may not work optimally")
        logger.warning("Hybrid search will fall back to semantic-only search if needed")


def add_user_role_column(engine):
    """
    Add role column to the 'users' table for role-based access control (RBAC).

    Adds:
    - role: User role (user, admin, super_admin) with default 'user'

    This enables admin features like queue monitoring and user management.
    """
    try:
        inspector = inspect(engine)

        # Check if users table exists
        table_names = inspector.get_table_names()
        if 'users' not in table_names:
            logger.info("users table does not exist, will be created by create_all()")
            return

        # Check existing columns
        existing_columns = {col['name'] for col in inspector.get_columns('users')}

        with engine.begin() as conn:
            # Add role column (default: user)
            if 'role' not in existing_columns:
                logger.info("Adding 'role' column to 'users' table...")
                conn.execute(text("""
                    ALTER TABLE users
                    ADD COLUMN role VARCHAR(20) DEFAULT 'user' NOT NULL
                """))
                logger.info("Successfully added 'role' column to 'users' table")
            else:
                logger.debug("'role' column already exists in 'users' table")

        logger.info("User role column migration completed successfully")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_user_role_column: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but admin features may not work")


def add_processing_time_columns(engine):
    """
    Add processing time tracking columns to the 'documents' table.

    Adds:
    - processing_started_at: Timestamp when document processing began
    - processing_completed_at: Timestamp when document processing finished

    These fields enable accurate calculation of average processing times
    for queue monitoring and performance analytics.
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
            # Add processing_started_at column
            if 'processing_started_at' not in existing_columns:
                logger.info("Adding 'processing_started_at' column to 'documents' table...")
                conn.execute(text("""
                    ALTER TABLE documents
                    ADD COLUMN processing_started_at TIMESTAMP WITH TIME ZONE
                """))
                logger.info("Successfully added 'processing_started_at' column")
            else:
                logger.debug("'processing_started_at' column already exists")

            # Add processing_completed_at column
            if 'processing_completed_at' not in existing_columns:
                logger.info("Adding 'processing_completed_at' column to 'documents' table...")
                conn.execute(text("""
                    ALTER TABLE documents
                    ADD COLUMN processing_completed_at TIMESTAMP WITH TIME ZONE
                """))
                logger.info("Successfully added 'processing_completed_at' column")
            else:
                logger.debug("'processing_completed_at' column already exists")

        logger.info("Processing time tracking columns migration completed successfully")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_processing_time_columns: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but processing time tracking may not work")


def add_audit_logs_table(engine):
    """
    Create audit_logs table for tracking admin actions.

    This table stores a complete audit trail of administrative actions
    for compliance, security, and debugging purposes.
    """
    try:
        inspector = inspect(engine)
        table_names = inspector.get_table_names()

        if 'audit_logs' not in table_names:
            logger.info("Creating 'audit_logs' table...")
            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE audit_logs (
                        id VARCHAR PRIMARY KEY,
                        user_id VARCHAR NOT NULL,
                        action VARCHAR NOT NULL,
                        resource_type VARCHAR,
                        resource_id VARCHAR,
                        details TEXT,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """))

                # Create indexes for efficient querying
                conn.execute(text("""
                    CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id)
                """))
                conn.execute(text("""
                    CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp)
                """))
                conn.execute(text("""
                    CREATE INDEX idx_audit_logs_action ON audit_logs(action)
                """))

                logger.info("Successfully created 'audit_logs' table with indexes")
        else:
            logger.debug("'audit_logs' table already exists")

        logger.info("Audit logs table migration completed successfully")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_audit_logs_table: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but audit logging may not work")

def add_chunk_type_column(engine):
    """
    Add chunk_type column to the 'document_chunks' table for multimodal support.

    Adds:
        - chunk_type: Tyoe of chunk (text, image, table, figure) with default 'text'

        This enable storing and differentiating between text chunks and image chunks
        for multimodal document processing with Docling.
    """
    try:
        inspector = inspect(engine)

        # check if document_chunks table exists
        table_names = inspector.get_table_names()
        if 'document_chunks' not in table_names:
            logger.info("document_chunks table does not exist, will be created by create_all()")
            return

        # check existing columns
        existing_columns = {
            col['name']
            for col in inspector.get_columns('document_chunks')
        }

        with engine.begin() as conn:
            # add chunk_type column (default: text)
            if 'chunk_type' not in existing_columns:
                logger.info("Adding 'chunk_type' column to 'document_chunks' table...")
                conn.execute(text("""
                    ALTER TABLE document_chunks
                    ADD COLUMN chunk_type VARCHAR(50) DEFAULT 'text'
                """))

                # Backfill existing rows with 'text
                conn.execute(text("""
                    UPDATE document_chunks
                    SET chunk_type='text'
                    WHERE chunk_type is NULL
                """))

                logger.info("Successfully added 'chunk_type' column")
            else:
                logger.debug("'chunk_type' column already exists in 'document_chunks' table")

        logger.info("Chunk type column migration completed successfully")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_chunk_type_column: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but image chunk support may not work")


def add_hierarchical_chunking_schema(engine):
    """
    Add hierarchical chunking support with parent-child chunk relationships.

    Phase 3: Hierarchical Parent-Child Chunking

    Adds:
        - chunk_level: Column to document_chunks to track chunk hierarchy (flat, parent, child)
        - parent_child_relationships: New table to track parent-child chunk relationships
        - Indexes for efficient parent-child lookups

    This enables precision retrieval (search child chunks) with context preservation
    (return parent chunks) for improved detail capture.
    """
    try:
        inspector = inspect(engine)

        # Check if document_chunks table exists
        table_names = inspector.get_table_names()
        if 'document_chunks' not in table_names:
            logger.info("document_chunks table does not exist, will be created by create_all()")
            return

        # Part 1: Add chunk_level column to document_chunks
        existing_columns = {col['name'] for col in inspector.get_columns('document_chunks')}

        with engine.begin() as conn:
            if 'chunk_level' not in existing_columns:
                logger.info("Adding 'chunk_level' column to 'document_chunks' table...")
                conn.execute(text("""
                    ALTER TABLE document_chunks
                    ADD COLUMN chunk_level VARCHAR(20) DEFAULT 'flat'
                """))

                # Backfill existing rows with 'flat' (default for non-hierarchical chunks)
                conn.execute(text("""
                    UPDATE document_chunks
                    SET chunk_level = 'flat'
                    WHERE chunk_level IS NULL
                """))

                logger.info("Successfully added 'chunk_level' column")
            else:
                logger.debug("'chunk_level' column already exists in 'document_chunks' table")

        # Part 2: Add index on chunk_level for filtering
        with engine.begin() as conn:
            existing_indexes = {idx['name'] for idx in inspector.get_indexes('document_chunks')}
            idx_name = 'idx_document_chunks_level'

            if idx_name not in existing_indexes:
                logger.info(f"Creating index '{idx_name}' on document_chunks.chunk_level...")
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_document_chunks_level
                    ON document_chunks(chunk_level)
                """))
                logger.info(f"Successfully created index '{idx_name}'")
            else:
                logger.debug(f"Index '{idx_name}' already exists")

        # Part 3: Create parent_child_relationships table
        if 'parent_child_relationships' not in table_names:
            logger.info("Creating 'parent_child_relationships' table...")
            with engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE parent_child_relationships (
                        id VARCHAR PRIMARY KEY,
                        parent_chunk_id VARCHAR NOT NULL,
                        child_chunk_id VARCHAR NOT NULL,
                        child_index INTEGER NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(parent_chunk_id, child_chunk_id),
                        FOREIGN KEY (parent_chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE,
                        FOREIGN KEY (child_chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
                    )
                """))
                logger.info("Successfully created 'parent_child_relationships' table")
        else:
            logger.debug("'parent_child_relationships' table already exists")

        # Part 4: Create indexes on parent_child_relationships table
        with engine.begin() as conn:
            existing_indexes = {idx['name'] for idx in inspector.get_indexes('parent_child_relationships')}

            # Index on parent_chunk_id (lookup children by parent)
            idx_parent = 'idx_parent_child_parent'
            if idx_parent not in existing_indexes:
                logger.info(f"Creating index '{idx_parent}' on parent_child_relationships.parent_chunk_id...")
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_parent_child_parent
                    ON parent_child_relationships(parent_chunk_id)
                """))
                logger.info(f"Successfully created index '{idx_parent}'")
            else:
                logger.debug(f"Index '{idx_parent}' already exists")

            # Index on child_chunk_id (lookup parent by child)
            idx_child = 'idx_parent_child_child'
            if idx_child not in existing_indexes:
                logger.info(f"Creating index '{idx_child}' on parent_child_relationships.child_chunk_id...")
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_parent_child_child
                    ON parent_child_relationships(child_chunk_id)
                """))
                logger.info(f"Successfully created index '{idx_child}'")
            else:
                logger.debug(f"Index '{idx_child}' already exists")

        logger.info("Hierarchical chunking schema migration completed successfully")

    except Exception as e:
        # Log error but don't crash the backend
        logger.error(f"Migration error (non-fatal) - add_hierarchical_chunking_schema: {e}", exc_info=True)
        logger.warning("Backend will continue to start, but hierarchical chunking may not work")