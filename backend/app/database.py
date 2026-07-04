from contextlib import asynccontextmanager
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# Create SQLAlchemy engine with optimized connection pooling
engine = create_engine(
    settings.DATABASE_URL,
    # Connection pool settings for better performance and concurrency
    pool_size=10,              # Number of persistent connections (increased from 5)
    max_overflow=20,           # Additional connections during high load (increased from 10)
    pool_pre_ping=True,        # Verify connections before use (prevents stale connections)
    pool_recycle=3600,         # Recycle connections after 1 hour (prevents long-lived connection issues)
    pool_timeout=30,           # Wait up to 30 seconds for available connection
    # Performance tuning
    echo=False,                # Disable SQL logging in production (set via env for debugging)
    echo_pool=False,           # Disable connection pool logging (set via env for debugging)
    future=True,               # Enable SQLAlchemy 2.0 behavior
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for models
Base = declarative_base()


# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def app_lifespan(app):
    """Modern FastAPI lifespan context manager.

    Handles startup/shutdown without performing any DDL. Schema changes are
    the responsibility of Alembic (see backend/scripts/run_migrations.sh).
    """
    logger = __import__("logging").getLogger(__name__)
    logger.info("Starting application initialization...")

    # NOTE: No migrations or Base.metadata.create_all() here. The app assumes
    # the database schema is managed externally via Alembic. For local dev,
    # run `alembic upgrade head` or `./scripts/run_migrations.sh` before
    # starting the app.

    # Initialize cache service
    try:
        logger.info("Initializing cache service...")
        from .services.cache_service import get_cache_service
        await get_cache_service()
        logger.info("Cache service initialized successfully")
    except Exception as e:
        logger.warning(f"Cache service initialization error: {e}. Caching will be disabled.")

    logger.info("Application initialization complete")

    yield

    # Shutdown
    logger.info("Shutting down application...")
    try:
        from .services.cache_service import close_cache_service
        await close_cache_service()
        logger.info("Cache service disconnected")
    except Exception as e:
        logger.warning(f"Error disconnecting cache service: {e}")
    logger.info("Application shutdown complete")
