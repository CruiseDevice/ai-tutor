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

