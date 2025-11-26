"""
Arq worker configuration for background job processing.
"""
import logging
from arq import create_pool
from arq.connections import RedisSettings
from app.config import settings

logger = logging.getLogger(__name__)

# Import job functions - this must be done at module level
# We'll import the actual function after defining the settings class
from app.workers import document_jobs


def parse_redis_url(redis_url: str) -> dict:
    """
    Parse Redis URL into components.

    Args:
        redis_url: Redis URL (e.g., redis://localhost:6379/0)

    Returns:
        Dict with host, port, and database
    """
    # Remove redis:// prefix
    url = redis_url.replace("redis://", "")

    # Split by / to get host:port and database
    parts = url.split("/")
    host_port = parts[0]
    database = int(parts[1]) if len(parts) > 1 else 0

    # Split host and port
    host_port_parts = host_port.split(":")
    host = host_port_parts[0]
    port = int(host_port_parts[1]) if len(host_port_parts) > 1 else 6379

    return {
        "host": host,
        "port": port,
        "database": database
    }


# Parse Redis URL from settings
redis_config = parse_redis_url(settings.REDIS_URL)

# Arq Redis settings
ARQ_REDIS_SETTINGS = RedisSettings(
    host=redis_config["host"],
    port=redis_config["port"],
    database=redis_config["database"],
)


# Worker settings class
class WorkerSettings:
    """Arq worker settings."""

    redis_settings = ARQ_REDIS_SETTINGS

    # Worker configuration
    max_jobs = 4  # Process up to 4 documents concurrently
    job_timeout = 600  # 10 minutes timeout per job
    keep_result = 3600  # Keep job results for 1 hour

    # Health check
    health_check_interval = 60  # Check health every 60 seconds

    # Job functions - register the process_document_job
    functions = [document_jobs.process_document_job]


async def startup(ctx):
    """
    Worker startup - initialize services.

    Args:
        ctx: Worker context for storing shared resources
    """
    from app.database import SessionLocal
    from app.services.embedding_service import get_embedding_service

    logger.info("Initializing worker services...")

    # Store shared resources in worker context
    ctx["db_session_factory"] = SessionLocal
    ctx["embedding_service"] = get_embedding_service()

    logger.info("Worker services initialized successfully")


async def shutdown(ctx):
    """
    Worker shutdown - cleanup.

    Args:
        ctx: Worker context
    """
    logger.info("Shutting down worker...")
    # Cleanup if needed
    logger.info("Worker shutdown complete")


# Attach lifecycle hooks
WorkerSettings.on_startup = startup
WorkerSettings.on_shutdown = shutdown
