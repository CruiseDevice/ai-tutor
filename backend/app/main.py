from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from .config import settings
from .api import auth, documents, chat, conversations, user, config, admin
from .services.embedding_service import EmbeddingService
from .core.rate_limiting import setup_rate_limiting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize logger
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="StudyFetch AI Tutor Backend",
    description="Backend API for StudyFetch AI Tutor",
    version="1.0.0"
)

# Setup rate limiting
setup_rate_limiting(app)

# Configure CORS
# Use specific origins from settings (allows credentials)
# For development, include common localhost variations
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup.

    Database schema is managed by Alembic and applied before the app starts (see
    the `alembic upgrade head` entrypoint in docker-compose.yml). Nothing to do
    here for the database.
    """
    logger.info("Starting application initialization...")

    # Initialize embedding service lazily (loads the model on first use)
    # This prevents blocking startup while downloading the model
    logger.info("Embedding service will be initialized on first use (lazy loading)")

    # Initialize cache service
    try:
        logger.info("Initializing cache service...")
        from .services.cache_service import get_cache_service
        await get_cache_service()  # This will connect to Redis
        logger.info("Cache service initialized successfully")
    except Exception as e:
        logger.warning(f"Cache service initialization error: {e}. Caching will be disabled.")
        # Don't crash - app can function without cache (just slower)

    logger.info("Application initialization complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    try:
        from .services.cache_service import close_cache_service
        await close_cache_service()
        logger.info("Cache service disconnected")
    except Exception as e:
        logger.warning(f"Error disconnecting cache service: {e}")


# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(user.router)
app.include_router(config.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "StudyFetch AI Tutor Backend API", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

