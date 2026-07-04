from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import uuid
from .config import settings
from .api import auth, documents, chat, conversations, user, config, admin
from .core.rate_limiting import setup_rate_limiting
from .core.exceptions import StudyFetchError
from .database import app_lifespan

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
    version="1.0.0",
    lifespan=app_lifespan,
)

# Setup rate limiting (Redis-backed slowapi)
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


@app.exception_handler(StudyFetchError)
async def study_fetch_error_handler(request: Request, exc: StudyFetchError):
    """Map domain exceptions to safe JSON responses."""
    request_id = str(uuid.uuid4())
    logger.warning(
        f"Domain error ({request_id}): {exc.__class__.__name__}: {exc.detail}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.default_detail, "request_id": request_id},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all: log full traceback server-side, return generic client message."""
    request_id = str(uuid.uuid4())
    logger.exception(
        f"Unhandled exception ({request_id}): {exc}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


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
