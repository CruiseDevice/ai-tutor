from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .api import auth, documents, chat, conversations, user
from .services.embedding_service import EmbeddingService

# Create FastAPI app
app = FastAPI(
    title="StudyFetch AI Tutor Backend",
    description="Backend API for StudyFetch AI Tutor",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    # Create database tables
    from .database import engine, Base
    # Import models to register them with SQLAlchemy Base
    from .models import user, document, conversation  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Initialize embedding service (loads the model once)
    from .services import embedding_service as emb_module
    if emb_module.embedding_service is None:
        emb_module.embedding_service = EmbeddingService()


# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(user.router)


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

