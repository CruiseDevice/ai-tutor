"""
Standalone Embedding Service
Handles all ML model operations separately from main backend
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import logging
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service")

# Global model instance
model = None
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DIMENSIONS = 768


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 5


class RerankResponse(BaseModel):
    scores: List[float]
    indices: List[int]


@app.on_event("startup")
async def load_model():
    """Load the embedding model on startup"""
    global model
    logger.info(f"Loading embedding model: {MODEL_NAME}")

    # Detect device
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    except:
        device = "cpu"

    logger.info(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    logger.info("Model loaded successfully")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "dimensions": DIMENSIONS,
        "model_loaded": model is not None
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for a list of texts"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        logger.info(f"Generating embeddings for {len(request.texts)} texts")
        embeddings = model.encode(request.texts, convert_to_numpy=True)

        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimensions=DIMENSIONS
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """Rerank documents based on query similarity"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        logger.info(f"Reranking {len(request.documents)} documents")

        # Encode query and documents
        query_embedding = model.encode([request.query], convert_to_numpy=True)[0]
        doc_embeddings = model.encode(request.documents, convert_to_numpy=True)

        # Calculate similarity scores
        from numpy import dot
        from numpy.linalg import norm

        scores = []
        for doc_emb in doc_embeddings:
            score = dot(query_embedding, doc_emb) / (norm(query_embedding) * norm(doc_emb))
            scores.append(float(score))

        # Get top-k indices
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in scored_indices[:request.top_k]]
        top_scores = [score for _, score in scored_indices[:request.top_k]]

        return RerankResponse(
            scores=top_scores,
            indices=top_indices
        )
    except Exception as e:
        logger.error(f"Error reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
