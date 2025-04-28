import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Initialize the model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


@app.get("/health")
async def health():
    return {"status": "ok"}


class EmbeddingRequest(BaseModel):
    text: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # generate embedding
    embedding = model.encode(request.text)

    # convert numpy array to list for JSON serialization
    embedding_list = embedding.tolist()

    return EmbeddingResponse(
        embedding=embedding_list,
        dimensions=len(embedding_list)
    )


class BatchEmbeddingRequest(BaseModel):
    texts: list[str]


class BatchEmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int


@app.post("/batch-embeddings", response_model=BatchEmbeddingResponse)
async def create_batch_embeddings(request: BatchEmbeddingRequest):
    if not request.texts or any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=400, detail="Texts cannot be empty")
    
    # generate embeddings for all texts
    embeddings = model.encode(request.texts)

    # convert numpy arrays to lists for JSON serialization
    embeddings_list = embeddings.tolist()

    return BatchEmbeddingResponse(
        embeddings=embeddings_list,
        dimensions=len(embeddings_list[0]) if embeddings_list else 0
    )
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)