import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from document_processor import process_document, ProcessDocumentResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:3000",  # Add this
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.post("/process-document", response_model=ProcessDocumentResponse)
async def process_document_endpoint(
    file: UploadFile = File(...),
    document_id: str = Form(...)
):
    try:
        # First verify we can read the file
        await file.seek(0)  # Rewind for actual processing

        # Now process the document
        result = await process_document(file, document_id)
        return result

    except Exception as e:      
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing document: {str(e)}"
        )
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)