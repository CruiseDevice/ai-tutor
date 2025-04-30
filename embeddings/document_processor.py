from fastapi import UploadFile, HTTPException
from typing import List, Dict, Optional
from pydantic import BaseModel
import io
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize the model (same as in embeddings_service.py)
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


class TextRange(BaseModel):
    from_pos: int
    to_pos: int


class DocumentChunk(BaseModel):
    content: str
    page_number: int
    embedding: List[float]
    document_id: str
    text_ranges: Optional[List[Dict[str, int]]] = None


class ProcessDocumentResponse(BaseModel):
    chunks: List[DocumentChunk] = []
    chunks_processed: int
    chunks_failed: int
    message: str
    success: bool


async def process_document(
        file: UploadFile, document_id: str) -> ProcessDocumentResponse:
    """
    Process a PDF document: load it, split it into chunks, and generate
    embeddings
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported")
    
    try:
        # Read file content into memory
        content = await file.read()

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(file_path=temp_file_path)
            documents = loader.load()
        finally:
            import os
            os.remove(temp_file_path)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)

        successful_chunks = 0
        failed_chunks = 0
        document_chunks = []

        # Process each chunk
        for chunk in chunks:
            try:
                # Get the text content
                text_content = chunk.page_content

                # Get metadata
                page_number = 1
                text_ranges = []

                if chunk.metadata and "page" in chunk.metadata:
                    page_number = int(chunk.metadata["page"]) + 1  # 0-indexed to 1-indexed

                # Extract line positions if available
                if "lines" in chunk.metadata:
                    lines = chunk.metadata["lines"]
                    for line in lines:
                        if isinstance(line, dict) and "from" in line and "to" in line:
                            text_ranges.append({
                                "from": int(line["from"]),
                                "to": int(line["to"])
                            })

                # Generate embedding
                embedding = model.encode(text_content).tolist()

                # Create document chunk
                document_chunk = DocumentChunk(
                    content=text_content,
                    page_number=page_number,
                    embedding=embedding,
                    document_id=document_id,
                    text_ranges=text_ranges if text_ranges else None
                )

                document_chunks.append(document_chunk)
                successful_chunks += 1

            except Exception as e:
                print(f"Error processing chunk: {e}")
                failed_chunks += 1

        # Return a properly structured response
        return ProcessDocumentResponse(
            chunks=document_chunks,
            chunks_processed=successful_chunks,
            chunks_failed=failed_chunks,
            message="Document processed successfully",
            success=True
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}")