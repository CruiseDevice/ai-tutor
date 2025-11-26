from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class DocumentBase(BaseModel):
    title: str


class DocumentCreate(DocumentBase):
    url: str
    blob_path: str


class DocumentResponse(DocumentBase):
    id: str
    user_id: str
    url: str
    blob_path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentProcessRequest(BaseModel):
    document_id: str


class DocumentProcessResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int
    chunks_failed: int


class DocumentProcessQueueResponse(BaseModel):
    """Response when a document processing job is queued."""
    document_id: str
    job_id: str
    status: str
    message: str


class DocumentProcessStatusResponse(BaseModel):
    """Response for checking document processing status."""
    document_id: str
    status: str  # pending, queued, processing, completed, failed
    job_id: Optional[str] = None
    error_message: Optional[str] = None
    chunks_processed: Optional[int] = None
    chunks_failed: Optional[int] = None


class ChunkResponse(BaseModel):
    id: str
    content: str
    page_number: int
    document_id: str

    class Config:
        from_attributes = True

