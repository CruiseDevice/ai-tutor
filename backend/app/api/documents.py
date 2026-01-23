from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List
from arq import create_pool
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User
from ..models.document import Document, DocumentChunk
from ..schemas.document import (
    DocumentResponse,
    DocumentProcessRequest,
    DocumentProcessResponse,
    DocumentProcessQueueResponse,
    DocumentProcessStatusResponse
)
from ..services.document_service import DocumentService
from ..workers.arq_config import ARQ_REDIS_SETTINGS
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Initialize document service
document_service = DocumentService()


@router.post("", response_model=dict)
async def upload_document(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a PDF document."""
    try:
        document, conversation = await document_service.create_document(db, user.id, file)

        # Generate signed URL for immediate access
        signed_url = document_service.get_signed_url(document.blob_path)

        return {
            "id": document.id,
            "title": document.title,
            "url": signed_url,
            "conversationId": conversation.id,
            "createdAt": document.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload document: {str(e)}"
        )


@router.post("/process", response_model=DocumentProcessQueueResponse)
async def process_document(
    request: DocumentProcessRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Queue document processing as a background job.
    Returns immediately with job ID for tracking.
    """
    try:
        # Verify document belongs to user
        document = db.query(Document).filter(
            Document.id == request.document_id,
            Document.user_id == user.id
        ).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Create Arq connection pool
        redis = await create_pool(ARQ_REDIS_SETTINGS)

        # Queue the background job
        job = await redis.enqueue_job(
            "process_document_job",  # Job function name
            request.document_id,     # document_id argument
        )

        # Update document with job ID and status
        document.job_id = job.job_id
        document.status = "queued"
        document.error_message = None
        db.commit()

        return DocumentProcessQueueResponse(
            document_id=request.document_id,
            job_id=job.job_id,
            status="queued",
            message="Document processing queued successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to queue document processing: {str(e)}"
        )


@router.get("/process/{document_id}/status", response_model=DocumentProcessStatusResponse)
async def get_processing_status(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get the current processing status of a document.
    """
    try:
        # Verify document belongs to user
        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user.id
        ).first()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        response = DocumentProcessStatusResponse(
            document_id=document_id,
            status=document.status,
            job_id=document.job_id,
            error_message=document.error_message
        )

        # If completed, include chunk counts
        if document.status == "completed":
            chunk_count = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).count()
            response.chunks_processed = chunk_count
            response.chunks_failed = 0

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing status: {str(e)}"
        )


@router.get("", response_model=List[DocumentResponse])
async def list_documents(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all documents for the current user."""
    documents = document_service.list_documents(db, user.id)
    return [DocumentResponse.model_validate(doc) for doc in documents]


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a document and all associated data."""
    success = await document_service.delete_document(db, document_id, user.id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    return {"message": "Document deleted successfully"}


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific document with signed URL."""
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Generate fresh signed URL
    signed_url = document_service.get_signed_url(document.blob_path)

    return {
        "id": document.id,
        "title": document.title,
        "url": signed_url,
        "createdAt": document.created_at.isoformat(),
        "updatedAt": document.updated_at.isoformat()
    }


@router.get("/{document_id}/pdf")
async def get_document_pdf(
    document_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Proxy endpoint to fetch PDF from S3 and stream it to the client.

    This bypasses CORS issues by having the backend fetch from S3 using
    its AWS credentials (not subject to browser CORS) and serving to
    the frontend with proper CORS headers.
    """
    # Verify document belongs to user
    document = db.query(Document).filter(
        Document.id == document_id,
        Document.user_id == user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    try:
        # Fetch PDF from S3 using boto3 (bypasses browser CORS)
        response = document_service.s3_client.get_object(
            Bucket=document_service.bucket_name,
            Key=document.blob_path
        )

        # Stream the PDF content to the client
        def iterfile():
            """Stream file in chunks to avoid loading entire file in memory."""
            chunk_size = 8192  # 8KB chunks
            while chunk := response['Body'].read(chunk_size):
                yield chunk

        logger.info(f"Serving PDF proxy for document {document_id} to user {user.id}")

        return StreamingResponse(
            iterfile(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{document.title}"',
                "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
            }
        )

    except document_service.s3_client.exceptions.NoSuchKey:
        logger.error(f"PDF not found in S3: {document.blob_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found in storage"
        )
    except Exception as e:
        logger.error(f"Error fetching PDF from S3: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch PDF: {str(e)}"
        )

