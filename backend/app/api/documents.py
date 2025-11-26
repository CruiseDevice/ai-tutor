from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User
from ..models.document import Document
from ..schemas.document import (
    DocumentResponse,
    DocumentProcessRequest,
    DocumentProcessResponse
)
from ..services.document_service import DocumentService

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


@router.post("/process", response_model=DocumentProcessResponse)
async def process_document(
    request: DocumentProcessRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Process a document: extract text, chunk it, and generate embeddings."""
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

        result = await document_service.process_document(db, request.document_id)

        return DocumentProcessResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
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
    success = document_service.delete_document(db, document_id, user.id)

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

