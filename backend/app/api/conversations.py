from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime, timezone
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User
from ..models.conversation import Conversation, Message
from ..models.document import Document
from ..schemas.chat import ConversationResponse, ConversationWithMessages, MessageResponse
from ..services.document_service import DocumentService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])

# Initialize document service for signed URLs
document_service = DocumentService()


class ConversationCreate(BaseModel):
    document_id: str


@router.post("", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new conversation for an existing document. Multiple conversations per document are allowed."""
    # Verify document belongs to user
    document = db.query(Document).filter(
        Document.id == conversation_data.document_id,
        Document.user_id == user.id
    ).first()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Create new conversation (multiple conversations per document are allowed)
    conversation = Conversation(
        user_id=user.id,
        document_id=document.id,
        title=None,  # Will be generated from first message
        updated_at=datetime.now(timezone.utc)
    )
    db.add(conversation)

    try:
        db.commit()
        db.refresh(conversation)
    except IntegrityError as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )

    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        document_id=conversation.document_id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at
    )


@router.get("")
async def list_conversations(
    group_by_document: bool = Query(False, description="Group conversations by document"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all conversations for the current user, optionally grouped by document."""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id
    ).order_by(Conversation.updated_at.desc()).all()

    if group_by_document:
        # Group conversations by document
        document_map: Dict[str, Dict[str, Any]] = {}

        for conv in conversations:
            doc_id = conv.document_id

            if doc_id not in document_map:
                # Get document info
                document = db.query(Document).filter(Document.id == doc_id).first()
                signed_url = document_service.get_signed_url(document.blob_path) if document else None
                document_map[doc_id] = {
                    "document": {
                        "id": document.id,
                        "title": document.title,
                        "url": signed_url
                    } if document else None,
                    "conversations": []
                }

            document_map[doc_id]["conversations"].append({
                "id": conv.id,
                "user_id": conv.user_id,
                "document_id": conv.document_id,
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
                "updated_at": conv.updated_at.isoformat()
            })

        # Convert to list format
        result = []
        for doc_id, data in document_map.items():
            result.append({
                "document": data["document"],
                "conversations": data["conversations"]
            })

        return result
    else:
        # Return flat list (backward compatible)
        result = []
        for conv in conversations:
            # Get document info
            document = db.query(Document).filter(Document.id == conv.document_id).first()
            signed_url = document_service.get_signed_url(document.blob_path) if document else None

            result.append({
                "id": conv.id,
                "user_id": conv.user_id,
                "document_id": conv.document_id,
                "title": conv.title,  # Smart conversation title
                "created_at": conv.created_at,
                "updated_at": conv.updated_at,
                "document": {
                    "id": document.id,
                    "title": document.title,
                    "url": signed_url
                } if document else None
            })

        return result


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific conversation with its messages."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Get messages
    messages = db.query(Message).filter(
        Message.conversation_id == conversation_id
    ).order_by(Message.created_at).all()

    # Get document
    document = db.query(Document).filter(Document.id == conversation.document_id).first()
    signed_url = document_service.get_signed_url(document.blob_path) if document else None

    # Build messages with annotations extracted from context
    message_list = []
    for msg in messages:
        message_data = {
            "id": msg.id,
            "content": msg.content,
            "role": msg.role,
            "created_at": msg.created_at,
            "context": msg.context.get("chunks") if isinstance(msg.context, dict) else msg.context,
            "annotations": msg.context.get("annotations") if isinstance(msg.context, dict) else None
        }
        message_list.append(message_data)

    return {
        "conversation": {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "document_id": conversation.document_id,
            "title": conversation.title,  # Smart conversation title
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "document": {
                "id": document.id,
                "title": document.title,
                "url": signed_url
            } if document else None
        },
        "messages": message_list
    }


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and its messages. The document and other conversations remain intact."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()

    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )

    # Delete all messages (cascade should handle this, but being explicit)
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()

    # Delete conversation only (document and other conversations remain)
    db.delete(conversation)

    db.commit()

    return {
        "success": True,
        "message": "Conversation deleted successfully"
    }
