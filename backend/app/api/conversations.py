from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User
from ..models.conversation import Conversation, Message
from ..models.document import Document
from ..schemas.chat import ConversationResponse, ConversationWithMessages, MessageResponse

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=List[ConversationResponse])
async def list_conversations(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all conversations for the current user."""
    conversations = db.query(Conversation).filter(
        Conversation.user_id == user.id
    ).order_by(Conversation.updated_at.desc()).all()
    
    result = []
    for conv in conversations:
        # Get document info
        document = db.query(Document).filter(Document.id == conv.document_id).first()
        
        result.append({
            "id": conv.id,
            "user_id": conv.user_id,
            "document_id": conv.document_id,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "document": {
                "id": document.id,
                "title": document.title,
                "url": document.url
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
    
    return {
        "conversation": {
            "id": conversation.id,
            "user_id": conversation.user_id,
            "document_id": conversation.document_id,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "document": {
                "id": document.id,
                "title": document.title,
                "url": document.url
            } if document else None
        },
        "messages": [
            {
                "id": msg.id,
                "content": msg.content,
                "role": msg.role,
                "created_at": msg.created_at,
                "context": msg.context
            }
            for msg in messages
        ]
    }


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a conversation and all associated data."""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id,
        Conversation.user_id == user.id
    ).first()
    
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    # Get associated document
    document = db.query(Document).filter(Document.id == conversation.document_id).first()
    
    # Delete all messages
    db.query(Message).filter(Message.conversation_id == conversation_id).delete()
    
    # Delete conversation
    db.delete(conversation)
    
    # Delete document chunks
    if document:
        from ..models.document import DocumentChunk
        db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).delete()
        
        # Delete document
        db.delete(document)
    
    db.commit()
    
    return {
        "success": True,
        "message": "Conversation and associated data deleted successfully"
    }

