from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from ..database import get_db
from ..core.deps import get_current_user
from ..models.user import User
from ..models.conversation import Conversation
from ..schemas.chat import MessageCreate, ChatResponse
from ..services.chat_service import ChatService

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Initialize chat service
chat_service = ChatService()


@router.post("", response_model=ChatResponse)
async def send_message(
    message_data: MessageCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message and get AI response."""
    try:
        # Verify conversation belongs to user
        conversation = db.query(Conversation).filter(
            Conversation.id == message_data.conversation_id,
            Conversation.user_id == user.id
        ).first()
        
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
        
        # Generate response
        result = await chat_service.generate_chat_response(
            db=db,
            user=user,
            content=message_data.content,
            conversation_id=message_data.conversation_id,
            document_id=message_data.document_id,
            model=message_data.model or "gpt-4"
        )
        
        return ChatResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )

