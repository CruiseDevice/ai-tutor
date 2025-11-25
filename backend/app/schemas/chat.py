from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class MessageBase(BaseModel):
    content: str


class MessageCreate(MessageBase):
    conversation_id: str
    document_id: str
    model: Optional[str] = "gpt-4"


class MessageResponse(MessageBase):
    id: str
    role: str
    created_at: datetime
    context: Optional[Any] = None
    
    class Config:
        from_attributes = True


class ChatResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse


class ConversationBase(BaseModel):
    pass


class ConversationResponse(ConversationBase):
    id: str
    user_id: str
    document_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse]
    document: Optional[Any] = None

