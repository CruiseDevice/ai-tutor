from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class MessageBase(BaseModel):
    content: str


class MessageCreate(MessageBase):
    conversation_id: str
    document_id: str
    model: Optional[str] = "gpt-4"


class AnnotationBounds(BaseModel):
    x: float
    y: float
    width: float
    height: float


class PDFAnnotation(BaseModel):
    id: str
    type: str  # 'highlight', 'circle', 'underline', 'box'
    pageNumber: int
    bounds: AnnotationBounds
    textContent: Optional[str] = None
    color: Optional[str] = None
    label: Optional[str] = None


class AnnotationReference(BaseModel):
    pageNumber: int
    annotations: List[PDFAnnotation]
    sourceText: str
    explanation: Optional[str] = None


class MessageResponse(MessageBase):
    id: str
    role: str
    created_at: datetime
    context: Optional[Any] = None
    annotations: Optional[List[AnnotationReference]] = None

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
    title: Optional[str] = None  # Smart conversation title
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationWithMessages(ConversationResponse):
    messages: List[MessageResponse]
    document: Optional[Any] = None

