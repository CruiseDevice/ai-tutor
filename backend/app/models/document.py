from sqlalchemy import Column, String, DateTime, Integer, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from ..database import Base
from datetime import datetime, timezone
from typing import Optional
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    url = Column(String, nullable=False)
    blob_path = Column(String, nullable=False)
    content_hash = Column(String, nullable=True, index=True)

    # Document processing status tracking
    status = Column(String, default="pending", nullable=False, index=True)
    # Possible values: pending, queued, processing, completed, failed
    error_message = Column(Text, nullable=True)  # Store error details if processing fails
    job_id = Column(String, nullable=True)  # Arq job ID for tracking background processing

    # Processing time tracking
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="document", uselist=True, cascade="all, delete-orphan")

    @property
    def processing_duration(self) -> Optional[float]:
        """
        Calculate processing duration in seconds.

        Returns:
            Processing duration in seconds, or None if processing hasn't completed
        """
        if self.processing_started_at and self.processing_completed_at:
            delta = self.processing_completed_at - self.processing_started_at
            return delta.total_seconds()
        return None


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=generate_uuid)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=False)
    chunk_type = Column(String, nullable=True, default='text', index=True)
    embedding = Column(Vector(768))  # pgvector type for 768-dimensional embeddings
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    position_data = Column(JSONB, nullable=True)
    content_hash = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now(), onupdate=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")

