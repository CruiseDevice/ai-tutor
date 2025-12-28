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
    chunk_level = Column(String, nullable=True, default='flat', index=True)  # flat, parent, or child
    embedding = Column(Vector(768))  # pgvector type for 768-dimensional embeddings
    document_id = Column(String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    position_data = Column(JSONB, nullable=True)
    content_hash = Column(String, nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now(), onupdate=func.now())

    # Relationships
    document = relationship("Document", back_populates="chunks")
    # Parent-child relationships for hierarchical chunking (Phase 3)
    parent_relationships = relationship(
        "ParentChildRelationship",
        foreign_keys="ParentChildRelationship.parent_chunk_id",
        back_populates="parent_chunk",
        cascade="all, delete-orphan"
    )
    child_relationships = relationship(
        "ParentChildRelationship",
        foreign_keys="ParentChildRelationship.child_chunk_id",
        back_populates="child_chunk",
        cascade="all, delete-orphan"
    )


class ParentChildRelationship(Base):
    """
    Model for tracking hierarchical parent-child relationships between chunks.

    Phase 3: Hierarchical Parent-Child Chunking

    This table enables:
    - Precision retrieval: Search against small child chunks for better keyword matching
    - Context preservation: Return larger parent chunks to LLM with full context
    - Flexible chunking: Different chunk sizes for indexing vs context delivery
    """
    __tablename__ = "parent_child_relationships"

    id = Column(String, primary_key=True, default=generate_uuid)
    parent_chunk_id = Column(String, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    child_chunk_id = Column(String, ForeignKey("document_chunks.id", ondelete="CASCADE"), nullable=False, index=True)
    child_index = Column(Integer, nullable=False)  # Order of child within parent (0, 1, 2, ...)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    parent_chunk = relationship(
        "DocumentChunk",
        foreign_keys=[parent_chunk_id],
        back_populates="parent_relationships"
    )
    child_chunk = relationship(
        "DocumentChunk",
        foreign_keys=[child_chunk_id],
        back_populates="child_relationships"
    )

