from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from ..database import Base
import uuid
import enum


def generate_uuid():
    return str(uuid.uuid4())


class UserRole(str, enum.Enum):
    """User role enumeration for role-based access control."""
    USER = "user"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    api_key = Column(String, nullable=True)  # Stored encrypted if encryption is enabled
    role = Column(
        Enum(UserRole),
        default=UserRole.USER,
        nullable=False,
        server_default=UserRole.USER.value
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # Relationships
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

    def get_decrypted_api_key(self) -> str | None:
        """Get the decrypted API key."""
        if not self.api_key:
            return None
        from ..services.encryption_service import get_encryption_service
        encryption_service = get_encryption_service()
        decrypted = encryption_service.decrypt(self.api_key)
        return decrypted if decrypted else None

    def set_encrypted_api_key(self, plaintext_key: str | None):
        """Set the API key, encrypting it before storage."""
        if plaintext_key is None:
            self.api_key = None
            return
        from ..services.encryption_service import get_encryption_service
        encryption_service = get_encryption_service()
        self.api_key = encryption_service.encrypt(plaintext_key)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="sessions")


class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, nullable=False)
    token = Column(String, unique=True, nullable=False, index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

