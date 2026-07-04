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
    # Legacy single-provider key. Kept as a backward-compatible read-alias for
    # OpenAI (migrated into openai_api_key on first use). New writes go to the
    # per-provider columns below.
    api_key = Column(String, nullable=True)  # Stored encrypted if encryption is enabled
    # Per-provider encrypted API keys.
    openai_api_key = Column(String, nullable=True)
    anthropic_api_key = Column(String, nullable=True)
    ollama_api_key = Column(String, nullable=True)
    role = Column(
        Enum(
            UserRole,
            name="userrole",
            values_callable=lambda enum_cls: [e.value for e in enum_cls]
        ),
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

    # Maps provider name -> column attribute. Used by the per-provider accessors.
    _PROVIDER_KEY_COLUMNS = {
        "openai": "openai_api_key",
        "anthropic": "anthropic_api_key",
        "ollama": "ollama_api_key",
    }

    def get_decrypted_api_key(self) -> str | None:
        """Get the decrypted OpenAI API key (legacy entry point).

        Reads from openai_api_key, falling back to the legacy api_key column so
        existing users keep working until they re-save a key.
        """
        return self.get_decrypted_key("openai")

    def set_encrypted_api_key(self, plaintext_key: str | None):
        """Set the OpenAI API key (legacy entry point)."""
        self.set_encrypted_key("openai", plaintext_key)

    def get_decrypted_key(self, provider: str) -> str | None:
        """Get the decrypted API key for a provider.

        For OpenAI, falls back to the legacy api_key column if the per-provider
        column is empty (transparent migration of pre-existing keys).
        """
        from ..services.encryption_service import get_encryption_service
        encryption_service = get_encryption_service()

        column_name = self._PROVIDER_KEY_COLUMNS.get(provider)
        if column_name is None:
            return None
        ciphertext = getattr(self, column_name, None)

        # Transparent migration: legacy api_key holds OpenAI keys.
        if not ciphertext and provider == "openai":
            ciphertext = self.api_key

        if not ciphertext:
            return None
        decrypted = encryption_service.decrypt(ciphertext)
        return decrypted if decrypted else None

    def set_encrypted_key(self, provider: str, plaintext_key: str | None):
        """Set the API key for a provider, encrypting it before storage."""
        from ..services.encryption_service import get_encryption_service
        encryption_service = get_encryption_service()

        column_name = self._PROVIDER_KEY_COLUMNS.get(provider)
        if column_name is None:
            raise ValueError(f"Unknown provider: {provider}")

        if plaintext_key is None or plaintext_key == "":
            setattr(self, column_name, None)
            return
        setattr(self, column_name, encryption_service.encrypt(plaintext_key))


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
