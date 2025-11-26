"""
Encryption service for sensitive data like API keys.
Uses Fernet symmetric encryption for API key storage.
"""
from cryptography.fernet import Fernet
from typing import Optional
import logging
from ..config import settings

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for encrypting and decrypting sensitive data."""

    def __init__(self):
        """Initialize encryption service with key from settings."""
        self._cipher_suite: Optional[Fernet] = None

        if settings.ENCRYPTION_KEY:
            try:
                # Validate key length (Fernet requires 32 bytes base64-encoded)
                key = settings.ENCRYPTION_KEY.encode()
                if len(key) == 44:  # Fernet key is 32 bytes base64-encoded = 44 chars
                    self._cipher_suite = Fernet(key)
                else:
                    logger.warning(
                        f"Invalid encryption key length: {len(key)}. "
                        "Expected 44 characters (32 bytes base64-encoded). "
                        "API keys will be stored unencrypted."
                    )
            except Exception as e:
                logger.error(f"Failed to initialize encryption service: {e}")
                logger.warning("API keys will be stored unencrypted.")
        else:
            logger.warning(
                "ENCRYPTION_KEY not set. API keys will be stored unencrypted. "
                "Set ENCRYPTION_KEY in environment for production."
            )

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.

        Args:
            plaintext: The string to encrypt

        Returns:
            Encrypted string (base64-encoded), or original string if encryption unavailable
        """
        if not self._cipher_suite or not plaintext:
            return plaintext

        try:
            encrypted_bytes = self._cipher_suite.encrypt(plaintext.encode())
            return encrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            # Return original to avoid breaking functionality
            return plaintext

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.

        Args:
            ciphertext: The encrypted string (base64-encoded)

        Returns:
            Decrypted string, or original string if decryption unavailable
        """
        if not self._cipher_suite or not ciphertext:
            return ciphertext

        try:
            decrypted_bytes = self._cipher_suite.decrypt(ciphertext.encode())
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            # If decryption fails, might be unencrypted data (backward compatibility)
            # Return original string
            return ciphertext

    def is_encryption_available(self) -> bool:
        """Check if encryption is properly configured."""
        return self._cipher_suite is not None


# Global instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """Get the global encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service

