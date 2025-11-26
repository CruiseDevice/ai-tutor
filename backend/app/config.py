from pydantic_settings import BaseSettings
from typing import Optional, List


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str

    # AWS S3
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    S3_PDFBUCKET_NAME: str

    # Security
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days

    # Cookie security settings
    COOKIE_SECURE: bool = False  # Set to True in production with HTTPS
    COOKIE_SAMESITE: str = "lax"  # Options: "lax", "strict", "none"

    # API Key encryption
    ENCRYPTION_KEY: Optional[str] = None  # 32-byte key for Fernet encryption

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_AUTH_PER_MINUTE: int = 5  # Stricter for auth endpoints

    # Redis Cache (optimized for performance)
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_ENABLED: bool = True
    CACHE_EMBEDDING_TTL: int = 86400  # 24 hours (embeddings stable per query)
    CACHE_RESPONSE_TTL: int = 7200  # 2 hours (increased from 1h for better hit rate)
    CACHE_CHUNK_TTL: int = 259200  # 72 hours (increased from 24h - vector results are stable)
    CACHE_SIMILARITY_THRESHOLD: float = 0.85  # 85% similarity for response cache
    CACHE_COMPRESSION_THRESHOLD: int = 1024  # Compress cache values larger than 1KB

    # Environment
    NODE_ENV: str = "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.NODE_ENV.lower() == "production"

    # File upload limits
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB in bytes

    # CORS - Allow common development origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # In case Next.js runs on different port
        "http://127.0.0.1:3001",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

