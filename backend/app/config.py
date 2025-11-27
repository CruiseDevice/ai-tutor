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
    MAX_FILE_SIZE: int = 30 * 1024 * 1024  # 30MB in bytes

    # Document Processing Optimizations (Phase 3)
    USE_STREAMING_PROCESSING: bool = True  # Enable streaming PDF processing for progressive availability

    # Hybrid Search Configuration
    # Controls the weighting between semantic and keyword search
    # semantic_weight + keyword_weight should = 1.0 for optimal results
    SEMANTIC_SEARCH_WEIGHT: float = 0.7  # Weight for pgvector semantic search (default: 70%)
    KEYWORD_SEARCH_WEIGHT: float = 0.3   # Weight for PostgreSQL full-text keyword search (default: 30%)

    # Re-Ranking Configuration
    # Cross-encoder re-ranking improves retrieval quality by re-ranking top candidates
    RERANK_ENABLED: bool = True  # Enable/disable cross-encoder re-ranking
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model for re-ranking
    RERANK_TOP_K: int = 20  # Number of candidates to retrieve before re-ranking
    RERANK_FINAL_K: int = 5  # Final number of chunks to return after re-ranking

    # Query Expansion Configuration
    # Multi-query retrieval generates query variations to improve retrieval accuracy for ambiguous/complex queries
    QUERY_EXPANSION_ENABLED: bool = True  # Enable/disable query expansion (default: off for backwards compatibility)
    QUERY_EXPANSION_MODEL: str = "gpt-4o-mini"  # LLM model for generating query variations (cheaper model recommended)
    QUERY_EXPANSION_NUM_VARIATIONS: int = 3  # Number of query variations to generate (including original query)
    QUERY_EXPANSION_TEMPERATURE: float = 0.7  # Temperature for variation generation (0.7 for diverse variations)
    RRF_K: int = 60  # Reciprocal Rank Fusion constant (standard value, controls score normalization)

    # Token Management and Context Window Configuration
    # Dynamic chunk selection based on token limits instead of fixed chunk count
    MAX_CONTEXT_TOKENS: int = 100000  # Maximum context window limit (GPT-4 supports up to 128k)
    TOKEN_RESERVE_BUFFER: int = 20000  # Reserve tokens for system prompt, history, and response generation
    CHUNK_TRUNCATION_ENABLED: bool = True  # Enable truncating chunks to fit within token limits
    TOKEN_TRACKING_ENABLED: bool = True  # Enable tracking token usage per request for analytics

    # Advanced Prompting & Answer Quality Configuration
    # Enhance system prompts with few-shot examples, chain-of-thought, and quality scoring
    ENABLE_QUERY_CLASSIFICATION: bool = True  # Enable automatic query type classification (factual, analytical, etc.)
    ENABLE_CHAIN_OF_THOUGHT: bool = True  # Enable chain-of-thought prompting for complex queries
    COT_COMPLEXITY_THRESHOLD: str = "moderate"  # Enable COT for queries with "moderate" or "complex" complexity
    ENABLE_CITATION_VERIFICATION: bool = True  # Enable post-processing verification of page citations
    ENABLE_ANSWER_QUALITY_SCORING: bool = True  # Enable LLM-based answer quality evaluation
    QUERY_CLASSIFICATION_MODEL: str = "gpt-4o-mini"  # Model for query classification (cost-efficient)
    QUALITY_SCORING_MODEL: str = "gpt-4o-mini"  # Model for quality scoring (cost-efficient)

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

