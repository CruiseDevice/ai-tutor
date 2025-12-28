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
    RATE_LIMIT_AUTH_PER_MINUTE: int = 20  # More lenient for development (can be overridden via env var)

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

    # Document Processing Optimizations
    USE_STREAMING_PROCESSING: bool = True  # Enable streaming PDF processing for progressive availability

    # Semantic Chunking Configuration
    # Advanced chunking strategy that preserves document structure and adapts to content types
    CHUNK_SIZE_MIN: int = 500  # Minimum chunk size in characters
    CHUNK_SIZE_MAX: int = 2000  # Maximum chunk size in characters
    CHUNK_SIZE_DEFAULT: int = 1000  # Default chunk size in characters
    CHUNK_OVERLAP_MIN: int = 50  # Minimum overlap in characters
    CHUNK_OVERLAP_MAX: int = 400  # Maximum overlap in characters
    CHUNK_OVERLAP_DEFAULT: int = 200  # Default overlap in characters
    CHUNK_OVERLAP_PERCENTAGE: float = 0.15  # Adaptive overlap as percentage of chunk size (15%)
    USE_SEMANTIC_CHUNKING: bool = True  # Enable semantic boundary detection using embeddings
    SEMANTIC_CHUNKING_BREAKPOINT_THRESHOLD: float = 0.5  # Threshold for detecting semantic boundaries (0.0-1.0)
    CHUNK_BY_CONTENT_TYPE: bool = True  # Enable content-type-based adaptive chunking (headers, tables, lists, etc.)
    PRESERVE_METADATA: bool = True  # Enable extraction and storage of structural metadata (headers, sections, content types)

    # Hybrid Search Configuration
    # Controls the weighting between semantic and keyword search
    # semantic_weight + keyword_weight should = 1.0 for optimal results
    SEMANTIC_SEARCH_WEIGHT: float = 0.7  # Weight for pgvector semantic search (default: 70%)
    KEYWORD_SEARCH_WEIGHT: float = 0.3   # Weight for PostgreSQL full-text keyword search (default: 30%)

    # Adaptive Hybrid Search
    # Dynamically adjust weights based on query characteristics for better retrieval
    ENABLE_ADAPTIVE_HYBRID_WEIGHTS: bool = True  # Enable query-adaptive weighting
    HYBRID_WEIGHT_KEYWORD_BOOST: float = 0.6     # Keyword weight for keyword-focused queries (definitions, facts)
    HYBRID_WEIGHT_SEMANTIC_BOOST: float = 0.85   # Semantic weight for semantic queries (explanations, concepts)

    # Re-Ranking Configuration
    # Cross-encoder re-ranking improves retrieval quality by re-ranking top candidates
    RERANK_ENABLED: bool = True  # Enable/disable cross-encoder re-ranking
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Cross-encoder model for re-ranking
    RERANK_TOP_K: int = 40  # Number of candidates to retrieve before re-ranking (increased for better coverage)
    RERANK_FINAL_K: int = 15  # Final number of chunks to return after re-ranking (increased for better detail capture)

    # Query Expansion Configuration
    # Multi-query retrieval generates query variations to improve retrieval accuracy for ambiguous/complex queries
    QUERY_EXPANSION_ENABLED: bool = True  # Enable/disable query expansion (default: off for backwards compatibility)
    QUERY_EXPANSION_MODEL: str = "gpt-4o-mini"  # LLM model for generating query variations (cheaper model recommended)
    QUERY_EXPANSION_NUM_VARIATIONS: int = 3  # Number of query variations to generate (including original query)
    QUERY_EXPANSION_TEMPERATURE: float = 0.7  # Temperature for variation generation (0.7 for diverse variations)
    RRF_K: int = 60  # Reciprocal Rank Fusion constant (standard value, controls score normalization)

    # Query Decomposition Configuration
    # Breaks down complex multi-part queries into simpler atomic sub-queries for better retrieval
    ENABLE_QUERY_DECOMPOSITION: bool = True  # Enable/disable query decomposition
    QUERY_DECOMPOSITION_MAX_SUBQUERIES: int = 5  # Maximum number of sub-queries to generate

    # Hierarchical Chunking Configuration
    # Two-level chunking: large parent chunks for context, small child chunks for precision
    ENABLE_HIERARCHICAL_CHUNKING: bool = True  # Enable/disable hierarchical chunking (start disabled for testing)
    HIERARCHICAL_PARENT_CHUNK_SIZE: int = 1500  # Parent chunk size in characters (provides context to LLM)
    HIERARCHICAL_PARENT_OVERLAP: int = 200  # Overlap between parent chunks
    HIERARCHICAL_CHILD_CHUNK_SIZE: int = 300  # Child chunk size in characters (for precise retrieval)
    HIERARCHICAL_CHILD_OVERLAP: int = 50  # Overlap between child chunks

    # Adaptive Chunk Sizing Configuration
    # Dynamically adjust chunk sizes based on content density
    ENABLE_ADAPTIVE_CHUNKING: bool = True  # Enable/disable adaptive chunk sizing (start disabled for testing)
    ADAPTIVE_DENSITY_HIGH_THRESHOLD: float = 0.6  # Density threshold for "high density" content (tables, code)
    ADAPTIVE_DENSITY_LOW_THRESHOLD: float = 0.4  # Density threshold for "low density" content (narrative)
    # Density metrics weights (sum to 1.0)
    DENSITY_WEIGHT_SPECIAL_CHAR: float = 0.30  # Weight for special characters (code, math symbols)
    DENSITY_WEIGHT_NUMERIC: float = 0.25  # Weight for numeric content (tables, data)
    DENSITY_WEIGHT_LINE_BREAK: float = 0.20  # Weight for line breaks (structured content)
    DENSITY_WEIGHT_PUNCTUATION: float = 0.10  # Weight for punctuation
    DENSITY_WEIGHT_TOKEN: float = 0.10  # Weight for token density
    DENSITY_WEIGHT_WHITESPACE: float = 0.05  # Weight for whitespace (inverted)

    # Sentence-Level Retrieval Configuration
    # Extract and index individual critical sentences for ultra-precise retrieval of specific details
    ENABLE_SENTENCE_RETRIEVAL: bool = True  # Enable/disable sentence-level retrieval
    SENTENCE_RETRIEVAL_TOP_K: int = 10  # Number of top critical sentences to retrieve per query
    SENTENCE_BOOST_FACTOR: float = 1.2  # Boost factor for sentences when answering detail queries
    SENTENCE_MIN_LENGTH: int = 10  # Minimum character length for a sentence to be considered
    SENTENCE_MAX_LENGTH: int = 500  # Maximum character length for a sentence to be indexed
    SENTENCE_INCLUDE_SHORT: bool = True  # Include short sentences (<150 chars) as potentially critical

    # Contextual Compression Configuration
    # Compress retrieved chunks by extracting only query-relevant sentences to reduce noise and fit more chunks
    ENABLE_CONTEXTUAL_COMPRESSION: bool = True  # Enable/disable contextual compression
    COMPRESSION_MIN_SENTENCES: int = 3  # Minimum sentences in chunk before applying compression
    COMPRESSION_KEEP_PERCENTAGE: float = 0.4  # Keep top 40% most relevant sentences
    COMPRESSION_MIN_CHUNKS: int = 10  # Apply compression only when retrieving this many chunks or more

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

    # Agent Configuration (LangGraph Integration)
    # Multi-step reasoning system with adaptive routing and quality verification
    AGENT_ENABLED: bool = True  # Feature flag for gradual rollout (default: disabled)
    AGENT_DEFAULT_MODEL: str = "gpt-4o-mini"  # Default model for agent nodes (cost-efficient)
    AGENT_COMPLEXITY_THRESHOLD: str = "medium"  # Use agents for queries with "medium" or "complex" complexity
    AGENT_STREAMING_ENABLED: bool = True  # Enable streaming support for agent workflow execution

    # Image Processing with Docling (Multimodal support)
    # Controls image extraction, captioning, and embedding generation
    ENABLE_IMAGE_EXTRACTION: bool = True    # Feature flag to enable/disable image processing
    VISION_MODEL: str = "gpt-4o-mini"   # Model for image captioning (gpt-4o-mini for cost, gpt-4o for quality)
    MAX_IMAGES_PER_DOCUMENT: int = 200  # Maximum number of images to process per document
    IMAGE_CAPTION_BATCH_SIZE: int = 5   # Number of images to caption in parallel (controls rate limits)
    S3_ASSETS_FOLDER_SUFFIX: str = "_assets"    # Suffic for S3 folder containing extracted images
    IMAGE_COMPRESSION_QUALITY: int = 85     # JPEG compression quality (1-100, higher = better quality)
    DOCLING_FALLBACK_TO_PYPDF: bool = True  # Fallback to PyPDF if Docling fails

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

