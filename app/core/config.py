"""Application Settings and Configuration."""
from typing import List, Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings loaded from environment variables."""
    PROJECT_NAME: str = "AIRA - Amharic Intelligent Research Assistant"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"

    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS configuration for the planned Next.js / React frontend
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # Primary LLM Provider (Google Gemini)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # Fallback LLM Provider (Groq)
    FALLBACK_PROVIDER: str = "groq"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Academic Search APIs
    OPENALEX_API_KEY: Optional[str] = None
    SEMANTIC_SCHOLAR_API_KEY: Optional[str] = None
    CROSSREF_MAILTO: Optional[str] = "researcher@example.com"

    # Database & Storage Paths
    DATABASE_URL: str = "sqlite:///./aira.db"
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    PAPERS_DIRECTORY: str = "./data/papers"
    PROCESSED_DIRECTORY: str = "./data/processed"
    REPORTS_DIRECTORY: str = "./data/reports"

    # Agent & Search Limits
    # MAX_AGENT_STEPS counts LangGraph node executions (one tool cycle is ~5
    # nodes), so a complete multi-paper workflow needs on the order of 100.
    MAX_AGENT_STEPS: int = 150
    MAX_RETRIES: int = 2
    # "llm": the LLM proposes the next tool call (native function calling) and
    # code validates it, falling back to the rule choice. "rules": rules only.
    TOOL_SELECTION_MODE: Literal["llm", "rules"] = "llm"
    TOOL_SELECTION_TIMEOUT_SECONDS: float = 20.0
    MAX_PAPERS_TO_EXTRACT: int = 5
    MAX_PLAN_REVISIONS: int = 1
    METADATA_VALIDATION_CONCURRENCY: int = 5

    # LLM request limits
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_DEGRADED_COOLDOWN_SECONDS: float = 120.0
    # After a 429 / 503 the provider is skipped for a while instead of being
    # retried seconds later. A provider's own "retry in Ns" hint is honoured,
    # capped at LLM_RETRY_HINT_MAX_SECONDS.
    LLM_RATE_LIMIT_COOLDOWN_SECONDS: float = 20.0
    LLM_OVERLOAD_COOLDOWN_SECONDS: float = 15.0
    LLM_RETRY_HINT_MAX_SECONDS: float = 30.0
    # A budgeted request (tool proposal) does not start the fallback provider
    # with less than this many seconds left.
    LLM_MIN_FALLBACK_BUDGET_SECONDS: float = 3.0
    # Pause LLM tool proposals for the rest of a run after this many
    # consecutive failed or timed-out proposals.
    TOOL_SELECTION_MAX_CONSECUTIVE_FAILURES: int = 2
    MAX_SEARCH_RESULTS: int = 20
    REQUEST_TIMEOUT_SECONDS: int = 30
    # Semantic Scholar allows ~1 request/second without an API key.
    SEMANTIC_SCHOLAR_MIN_INTERVAL_SECONDS: float = 1.1
    ACADEMIC_MAX_RETRIES: int = 2
    ACADEMIC_INITIAL_BACKOFF_SECONDS: float = 1.0

    # PDF Processing & Security Configuration (Phase 4)
    MAX_PDF_SIZE_BYTES: int = 30 * 1024 * 1024  # 30 MB maximum file size
    PDF_DOWNLOAD_TIMEOUT: float = 30.0  # seconds
    DEFAULT_EXTRACTION_METHOD: str = "auto"  # 'auto', 'pymupdf', 'pdfplumber'
    MAX_REDIRECTS: int = 3
    MAX_PDF_PAGES: int = 200  # maximum pages per document to prevent memory bombs
    MAX_EXTRACTED_CHARS: int = 1_000_000  # 1 million characters limit
    PDF_PROCESSING_TIMEOUT: float = 30.0  # seconds

    # Environment & Deployment Safety
    ENVIRONMENT: str = "development"  # 'development', 'staging', 'production'
    ALLOW_MOCK_EMBEDDINGS: bool = False  # Explicit override to allow mock embeddings in production

    # RAG & Embedding Configuration (Phase 5)
    EMBEDDING_PROVIDER: str = "mock"  # 'gemini', 'mock', 'none'
    EMBEDDING_MODEL: str = "gemini-embedding-2"
    EMBEDDING_DIMENSION: int = 768
    EMBEDDING_BATCH_SIZE: int = 16
    EMBEDDING_TIMEOUT_SECONDS: float = 15.0
    RAG_CHUNK_SIZE: int = 1000
    RAG_CHUNK_OVERLAP: int = 150
    RAG_DEFAULT_TOP_K: int = 5
    RAG_MAX_TOP_K: int = 50

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
