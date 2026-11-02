"""Schemas for Papers API Endpoints."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from validation.schemas import (
    FieldComparisonSchema,
    PaperMetadataSchema,
    ValidationStatus,
)


class PaperSearchRequest(BaseModel):
    """Request payload for academic paper search."""
    query: str = Field(..., min_length=2, description="Academic research query string")
    year_from: Optional[int] = Field(None, ge=1900, le=2100, description="Earliest publication year")
    year_to: Optional[int] = Field(None, ge=1900, le=2100, description="Latest publication year")
    sources: Optional[List[str]] = Field(
        default=None,
        description="List of sources to search (openalex, semantic_scholar, crossref). Defaults to all.",
    )
    max_results: int = Field(default=20, ge=1, le=100, description="Maximum number of deduplicated results to return")


class PaperSearchResponse(BaseModel):
    """Response payload for academic paper search."""
    query: str
    total_results: int
    raw_results_count: int
    duplicates_removed: int
    sources_searched: List[str]
    provider_counts: Dict[str, int]
    provider_errors: Dict[str, str] = Field(default_factory=dict)
    execution_time_seconds: float
    papers: List[PaperMetadataSchema]


class PaperValidationRequest(BaseModel):
    """Request payload for validating paper metadata."""
    paper: PaperMetadataSchema = Field(..., description="Paper metadata to cross-check")
    paper_id: Optional[str] = Field(None, description="Optional paper identifier or tracking ID")


class PaperValidationResponse(BaseModel):
    """Response payload for paper metadata validation."""
    status: ValidationStatus
    paper_id: str
    doi: Optional[str] = None
    field_comparisons: Dict[str, FieldComparisonSchema]
    sources_checked: List[str]
    explanation: str
    original_metadata: Dict[str, Any]


class PaperReadRequest(BaseModel):
    """Request payload for reading a paper from a remote URL."""
    url: str = Field(..., max_length=2048, description="Publicly accessible HTTP/HTTPS URL of the academic PDF")
    paper_id: Optional[str] = Field(None, max_length=200, description="Optional paper identifier")
    extraction_method: str = Field(default="auto", description="Extraction method: 'auto', 'pymupdf', or 'pdfplumber'")

    @field_validator("url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        # The reader treats non-URL strings as local file paths; an API client
        # must never be able to make the server read its own filesystem.
        cleaned = value.strip()
        if not cleaned.lower().startswith(("http://", "https://")):
            raise ValueError("url must be an http:// or https:// URL")
        return cleaned


class PaperExtractionResponse(BaseModel):
    """Response payload for research paper PDF reading and extraction."""
    paper_id: str
    status: str
    extraction_method: str
    page_count: int
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_data: Optional[Dict[str, Any]] = None
    is_scanned: bool = False
    ocr_required: bool = False
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None


# =====================================================================
# Phase 5: RAG & Persistent Vector Retrieval Schemas
# =====================================================================

class PaperIndexRequest(BaseModel):
    """Request payload for indexing an extracted academic paper into ChromaDB."""
    chunk_size: Optional[int] = Field(None, ge=100, le=5000, description="Maximum characters per chunk")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=1000, description="Character overlap between chunks")


class PaperIndexResponse(BaseModel):
    """Response payload after indexing a paper into ChromaDB."""
    paper_id: str
    chunk_count: int
    embedding_model: str
    dimension: int
    status: str
    message: Optional[str] = None


class ContextRetrievalRequest(BaseModel):
    """Request payload for semantic research context retrieval."""
    question: str = Field(..., min_length=2, description="Research query or topic to retrieve context for")
    paper_ids: Optional[List[str]] = Field(None, description="Optional paper IDs to filter results against")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of relevant chunks to retrieve")


class ContextRetrievalResponse(BaseModel):
    """Response payload containing retrieved research passages with citations."""
    question: str
    results_count: int
    top_k: int
    embedding_model: str
    items: List[Dict[str, Any]]
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    is_degraded: bool = False
    is_synthetic: bool = False
    warning: Optional[str] = None


class PaperIndexingStatusResponse(BaseModel):
    """Response payload for inspecting paper indexing state in ChromaDB."""
    paper_id: str
    is_indexed: bool
    chunk_count: int
    total_collection_chunks: int
    embedding_model: str
    dimension: int


class DeletePaperIndexResponse(BaseModel):
    """Response payload after deleting a paper's indexed chunks from ChromaDB."""
    paper_id: str
    deleted_chunks: int
    remaining_collection_chunks: int
    message: str
