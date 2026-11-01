"""AIRA Pydantic Validation Schemas.

Defines schemas for research queries, paper metadata, analysis results, and citations.
Scheduled for implementation across Phases 2-7.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ResearchQuerySchema(BaseModel):
    """Schema for validating research queries."""
    query: str = Field(..., min_length=3, description="Academic research question or topic")
    year_from: Optional[int] = Field(None, ge=1900, le=2100)
    year_to: Optional[int] = Field(None, ge=1900, le=2100)
    sources: Optional[List[str]] = Field(default=None, description="Selected academic providers")
    max_results: int = Field(default=20, ge=1, le=100)


class PaperMetadataSchema(BaseModel):
    """Schema for validating academic paper metadata."""
    id: Optional[str] = None
    title: str = Field(..., min_length=1)
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    venue: Optional[str] = None
    url: Optional[str] = None
    citation_count: Optional[int] = None
    is_open_access: Optional[bool] = None
    open_access_url: Optional[str] = None
    source: str = Field(default="unknown")
    sources: List[str] = Field(default_factory=list)
    provider_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Original unmerged provider metadata")
    conflicts: List[Dict[str, Any]] = Field(default_factory=list, description="Exposed differences between merged provider records")


class ValidationStatus(str, Enum):
    """Metadata cross-verification status."""
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    CONFLICTING = "CONFLICTING"
    UNVERIFIED = "UNVERIFIED"


class FieldComparisonSchema(BaseModel):
    """Detailed comparison for a specific metadata field."""
    field_name: str
    input_value: Optional[Any] = None
    source_values: Dict[str, Any] = Field(default_factory=dict)
    is_match: Optional[bool] = None
    details: str = ""


class PaperValidationResultSchema(BaseModel):
    """Structured result of metadata cross-verification across authoritative registries."""
    status: ValidationStatus
    paper_id: str
    doi: Optional[str] = None
    field_comparisons: Dict[str, FieldComparisonSchema] = Field(default_factory=dict)
    sources_checked: List[str] = Field(default_factory=list)
    explanation: str = ""
    original_metadata: Dict[str, Any] = Field(default_factory=dict)


# =====================================================================
# Phase 4: PDF Reading & Research Information Extraction Schemas
# =====================================================================

class EvidenceType(str, Enum):
    """Provenance type of extracted finding."""
    DIRECT_EXTRACT = "DIRECT_EXTRACT"
    LLM_SUMMARY = "LLM_SUMMARY"
    INFERENCE = "INFERENCE"


class EvidenceItemSchema(BaseModel):
    """Evidence finding with strict provenance and page reference."""
    paper_id: str = Field(..., description="Unique paper identifier or file name")
    page_number: Optional[int] = Field(default=None, ge=1, description="1-indexed physical page number in PDF if established; None if unknown")
    section_heading: Optional[str] = Field(None, description="Section heading where passage appears")
    passage: str = Field(..., description="Supporting verbatim extracted text passage")
    evidence_type: EvidenceType = Field(default=EvidenceType.DIRECT_EXTRACT, description="Distinguishes extract vs summary vs inference")


class ExtractionStatus(str, Enum):
    """Status of research paper PDF extraction."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    SCANNED_NEEDS_OCR = "SCANNED_NEEDS_OCR"
    FAILED = "FAILED"


class ExtractedPaperDataSchema(BaseModel):
    """Extracted and organized research paper dimensions."""
    paper_id: str
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    research_objectives: Optional[str] = None
    research_questions: Optional[List[str]] = None
    methodology: Optional[str] = None
    dataset: Optional[str] = None
    models_and_algorithms: Optional[str] = None
    experimental_setup: Optional[str] = None
    evaluation_metrics: Optional[str] = None
    results: Optional[str] = None
    limitations: Optional[str] = None
    future_work: Optional[str] = None
    conclusion: Optional[str] = None
    evidence: List[EvidenceItemSchema] = Field(default_factory=list)
    missing_sections: List[str] = Field(default_factory=list, description="Sections searched for but not found in text")


class PaperExtractionResultSchema(BaseModel):
    """Complete response payload for paper extraction."""
    paper_id: str
    status: ExtractionStatus
    extraction_method: str = Field(..., description="Engine used: 'pymupdf', 'pdfplumber', or 'none'")
    page_count: int = Field(default=0, ge=0)
    document_metadata: Dict[str, Any] = Field(default_factory=dict, description="PDF trailer metadata (title, author, creationDate)")
    extracted_data: Optional[ExtractedPaperDataSchema] = None
    is_scanned: bool = False
    ocr_required: bool = False
    warnings: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    raw_text_by_page: Dict[int, str] = Field(default_factory=dict, description="1-indexed page text map")


# =====================================================================
# Phase 5: RAG & Persistent Vector Retrieval Schemas
# =====================================================================

class DocumentChunkSchema(BaseModel):
    """Schema for an individual academic paper chunk with provenance metadata."""
    paper_id: str = Field(..., description="Unique paper identifier")
    chunk_id: str = Field(..., description="Stable, deterministic chunk identifier")
    page_start: int = Field(..., ge=1, description="First page number containing this chunk text")
    page_end: int = Field(..., ge=1, description="Last page number containing this chunk text")
    section_heading: Optional[str] = Field(None, description="Section heading containing this chunk")
    text: str = Field(..., min_length=1, description="Verbatim chunk text")
    source_type: str = Field(default="academic_paper", description="Type of source document")
    document_title: Optional[str] = Field(None, description="Title of the source paper")
    authors: List[str] = Field(default_factory=list, description="Paper authors")
    publication_year: Optional[int] = Field(None, description="Publication year")


class IndexingStatus(str, Enum):
    """Status of academic paper vector indexing."""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNAVAILABLE = "UNAVAILABLE"


class IndexingSummarySchema(BaseModel):
    """Summary of vector indexing execution for a paper."""
    paper_id: str
    chunk_count: int
    embedding_model: str
    dimension: int
    status: IndexingStatus
    message: Optional[str] = None


class RetrievedContextItemSchema(BaseModel):
    """Single relevant research context passage retrieved via semantic search."""
    chunk_id: str
    paper_id: str
    text: str
    document_title: Optional[str] = None
    section_heading: Optional[str] = None
    page_start: int
    page_end: int
    similarity_score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    distance: float = Field(..., description="ChromaDB raw cosine distance")
    authors: List[str] = Field(default_factory=list)
    publication_year: Optional[int] = None
    evidence_type: EvidenceType = Field(default=EvidenceType.DIRECT_EXTRACT)
    is_synthetic: bool = Field(default=False, description="True if generated using mock/synthetic embeddings")


class RetrievalResultSchema(BaseModel):
    """Complete semantic search result for a research query."""
    question: str
    results_count: int
    top_k: int
    embedding_model: str
    items: List[RetrievedContextItemSchema] = Field(default_factory=list)
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    is_degraded: bool = False
    is_synthetic: bool = Field(default=False, description="True if retrieval used synthetic/mock embeddings")
    warning: Optional[str] = None


# =====================================================================
# Phase 6: Research Analysis and Comparison Schemas
# =====================================================================

class GapType(str, Enum):
    """Categorization of an identified academic research gap."""
    AUTHOR_ACKNOWLEDGED_GAP = "AUTHOR_ACKNOWLEDGED_GAP"
    CROSS_PAPER_GAP = "CROSS_PAPER_GAP"
    POTENTIAL_RESEARCH_DIRECTION = "POTENTIAL_RESEARCH_DIRECTION"


class MetricCompatibility(str, Enum):
    """Compatibility classification for experimental metrics."""
    COMPATIBLE = "COMPATIBLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


class MetricGoal(str, Enum):
    """Optimization objective for evaluation metrics."""
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class MetricUnitType(str, Enum):
    """Representation scale of numerical metric value."""
    PERCENTAGE = "PERCENTAGE"  # 0.0 - 100.0%
    FRACTION = "FRACTION"      # 0.0 - 1.0
    SCORE = "SCORE"            # arbitrary score (e.g. BLEU 0-100, perplexity)
    ABSOLUTE = "ABSOLUTE"      # raw count or count delta


class ExtractedMetricValueSchema(BaseModel):
    """Individual reported experimental metric measurement."""
    paper_id: str
    model_name: str
    metric_name: str
    raw_value: Union[float, int, str]
    normalized_value: Optional[float] = None
    unit_type: MetricUnitType = MetricUnitType.PERCENTAGE
    dataset_name: Optional[str] = None
    dataset_version: Optional[str] = None
    task: Optional[str] = None
    language: Optional[str] = None
    evaluation_protocol: Optional[str] = None
    num_classes: Optional[int] = None
    dataset_split: Optional[str] = None  # e.g., 'test', 'validation'
    variant: Optional[str] = None        # e.g., 'macro', 'micro', 'weighted'
    sample_size: Optional[int] = None
    confidence_interval: Optional[str] = None
    statistical_significance: Optional[str] = None
    page_number: Optional[int] = Field(default=None, ge=1)
    evidence_passage: Optional[str] = None


class MetricComparisonResultSchema(BaseModel):
    """Comparative analysis of experimental metrics across models and papers."""
    metric_name: str
    goal: MetricGoal
    compatibility: MetricCompatibility
    incompatibility_reason: Optional[str] = None
    ranked_models: List[Dict[str, Any]] = Field(default_factory=list)
    summary_table: List[Dict[str, Any]] = Field(default_factory=list)
    missing_data_warnings: List[str] = Field(default_factory=list)
    trade_off_notes: Optional[str] = None
    notes: Optional[str] = None


class MethodologyDimensionSchema(BaseModel):
    """Specific analytical finding regarding research methodology."""
    aspect: str  # e.g. 'Research Design', 'Data Collection & Characteristics', 'Preprocessing', 'Architecture', 'Training', 'Evaluation', 'Artifact Availability'
    description: str
    evidence_type: EvidenceType = EvidenceType.DIRECT_EXTRACT
    confidence: str = Field(default="HIGH", description="Confidence level: HIGH, MEDIUM, LOW")
    paper_id: str
    page_number: Optional[int] = Field(default=None, ge=1)
    evidence_passage: Optional[str] = None


class MethodologyAnalysisSchema(BaseModel):
    """Complete methodology breakdown for a single academic research paper."""
    paper_id: str
    document_title: Optional[str] = None
    dimensions: List[MethodologyDimensionSchema] = Field(default_factory=list)
    artifact_availability: Optional[Dict[str, Any]] = None
    reproducibility_score: Optional[str] = None
    reported_limitations: List[EvidenceItemSchema] = Field(default_factory=list)
    is_degraded: bool = False


class PaperComparisonDimensionSchema(BaseModel):
    """Comparison of a single dimension across multiple papers."""
    dimension_name: str
    values_by_paper: Dict[str, Optional[str]] = Field(default_factory=dict)
    differences_noted: List[str] = Field(default_factory=list)
    missing_papers: List[str] = Field(default_factory=list)


class PaperComparisonSchema(BaseModel):
    """Structured matrix comparing multiple research papers."""
    paper_ids: List[str]
    dimensions_compared: List[str]
    matrix: List[PaperComparisonDimensionSchema] = Field(default_factory=list)
    synthesis: Optional[str] = None
    unverified_claims: List[str] = Field(default_factory=list)
    is_degraded: bool = False


class ResearchGapSchema(BaseModel):
    """Evidence-supported research gap or open problem identified in corpus."""
    gap_id: str
    title: str
    description: str
    gap_type: GapType
    supporting_papers: List[str]
    evidence: List[EvidenceItemSchema] = Field(default_factory=list)
    suggested_direction: Optional[str] = None


class ResearchGapsResultSchema(BaseModel):
    """Result of research gap identification over an indexed research corpus."""
    research_area: str
    paper_ids_analyzed: List[str]
    gaps: List[ResearchGapSchema] = Field(default_factory=list)
    insufficient_evidence: bool = False
    corpus_scope_disclaimer: str = Field(
        default="Identified gaps are strictly derived from the analyzed research corpus and do not imply global literature nonexistence."
    )
    is_degraded: bool = False


class ResearchComparisonReportSchema(BaseModel):
    """Comprehensive 11-section cited research comparison report."""
    report_id: str
    research_question: str
    papers_analyzed: List[str]
    objectives_and_methodology_comparison: str
    dataset_comparison: str
    model_architecture_comparison: str
    experimental_results_comparison: str
    strengths_and_limitations: str
    research_gaps: List[ResearchGapSchema] = Field(default_factory=list)
    evidence_backed_conclusions: str
    limitations_of_analysis: str
    references: List[Dict[str, Any]] = Field(default_factory=list)
    metric_tables: List[MetricComparisonResultSchema] = Field(default_factory=list)
    is_degraded: bool = False
    created_at: str
