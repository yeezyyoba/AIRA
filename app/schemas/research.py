"""Research API Schemas."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    """Schema for initiating an agentic research session."""
    query: str = Field(..., min_length=3, max_length=1000, description="Academic research question or topic")
    year_from: Optional[int] = Field(None, ge=1900, le=2100)
    year_to: Optional[int] = Field(None, ge=1900, le=2100)
    max_results: int = Field(default=20, ge=1, le=100)
    run_in_background: bool = Field(
        default=False,
        description="Return immediately with status IN_PROGRESS; poll the session and stream /events for progress",
    )


class ResearchResponse(BaseModel):
    """Schema for research session status or results."""
    session_id: str
    status: str
    research_goal: str
    query: Optional[str] = None
    clarification_question: Optional[str] = None
    message: Optional[str] = Field(None, description="Explanation shown to the user when a request is refused or out of scope")
    missing_information: Optional[List[str]] = None
    plan: Optional[List[Dict[str, Any]]] = None
    final_report: Optional[Dict[str, Any]] = None
    observable_logs: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    discovered_papers: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    validated_papers: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    extracted_papers: Optional[List[str]] = Field(default_factory=list)
    metric_comparisons: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    identified_gaps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    citations: Optional[Dict[str, Any]] = Field(default_factory=dict)
    is_degraded: Optional[bool] = False
    retrieved_evidence: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    phase_statuses: Optional[Dict[str, str]] = Field(default_factory=dict)
    step_count: Optional[int] = None
    tool_selection_stats: Dict[str, int] = Field(
        default_factory=dict,
        description="How the next tool was chosen: LLM proposals accepted/rejected, rule fallbacks, single-option steps",
    )
    max_steps: Optional[int] = None
    api_usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "External API requests made by this session: per LLM provider (calls, successes, failures, "
            "rate_limited, timeouts, tokens), embeddings, and academic APIs. None when this server "
            "process has no record of the session (e.g. after a restart)."
        ),
    )


class ResearchResumeRequest(BaseModel):
    """Request payload for resuming a paused research session with clarification."""
    clarification_response: str = Field(..., min_length=1, max_length=2000, description="User's clarification response or answers")
    run_in_background: bool = Field(default=False, description="Resume without waiting for the run to finish")


# =====================================================================
# Phase 6: Research Analysis and Comparison Schemas
# =====================================================================

class PaperComparisonRequest(BaseModel):
    """Request payload for comparing multiple research papers."""
    paper_ids: List[str] = Field(..., min_length=2, description="List of paper IDs to compare (minimum 2)")
    comparison_dimensions: Optional[List[str]] = Field(None, description="Optional custom dimensions to compare")


class MethodologyAnalysisRequest(BaseModel):
    """Request payload for analyzing a paper's methodology."""
    paper_id: str = Field(..., description="Unique paper identifier")


class ExperimentalResultsRequest(BaseModel):
    """Request payload for experimental result comparison."""
    paper_ids: List[str] = Field(..., min_length=1, description="List of paper IDs to extract and compare results for")
    metrics: Optional[List[str]] = Field(None, description="Optional list of specific metrics to analyze (e.g. ['bleu', 'f1'])")


class ResearchGapsRequest(BaseModel):
    """Request payload for research gap identification."""
    paper_ids: List[str] = Field(..., min_length=1, description="List of paper IDs in corpus")
    research_area: str = Field(..., min_length=2, description="Target domain or research question")
