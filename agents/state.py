"""State definitions for AIRA LangGraph Agentic Workflow.

Provides ResearchState and associated data structures for orchestrating
multi-step academic research across planning, tool selection, execution,
validation, reflection, and report generation.
"""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict, total=False):
    """Central working memory and execution state for the LangGraph workflow."""

    # Session & Goal definition
    session_id: str
    research_goal: str
    query: str
    research_questions: List[str]
    domain: Optional[str]

    # Planning & Task tracking
    research_plan: List[Dict[str, Any]]
    completed_tasks: List[str]
    current_task: Optional[str]

    # Tool selection & Execution
    selected_tool: Optional[str]
    tool_inputs: Optional[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    validation_status: str  # "VALID", "INVALID", "ERROR"
    executed_tool_signatures: List[str]

    # Evidence, Papers & Analysis Artifacts
    discovered_papers: List[Dict[str, Any]]
    validated_papers: List[Dict[str, Any]]
    extracted_papers: List[str]  # paper_ids
    indexed_papers: List[str]
    retrieved_evidence: List[Dict[str, Any]]
    metric_comparisons: List[Dict[str, Any]]
    identified_gaps: List[Dict[str, Any]]
    citations: Dict[str, str]

    # Reflection & Critique
    reflections: List[str]
    critique_status: Optional[str]  # "APPROVED", "NEEDS_REVISION", "INSUFFICIENT_EVIDENCE"

    # Execution control, Retries & Stopping conditions
    step_count: int
    max_steps: int
    retry_counts: Dict[str, int]  # operation_key -> int (max 2)
    missing_information: List[str]
    clarification_question: Optional[str]
    clarification_response: Optional[str]
    stopping_reason: Optional[str]

    # Observability & Reporting
    observable_logs: List[Dict[str, Any]]
    is_degraded: bool
    final_report: Optional[Dict[str, Any]]
