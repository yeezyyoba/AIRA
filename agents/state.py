"""Typed state shared by AIRA's LangGraph research workflow."""
from typing import Any, Dict, List, Optional, TypedDict


class ResearchState(TypedDict, total=False):
    # Request and research framing
    session_id: str
    query: str
    research_goal: str
    search_query: str
    research_questions: List[str]
    domain: Optional[str]
    max_results: int
    year_from: Optional[int]
    year_to: Optional[int]

    # Plan and workflow control
    research_plan: List[Dict[str, Any]]
    completed_tasks: List[str]
    current_task: Optional[str]
    selected_tool: Optional[str]
    tool_inputs: Optional[Dict[str, Any]]
    validation_status: str
    step_count: int
    max_steps: int
    retry_counts: Dict[str, int]
    executed_tool_signatures: List[str]
    successful_tool_signatures: List[str]
    failed_items: List[str]
    stopping_reason: Optional[str]

    # Plan revision (reflection-driven replanning)
    pending_revision: Optional[str]
    plan_revisions: int
    expanded_search_pending: bool

    # Search and metadata validation
    discovered_papers: List[Dict[str, Any]]
    validated_papers: List[Dict[str, Any]]

    # Full-text extraction and indexing
    extracted_papers: List[str]
    extraction_results: List[Dict[str, Any]]
    indexed_papers: List[str]
    indexing_results: List[Dict[str, Any]]

    # Analysis outputs
    retrieved_evidence: List[Dict[str, Any]]
    metric_comparisons: List[Dict[str, Any]]
    methodology_analyses: List[Dict[str, Any]]
    paper_comparisons: List[Dict[str, Any]]
    identified_gaps: List[Dict[str, Any]]
    citations: Dict[str, Any]

    # Explicit phase accounting
    phase_statuses: Dict[str, str]
    analysis_actions_completed: List[str]

    # Safety
    risk_flags: List[str]
    security_flags: List[Dict[str, Any]]

    # Execution history and report
    tool_results: List[Dict[str, Any]]
    reflections: List[str]
    critique_status: Optional[str]
    observable_logs: List[Dict[str, Any]]
    is_degraded: bool
    final_report: Optional[Dict[str, Any]]
    saved_report_paths: Dict[str, str]

    # Clarification flow
    missing_information: List[str]
    user_message: Optional[str]
    clarification_question: Optional[str]
    clarification_response: Optional[str]
