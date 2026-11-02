"""Research Sessions and Analysis API Endpoints."""
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, HTTPException, status

from app.schemas.research import (
    ExperimentalResultsRequest,
    MethodologyAnalysisRequest,
    PaperComparisonRequest,
    ResearchGapsRequest,
    ResearchRequest,
    ResearchResponse,
    ResearchResumeRequest,
)
from agents.graph import build_research_graph
from app.api.v1.endpoints.reports import _REPORTS_STORE, _SESSION_REPORTS
from app.core.config import settings
from logging_config.logger import get_logger
from langgraph.checkpoint.memory import MemorySaver
from tools.gap_identifier import identify_research_gaps
from tools.methodology_analyzer import analyze_research_methodology
from tools.paper_comparator import compare_research_papers
from tools.result_analyzer import analyze_experimental_results
from validation.schemas import (
    MethodologyAnalysisSchema,
    MetricComparisonResultSchema,
    PaperComparisonSchema,
    ResearchComparisonReportSchema,
    ResearchGapsResultSchema,
)

router = APIRouter()
logger = get_logger("api.research")

# Shared in-memory checkpointer for session state across endpoints.
# Sessions survive across requests but not across process restarts; generated
# reports are additionally persisted to disk by the report node.
_api_checkpointer = MemorySaver()
_shared_graph = build_research_graph(checkpointer=_api_checkpointer)


def _graph_config(session_id: str) -> Dict[str, Any]:
    # The recursion limit is only a safety net above the explicit max-step
    # stopping rule enforced inside the graph (reflect_and_revise).
    return {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.MAX_AGENT_STEPS + 50,
    }


def _register_report_if_present(final_report: Optional[dict], session_id: Optional[str] = None) -> Optional[str]:
    """Register a generated report in the archive. Returns the report_id or None.

    Reports are validated against the report schema as-is; missing sections are
    never filled with invented text. A report that does not validate is logged
    and left out of the archive instead of being silently dropped.
    """
    if not final_report or not isinstance(final_report, dict):
        return None
    try:
        schema = ResearchComparisonReportSchema(**final_report)
    except Exception as exc:
        logger.error(
            "Report from session '%s' failed schema validation and was not archived: %s",
            session_id, exc,
        )
        return None
    _REPORTS_STORE[schema.report_id] = schema
    if session_id:
        _SESSION_REPORTS[session_id] = schema.report_id
    return schema.report_id


def _build_response(values: Dict[str, Any], session_id: str, default_status: str, query: Optional[str] = None) -> ResearchResponse:
    return ResearchResponse(
        session_id=values.get("session_id") or session_id,
        status=values.get("stopping_reason") or default_status,
        research_goal=values.get("research_goal") or query or "",
        query=values.get("query") or query,
        clarification_question=values.get("clarification_question"),
        missing_information=values.get("missing_information"),
        plan=values.get("research_plan"),
        final_report=values.get("final_report"),
        observable_logs=values.get("observable_logs", []),
        discovered_papers=values.get("discovered_papers", []),
        validated_papers=values.get("validated_papers", []),
        extracted_papers=values.get("extracted_papers", []),
        metric_comparisons=values.get("metric_comparisons", []),
        identified_gaps=values.get("identified_gaps", []),
        citations=values.get("citations", {}),
        is_degraded=bool(values.get("is_degraded", False)),
        retrieved_evidence=values.get("retrieved_evidence", []),
        phase_statuses=values.get("phase_statuses", {}),
        step_count=values.get("step_count"),
        max_steps=values.get("max_steps"),
    )


async def _run_graph(payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    try:
        return await _shared_graph.ainvoke(payload, config=_graph_config(session_id))
    except Exception as exc:
        logger.exception("Research workflow for session '%s' crashed.", session_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research workflow failed unexpectedly ({type(exc).__name__}). See server logs for session '{session_id}'.",
        ) from exc


@router.post("/sessions", response_model=ResearchResponse, tags=["Research"])
async def start_research_session(request: ResearchRequest) -> ResearchResponse:
    """Initiate an agentic research session with AIRA LangGraph workflow."""
    if request.year_from and request.year_to and request.year_from > request.year_to:
        raise HTTPException(status_code=422, detail="year_from must not be after year_to.")
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    initial_state = {
        "session_id": session_id,
        "query": request.query,
        "research_goal": request.query,
        "max_results": request.max_results,
        "year_from": request.year_from,
        "year_to": request.year_to,
        "max_steps": settings.MAX_AGENT_STEPS,
    }
    final_state = await _run_graph(initial_state, session_id)
    _register_report_if_present(final_state.get("final_report"), session_id=session_id)
    return _build_response(final_state, session_id, "COMPLETED", query=request.query)


@router.get("/sessions/{session_id}", response_model=ResearchResponse, tags=["Research"])
async def get_research_session(session_id: str) -> ResearchResponse:
    """Retrieve current state and inspection details of an active or paused research session."""
    state_snapshot = await _shared_graph.aget_state({"configurable": {"thread_id": session_id}})
    if not state_snapshot or not state_snapshot.values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research session '{session_id}' not found.")
    return _build_response(state_snapshot.values, session_id, "IN_PROGRESS")


@router.post("/sessions/{session_id}/resume", response_model=ResearchResponse, tags=["Research"])
async def resume_research_session(session_id: str, request: ResearchResumeRequest) -> ResearchResponse:
    """Resume a paused research session with user-provided clarification."""
    state_snapshot = await _shared_graph.aget_state({"configurable": {"thread_id": session_id}})
    if not state_snapshot or not state_snapshot.values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research session '{session_id}' not found.")

    current_values = state_snapshot.values
    if current_values.get("stopping_reason") != "USER_CLARIFICATION_REQUIRED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session '{session_id}' is not awaiting clarification (current status: {current_values.get('stopping_reason')}).",
        )

    resume_input = {
        "clarification_response": request.clarification_response.strip(),
        "missing_information": [],
        "clarification_question": None,
        "stopping_reason": None,
        "max_steps": settings.MAX_AGENT_STEPS,
    }
    final_state = await _run_graph(resume_input, session_id)
    _register_report_if_present(final_state.get("final_report"), session_id=session_id)
    return _build_response(final_state, session_id, "COMPLETED")


@router.post("/compare", response_model=PaperComparisonSchema, tags=["Research"])
async def compare_papers_endpoint(request: PaperComparisonRequest) -> PaperComparisonSchema:
    """Compare multiple extracted research papers across standardized dimensions."""
    try:
        result = await compare_research_papers(
            paper_ids=request.paper_ids,
            comparison_dimensions=request.comparison_dimensions,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Comparison failed: {e}")


@router.get("/methodology/{paper_id}", response_model=MethodologyAnalysisSchema, tags=["Research"])
async def analyze_methodology_endpoint(paper_id: str) -> MethodologyAnalysisSchema:
    """Analyze research methodology, design, and reproducibility for a single paper."""
    try:
        result = await analyze_research_methodology(paper_id=paper_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Methodology analysis failed: {e}")


@router.post("/results", response_model=List[MetricComparisonResultSchema], tags=["Research"])
def analyze_results_endpoint(request: ExperimentalResultsRequest) -> List[MetricComparisonResultSchema]:
    """Extract, normalize, and compare experimental results across papers."""
    try:
        result = analyze_experimental_results(
            paper_ids=request.paper_ids,
            metrics=request.metrics,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Results analysis failed: {e}")


@router.post("/gaps", response_model=ResearchGapsResultSchema, tags=["Research"])
async def identify_gaps_endpoint(request: ResearchGapsRequest) -> ResearchGapsResultSchema:
    """Identify evidence-supported research gaps and open problems across corpus."""
    try:
        result = await identify_research_gaps(
            paper_ids=request.paper_ids,
            research_area=request.research_area,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Gap identification failed: {e}")
