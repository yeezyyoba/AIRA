"""Research Sessions and Analysis API Endpoints."""
from typing import List, Optional
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

# Shared in-memory checkpointer for session state across endpoints
_api_checkpointer = MemorySaver()
_shared_graph = build_research_graph(checkpointer=_api_checkpointer)


def _register_report_if_present(final_report: Optional[dict], session_id: Optional[str] = None) -> None:
    """Register generated session report in reports store for archive & export."""
    if not final_report or not isinstance(final_report, dict):
        return
    try:
        from datetime import datetime
        from app.api.v1.endpoints.reports import _REPORTS_STORE
        rep_id = final_report.get("report_id") or (f"rep_{session_id}" if session_id else None)
        if not rep_id:
            return

        normalized = dict(final_report)
        normalized.setdefault("report_id", rep_id)
        normalized.setdefault("research_question", normalized.get("title") or "Academic Research Investigation")
        normalized.setdefault("papers_analyzed", [])
        normalized.setdefault("objectives_and_methodology_comparison", normalized.get("executive_summary") or "Methodology analysis completed.")
        normalized.setdefault("dataset_comparison", "Benchmarks evaluated across extracted literature.")
        normalized.setdefault("model_architecture_comparison", "Model architectures evaluated across corpus.")
        normalized.setdefault("experimental_results_comparison", "Experimental measurements extracted.")
        normalized.setdefault("strengths_and_limitations", "Evaluated strengths and empirical limitations.")
        normalized.setdefault("evidence_backed_conclusions", normalized.get("executive_summary") or "Conclusions established.")
        normalized.setdefault("limitations_of_analysis", "Analysis bounded to retrieved evidence.")
        normalized.setdefault("created_at", datetime.utcnow().isoformat() + "Z")

        schema = ResearchComparisonReportSchema(**normalized)
        _REPORTS_STORE[rep_id] = schema
        if session_id:
            _REPORTS_STORE[session_id] = schema
    except Exception:
        pass


@router.post("/sessions", response_model=ResearchResponse, tags=["Research"])
async def start_research_session(request: ResearchRequest) -> ResearchResponse:
    """Initiate an agentic research session with AIRA LangGraph workflow."""
    import uuid
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    initial_state = {
        "session_id": session_id,
        "query": request.query,
        "research_goal": request.query,
    }
    final_state = await _shared_graph.ainvoke(
        initial_state,
        config={"configurable": {"thread_id": session_id}},
    )
    _register_report_if_present(final_state.get("final_report"), session_id=session_id)
    return ResearchResponse(
        session_id=final_state.get("session_id", session_id),
        status=final_state.get("stopping_reason", "COMPLETED"),
        research_goal=final_state.get("research_goal", request.query),
        query=request.query,
        clarification_question=final_state.get("clarification_question"),
        missing_information=final_state.get("missing_information"),
        plan=final_state.get("research_plan"),
        final_report=final_state.get("final_report"),
        observable_logs=final_state.get("observable_logs", []),
        discovered_papers=final_state.get("discovered_papers", []),
        validated_papers=final_state.get("validated_papers", []),
        extracted_papers=final_state.get("extracted_papers", []),
        metric_comparisons=final_state.get("metric_comparisons", []),
        identified_gaps=final_state.get("identified_gaps", []),
        citations=final_state.get("citations", {}),
        is_degraded=final_state.get("is_degraded", False),
    )


@router.get("/sessions/{session_id}", response_model=ResearchResponse, tags=["Research"])
async def get_research_session(session_id: str) -> ResearchResponse:
    """Retrieve current state and inspection details of an active or paused research session."""
    state_snapshot = await _shared_graph.aget_state({"configurable": {"thread_id": session_id}})
    if not state_snapshot or not state_snapshot.values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research session '{session_id}' not found.")
    
    vals = state_snapshot.values
    _register_report_if_present(vals.get("final_report"), session_id=session_id)
    return ResearchResponse(
        session_id=vals.get("session_id", session_id),
        status=vals.get("stopping_reason", "IN_PROGRESS"),
        research_goal=vals.get("research_goal", ""),
        query=vals.get("query"),
        clarification_question=vals.get("clarification_question"),
        missing_information=vals.get("missing_information"),
        plan=vals.get("research_plan"),
        final_report=vals.get("final_report"),
        observable_logs=vals.get("observable_logs", []),
        discovered_papers=vals.get("discovered_papers", []),
        validated_papers=vals.get("validated_papers", []),
        extracted_papers=vals.get("extracted_papers", []),
        metric_comparisons=vals.get("metric_comparisons", []),
        identified_gaps=vals.get("identified_gaps", []),
        citations=vals.get("citations", {}),
        is_degraded=vals.get("is_degraded", False),
    )


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
        "clarification_response": request.clarification_response,
        "missing_information": [],
        "clarification_question": None,
        "stopping_reason": None,
    }
    
    final_state = await _shared_graph.ainvoke(
        resume_input,
        config={"configurable": {"thread_id": session_id}},
    )
    _register_report_if_present(final_state.get("final_report"), session_id=session_id)
    return ResearchResponse(
        session_id=final_state.get("session_id", session_id),
        status=final_state.get("stopping_reason", "COMPLETED"),
        research_goal=final_state.get("research_goal", ""),
        query=final_state.get("query"),
        clarification_question=final_state.get("clarification_question"),
        missing_information=final_state.get("missing_information"),
        plan=final_state.get("research_plan"),
        final_report=final_state.get("final_report"),
        observable_logs=final_state.get("observable_logs", []),
        discovered_papers=final_state.get("discovered_papers", []),
        validated_papers=final_state.get("validated_papers", []),
        extracted_papers=final_state.get("extracted_papers", []),
        metric_comparisons=final_state.get("metric_comparisons", []),
        identified_gaps=final_state.get("identified_gaps", []),
        citations=final_state.get("citations", {}),
        is_degraded=final_state.get("is_degraded", False),
    )


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
