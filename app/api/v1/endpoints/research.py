"""Research Sessions and Analysis API Endpoints."""
import asyncio
import json
from typing import Any, Dict, List, Optional
import uuid

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

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
from app.core.session_events import current_session_id, event_bus, install_session_log_handler
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

install_session_log_handler()

# Background runs: strong references keep the tasks alive; failures are kept
# so polling clients see FAILED instead of a session that never finishes.
_BACKGROUND_TASKS: Dict[str, "asyncio.Task[Any]"] = {}
_FAILED_RUNS: Dict[str, str] = {}
_SESSION_QUERIES: Dict[str, str] = {}

_NODE_LABELS = {
    "receive_request": "Received research request",
    "analyze_goal": "Analyzed research goal",
    "check_information": "Checked whether the request has enough information",
    "ask_user": "Paused to ask for clarification",
    "create_plan": "Created research plan",
    "select_tools": "Selected next tools",
    "execute_tools": "Executed tools",
    "validate_tool_outputs": "Validated tool outputs",
    "retry_handler": "Handled retry",
    "store_evidence": "Stored evidence",
    "reflect_and_revise": "Reflected on progress",
    "critique_findings": "Critiqued findings",
    "generate_report": "Generated report",
    "finish_task": "Finished",
}


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
        message=values.get("user_message"),
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


def _publish_node_update(session_id: str, node: str, update: Any, seen_logs: List[int]) -> None:
    """Turn one graph step into live-log events."""
    event_bus.publish(session_id, _NODE_LABELS.get(node, node), kind="node", source=node)
    if not isinstance(update, dict):
        return
    logs = update.get("observable_logs")
    if isinstance(logs, list) and len(logs) > seen_logs[0]:
        for entry in logs[seen_logs[0]:]:
            if not isinstance(entry, dict):
                continue
            outcome = str(entry.get("outcome") or "")
            level = "ERROR" if outcome in ("FAILURE", "FAILED", "ERROR") else "WARNING" if outcome in ("PARTIAL", "RETRY", "DEGRADED") else "INFO"
            tool = entry.get("tool")
            message = entry.get("decision") or ""
            if tool:
                message = f"[{tool}] {message}"
            event_bus.publish(
                session_id,
                message,
                kind="decision",
                level=level,
                source=entry.get("node") or node,
                data={"step": entry.get("step"), "outcome": outcome or None},
            )
        seen_logs[0] = len(logs)


async def _run_graph(payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    config = _graph_config(session_id)
    token = current_session_id.set(session_id)
    event_bus.start(session_id)
    try:
        snapshot = await _shared_graph.aget_state(config)
        seen_logs = [len((snapshot.values or {}).get("observable_logs", []) or [])] if snapshot else [0]
        async for chunk in _shared_graph.astream(payload, config=config, stream_mode="updates"):
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                if node.startswith("__"):
                    continue
                _publish_node_update(session_id, node, update, seen_logs)
        final = await _shared_graph.aget_state(config)
        values = dict(final.values or {})
        event_bus.publish(
            session_id,
            f"Run ended: {values.get('stopping_reason') or 'COMPLETED'}",
            kind="status",
            source="workflow",
        )
        return values
    except Exception as exc:
        logger.exception("Research workflow for session '%s' crashed.", session_id)
        event_bus.publish(
            session_id,
            f"Workflow crashed ({type(exc).__name__}): {exc}",
            kind="status",
            level="ERROR",
            source="workflow",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research workflow failed unexpectedly ({type(exc).__name__}). See server logs for session '{session_id}'.",
        ) from exc
    finally:
        event_bus.finish(session_id)
        current_session_id.reset(token)


async def _run_and_register(payload: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    final_state = await _run_graph(payload, session_id)
    _register_report_if_present(final_state.get("final_report"), session_id=session_id)
    return final_state


def _launch_background(payload: Dict[str, Any], session_id: str) -> None:
    """Run the workflow without holding the HTTP request open."""
    _FAILED_RUNS.pop(session_id, None)
    # Mark running before returning so pollers and the event stream never see
    # a gap between the response and the task's first step.
    event_bus.start(session_id)

    async def runner() -> None:
        try:
            await _run_and_register(payload, session_id)
        except HTTPException as exc:
            _FAILED_RUNS[session_id] = str(exc.detail)
        except Exception as exc:  # pragma: no cover - _run_graph wraps errors
            _FAILED_RUNS[session_id] = f"{type(exc).__name__}: {exc}"
        finally:
            event_bus.finish(session_id)
            _BACKGROUND_TASKS.pop(session_id, None)

    _BACKGROUND_TASKS[session_id] = asyncio.create_task(runner())


def _in_progress_response(session_id: str, values: Optional[Dict[str, Any]] = None) -> ResearchResponse:
    values = values or {}
    response = _build_response(values, session_id, "IN_PROGRESS", query=_SESSION_QUERIES.get(session_id))
    # Mid-run the graph may already hold a provisional stopping reason (e.g.
    # GOAL_ACHIEVED before the report node runs); clients must keep polling.
    response.status = "IN_PROGRESS"
    return response


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
    _SESSION_QUERIES[session_id] = request.query
    if request.run_in_background:
        _launch_background(initial_state, session_id)
        return _in_progress_response(session_id, {"session_id": session_id, "query": request.query, "research_goal": request.query})
    final_state = await _run_and_register(initial_state, session_id)
    return _build_response(final_state, session_id, "COMPLETED", query=request.query)


@router.get("/sessions/{session_id}", response_model=ResearchResponse, tags=["Research"])
async def get_research_session(session_id: str) -> ResearchResponse:
    """Retrieve current state and inspection details of an active or paused research session."""
    state_snapshot = await _shared_graph.aget_state({"configurable": {"thread_id": session_id}})
    values = state_snapshot.values if state_snapshot else None
    if event_bus.is_running(session_id):
        return _in_progress_response(session_id, values)
    if session_id in _FAILED_RUNS:
        response = _build_response(values or {}, session_id, "FAILED", query=_SESSION_QUERIES.get(session_id))
        response.status = "FAILED"
        response.message = _FAILED_RUNS[session_id]
        return response
    if not values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research session '{session_id}' not found.")
    return _build_response(values, session_id, "IN_PROGRESS")


@router.post("/sessions/{session_id}/resume", response_model=ResearchResponse, tags=["Research"])
async def resume_research_session(session_id: str, request: ResearchResumeRequest) -> ResearchResponse:
    """Resume a paused research session with user-provided clarification."""
    state_snapshot = await _shared_graph.aget_state({"configurable": {"thread_id": session_id}})
    if not state_snapshot or not state_snapshot.values:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Research session '{session_id}' not found.")

    if event_bus.is_running(session_id):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Session '{session_id}' is already running.")

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
    if request.run_in_background:
        _launch_background(resume_input, session_id)
        return _in_progress_response(session_id, current_values)
    final_state = await _run_and_register(resume_input, session_id)
    return _build_response(final_state, session_id, "COMPLETED")


@router.get("/sessions/{session_id}/events", tags=["Research"])
async def stream_session_events(
    session_id: str,
    request: Request,
    after: int = Query(0, ge=0, description="Only send events with a sequence number above this"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """Stream a session's live event log as Server-Sent Events.

    Sends buffered events first, then new ones as the run produces them, and
    an ``end`` event once the run is no longer active.
    """
    cursor = after
    if last_event_id and last_event_id.isdigit():
        cursor = max(cursor, int(last_event_id))

    async def event_source():
        nonlocal cursor
        idle_ticks = 0
        while True:
            if await request.is_disconnected():
                return
            events = event_bus.events_since(session_id, cursor)
            for event in events:
                cursor = event["seq"]
                yield f"id: {cursor}\nevent: log\ndata: {json.dumps(event, default=str)}\n\n"
            if not events and not event_bus.is_running(session_id):
                yield "event: end\ndata: {}\n\n"
                return
            idle_ticks = 0 if events else idle_ticks + 1
            if idle_ticks and idle_ticks % 30 == 0:
                yield ": keep-alive\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
