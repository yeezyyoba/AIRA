"""AIRA LangGraph Stateful Graph Definition."""

from typing import Any, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agents.nodes import (
    analyze_goal,
    ask_user,
    check_information,
    create_plan,
    critique_findings,
    execute_tools,
    finish_task,
    generate_report_node,
    receive_request,
    reflect_and_revise,
    retry_handler,
    select_tools,
    store_evidence,
    validate_tool_outputs,
)
from agents.state import ResearchState
from app.core.config import settings
from llm.model_router import ModelRouter
from memory.vector_store import ChromaVectorStore
from services.paper_repository import PaperRepository


def _plan_tasks(state: ResearchState):
    """Normalise plan entries; tolerate legacy string task ids."""
    tasks = []
    for task in state.get("research_plan", []) or []:
        if isinstance(task, dict):
            tasks.append(task)
        elif task is not None:
            tasks.append({"task_id": str(task), "status": "PENDING"})
    return tasks


def _step_budget_exhausted(state: ResearchState) -> bool:
    return state.get("step_count", 0) >= state.get("max_steps", settings.MAX_AGENT_STEPS)


def route_after_check_information(state: ResearchState) -> str:
    if state.get("stopping_reason") == "UNACCEPTABLE_RISK" or state.get("risk_flags"):
        return "finish_task"
    if state.get("missing_information"):
        return "ask_user"
    return "create_plan"


def route_after_select_tools(state: ResearchState) -> str:
    if state.get("selected_tool"):
        return "execute_tools"
    return "reflect_and_revise"


def route_after_validation(state: ResearchState) -> str:
    status = state.get("validation_status")
    if status in ("INVALID", "ERROR"):
        return "retry_handler"
    # An initial search that finds nothing is retried with a reformulated
    # query; an empty *expanded* search simply adds no new candidates.
    if (
        status == "EMPTY"
        and state.get("selected_tool") == "search_academic_papers"
        and not state.get("discovered_papers")
    ):
        return "retry_handler"
    return "store_evidence"


def route_after_retry(state: ResearchState) -> str:
    if state.get("validation_status") == "RETRY_REQUESTED":
        return "execute_tools"
    return "reflect_and_revise"


def route_after_reflection(state: ResearchState) -> str:
    if state.get("stopping_reason") or _step_budget_exhausted(state):
        return "critique_findings"

    if state.get("final_report"):
        return "critique_findings"

    plan = _plan_tasks(state)
    completed = set(state.get("completed_tasks", []))
    if plan and all(task.get("task_id") in completed for task in plan):
        return "critique_findings"

    return "select_tools"


def route_after_critique(state: ResearchState) -> str:
    """Loop back only when a genuinely unresolved phase remains and budget allows."""
    if state.get("stopping_reason") or _step_budget_exhausted(state):
        return "generate_report"

    completed = set(state.get("completed_tasks", []))
    unresolved = any(
        task.get("task_id") not in completed
        and task.get("status", "PENDING") in ("PENDING", "IN_PROGRESS")
        for task in _plan_tasks(state)
    )

    if state.get("critique_status") == "NEEDS_REVISION" and unresolved:
        return "select_tools"

    return "generate_report"


def build_research_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    router: Optional[ModelRouter] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> Any:
    """Build and compile the AIRA research workflow."""

    workflow = StateGraph(ResearchState)

    workflow.add_node("receive_request", receive_request)
    workflow.add_node("analyze_goal", analyze_goal)
    workflow.add_node("check_information", check_information)
    workflow.add_node("ask_user", ask_user)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("select_tools", select_tools)

    async def bound_execute_tools(
        state: ResearchState,
    ) -> Dict[str, Any]:
        return await execute_tools(
            state,
            router=router,
            repository=repository,
            vector_db=vector_db,
        )

    async def bound_generate_report(
        state: ResearchState,
    ) -> Dict[str, Any]:
        return await generate_report_node(
            state,
            router=router,
            repository=repository,
            vector_db=vector_db,
        )

    workflow.add_node("execute_tools", bound_execute_tools)
    workflow.add_node("validate_tool_outputs", validate_tool_outputs)
    workflow.add_node("retry_handler", retry_handler)
    workflow.add_node("store_evidence", store_evidence)
    workflow.add_node("reflect_and_revise", reflect_and_revise)
    workflow.add_node("critique_findings", critique_findings)
    workflow.add_node("generate_report", bound_generate_report)
    workflow.add_node("finish_task", finish_task)

    workflow.add_edge(START, "receive_request")
    workflow.add_edge("receive_request", "analyze_goal")
    workflow.add_edge("analyze_goal", "check_information")

    workflow.add_conditional_edges(
        "check_information",
        route_after_check_information,
        {
            "ask_user": "ask_user",
            "create_plan": "create_plan",
            "finish_task": "finish_task",
        },
    )

    workflow.add_edge("ask_user", "finish_task")
    workflow.add_edge("create_plan", "select_tools")

    workflow.add_conditional_edges(
        "select_tools",
        route_after_select_tools,
        {
            "execute_tools": "execute_tools",
            "reflect_and_revise": "reflect_and_revise",
        },
    )

    workflow.add_edge("execute_tools", "validate_tool_outputs")

    workflow.add_conditional_edges(
        "validate_tool_outputs",
        route_after_validation,
        {
            "retry_handler": "retry_handler",
            "store_evidence": "store_evidence",
        },
    )

    workflow.add_conditional_edges(
        "retry_handler",
        route_after_retry,
        {
            "execute_tools": "execute_tools",
            "reflect_and_revise": "reflect_and_revise",
        },
    )

    workflow.add_edge("store_evidence", "reflect_and_revise")

    workflow.add_conditional_edges(
        "reflect_and_revise",
        route_after_reflection,
        {
            "select_tools": "select_tools",
            "critique_findings": "critique_findings",
        },
    )

    workflow.add_conditional_edges(
        "critique_findings",
        route_after_critique,
        {
            "select_tools": "select_tools",
            "generate_report": "generate_report",
        },
    )

    workflow.add_edge("generate_report", "finish_task")
    workflow.add_edge("finish_task", END)

    saver = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=saver)
