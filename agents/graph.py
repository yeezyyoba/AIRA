"""AIRA LangGraph Stateful Graph Definition.

Constructs and compiles the StateGraph workflow with conditional routing,
bounded retries, dynamic loops, and SQLite / Memory checkpointing.
"""
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
from llm.model_router import ModelRouter
from memory.vector_store import ChromaVectorStore
from services.paper_repository import PaperRepository


# ---------------------------------------------------------------------------
# Conditional Routing Functions
# ---------------------------------------------------------------------------

def route_after_check_information(state: ResearchState) -> str:
    """Route to ask_user if information is missing; otherwise proceed to planning."""
    if state.get("missing_information"):
        return "ask_user"
    return "create_plan"


def route_after_select_tools(state: ResearchState) -> str:
    """Route to tool execution if a tool is selected; otherwise reflect on plan."""
    if state.get("selected_tool"):
        return "execute_tools"
    return "reflect_and_revise"


def route_after_validation(state: ResearchState) -> str:
    """Route based on tool validation status: retry on failure, store evidence on success."""
    status = state.get("validation_status")
    if status in ("INVALID", "ERROR"):
        return "retry_handler"
    return "store_evidence"


def route_after_retry(state: ResearchState) -> str:
    """Route after retry decision: re-execute if retry requested, otherwise adapt plan."""
    if state.get("validation_status") == "RETRY_REQUESTED":
        return "execute_tools"
    return "reflect_and_revise"


def route_after_reflection(state: ResearchState) -> str:
    """Route after reflection: loop back for more evidence, or proceed to critique."""
    step = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)

    if step >= max_steps:
        return "critique_findings"

    completed = set(state.get("completed_tasks", []))
    plan = state.get("research_plan", [])

    # If all tasks in plan are completed or report already generated
    if state.get("final_report") or (plan and len(completed) >= len(plan)):
        return "critique_findings"

    # If tools are still needed to satisfy tasks
    return "select_tools"


def route_after_critique(state: ResearchState) -> str:
    """Route after critique: revise plan if needed and steps remain; otherwise generate report."""
    status = state.get("critique_status")
    step = state.get("step_count", 0)
    max_steps = state.get("max_steps", 15)

    if status == "NEEDS_REVISION" and step < max_steps:
        return "select_tools"
    return "generate_report"


# ---------------------------------------------------------------------------
# Graph Builder
# ---------------------------------------------------------------------------

def build_research_graph(
    checkpointer: Optional[BaseCheckpointSaver] = None,
    router: Optional[ModelRouter] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
) -> Any:
    """Build and compile the AIRA Research StateGraph workflow.

    Args:
        checkpointer: Checkpoint saver for state persistence (MemorySaver or SqliteSaver).
        router: ModelRouter instance for LLM reasoning and degraded mode.
        repository: PaperRepository instance for paper storage.
        vector_db: ChromaVectorStore instance for RAG retrieval.

    Returns:
        Compiled LangGraph Pregel executable workflow.
    """
    workflow = StateGraph(ResearchState)

    # 1. Register Nodes
    workflow.add_node("receive_request", receive_request)
    workflow.add_node("analyze_goal", analyze_goal)
    workflow.add_node("check_information", check_information)
    workflow.add_node("ask_user", ask_user)
    workflow.add_node("create_plan", create_plan)
    workflow.add_node("select_tools", select_tools)

    # Wrap async execution nodes with injected dependencies
    async def bound_execute_tools(state: ResearchState) -> Dict[str, Any]:
        return await execute_tools(state, router=router, repository=repository, vector_db=vector_db)

    async def bound_generate_report(state: ResearchState) -> Dict[str, Any]:
        return await generate_report_node(state, router=router, repository=repository, vector_db=vector_db)

    workflow.add_node("execute_tools", bound_execute_tools)
    workflow.add_node("validate_tool_outputs", validate_tool_outputs)
    workflow.add_node("retry_handler", retry_handler)
    workflow.add_node("store_evidence", store_evidence)
    workflow.add_node("reflect_and_revise", reflect_and_revise)
    workflow.add_node("critique_findings", critique_findings)
    workflow.add_node("generate_report", bound_generate_report)
    workflow.add_node("finish_task", finish_task)

    # 2. Register Edges
    workflow.add_edge(START, "receive_request")
    workflow.add_edge("receive_request", "analyze_goal")
    workflow.add_edge("analyze_goal", "check_information")

    # Branch after check_information
    workflow.add_conditional_edges(
        "check_information",
        route_after_check_information,
        {
            "ask_user": "ask_user",
            "create_plan": "create_plan",
        },
    )

    # User clarification pauses workflow
    workflow.add_edge("ask_user", "finish_task")

    # Planning flow
    workflow.add_edge("create_plan", "select_tools")

    # Branch after select_tools
    workflow.add_conditional_edges(
        "select_tools",
        route_after_select_tools,
        {
            "execute_tools": "execute_tools",
            "reflect_and_revise": "reflect_and_revise",
        },
    )

    # Tool execution to validation
    workflow.add_edge("execute_tools", "validate_tool_outputs")

    # Branch after validation
    workflow.add_conditional_edges(
        "validate_tool_outputs",
        route_after_validation,
        {
            "retry_handler": "retry_handler",
            "store_evidence": "store_evidence",
        },
    )

    # Branch after retry_handler
    workflow.add_conditional_edges(
        "retry_handler",
        route_after_retry,
        {
            "execute_tools": "execute_tools",
            "reflect_and_revise": "reflect_and_revise",
        },
    )

    # Evidence stored leads to reflection
    workflow.add_edge("store_evidence", "reflect_and_revise")

    # Branch after reflection
    workflow.add_conditional_edges(
        "reflect_and_revise",
        route_after_reflection,
        {
            "select_tools": "select_tools",
            "critique_findings": "critique_findings",
        },
    )

    # Branch after critique
    workflow.add_conditional_edges(
        "critique_findings",
        route_after_critique,
        {
            "select_tools": "select_tools",
            "generate_report": "generate_report",
        },
    )

    # Finalization
    workflow.add_edge("generate_report", "finish_task")
    workflow.add_edge("finish_task", END)

    # 3. Compile Graph with Checkpointing
    saver = checkpointer if checkpointer is not None else MemorySaver()
    compiled_app = workflow.compile(checkpointer=saver)
    return compiled_app
