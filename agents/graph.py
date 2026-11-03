"""AIRA LangGraph Stateful Graph Definition.

Wires the nodes (``agents/nodes.py``) together with the conditional edges
(``agents/routing.py``). The ``route_after_*`` functions are re-exported here
so existing imports such as ``from agents.graph import route_after_validation``
keep working.
"""

from typing import Any, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
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
from agents.routing import (
    route_after_check_information,
    route_after_critique,
    route_after_reflection,
    route_after_retry,
    route_after_select_tools,
    route_after_validation,
)
from agents.state import ResearchState
from llm.model_router import ModelRouter
from memory.checkpointing import create_checkpointer
from memory.vector_store import ChromaVectorStore
from services.paper_repository import PaperRepository

__all__ = [
    "build_research_graph",
    "route_after_check_information",
    "route_after_critique",
    "route_after_reflection",
    "route_after_retry",
    "route_after_select_tools",
    "route_after_validation",
]


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

    saver = checkpointer if checkpointer is not None else create_checkpointer()
    return workflow.compile(checkpointer=saver)
