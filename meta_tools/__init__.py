"""AIRA Meta-Tools: tools that act on the agent's own process.

Execution tools (``tools/``) act on the outside world: search, read PDFs,
retrieve passages. Meta-tools act on the agent itself:

* ``planner``: build the phase plan (to-do list).
* ``plan_updater``: check off, block, re-open and revise plan phases.
* ``ask_user``: detect missing information or unacceptable risk.
* ``critic``: review the gathered evidence before reporting.
* ``finish``: declare how the run ended.

They are pure functions. The LangGraph nodes in ``agents/nodes.py`` call them,
then log the decision and update the graph state.
"""
from meta_tools.ask_user import GENERIC_QUERY_TERMS, RISK_PATTERN, assess_request
from meta_tools.critic import REQUIRED_PHASES, critique_research
from meta_tools.finish import determine_stopping_reason
from meta_tools.plan_updater import (
    reopen_phase,
    resolve_phase,
    revise_plan_for_expanded_search,
    update_plan_task,
)
from meta_tools.planner import PHASE_DESCRIPTIONS, PHASES, RESOLVED_STATUSES, build_research_plan

__all__ = [
    "GENERIC_QUERY_TERMS",
    "PHASES",
    "PHASE_DESCRIPTIONS",
    "REQUIRED_PHASES",
    "RESOLVED_STATUSES",
    "RISK_PATTERN",
    "assess_request",
    "build_research_plan",
    "critique_research",
    "determine_stopping_reason",
    "reopen_phase",
    "resolve_phase",
    "revise_plan_for_expanded_search",
    "update_plan_task",
]
