"""AIRA LangGraph Checkpointer.

LangGraph saves the research state after every node so a paused session (for
example one waiting for the user's clarification) can be resumed by
``session_id``. AIRA uses the in-process ``MemorySaver``: checkpoints live as
long as the backend process. A persistent (e.g. SQLite) checkpointer can be
swapped in here without touching the graph.
"""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer() -> BaseCheckpointSaver:
    """Return the checkpointer used when the caller does not supply one."""
    return MemorySaver()
