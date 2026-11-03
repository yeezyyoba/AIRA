"""Meta-Tool: Ask User for Clarification.

Checks a research request before any tool runs and decides whether the agent
can proceed, must ask the user for missing information, or must refuse.

Two stopping rules live here:

* **Essential information missing**: the topic is empty or too broad
  (for example just "papers"). The agent pauses and asks a question.
* **Unacceptable risk**: the request tries to override the agent's
  instructions, leak secrets, delete data, or fabricate citations.

``assess_request`` is pure; the ``check_information`` and ``ask_user`` nodes
log the outcome and pause the graph.
"""
import re
from typing import Any, Dict, List, Optional

# Requests that try to subvert the agent or ask for destructive / secret-leaking
# actions are refused outright (the "unacceptable risk" stopping rule).
RISK_PATTERN = re.compile(
    r"(?i)("
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|your)\s+(?:instructions|rules|prompts?)"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|your)\s+(?:instructions|rules)"
    r"|(?:reveal|print|show|leak|dump|exfiltrate)\s+(?:the\s+|your\s+)?(?:system\s+prompt|api[\s_-]*keys?|secrets?|passwords?|credentials|\.env)"
    r"|override\s+(?:the\s+)?system\s+prompt"
    r"|rm\s+-rf"
    r"|drop\s+(?:table|database)"
    r"|delete\s+(?:all\s+)?(?:files|the\s+database|everything)"
    r"|fabricate\s+(?:citations?|results|data|references)"
    r"|(?:invent|make\s+up)\s+(?:citations?|references|results)"
    r")"
)

# A request made only of one of these words is too broad to research.
GENERIC_QUERY_TERMS = {"research", "papers", "paper", "study", "analysis", "compare", "review", "topic"}


def assess_request(query: Optional[str], clarification: Optional[str] = None) -> Dict[str, Any]:
    """Decide whether a request can proceed.

    Args:
        query: The user's research request.
        clarification: The user's answer to an earlier clarification question.

    Returns:
        A dict with ``missing`` (list of missing-information messages),
        ``question`` (the clarification question, or ``None``) and ``risks``
        (list of reasons the request must be refused).
    """
    combined = f"{(query or '').strip()} {(clarification or '').strip()}".strip()
    clean = re.sub(r"[^\w\s]", "", combined).strip()
    missing: List[str] = []
    question: Optional[str] = None
    risks: List[str] = []

    risk_match = RISK_PATTERN.search(combined)
    if risk_match:
        risks.append(
            f"Request contains an instruction the agent must not follow: '{risk_match.group(0)}'."
        )
    elif len(clean) < 4:
        missing.append("A specific research topic is required.")
        question = "Please provide the specific research topic or question."
    elif clean.lower() in GENERIC_QUERY_TERMS:
        missing.append("The research scope is too broad.")
        question = "Which specific topic, models, datasets, or language should the research focus on?"

    return {"missing": missing, "question": question, "risks": risks}
