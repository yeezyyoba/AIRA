"""Meta-Tool: Ask User for Clarification.

Checks a research request before any tool runs and decides whether the agent
can proceed, must ask the user for missing information, or must refuse.

Two stopping rules live here:

* **Essential information missing**: the topic is empty or too broad
  (for example just "papers"). The agent pauses and asks a question.
* **Unacceptable risk**: the request tries to override the agent's
  instructions, leak secrets, delete data, or fabricate citations.
* **Out of scope**: AIRA is a *language* research assistant (NLP, speech,
  translation, Amharic and other low-resource languages). Requests with no
  language-research angle (football trivia, jokes, cover letters) are refused
  before any tool runs; AI/ML topics without a language angle get a
  clarification question instead of a refusal.

The scope check is deliberately keyword based: it is free, instant, works when
the LLM providers are down, and its decisions are easy to inspect and test.

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

# Evidence that a request is about language research. Matching any one is
# enough to be in scope.
#   * whole words: short terms must match exactly ("wer" must not match "were",
#     "text" must not match "texture");
#   * word-start stems: "translat" matches "translation" and "translating";
#   * anywhere: a few stems that appear inside compounds ("LibriSpeech",
#     "multilingual", "tokenizer").
_WHOLE_WORDS = (
    r"nlp|nlu|nlg|language|languages|llms?|corpus|corpora|texts?|textual|words?|ner|asr|tts|rag|gpt|"
    r"ocr|glue|squad|wer|cer|bleu|rouge|bert|roberta|xlm|xlm-r|mbert|mt5|byt5|glove|pos"
)
_WORD_STEMS = (
    r"natural\s+language|sentiment|emotion|sarcasm|named\s+entit|part[-\s]of[-\s]speech|syntax|syntactic|"
    r"pars(?:er|ing)|translat|transliterat|spoken|dialect|question\s+answering|summari|"
    r"information\s+retrieval|retrieval[-\s]augmented|chatbot|conversational|dialog|embedding|word2vec|"
    r"fasttext|afro-?xlmr|afriberta|handwrit|spell|grammar|multilingual|cross[-\s]lingual|"
    r"low[-\s]resource|under[-\s]resourced|amharic|ge'?ez|ethiopic|tigrinya|tigre|oromo|afaan|somali|"
    r"swahili|hausa|yoruba|igbo|wolof|zulu|xhosa|ethiopian|african|afrisenti|masakhane|librispeech|common\s+voice"
)
_ANYWHERE = r"speech|lingu|tokeni[sz]|lexic|semantic|morpholog|phonet|phonem"
LANGUAGE_RESEARCH_PATTERN = re.compile(
    rf"(?i)(\b(?:{_WHOLE_WORDS})\b|\b(?:{_WORD_STEMS})|{_ANYWHERE})"
)

# AI/ML vocabulary without a language angle: close to AIRA's field, so the
# agent asks how the topic relates to language instead of refusing.
AI_RESEARCH_PATTERN = re.compile(
    r"(?i)\b(machine\s+learning|deep\s+learning|neural\s+network|transformers?|attention|cnns?|rnns?|"
    r"lstm|reinforcement\s+learning|computer\s+vision|vision|images?|artificial\s+intelligence|ai|"
    r"fine[-\s]?tun|pre[-\s]?train|classification|classifier|datasets?|benchmarks?|"
    r"models?|algorithms?|optimi[sz]ation|federated|generative|diffusion|gans?)\b"
)

SCOPE_DESCRIPTION = (
    "AIRA researches language technology: NLP, speech, machine translation, and "
    "Amharic and other low-resource languages"
)
SCOPE_EXAMPLE = "Amharic sentiment analysis with transformer models"


def assess_request(query: Optional[str], clarification: Optional[str] = None) -> Dict[str, Any]:
    """Decide whether a request can proceed.

    Args:
        query: The user's research request.
        clarification: The user's answer to an earlier clarification question.

    Returns:
        A dict with ``missing`` (list of missing-information messages),
        ``question`` (the clarification question, or ``None``), ``risks``
        (list of reasons the request must be refused) and ``out_of_scope``
        (the message for a request outside AIRA's research scope, or ``None``).
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

    out_of_scope: Optional[str] = None
    if not risks and not missing:
        if not LANGUAGE_RESEARCH_PATTERN.search(combined):
            if AI_RESEARCH_PATTERN.search(combined):
                missing.append("The request has no language-research angle.")
                question = (
                    f"{SCOPE_DESCRIPTION}. How does this topic relate to language? "
                    "For example, which language or NLP task should the research focus on?"
                )
            else:
                out_of_scope = (
                    f"{SCOPE_DESCRIPTION}, so this request is outside its scope. "
                    f"Try a research question such as: '{SCOPE_EXAMPLE}'."
                )

    return {"missing": missing, "question": question, "risks": risks, "out_of_scope": out_of_scope}
