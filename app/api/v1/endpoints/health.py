"""Health Check API Endpoint."""
from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse

from llm.model_router import model_router

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def get_health() -> HealthResponse:
    """Return system health status and active configuration."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        phase="Phase 9: Agentic research workflow with evaluation",
        services={
            "orchestration": "LangGraph (Configured)",
            "llm_router": f"Active: {model_router.get_active_provider_name()} (Degraded: {model_router.is_degraded_mode()})",
            "llm_primary": f"Gemini ({settings.GEMINI_MODEL})",
            "llm_fallback": f"{settings.FALLBACK_PROVIDER.capitalize()} ({settings.GROQ_MODEL})",
            "llm_keys": (
                f"gemini={'set' if settings.GEMINI_API_KEY else 'missing'}, "
                f"groq={'set' if settings.GROQ_API_KEY else 'missing'}"
            ),
            "session_store": "In-memory LangGraph checkpointer (sessions do not survive a restart)",
            "report_store": f"JSON/Markdown files in {settings.REPORTS_DIRECTORY}",
            "paper_store": f"JSON files in {settings.PROCESSED_DIRECTORY}",
            "vector_store": f"ChromaDB at {settings.CHROMA_PERSIST_DIRECTORY} (embeddings: {settings.EMBEDDING_PROVIDER})",
        },
    )
