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
        phase="Phase 2: LLM Provider Integration and Central Model Router",
        services={
            "orchestration": "LangGraph (Configured)",
            "llm_router": f"Active: {model_router.get_active_provider_name()} (Degraded: {model_router.is_degraded_mode()})",
            "llm_primary": f"Gemini ({settings.GEMINI_MODEL})",
            "llm_fallback": f"{settings.FALLBACK_PROVIDER.capitalize()} ({settings.GROQ_MODEL})",
            "database": "SQLite (Configured)",
            "vector_store": "ChromaDB (Configured)",
            "frontend": "Next.js / React (Planned separate component)",
        },
    )
