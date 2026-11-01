"""AIRA - Amharic Intelligent Research Assistant
Main FastAPI Backend Application Entrypoint.
"""
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# Ensure root directory is on the Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.api import api_router
from app.core.config import settings
from app.schemas.health import HealthResponse
from logging_config.logger import get_logger, setup_logging

setup_logging()
logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown hooks."""
    logger.info("Initializing AIRA FastAPI Backend...")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Primary LLM: Gemini ({settings.GEMINI_MODEL})")
    logger.info(f"Fallback LLM: {settings.FALLBACK_PROVIDER} ({settings.GROQ_MODEL})")
    yield
    logger.info("Shutting down AIRA FastAPI Backend.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Autonomous Agentic Research Assistant for Low-Resource NLP (Backend API)",
    lifespan=lifespan,
)

# CORS middleware for Next.js / React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api.v1.endpoints.health import get_health


# Top-level direct health check endpoint (delegates to canonical implementation)
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """System health check endpoint (top-level alias for /api/v1/health)."""
    return get_health()


# Root endpoint
@app.get("/", tags=["Root"])
def root() -> dict:
    """API root providing metadata and documentation links."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "documentation": "/docs",
        "openapi": "/openapi.json",
        "health": "/health",
        "health_v1": f"{settings.API_V1_STR}/health",
    }


# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)
