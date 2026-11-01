"""Automated verification tests for Phase 1: Architecture and Project Setup (FastAPI Backend)."""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_required_directories_exist():
    """Verify that all required project directories are created."""
    required_dirs = [
        "app",
        "app/api",
        "app/schemas",
        "app/core",
        "agents",
        "llm",
        "tools",
        "meta_tools",
        "memory",
        "services",
        "validation",
        "logging_config",
        "data/papers",
        "data/processed",
        "data/reports",
        "data/chroma",
        "tests",
        "evaluation",
    ]
    for rel_dir in required_dirs:
        d = ROOT_DIR / rel_dir
        assert d.is_dir(), f"Required directory {rel_dir} is missing"


def test_required_files_exist():
    """Verify that all foundational files exist."""
    required_files = [
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
        "run.py",
        "app/main.py",
        "app/core/config.py",
        "logging_config/logger.py",
        "agents/state.py",
        "llm/provider_errors.py",
        "validation/schemas.py",
        "evaluation/test_cases.json",
        "evaluation/evaluation_report.md",
    ]
    for rel_file in required_files:
        f = ROOT_DIR / rel_file
        assert f.is_file(), f"Required file {rel_file} is missing"


def test_fastapi_root_endpoint():
    """Verify that root endpoint GET / returns 200 with API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "project" in data
    assert data["health"] == "/health"
    assert data["health_v1"] == "/api/v1/health"


def test_fastapi_health_endpoint():
    """Verify that GET /health returns 200, healthy status, and updated Groq model."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Phase" in data["phase"]
    assert "llm_router" in data["services"]
    assert "llm_primary" in data["services"]
    assert data["services"]["llm_fallback"] == "Groq (openai/gpt-oss-120b)"


def test_fastapi_api_v1_health():
    """Verify that GET /api/v1/health returns 200, healthy status, and updated Groq model."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Phase" in data["phase"]
    assert "llm_router" in data["services"]
    assert "llm_primary" in data["services"]
    assert data["services"]["llm_fallback"] == "Groq (openai/gpt-oss-120b)"

    # Verify both /health and /api/v1/health return consistent responses
    health_top_response = client.get("/health").json()
    assert data == health_top_response


def test_registered_routes_contain_both_health_endpoints():
    """Verify that route registration explicitly includes /health and /api/v1/health via OpenAPI schema."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    paths = schema.get("paths", {})
    assert "/health" in paths, "Expected '/health' in OpenAPI paths"
    assert "/api/v1/health" in paths, "Expected '/api/v1/health' in OpenAPI paths"
    assert "/" in paths, "Expected '/' in OpenAPI paths"
    assert "/api/v1/research/sessions" in paths, "Expected '/api/v1/research/sessions' in OpenAPI paths"
    assert "/api/v1/papers/" in paths, "Expected '/api/v1/papers/' in OpenAPI paths"
    assert "/api/v1/reports/" in paths, "Expected '/api/v1/reports/' in OpenAPI paths"


def test_api_v1_endpoints_http_behavior():
    """Verify actual HTTP behavior of registered API v1 endpoints."""
    # 1. Research sessions endpoint
    res_research = client.post("/api/v1/research/sessions", json={"query": "Amharic NER models"})
    assert res_research.status_code == 200
    assert res_research.json()["query"] == "Amharic NER models"

    # 2. Papers endpoint
    res_papers = client.get("/api/v1/papers/")
    assert res_papers.status_code == 200
    assert "papers" in res_papers.json()

    # 3. Reports endpoint
    res_reports = client.get("/api/v1/reports/")
    assert res_reports.status_code == 200
    assert "reports" in res_reports.json()


def test_logger_setup():
    """Verify that logging setup and get_logger work properly."""
    from logging_config.logger import get_logger, setup_logging
    logger = setup_logging()
    assert logger is not None
    named = get_logger("test")
    assert named.name == "aira.test"


def test_pydantic_schemas():
    """Verify foundational Pydantic schemas work."""
    from validation.schemas import ResearchQuerySchema, PaperMetadataSchema

    query = ResearchQuerySchema(query="Amharic sentiment analysis", year_from=2020)
    assert query.query == "Amharic sentiment analysis"
    assert query.year_from == 2020
    assert query.max_results == 20

    paper = PaperMetadataSchema(
        title="Sample Amharic Paper",
        authors=["Author One"],
        year=2023,
        source="OpenAlex",
    )
    assert paper.title == "Sample Amharic Paper"
    assert paper.source == "OpenAlex"


def test_provider_status_enum():
    """Verify LLM provider status enum definition."""
    from llm.provider_errors import ProviderStatus

    assert ProviderStatus.AVAILABLE == "AVAILABLE"
    assert ProviderStatus.RATE_LIMITED == "RATE_LIMITED"
    assert ProviderStatus.QUOTA_EXHAUSTED == "QUOTA_EXHAUSTED"
    assert ProviderStatus.TEMPORARILY_UNAVAILABLE == "TEMPORARILY_UNAVAILABLE"
    assert ProviderStatus.AUTHENTICATION_ERROR == "AUTHENTICATION_ERROR"
