"""Tests for Papers FastAPI Endpoints (Search & Validation).

Verifies OpenAPI schema, HTTP request handling, response parsing,
and 502 Bad Gateway when all search providers fail.
"""
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import pytest

from app.main import app
from tools.academic_search import AllProvidersFailedError
from validation.schemas import PaperMetadataSchema, PaperValidationResultSchema, ValidationStatus

client = TestClient(app)


def test_papers_endpoints_in_openapi():
    """Verify papers search and validation endpoints are registered in OpenAPI."""
    response = client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json().get("paths", {})
    assert "/api/v1/papers/search" in paths
    assert "/api/v1/papers/validate" in paths
    assert "/api/v1/papers/" in paths


@pytest.mark.anyio
async def test_search_papers_endpoint_success():
    """Verify POST /api/v1/papers/search returns 200 with structured response."""
    mock_search_return = {
        "query": "Amharic NER",
        "total_results": 1,
        "raw_results_count": 1,
        "duplicates_removed": 0,
        "sources_searched": ["openalex"],
        "provider_counts": {"openalex": 1},
        "provider_errors": {},
        "execution_time_seconds": 0.12,
        "papers": [
            {
                "id": "10.1000/1",
                "title": "Amharic NER with Transformers",
                "authors": ["Tesfaye Lemma"],
                "year": 2022,
                "doi": "10.1000/1",
                "abstract": "We build Amharic NER.",
                "venue": "ACL",
                "url": "https://doi.org/10.1000/1",
                "citation_count": 10,
                "is_open_access": True,
                "open_access_url": None,
                "source": "openalex",
                "sources": ["openalex"],
            }
        ],
    }

    with patch("app.api.v1.endpoints.papers.search_academic_papers", new_callable=AsyncMock) as mock_search:
        mock_search.return_value = mock_search_return

        response = client.post(
            "/api/v1/papers/search",
            json={
                "query": "Amharic NER",
                "year_from": 2020,
                "year_to": 2023,
                "sources": ["openalex"],
                "max_results": 10,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Amharic NER"
        assert data["total_results"] == 1
        assert data["papers"][0]["title"] == "Amharic NER with Transformers"


def test_search_papers_endpoint_all_providers_failed():
    """Verify POST /api/v1/papers/search returns 502 when all providers fail."""
    with patch("app.api.v1.endpoints.papers.search_academic_papers", new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = AllProvidersFailedError(
            "All providers down",
            errors={"openalex": "503", "semantic_scholar": "timeout", "crossref": "500"},
        )

        response = client.post(
            "/api/v1/papers/search",
            json={"query": "Failure Test"},
        )

        assert response.status_code == 502
        assert "All providers down" in str(response.json())


def test_validate_paper_endpoint_success():
    """Verify POST /api/v1/papers/validate returns 200 with structured validation result."""
    mock_val_result = PaperValidationResultSchema(
        status=ValidationStatus.VERIFIED,
        paper_id="10.1000/test",
        doi="10.1000/test",
        field_comparisons={},
        sources_checked=["crossref", "openalex"],
        explanation="Verified across Crossref and OpenAlex.",
        original_metadata={"title": "Test Title"},
    )

    with patch("app.api.v1.endpoints.papers.validate_paper_metadata", new_callable=AsyncMock) as mock_val:
        mock_val.return_value = mock_val_result

        response = client.post(
            "/api/v1/papers/validate",
            json={
                "paper": {
                    "title": "Test Title",
                    "doi": "10.1000/test",
                    "source": "manual",
                },
                "paper_id": "custom_id_123",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "VERIFIED"
        assert data["paper_id"] == "10.1000/test"
        assert "Verified across" in data["explanation"]
