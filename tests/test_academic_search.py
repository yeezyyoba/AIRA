"""Offline Unit Tests for Academic Paper Search and Client Layer.

Tests all search providers, deduplication, error handling, rate limits,
missing metadata handling, and LLM independence using mocked HTTP responses.
Zero live API quota consumed.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest

from services.academic_clients import (
    AcademicAuthError,
    AcademicClientError,
    AcademicNotFoundError,
    AcademicRateLimitError,
    CrossrefClient,
    OpenAlexClient,
    SemanticScholarClient,
    normalize_doi,
    normalize_title,
    reconstruct_openalex_abstract,
)
from tools.academic_search import (
    AllProvidersFailedError,
    deduplicate_papers,
    search_academic_papers,
)
from validation.schemas import PaperMetadataSchema


def test_doi_normalization():
    """Verify DOI normalization strips URLs, prefixes, and punctuation."""
    assert normalize_doi("https://doi.org/10.1145/3411764.3445518") == "10.1145/3411764.3445518"
    assert normalize_doi("http://dx.doi.org/10.1145/3411764.3445518.") == "10.1145/3411764.3445518"
    assert normalize_doi("doi:10.1145/3411764.3445518;") == "10.1145/3411764.3445518"
    assert normalize_doi("  10.1145/3411764.3445518  ") == "10.1145/3411764.3445518"
    assert normalize_doi("not-a-doi") is None
    assert normalize_doi(None) is None


def test_title_normalization():
    """Verify title normalization for deduplication."""
    t1 = "Amharic sentiment analysis: A comparative study!"
    t2 = "amharic  sentiment analysis a comparative study."
    assert normalize_title(t1) == normalize_title(t2)
    assert normalize_title(None) == ""


def test_openalex_abstract_reconstruction():
    """Verify reconstruction of abstract from OpenAlex inverted index."""
    inverted = {
        "Amharic": [0],
        "NLP": [1],
        "is": [2],
        "advancing": [3],
        "rapidly": [4],
    }
    reconstructed = reconstruct_openalex_abstract(inverted)
    assert reconstructed == "Amharic NLP is advancing rapidly"
    assert reconstruct_openalex_abstract(None) is None
    assert reconstruct_openalex_abstract({}) is None


@pytest.mark.anyio
async def test_search_openalex_success():
    """Verify successful OpenAlex search with metadata extraction."""
    mock_response = httpx.Response(
        status_code=200,
        json={
            "results": [
                {
                    "id": "https://openalex.org/W12345",
                    "title": "Ethiopic Sentiment Classification with BiLSTM",
                    "authorships": [
                        {"author": {"display_name": "Abebe Bikila"}},
                        {"author": {"display_name": "Derartu Tulu"}},
                    ],
                    "publication_year": 2022,
                    "doi": "https://doi.org/10.1007/s10579-022-09600-1",
                    "abstract_inverted_index": {"We": [0], "evaluate": [1], "sentiment": [2]},
                    "primary_location": {
                        "source": {"display_name": "Language Resources and Evaluation"},
                        "landing_page_url": "https://doi.org/10.1007/s10579-022-09600-1",
                    },
                    "open_access": {"is_oa": True, "oa_url": "https://arxiv.org/pdf/2201.0001"},
                    "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2201.0001.pdf"},
                    "cited_by_count": 18,
                }
            ]
        },
        request=httpx.Request("GET", "https://api.openalex.org/works"),
    )

    client = OpenAlexClient()
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.return_value = mock_response

    papers = await client.search("Ethiopic Sentiment", max_results=5, client=mock_http)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Ethiopic Sentiment Classification with BiLSTM"
    assert p.authors == ["Abebe Bikila", "Derartu Tulu"]
    assert p.year == 2022
    assert p.doi == "10.1007/s10579-022-09600-1"
    assert p.abstract == "We evaluate sentiment"
    assert p.venue == "Language Resources and Evaluation"
    assert p.is_open_access is True
    assert p.citation_count == 18
    assert p.source == "openalex"
    assert p.provider_metadata["openalex"]["pdf_url"] == "https://arxiv.org/pdf/2201.0001.pdf"


@pytest.mark.anyio
async def test_search_semantic_scholar_success():
    """Verify successful Semantic Scholar search with metadata extraction."""
    mock_response = httpx.Response(
        status_code=200,
        json={
            "data": [
                {
                    "paperId": "s2_98765",
                    "title": "Low-Resource NLP for Ethiopian Languages",
                    "authors": [{"name": "Haile Gebrselassie"}],
                    "year": 2021,
                    "externalIds": {"DOI": "10.18653/v1/2021.findings-emnlp.45"},
                    "abstract": "Comprehensive survey on Ethiopian NLP.",
                    "venue": "Findings of EMNLP",
                    "url": "https://www.semanticscholar.org/paper/s2_98765",
                    "citationCount": 42,
                    "isOpenAccess": True,
                    "openAccessPdf": {"url": "https://aclanthology.org/2021.findings-emnlp.45.pdf"},
                }
            ]
        },
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
    )

    client = SemanticScholarClient()
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.return_value = mock_response

    papers = await client.search("Ethiopian Languages", max_results=5, client=mock_http)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Low-Resource NLP for Ethiopian Languages"
    assert p.authors == ["Haile Gebrselassie"]
    assert p.year == 2021
    assert p.doi == "10.18653/v1/2021.findings-emnlp.45"
    assert p.citation_count == 42
    assert p.source == "semantic_scholar"


@pytest.mark.anyio
async def test_search_crossref_success():
    """Verify successful Crossref search with metadata extraction."""
    mock_response = httpx.Response(
        status_code=200,
        json={
            "message": {
                "items": [
                    {
                        "title": ["Morphological Segmentation for Amharic"],
                        "author": [
                            {"given": "Kenenisa", "family": "Bekele"},
                            {"given": "Tirunesh", "family": "Dibaba"},
                        ],
                        "published-print": {"date-parts": [[2020, 5, 10]]},
                        "DOI": "10.1145/3383583.3398544",
                        "abstract": "<jats:p>In this paper we present morphology.</jats:p>",
                        "container-title": ["ACM Transactions on Asian and Low-Resource Language Information Processing"],
                        "URL": "https://doi.org/10.1145/3383583.3398544",
                        "link": [
                            {"URL": "https://example.org/article.html", "content-type": "text/html"},
                            {"URL": "https://example.org/paper.pdf", "content-type": "application/pdf"},
                        ],
                        "is-referenced-by-count": 12,
                    }
                ]
            }
        },
        request=httpx.Request("GET", "https://api.crossref.org/works"),
    )

    client = CrossrefClient()
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.return_value = mock_response

    papers = await client.search("Amharic Morphology", max_results=5, client=mock_http)
    assert len(papers) == 1
    p = papers[0]
    assert p.title == "Morphological Segmentation for Amharic"
    assert p.authors == ["Kenenisa Bekele", "Tirunesh Dibaba"]
    assert p.year == 2020
    assert p.doi == "10.1145/3383583.3398544"
    assert p.abstract == "In this paper we present morphology."
    assert "ACM Transactions" in (p.venue or "")
    assert p.citation_count == 12
    assert p.source == "crossref"
    assert p.provider_metadata["crossref"]["pdf_url"] == "https://example.org/paper.pdf"


def test_doi_based_deduplication():
    """Verify deduplication when two providers return the same DOI in different formats."""
    p1 = PaperMetadataSchema(
        title="Amharic Sentiment Analysis",
        authors=["Author One"],
        year=2021,
        doi="https://doi.org/10.1007/s10579-021-0001-x",
        abstract="Abstract from OpenAlex",
        source="openalex",
        sources=["openalex"],
    )
    p2 = PaperMetadataSchema(
        title="Amharic Sentiment Analysis",
        authors=["Author One", "Author Two"],
        year=2021,
        doi="10.1007/S10579-021-0001-X",
        venue="LRE Journal",
        citation_count=15,
        source="semantic_scholar",
        sources=["semantic_scholar"],
    )

    deduped, removed_count = deduplicate_papers([p1, p2])
    assert len(deduped) == 1
    assert removed_count == 1
    merged = deduped[0]
    assert merged.doi == "10.1007/s10579-021-0001-x"
    assert merged.abstract == "Abstract from OpenAlex"
    assert merged.venue == "LRE Journal"
    assert merged.citation_count == 15
    assert set(merged.sources) == {"openalex", "semantic_scholar"}


def test_title_year_based_deduplication():
    """Verify deduplication when DOI is absent but title and year match identically."""
    p1 = PaperMetadataSchema(
        title="Benchmark Dataset for Amharic Named Entity Recognition",
        authors=["Yonas Test"],
        year=2023,
        doi=None,
        abstract="Preprint abstract",
        source="openalex",
        sources=["openalex"],
    )
    p2 = PaperMetadataSchema(
        title="Benchmark Dataset for Amharic Named Entity Recognition!",
        authors=["Yonas Test"],
        year=2023,
        doi=None,
        citation_count=5,
        source="semantic_scholar",
        sources=["semantic_scholar"],
    )

    deduped, removed_count = deduplicate_papers([p1, p2])
    assert len(deduped) == 1
    assert removed_count == 1
    assert deduped[0].citation_count == 5
    assert set(deduped[0].sources) == {"openalex", "semantic_scholar"}


def test_no_invented_metadata():
    """Verify that absent fields remain None and are never fabricated."""
    p = PaperMetadataSchema(
        title="Minimal Paper",
        source="crossref",
        sources=["crossref"],
    )
    assert p.doi is None
    assert p.abstract is None
    assert p.venue is None
    assert p.year is None
    assert p.citation_count is None
    assert p.open_access_url is None


@pytest.mark.anyio
async def test_search_academic_papers_multi_provider_aggregation():
    """Verify search_academic_papers aggregates and deduplicates across mock providers."""
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.search = AsyncMock(
        return_value=[
            PaperMetadataSchema(
                title="Ethiopic NLP Survey",
                authors=["Almaz Ayana"],
                year=2022,
                doi="10.1000/182",
                source="openalex",
                sources=["openalex"],
            )
        ]
    )

    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.search = AsyncMock(
        return_value=[
            PaperMetadataSchema(
                title="Ethiopic NLP Survey",
                authors=["Almaz Ayana"],
                year=2022,
                doi="10.1000/182",
                citation_count=20,
                source="semantic_scholar",
                sources=["semantic_scholar"],
            ),
            PaperMetadataSchema(
                title="Tigrinya Part of Speech Tagging",
                authors=["Meseret Defar"],
                year=2021,
                doi="10.1000/183",
                source="semantic_scholar",
                sources=["semantic_scholar"],
            ),
        ]
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.search = AsyncMock(return_value=[])

    result = await search_academic_papers(
        query="Ethiopic NLP",
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
        crossref_client=mock_cr,
    )

    assert result["total_results"] == 2
    assert result["duplicates_removed"] == 1
    assert result["raw_results_count"] == 3
    assert result["provider_counts"]["openalex"] == 1
    assert result["provider_counts"]["semantic_scholar"] == 2
    assert result["provider_counts"]["crossref"] == 0


@pytest.mark.anyio
async def test_partial_provider_failure_returns_partial_results():
    """Verify that if one provider fails with an error, other providers' results are safely returned."""
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.search = AsyncMock(side_effect=AcademicClientError("503 Service Unavailable", provider="openalex"))

    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.search = AsyncMock(
        return_value=[
            PaperMetadataSchema(
                title="Surviving Paper",
                authors=["Resilient Researcher"],
                year=2023,
                doi="10.1000/999",
                source="semantic_scholar",
                sources=["semantic_scholar"],
            )
        ]
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.search = AsyncMock(return_value=[])

    result = await search_academic_papers(
        query="NLP Resilience",
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
        crossref_client=mock_cr,
    )

    assert result["total_results"] == 1
    assert "openalex" in result["provider_errors"]
    assert "503" in result["provider_errors"]["openalex"]
    assert result["papers"][0]["title"] == "Surviving Paper"


@pytest.mark.anyio
async def test_total_provider_failure_raises_all_failed_error():
    """Verify that when all providers fail, a clear AllProvidersFailedError is raised."""
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.search = AsyncMock(side_effect=AcademicClientError("OpenAlex down", provider="openalex"))

    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.search = AsyncMock(side_effect=AcademicClientError("S2 timeout", provider="semantic_scholar"))

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.search = AsyncMock(side_effect=AcademicClientError("Crossref 500", provider="crossref"))

    with pytest.raises(AllProvidersFailedError) as exc_info:
        await search_academic_papers(
            query="Complete Outage",
            openalex_client=mock_oa,
            semantic_scholar_client=mock_s2,
            crossref_client=mock_cr,
        )
    assert "openalex" in exc_info.value.errors
    assert "semantic_scholar" in exc_info.value.errors
    assert "crossref" in exc_info.value.errors


@pytest.mark.anyio
async def test_http_429_retry_handling():
    """Verify HTTP 429 triggers backoff and honors Retry-After header."""
    client = OpenAlexClient()
    client.initial_backoff = 0.01  # Fast test execution

    # First call: 429 with Retry-After 1s, second call: 200 OK
    resp_429 = httpx.Response(
        status_code=429,
        headers={"Retry-After": "1"},
        request=httpx.Request("GET", "https://api.openalex.org/works"),
    )
    resp_200 = httpx.Response(
        status_code=200,
        json={"results": []},
        request=httpx.Request("GET", "https://api.openalex.org/works"),
    )

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.side_effect = [resp_429, resp_200]

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        papers = await client.search("test", client=mock_http)
        assert papers == []
        assert mock_http.request.call_count == 2
        mock_sleep.assert_called_once()


@pytest.mark.anyio
async def test_http_auth_failure_immediate_no_retry():
    """Verify HTTP 401/403 fails immediately without wasteful retries."""
    client = SemanticScholarClient()
    client.max_retries = 3

    resp_401 = httpx.Response(
        status_code=401,
        text="Unauthorized API key",
        request=httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search"),
    )

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.request.return_value = resp_401

    with pytest.raises(AcademicAuthError):
        await client.search("test", client=mock_http)
    # Called exactly ONCE - not 3 times!
    assert mock_http.request.call_count == 1


@pytest.mark.anyio
async def test_tools_work_without_llm_availability():
    """Requirement: Academic search operates completely independently of Gemini and Groq."""
    from llm.model_router import ModelRouter
    from llm.provider_errors import DegradedModeError

    # Force router into degraded mode
    router = ModelRouter()
    router.degraded_mode = True
    assert router.is_degraded_mode()

    with pytest.raises(DegradedModeError):
        router.generate("Reasoning should fail in degraded mode")

    # Academic search must execute without error
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.search = AsyncMock(
        return_value=[
            PaperMetadataSchema(
                title="Independent Search Paper",
                source="openalex",
                sources=["openalex"],
            )
        ]
    )
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.search = AsyncMock(return_value=[])
    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.search = AsyncMock(return_value=[])

    res = await search_academic_papers(
        query="Amharic Morphology",
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
        crossref_client=mock_cr,
    )
    assert res["total_results"] == 1
    assert res["papers"][0]["title"] == "Independent Search Paper"


def test_merge_records_preserves_provider_metadata_and_exposes_conflicts():
    """Verify that merging paper records preserves individual provider data and exposes conflicts."""
    from tools.academic_search import _merge_paper_records

    p1 = PaperMetadataSchema(
        id="10.1000/test",
        title="Amharic Morphology with Neural Networks",
        authors=["Abebe Bikila"],
        year=2021,
        venue="Computational Linguistics",
        doi="10.1000/test",
        source="openalex",
        sources=["openalex"],
        provider_metadata={
            "openalex": {
                "title": "Amharic Morphology with Neural Networks",
                "year": 2021,
                "venue": "Computational Linguistics",
            }
        },
    )

    p2 = PaperMetadataSchema(
        id="10.1000/test",
        title="Amharic Morphology with Neural Networks: Extended Edition",
        authors=["Bikila, Abebe", "Tulu, Derartu"],
        year=2022,  # Discrepancy!
        venue="Comp Ling Journal",  # Discrepancy!
        doi="10.1000/test",
        source="crossref",
        sources=["crossref"],
        provider_metadata={
            "crossref": {
                "title": "Amharic Morphology with Neural Networks: Extended Edition",
                "year": 2022,
                "venue": "Comp Ling Journal",
            }
        },
    )

    merged = _merge_paper_records(p1, p2)

    # 1. Both provider records are preserved
    assert "openalex" in merged.provider_metadata
    assert "crossref" in merged.provider_metadata
    assert merged.provider_metadata["openalex"]["year"] == 2021
    assert merged.provider_metadata["crossref"]["year"] == 2022

    # 2. Conflicts are explicitly tracked, not silently overwritten
    conflict_fields = {c["field"] for c in merged.conflicts}
    assert "year" in conflict_fields
    assert "venue" in conflict_fields
    assert "title" in conflict_fields

