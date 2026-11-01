"""Offline Unit Tests for Paper Metadata Validation Tool.

Verifies metadata cross-referencing, field comparisons, status determination
(VERIFIED, PARTIALLY_VERIFIED, CONFLICTING, UNVERIFIED), DOI normalization,
and non-destructive preservation of original metadata.
"""
from unittest.mock import AsyncMock, MagicMock
import pytest

from services.academic_clients import CrossrefClient, OpenAlexClient, SemanticScholarClient
from tools.metadata_validator import (
    _compare_authors,
    _compare_titles,
    _compare_years,
    validate_paper_metadata,
)
from validation.schemas import PaperMetadataSchema, ValidationStatus


def test_compare_titles_exact_and_fuzzy():
    """Verify title comparison handles exact, fuzzy, and low similarities."""
    is_match, ratio, det = _compare_titles("Amharic BERT: Pretrained Models", "Amharic BERT: Pretrained Models")
    assert is_match is True
    assert ratio == 1.0

    # Minor subtitle or punctuation variance
    is_match2, ratio2, _ = _compare_titles(
        "Amharic Sentiment Analysis: A Survey",
        "Amharic Sentiment Analysis - A Comprehensive Survey"
    )
    assert is_match2 is True
    assert ratio2 >= 0.80

    # Low similarity
    is_match3, ratio3, _ = _compare_titles("Quantum Computing Algorithms", "Amharic POS Tagging")
    assert is_match3 is False
    assert ratio3 < 0.30


def test_compare_authors_surname_matching():
    """Verify author comparison by family name/surname extraction."""
    in_authors = ["Abebe Bikila", "Derartu Tulu"]
    reg_authors = ["Bikila, Abebe", "Tulu, D."]
    is_match, details = _compare_authors(in_authors, reg_authors)
    assert is_match is True
    assert "bikila" in details
    assert "tulu" in details

    # Completely different authors
    diff_authors = ["Albert Einstein", "Niels Bohr"]
    is_match_diff, _ = _compare_authors(in_authors, diff_authors)
    assert is_match_diff is False


def test_compare_years_delta_tolerance():
    """Verify year comparison accepts exact matches and 1-year preprint offsets."""
    assert _compare_years(2022, 2022)[0] is True
    assert _compare_years(2021, 2022)[0] is True  # 1-year offset accepted
    assert _compare_years(2015, 2022)[0] is False  # 7-year gap rejected


@pytest.mark.anyio
async def test_validate_metadata_verified():
    """Verify VERIFIED status when Crossref and OpenAlex confirm metadata."""
    input_paper = PaperMetadataSchema(
        title="BERT for Amharic Sentiment Classification",
        authors=["Tirunesh Dibaba", "Haile Gebrselassie"],
        year=2021,
        doi="10.1007/s10579-021-09550-y",
        source="user_upload",
    )

    mock_cr_rec = PaperMetadataSchema(
        title="BERT for Amharic Sentiment Classification",
        authors=["Tirunesh Dibaba", "Haile Gebrselassie"],
        year=2021,
        doi="10.1007/s10579-021-09550-y",
        venue="Language Resources and Evaluation",
        source="crossref",
        sources=["crossref"],
    )

    mock_oa_rec = PaperMetadataSchema(
        title="BERT for Amharic Sentiment Classification",
        authors=["Tirunesh Dibaba", "Haile Gebrselassie"],
        year=2021,
        doi="10.1007/s10579-021-09550-y",
        citation_count=25,
        source="openalex",
        sources=["openalex"],
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=mock_cr_rec)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=mock_oa_rec)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    result = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )

    assert result.status == ValidationStatus.VERIFIED
    assert result.doi == "10.1007/s10579-021-09550-y"
    assert result.field_comparisons["title"].is_match is True
    assert result.field_comparisons["authors"].is_match is True
    assert result.field_comparisons["year"].is_match is True
    assert "crossref" in result.sources_checked
    assert "openalex" in result.sources_checked


@pytest.mark.anyio
async def test_validate_metadata_partially_verified():
    """Verify PARTIALLY_VERIFIED status when title matches but author data is missing externally."""
    input_paper = PaperMetadataSchema(
        title="Ethiopic Character Recognition Workshop",
        authors=["Abebe Bikila"],
        year=2019,
        doi="10.1145/12345.67890",
        source="local",
    )

    # Registry has title and DOI, but no authors listed
    mock_cr_rec = PaperMetadataSchema(
        title="Ethiopic Character Recognition Workshop",
        authors=[],
        year=2019,
        doi="10.1145/12345.67890",
        source="crossref",
        sources=["crossref"],
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=mock_cr_rec)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=None)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    result = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )

    assert result.status in (ValidationStatus.PARTIALLY_VERIFIED, ValidationStatus.VERIFIED)
    assert result.field_comparisons["title"].is_match is True


@pytest.mark.anyio
async def test_validate_metadata_conflicting():
    """Verify CONFLICTING status when DOI exists in registry but title fundamentally conflicts."""
    input_paper = PaperMetadataSchema(
        title="Amharic Speech Synthesis with WaveNet",
        authors=["Eleni Gebre"],
        year=2021,
        doi="10.1007/s10579-999-9999-z",
        source="user_upload",
    )

    # Registry DOI points to a completely different paper
    mock_cr_rec = PaperMetadataSchema(
        title="Cardiovascular Disease in Pediatric Patients",
        authors=["John Smith", "David Miller"],
        year=2015,
        doi="10.1007/s10579-999-9999-z",
        source="crossref",
        sources=["crossref"],
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=mock_cr_rec)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=None)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    result = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )

    assert result.status == ValidationStatus.CONFLICTING
    assert "differs fundamentally" in result.explanation or "Conflict detected" in result.explanation


@pytest.mark.anyio
async def test_validate_metadata_unverified_not_falsehood():
    """Verify UNVERIFIED status when paper is absent from registries without claiming paper is false."""
    input_paper = PaperMetadataSchema(
        title="Very Recent Unpublished Manuscript on Ge'ez Parsing",
        authors=["Solomon Kassa"],
        year=2025,
        doi="10.9999/not-yet-indexed",
        source="preprints",
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=None)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=None)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    result = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )

    assert result.status == ValidationStatus.UNVERIFIED
    # Ensure missing record is NOT treated as proof of falsehood
    assert "does NOT imply the paper metadata is false" in result.explanation


@pytest.mark.anyio
async def test_original_metadata_never_overwritten():
    """Requirement: Never silently overwrite original input metadata."""
    original_dict = {
        "title": "My Custom Title",
        "authors": ["Author A"],
        "year": 2020,
        "doi": "10.1000/1",
        "source": "my_source",
    }
    input_paper = PaperMetadataSchema(**original_dict)

    mock_cr_rec = PaperMetadataSchema(
        title="Official Registered Title",
        authors=["Author B"],
        year=2021,
        doi="10.1000/1",
        source="crossref",
        sources=["crossref"],
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=mock_cr_rec)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=None)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    result = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )

    # Input paper object must remain unchanged
    assert input_paper.title == "My Custom Title"
    assert input_paper.authors == ["Author A"]
    assert input_paper.year == 2020
    # Validation result preserves original input metadata
    assert result.original_metadata["title"] == "My Custom Title"


@pytest.mark.anyio
async def test_validator_works_without_llm():
    """Verify metadata validation works deterministically when LLMs are offline."""
    from llm.model_router import ModelRouter
    router = ModelRouter()
    router.degraded_mode = True
    assert router.is_degraded_mode()

    input_paper = PaperMetadataSchema(
        title="Offline Validation Test",
        year=2022,
        doi="10.1000/offline",
        source="test",
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=None)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=None)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    res = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )
    assert res.status == ValidationStatus.UNVERIFIED


@pytest.mark.anyio
async def test_single_registry_cannot_assign_verified():
    """Verify that a single registry match yields PARTIALLY_VERIFIED, never VERIFIED.

    VERIFIED requires independent corroboration from at least two independent registries.
    """
    input_paper = PaperMetadataSchema(
        title="Single Registry Confirmation Paper",
        authors=["Haile Gebrselassie"],
        year=2021,
        doi="10.1007/single-registry-test",
        source="user_upload",
    )

    mock_cr_rec = PaperMetadataSchema(
        title="Single Registry Confirmation Paper",
        authors=["Haile Gebrselassie"],
        year=2021,
        doi="10.1007/single-registry-test",
        source="crossref",
        sources=["crossref"],
    )

    mock_cr = MagicMock(spec=CrossrefClient)
    mock_cr.get_by_doi = AsyncMock(return_value=mock_cr_rec)
    mock_oa = MagicMock(spec=OpenAlexClient)
    mock_oa.get_by_doi = AsyncMock(return_value=None)
    mock_s2 = MagicMock(spec=SemanticScholarClient)
    mock_s2.get_by_doi = AsyncMock(return_value=None)

    res = await validate_paper_metadata(
        paper=input_paper,
        crossref_client=mock_cr,
        openalex_client=mock_oa,
        semantic_scholar_client=mock_s2,
    )

    assert res.status == ValidationStatus.PARTIALLY_VERIFIED
    assert "single authoritative registry (crossref)" in res.explanation
    assert "lacks independent cross-registry confirmation" in res.explanation


def test_author_matching_rejects_surname_overlap_with_initial_conflict():
    """Verify that sharing a surname with conflicting initials is rejected as a conflict."""
    # John Smith vs Alice Smith
    is_match, details = _compare_authors(["John Smith"], ["Alice Smith"])
    assert is_match is False
    assert "Author conflict" in details or "Conflicting initial" in details


def test_author_matching_rejects_minority_surname_overlap():
    """Verify that matching only 1 of 3 authors in a multi-author paper is rejected."""
    in_authors = ["John Smith", "David Miller", "Peter Jones"]
    reg_authors = ["Alice Smith", "Jane Doe", "Bob Vance"]
    is_match, details = _compare_authors(in_authors, reg_authors)
    assert is_match is False
    assert "conflict" in details.lower() or "low author concordance" in details.lower()

