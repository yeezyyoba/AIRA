"""Tool 1: Academic Paper Search (Category: Search).

Searches OpenAlex, Semantic Scholar, and Crossref asynchronously,
normalizes metadata, deduplicates by DOI and title+year, and provides
resilient fallback with partial failure preservation.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from logging_config.logger import get_logger
from services.academic_clients import (
    AcademicClientError,
    CrossrefClient,
    OpenAlexClient,
    SemanticScholarClient,
    normalize_doi,
    normalize_title,
)
from validation.schemas import PaperMetadataSchema

logger = get_logger("academic_search")


class AllProvidersFailedError(Exception):
    """Raised when all selected academic providers fail."""
    def __init__(self, message: str, errors: Dict[str, str]):
        super().__init__(message)
        self.errors = errors


def _merge_paper_records(
    primary: PaperMetadataSchema,
    secondary: PaperMetadataSchema,
) -> PaperMetadataSchema:
    """Merge metadata from two records representing the same paper without hallucination or silent overwriting."""
    merged_sources: List[str] = list(dict.fromkeys(primary.sources + secondary.sources))
    norm_doi = normalize_doi(primary.doi) or normalize_doi(secondary.doi)

    # 1. Preserve provider-specific metadata dictionaries
    merged_prov_meta: Dict[str, Dict[str, Any]] = {}
    if primary.provider_metadata:
        merged_prov_meta.update(primary.provider_metadata)
    else:
        merged_prov_meta[primary.source] = {
            "title": primary.title,
            "authors": primary.authors,
            "year": primary.year,
            "doi": primary.doi,
            "venue": primary.venue,
            "citation_count": primary.citation_count,
        }

    if secondary.provider_metadata:
        merged_prov_meta.update(secondary.provider_metadata)
    else:
        merged_prov_meta[secondary.source] = {
            "title": secondary.title,
            "authors": secondary.authors,
            "year": secondary.year,
            "doi": secondary.doi,
            "venue": secondary.venue,
            "citation_count": secondary.citation_count,
        }

    # 2. Track discrepancies between providers without silent overwrite
    conflicts: List[Dict[str, Any]] = list(primary.conflicts) + list(secondary.conflicts)

    # Title discrepancy check
    norm_p_title = normalize_title(primary.title)
    norm_s_title = normalize_title(secondary.title)
    if norm_p_title and norm_s_title and norm_p_title != norm_s_title:
        chosen_title = primary.title if len(primary.title) >= len(secondary.title) else secondary.title
        conflicts.append({
            "field": "title",
            "values": {primary.source: primary.title, secondary.source: secondary.title},
            "resolution": f"Preserved longest title: '{chosen_title}'",
        })
    else:
        chosen_title = primary.title or secondary.title

    # Year discrepancy check
    if primary.year is not None and secondary.year is not None and primary.year != secondary.year:
        conflicts.append({
            "field": "year",
            "values": {primary.source: primary.year, secondary.source: secondary.year},
            "resolution": f"Retained primary year {primary.year} from {primary.source} (secondary reported {secondary.year})",
        })

    # Authors discrepancy check
    if primary.authors and secondary.authors and primary.authors != secondary.authors:
        conflicts.append({
            "field": "authors",
            "values": {primary.source: primary.authors, secondary.source: secondary.authors},
            "resolution": f"Retained primary author list from {primary.source}",
        })

    # Venue discrepancy check
    if primary.venue and secondary.venue and primary.venue.lower() != secondary.venue.lower():
        conflicts.append({
            "field": "venue",
            "values": {primary.source: primary.venue, secondary.source: secondary.venue},
            "resolution": f"Retained primary venue '{primary.venue}' from {primary.source}",
        })

    return PaperMetadataSchema(
        id=norm_doi or primary.id or secondary.id,
        title=chosen_title,
        authors=primary.authors if primary.authors else secondary.authors,
        year=primary.year or secondary.year,
        doi=norm_doi,
        abstract=primary.abstract or secondary.abstract,
        venue=primary.venue or secondary.venue,
        url=primary.url or secondary.url,
        citation_count=(
            max(primary.citation_count or 0, secondary.citation_count or 0)
            if (primary.citation_count is not None or secondary.citation_count is not None)
            else None
        ),
        is_open_access=(
            primary.is_open_access if primary.is_open_access is not None else secondary.is_open_access
        ),
        open_access_url=primary.open_access_url or secondary.open_access_url,
        source=", ".join(merged_sources),
        sources=merged_sources,
        provider_metadata=merged_prov_meta,
        conflicts=conflicts,
    )


def deduplicate_papers(
    papers: List[PaperMetadataSchema],
) -> Tuple[List[PaperMetadataSchema], int]:
    """Deduplicate papers: 1st by normalized DOI, 2nd by normalized title + year."""
    doi_map: Dict[str, PaperMetadataSchema] = {}
    title_year_map: Dict[Tuple[str, Optional[int]], PaperMetadataSchema] = {}
    remaining_papers: List[PaperMetadataSchema] = []
    duplicate_count = 0

    for paper in papers:
        doi = normalize_doi(paper.doi)
        norm_title = normalize_title(paper.title)
        title_key = (norm_title, paper.year) if norm_title else None

        # 1. Match by normalized DOI
        if doi and doi in doi_map:
            existing = doi_map[doi]
            merged = _merge_paper_records(existing, paper)
            doi_map[doi] = merged
            if title_key:
                title_year_map[title_key] = merged
            duplicate_count += 1
            continue

        # 2. Match by normalized title + year (when title is meaningful)
        if title_key and len(norm_title) > 5 and title_key in title_year_map:
            existing = title_year_map[title_key]
            merged = _merge_paper_records(existing, paper)
            title_year_map[title_key] = merged
            if merged.doi:
                doi_map[merged.doi] = merged
            duplicate_count += 1
            continue

        # New unique paper
        if doi:
            doi_map[doi] = paper
        if title_key and len(norm_title) > 5:
            title_year_map[title_key] = paper
        remaining_papers.append(paper)

    # Reassemble in order of first appearance
    deduped: List[PaperMetadataSchema] = []
    seen_ids: Set[str] = set()

    for paper in remaining_papers:
        doi = normalize_doi(paper.doi)
        norm_title = normalize_title(paper.title)
        title_key = (norm_title, paper.year) if norm_title else None

        current = paper
        if doi and doi in doi_map:
            current = doi_map[doi]
        elif title_key and title_key in title_year_map:
            current = title_year_map[title_key]

        identifier = current.doi or (norm_title, current.year)
        if identifier not in seen_ids:
            seen_ids.add(identifier)
            deduped.append(current)

    return deduped, duplicate_count


async def search_academic_papers(
    query: str,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    sources: Optional[List[str]] = None,
    max_results: int = 20,
    openalex_client: Optional[OpenAlexClient] = None,
    semantic_scholar_client: Optional[SemanticScholarClient] = None,
    crossref_client: Optional[CrossrefClient] = None,
) -> Dict[str, Any]:
    """Execute asynchronous multi-provider academic search with deduplication and partial failure handling."""
    start_time = time.perf_counter()
    max_results = min(max_results or settings.MAX_SEARCH_RESULTS, 100)

    # Determine requested sources (default to all three)
    allowed_sources = {"openalex", "semantic_scholar", "crossref"}
    if sources:
        selected_sources = [s.lower().strip() for s in sources if s.lower().strip() in allowed_sources]
        if not selected_sources:
            selected_sources = ["openalex", "semantic_scholar", "crossref"]
    else:
        selected_sources = ["openalex", "semantic_scholar", "crossref"]

    logger.info(
        f"Initiating academic search | Query: '{query}' | Sources: {selected_sources} | "
        f"Years: {year_from or '*'}-{year_to or '*'} | Max: {max_results}"
    )

    oa_client = openalex_client or OpenAlexClient()
    s2_client = semantic_scholar_client or SemanticScholarClient()
    cr_client = crossref_client or CrossrefClient()

    tasks: Dict[str, Any] = {}
    if "openalex" in selected_sources:
        tasks["openalex"] = oa_client.search(query=query, year_from=year_from, year_to=year_to, max_results=max_results)
    if "semantic_scholar" in selected_sources:
        tasks["semantic_scholar"] = s2_client.search(query=query, year_from=year_from, year_to=year_to, max_results=max_results)
    if "crossref" in selected_sources:
        tasks["crossref"] = cr_client.search(query=query, year_from=year_from, year_to=year_to, max_results=max_results)

    # Run searches concurrently
    task_keys = list(tasks.keys())
    responses = await asyncio.gather(*tasks.values(), return_exceptions=True)

    collected_papers: List[PaperMetadataSchema] = []
    provider_results_count: Dict[str, int] = {}
    provider_errors: Dict[str, str] = {}

    for provider, resp in zip(task_keys, responses):
        if isinstance(resp, Exception):
            err_msg = str(resp)
            logger.error(f"Provider {provider} failed during search: {err_msg}")
            provider_errors[provider] = err_msg
            provider_results_count[provider] = 0
        elif isinstance(resp, list):
            collected_papers.extend(resp)
            provider_results_count[provider] = len(resp)
            logger.info(f"Provider {provider} returned {len(resp)} raw records")
        else:
            provider_results_count[provider] = 0

    # Total failure detection
    if len(provider_errors) == len(task_keys) and not collected_papers:
        duration = time.perf_counter() - start_time
        logger.critical(f"All academic providers failed for query '{query}' in {duration:.2f}s: {provider_errors}")
        raise AllProvidersFailedError(
            f"All academic search providers failed for query '{query}'. Errors: {provider_errors}",
            errors=provider_errors,
        )

    # Deduplicate results
    deduped_papers, duplicates_removed = deduplicate_papers(collected_papers)

    # Sort deterministically (prefer citation count if available, then year descending)
    deduped_papers.sort(
        key=lambda p: (p.citation_count or 0, p.year or 0),
        reverse=True,
    )
    final_papers = deduped_papers[:max_results]
    duration = time.perf_counter() - start_time

    logger.info(
        f"Search completed in {duration:.2f}s | Raw: {len(collected_papers)} | "
        f"Duplicates Removed: {duplicates_removed} | Final: {len(final_papers)} | "
        f"Failures: {list(provider_errors.keys()) if provider_errors else 'None'}"
    )

    return {
        "query": query,
        "total_results": len(final_papers),
        "raw_results_count": len(collected_papers),
        "duplicates_removed": duplicates_removed,
        "sources_searched": selected_sources,
        "provider_counts": provider_results_count,
        "provider_errors": provider_errors,
        "execution_time_seconds": round(duration, 3),
        "papers": [p.model_dump() for p in final_papers],
    }
