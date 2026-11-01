"""Tool 5: Paper Metadata Validator (Category: Function Calling).

Verifies DOIs, titles, authors, and publication dates against authoritative
academic registries (Crossref, OpenAlex, Semantic Scholar).
Statuses: VERIFIED, PARTIALLY_VERIFIED, CONFLICTING, UNVERIFIED.
Never invents data or silently overwrites input metadata.
"""
import difflib
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from logging_config.logger import get_logger
from services.academic_clients import (
    AcademicNotFoundError,
    CrossrefClient,
    OpenAlexClient,
    SemanticScholarClient,
    normalize_doi,
    normalize_title,
)
from validation.schemas import (
    FieldComparisonSchema,
    PaperMetadataSchema,
    PaperValidationResultSchema,
    ValidationStatus,
)

logger = get_logger("metadata_validator")


def _compare_titles(input_title: str, reg_title: str) -> Tuple[bool, float, str]:
    """Compare normalized titles returning (is_match, similarity_ratio, details)."""
    norm_in = normalize_title(input_title)
    norm_reg = normalize_title(reg_title)

    if not norm_in or not norm_reg:
        return False, 0.0, "Missing title in comparison"

    if norm_in == norm_reg:
        return True, 1.0, "Exact normalized title match"

    ratio = difflib.SequenceMatcher(None, norm_in, norm_reg).ratio()
    # Check token overlap for subtitles or minor word reorderings
    tokens_in = set(norm_in.split())
    tokens_reg = set(norm_reg.split())
    if tokens_in and tokens_reg:
        token_jaccard = len(tokens_in & tokens_reg) / len(tokens_in | tokens_reg)
    else:
        token_jaccard = 0.0

    effective_sim = max(ratio, token_jaccard)
    if effective_sim >= 0.80:
        return True, round(effective_sim, 2), f"High title similarity ({effective_sim:.2f})"
    elif effective_sim >= 0.50:
        return False, round(effective_sim, 2), f"Moderate title similarity ({effective_sim:.2f}) - potential revision or subtitle mismatch"
    else:
        return False, round(effective_sim, 2), f"Low title similarity ({effective_sim:.2f})"


def _parse_author_entity(author_str: str) -> Dict[str, str]:
    """Parse author string into normalized surname and given initials.

    Handles:
    - 'Surname, Given Name' -> surname: 'surname', initials: 'g'
    - 'Given Name Surname' -> surname: 'surname', initials: 'g'
    - 'Surname' -> surname: 'surname', initials: ''
    - 'F. Last' -> surname: 'last', initials: 'f'
    """
    cleaned = re.sub(r"[^\w\s,-]", "", author_str).strip()
    if not cleaned:
        return {"surname": "", "initials": "", "raw": author_str}

    if "," in cleaned:
        parts = cleaned.split(",", 1)
        surname = parts[0].strip().lower()
        given_parts = parts[1].strip().split()
        initials = "".join(p[0].lower() for p in given_parts if p)
    else:
        tokens = cleaned.split()
        if len(tokens) == 1:
            surname = tokens[0].lower()
            initials = ""
        else:
            surname = tokens[-1].lower()
            initials = "".join(t[0].lower() for t in tokens[:-1] if t)

    return {"surname": surname, "initials": initials, "raw": author_str}


def _match_author_entities(
    in_author: Dict[str, str],
    reg_author: Dict[str, str],
) -> Tuple[bool, bool, str]:
    """Compare two author entities.

    Returns (is_match, is_conflict, reason).
    """
    s1, s2 = in_author["surname"], reg_author["surname"]
    if not s1 or not s2:
        return False, False, "Missing surname"

    surname_ratio = difflib.SequenceMatcher(None, s1, s2).ratio()
    if surname_ratio < 0.85:
        return False, False, f"Different surnames ('{s1}' vs '{s2}')"

    # Surnames match; verify given name initials if available
    init1 = in_author["initials"]
    init2 = reg_author["initials"]

    if init1 and init2:
        # If both provide initials, first initials must match
        if init1[0] != init2[0]:
            return False, True, f"Conflicting initial for surname '{s1}': '{init1[0]}' vs '{init2[0]}'"
        return True, False, f"Strong match: '{s1}' with matching initial '{init1[0]}'"

    # One or both lack initials -> weak surname match
    return True, False, f"Surname match without initial corroboration: '{s1}'"


def _compare_authors(input_authors: List[str], reg_authors: List[str]) -> Tuple[bool, str]:
    """Compare author lists ensuring surname overlap alone cannot establish identity."""
    if not input_authors and not reg_authors:
        return True, "Both sources report no authors"
    if not input_authors or not reg_authors:
        return False, "One source reports no authors"

    in_entities = [_parse_author_entity(a) for a in input_authors if a.strip()]
    reg_entities = [_parse_author_entity(a) for a in reg_authors if a.strip()]

    if not in_entities or not reg_entities:
        return False, "Unable to extract author entities"

    # Greedy bipartite matching
    matched_in_indices: Set[int] = set()
    matched_reg_indices: Set[int] = set()
    matches_details: List[str] = []
    conflict_details: List[str] = []
    strong_matches = 0

    for i, a_in in enumerate(in_entities):
        best_match_idx = None
        best_is_strong = False
        best_reason = ""

        for j, a_reg in enumerate(reg_entities):
            if j in matched_reg_indices:
                continue
            is_match, is_conflict, reason = _match_author_entities(a_in, a_reg)
            if is_conflict:
                conflict_details.append(reason)
            elif is_match:
                best_match_idx = j
                best_is_strong = ("Strong match" in reason)
                best_reason = reason
                if best_is_strong:
                    break

        if best_match_idx is not None:
            matched_in_indices.add(i)
            matched_reg_indices.add(best_match_idx)
            matches_details.append(best_reason)
            if best_is_strong:
                strong_matches += 1

    # Conflict check: If there are initial conflicts and no strong matches to override
    if conflict_details and strong_matches == 0:
        return False, f"Author conflict: {'; '.join(conflict_details)}"

    matched_count = len(matched_in_indices)
    total_distinct = max(len(in_entities), len(reg_entities))
    concordance = matched_count / total_distinct

    if concordance >= 0.50:
        has_initials_available = any(e["initials"] for e in in_entities) and any(e["initials"] for e in reg_entities)
        if has_initials_available and strong_matches == 0:
            return False, f"Unreliable author match: Surnames overlap ({matched_count}/{total_distinct}) but none have matching initials"
        matched_surnames = [in_entities[idx]["surname"] for idx in matched_in_indices]
        return True, f"Author concordance: {matched_count}/{total_distinct} authors matched ({strong_matches} strong with initials: {', '.join(sorted(matched_surnames))})"
    else:
        return False, f"Low author concordance: only {matched_count} of {total_distinct} authors matched"


def _compare_years(input_year: Optional[int], reg_year: Optional[int]) -> Tuple[bool, str]:
    """Compare publication years with tolerance for preprint vs publisher date."""
    if input_year is None or reg_year is None:
        return True, "One or both years not reported"
    if input_year == reg_year:
        return True, f"Exact publication year match ({input_year})"
    diff = abs(input_year - reg_year)
    if diff == 1:
        return True, f"Year within 1-year delta ({input_year} vs {reg_year}, typical preprint/print offset)"
    return False, f"Year mismatch: input {input_year} vs registry {reg_year} (difference {diff} years)"


async def validate_paper_metadata(
    paper: Union[PaperMetadataSchema, Dict[str, Any]],
    paper_id: Optional[str] = None,
    openalex_client: Optional[OpenAlexClient] = None,
    crossref_client: Optional[CrossrefClient] = None,
    semantic_scholar_client: Optional[SemanticScholarClient] = None,
) -> PaperValidationResultSchema:
    """Cross-verify paper metadata against Crossref, OpenAlex, and Semantic Scholar."""
    start_time = time.perf_counter()

    if isinstance(paper, dict):
        input_data = paper.copy()
    else:
        input_data = paper.model_dump()

    input_title = input_data.get("title", "")
    input_authors = input_data.get("authors", []) or []
    input_year = input_data.get("year")
    raw_doi = input_data.get("doi")
    norm_in_doi = normalize_doi(raw_doi)
    effective_id = paper_id or norm_in_doi or input_data.get("id") or "unknown_paper"

    logger.info(f"Validating paper metadata | ID: {effective_id} | Title: '{input_title[:40]}...' | DOI: {norm_in_doi}")

    oa_client = openalex_client or OpenAlexClient()
    cr_client = crossref_client or CrossrefClient()
    s2_client = semantic_scholar_client or SemanticScholarClient()

    registry_records: Dict[str, PaperMetadataSchema] = {}
    sources_checked: List[str] = []

    # 1. Fetch authoritative records by DOI if DOI is present
    if norm_in_doi:
        sources_checked.extend(["crossref", "openalex", "semantic_scholar"])
        try:
            cr_rec = await cr_client.get_by_doi(norm_in_doi)
            if cr_rec:
                registry_records["crossref"] = cr_rec
        except Exception as e:
            logger.warning(f"Crossref lookup failed for DOI {norm_in_doi}: {e}")

        try:
            oa_rec = await oa_client.get_by_doi(norm_in_doi)
            if oa_rec:
                registry_records["openalex"] = oa_rec
        except Exception as e:
            logger.warning(f"OpenAlex lookup failed for DOI {norm_in_doi}: {e}")

        try:
            s2_rec = await s2_client.get_by_doi(norm_in_doi)
            if s2_rec:
                registry_records["semantic_scholar"] = s2_rec
        except Exception as e:
            logger.warning(f"Semantic Scholar lookup failed for DOI {norm_in_doi}: {e}")

    # 2. If no DOI or no records found by DOI, search registries by title
    if not registry_records and input_title:
        if "crossref" not in sources_checked:
            sources_checked.append("crossref")
        try:
            cr_search = await cr_client.search(query=input_title, max_results=3)
            for cand in cr_search:
                is_match, ratio, _ = _compare_titles(input_title, cand.title)
                if is_match and ratio >= 0.85:
                    registry_records["crossref"] = cand
                    break
        except Exception as e:
            logger.warning(f"Crossref title fallback search failed: {e}")

        if "openalex" not in sources_checked:
            sources_checked.append("openalex")
        try:
            oa_search = await oa_client.search(query=input_title, max_results=3)
            for cand in oa_search:
                is_match, ratio, _ = _compare_titles(input_title, cand.title)
                if is_match and ratio >= 0.85:
                    registry_records["openalex"] = cand
                    break
        except Exception as e:
            logger.warning(f"OpenAlex title fallback search failed: {e}")

    # 3. Handle UNVERIFIED case (no records located in any registry)
    if not registry_records:
        duration = time.perf_counter() - start_time
        logger.info(f"Validation UNVERIFIED for '{effective_id}' in {duration:.2f}s - no external registry records found")
        return PaperValidationResultSchema(
            status=ValidationStatus.UNVERIFIED,
            paper_id=str(effective_id),
            doi=norm_in_doi,
            field_comparisons={},
            sources_checked=sources_checked,
            explanation=(
                f"No matching records found in checked registries ({', '.join(sources_checked)}). "
                "This does NOT imply the paper metadata is false; the paper may be very recent, unindexed, or proprietary."
            ),
            original_metadata=input_data,
        )

    # 4. Compare fields against retrieved registry records
    field_comparisons: Dict[str, FieldComparisonSchema] = {}

    # DOI Comparison
    source_dois = {src: rec.doi for src, rec in registry_records.items() if rec.doi}
    if norm_in_doi:
        doi_matches = all(d == norm_in_doi for d in source_dois.values()) if source_dois else False
        field_comparisons["doi"] = FieldComparisonSchema(
            field_name="doi",
            input_value=norm_in_doi,
            source_values=source_dois,
            is_match=doi_matches,
            details=f"DOI matches in {len(source_dois)} registries" if doi_matches else "DOI mismatch in external registry",
        )
    elif source_dois:
        # Input had no DOI, but registry discovered one
        found_doi = next(iter(source_dois.values()))
        field_comparisons["doi"] = FieldComparisonSchema(
            field_name="doi",
            input_value=None,
            source_values=source_dois,
            is_match=True,
            details=f"Discovered authoritative DOI: {found_doi}",
        )

    # Title Comparison
    source_titles = {src: rec.title for src, rec in registry_records.items() if rec.title}
    title_matches: List[bool] = []
    title_details_list: List[str] = []
    conflict_detected = False

    for src, reg_t in source_titles.items():
        is_match, ratio, det = _compare_titles(input_title, reg_t)
        title_matches.append(is_match)
        title_details_list.append(f"{src}: {det}")
        if ratio < 0.40 and norm_in_doi:
            # High-confidence DOI resolves to completely different paper title!
            conflict_detected = True

    field_comparisons["title"] = FieldComparisonSchema(
        field_name="title",
        input_value=input_title,
        source_values=source_titles,
        is_match=any(title_matches) and not conflict_detected,
        details="; ".join(title_details_list),
    )

    # Author Comparison
    source_authors = {src: rec.authors for src, rec in registry_records.items() if rec.authors}
    author_matches: List[bool] = []
    author_details_list: List[str] = []

    for src, reg_a in source_authors.items():
        is_match, det = _compare_authors(input_authors, reg_a)
        author_matches.append(is_match)
        author_details_list.append(f"{src}: {det}")

    field_comparisons["authors"] = FieldComparisonSchema(
        field_name="authors",
        input_value=input_authors,
        source_values=source_authors,
        is_match=any(author_matches) if source_authors else None,
        details="; ".join(author_details_list) if author_details_list else "No author data in registries",
    )

    # Year Comparison
    source_years = {src: rec.year for src, rec in registry_records.items() if rec.year is not None}
    year_matches: List[bool] = []
    year_details_list: List[str] = []

    for src, reg_y in source_years.items():
        is_match, det = _compare_years(input_year, reg_y)
        year_matches.append(is_match)
        year_details_list.append(f"{src}: {det}")

    field_comparisons["year"] = FieldComparisonSchema(
        field_name="year",
        input_value=input_year,
        source_values=source_years,
        is_match=all(year_matches) if year_matches else None,
        details="; ".join(year_details_list) if year_details_list else "No year data in registries",
    )

    # 5. Determine overall ValidationStatus
    duration = time.perf_counter() - start_time
    found_sources = list(registry_records.keys())

    title_comp = field_comparisons.get("title")
    authors_comp = field_comparisons.get("authors")
    year_comp = field_comparisons.get("year")
    doi_comp = field_comparisons.get("doi")

    title_ok = title_comp is not None and title_comp.is_match is True
    authors_ok = authors_comp is None or authors_comp.is_match is not False
    year_ok = year_comp is None or year_comp.is_match is not False
    doi_ok = doi_comp is None or doi_comp.is_match is not False

    if conflict_detected or (title_comp and title_comp.is_match is False) or (authors_comp and authors_comp.is_match is False):
        status = ValidationStatus.CONFLICTING
        discrepancies: List[str] = []
        if title_comp and not title_comp.is_match:
            discrepancies.append(f"Title: {title_comp.details}")
        if authors_comp and authors_comp.is_match is False:
            discrepancies.append(f"Authors: {authors_comp.details}")
        if doi_comp and doi_comp.is_match is False:
            discrepancies.append(f"DOI: {doi_comp.details}")
        explanation = (
            f"Conflict detected across authoritative registries ({', '.join(found_sources)}). "
            f"Discrepancies: {'; '.join(discrepancies) if discrepancies else 'Fundamental metadata mismatch'}"
        )
    elif len(found_sources) >= 2 and title_ok and authors_ok and year_ok and doi_ok:
        # Defensible standard: Requires independent confirmation from AT LEAST TWO registries
        status = ValidationStatus.VERIFIED
        explanation = (
            f"Successfully verified across {len(found_sources)} independent authoritative registries ({', '.join(found_sources)}). "
            "Title, DOI, author concordance, and publication date are independently corroborated."
        )
    elif len(found_sources) >= 1 and title_ok and authors_ok:
        # Single registry confirmation or partial data in secondary registry
        status = ValidationStatus.PARTIALLY_VERIFIED
        if len(found_sources) == 1:
            explanation = (
                f"Corroborated by single authoritative registry ({found_sources[0]}), but lacks independent cross-registry confirmation."
            )
        else:
            explanation = (
                f"Partially verified across ({', '.join(found_sources)}). Core title and author concordance match, "
                "but publication year or secondary fields require further confirmation."
            )
    else:
        status = ValidationStatus.CONFLICTING
        explanation = (
            f"Metadata conflicts across sources ({', '.join(found_sources)}). Details: {field_comparisons.get('title', FieldComparisonSchema(field_name='title')).details}"
        )

    logger.info(f"Validation completed in {duration:.2f}s | Status: {status} | Found: {found_sources}")

    return PaperValidationResultSchema(
        status=status,
        paper_id=str(effective_id),
        doi=norm_in_doi,
        field_comparisons=field_comparisons,
        sources_checked=sources_checked,
        explanation=explanation,
        original_metadata=input_data,
    )
