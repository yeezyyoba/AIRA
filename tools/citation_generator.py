"""Tool 7: Citation Generator (Category: Function Calling).

Generates verified APA 7, IEEE, and BibTeX citations without hallucination.
Extracts verified metadata and formats academic references deterministically.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from validation.schemas import PaperMetadataSchema


class CitationStyle(str, Enum):
    APA7 = "APA7"
    IEEE = "IEEE"
    BIBTEX = "BIBTEX"


class CitationResultSchema(BaseModel):
    paper_id: str
    style: CitationStyle
    citation_text: str
    doi: Optional[str] = None
    verification_status: Optional[str] = None


def _format_author_apa(author: str) -> str:
    """Format 'First Last' or 'Last, First' as 'Last, F.'"""
    parts = author.strip().split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if "," in author:
        last, first = [p.strip() for p in author.split(",", 1)]
        initial = f"{first[0]}." if first else ""
        return f"{last}, {initial}".strip()
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{last}, {initials}".strip()


def _format_author_ieee(author: str) -> str:
    """Format 'First Last' as 'F. Last'"""
    parts = author.strip().split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if "," in author:
        last, first = [p.strip() for p in author.split(",", 1)]
        initial = f"{first[0]}." if first else ""
        return f"{initial} {last}".strip()
    last = parts[-1]
    initials = " ".join(f"{p[0]}." for p in parts[:-1] if p)
    return f"{initials} {last}".strip()


def generate_citation(
    paper: Union[PaperMetadataSchema, Dict[str, Any]],
    style: CitationStyle = CitationStyle.APA7,
) -> CitationResultSchema:
    """Generate a verified academic citation in APA 7, IEEE, or BibTeX format.

    Never fabricates missing DOIs, authors, publication years, or venues.
    """
    if hasattr(paper, "model_dump"):
        data = paper.model_dump()
    elif isinstance(paper, dict):
        data = paper
    else:
        data = {
            "paper_id": getattr(paper, "paper_id", "unknown"),
            "title": getattr(paper, "title", "Untitled Document"),
            "authors": getattr(paper, "authors", []),
            "year": getattr(paper, "year", None),
            "doi": getattr(paper, "doi", None),
        }

    pid = str(data.get("paper_id") or "paper")
    title = str(data.get("title") or "Untitled Document").strip().rstrip(".")
    authors_raw: List[str] = data.get("authors") or []
    year = data.get("year")
    year_str = str(year) if year else "n.d."
    doi = data.get("doi")
    venue = data.get("venue") or data.get("journal")

    doi_url = f"https://doi.org/{doi}" if doi and not str(doi).startswith("http") else (doi or "")

    if style == CitationStyle.APA7:
        if authors_raw:
            formatted_authors = [_format_author_apa(a) for a in authors_raw if a.strip()]
            if len(formatted_authors) == 1:
                authors_str = formatted_authors[0]
            elif len(formatted_authors) == 2:
                authors_str = f"{formatted_authors[0]} & {formatted_authors[1]}"
            else:
                authors_str = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
        else:
            authors_str = "Unknown Author"

        cit = f"{authors_str} ({year_str}). {title}."
        if venue:
            cit += f" {venue}."
        if doi_url:
            cit += f" {doi_url}"

    elif style == CitationStyle.IEEE:
        if authors_raw:
            formatted_authors = [_format_author_ieee(a) for a in authors_raw if a.strip()]
            if len(formatted_authors) == 1:
                authors_str = formatted_authors[0]
            elif len(formatted_authors) == 2:
                authors_str = f"{formatted_authors[0]} and {formatted_authors[1]}"
            else:
                authors_str = f"{formatted_authors[0]} et al."
        else:
            authors_str = "Anon."

        cit = f'{authors_str}, "{title},"'
        if venue:
            cit += f" {venue},"
        cit += f" {year_str}."
        if doi:
            cit += f" doi: {doi}."

    elif style == CitationStyle.BIBTEX:
        first_author_last = authors_raw[0].split()[-1].lower() if authors_raw else "paper"
        bib_key = f"{first_author_last}{year or ''}_{pid[:6]}"
        authors_bib = " and ".join(authors_raw) if authors_raw else "Unknown"

        lines = [
            f"@article{{{bib_key},",
            f"  title = {{{title}}},",
            f"  author = {{{authors_bib}}},",
            f"  year = {{{year_str}}},",
        ]
        if venue:
            lines.append(f"  journal = {{{venue}}},")
        if doi:
            lines.append(f"  doi = {{{doi}}},")
        lines.append("}")
        cit = "\n".join(lines)

    else:
        cit = f"{title} ({year_str})"

    return CitationResultSchema(
        paper_id=pid,
        style=style,
        citation_text=cit,
        doi=doi,
        verification_status=data.get("verification_status"),
    )
