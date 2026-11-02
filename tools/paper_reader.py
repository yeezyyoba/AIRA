"""Tool 2: Research Paper Reader (Category: System & Environment).

Extracts and organizes academic paper dimensions from PDFs using PyMuPDF (primary)
and pdfplumber (fallback). Preserves 1-indexed page numbers, exact section headings,
and verbatim evidence passages. Treats all PDF content as untrusted input.

Section detection has two strategies:

1. Layout headings (primary): pymupdf4llm's layout analysis marks heading boxes from
   font size, weight and position. Boxes are put into two-column reading order from
   their coordinates, headings are accepted only if numbered ("4.1 Datasets") or a
   known section name, and numbering is used to verify the order. Section *text* is
   still taken from PyMuPDF's raw page text (clipped to each box), so excerpts,
   evidence passages and chunk page attribution stay verbatim with ``raw_text_by_page``.
2. Regex on plain text lines (fallback): used when no headings are found, when the
   layout pass fails or exceeds its time budget, or when heading numbering is still
   out of order after reordering (boundaries would be unreliable).
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pdfplumber
import pymupdf

from app.core.config import settings
from logging_config.logger import get_logger
from services.pdf_downloader import (
    InvalidPDFError,
    PDFDownloadError,
    PDFSizeExceededError,
    SSRFProtectionError,
    download_pdf_safely,
)
from validation.schemas import (
    EvidenceItemSchema,
    EvidenceType,
    ExtractedPaperDataSchema,
    ExtractionStatus,
    PaperExtractionResultSchema,
)

logger = get_logger("paper_reader")


def format_untrusted_content_block(text: str) -> str:
    """Format untrusted extracted paper text into an isolated structural XML container.

    Prevents prompt injection by structurally isolating academic text from system instructions.
    """
    cleaned = text.replace("\x00", "").replace("\r\n", "\n")
    return f"<untrusted_academic_content>\n{cleaned}\n</untrusted_academic_content>"

# Common academic section heading patterns
SECTION_REGEXES = {
    "abstract": re.compile(r"^(?:abstract|summary)\b[:\s\-]*", re.IGNORECASE),
    "introduction": re.compile(r"^(?:\d+[\.\s]+)?(?:introduction|overview|background)\b[:\s\-]*", re.IGNORECASE),
    "research_objectives": re.compile(r"^(?:\d+[\.\s]+)?(?:research\s+objectives?|goals?|aims?)\b[:\s\-]*", re.IGNORECASE),
    "research_questions": re.compile(r"^(?:\d+[\.\s]+)?(?:research\s+questions?|hypothes[ie]s)\b[:\s\-]*", re.IGNORECASE),
    "methodology": re.compile(r"^(?:\d+[\.\s]+)?(?:methodology|methods?|proposed\s+approach|architecture|system\s+model|framework)\b[:\s\-]*", re.IGNORECASE),
    "dataset": re.compile(r"^(?:\d+[\.\s]+)?(?:datasets?|corpus|data\s+collection|data\s+preparation|benchmark\s+data)\b[:\s\-]*", re.IGNORECASE),
    "models_and_algorithms": re.compile(r"^(?:\d+[\.\s]+)?(?:models?(?:\s+and\s+algorithms?)?|algorithmic\s+framework|neural\s+architecture)\b[:\s\-]*", re.IGNORECASE),
    "experimental_setup": re.compile(r"^(?:\d+[\.\s]+)?(?:experiment(?:s|al)?\s+(?:setup|design|settings?|environment))\b[:\s\-]*", re.IGNORECASE),
    "evaluation_metrics": re.compile(r"^(?:\d+[\.\s]+)?(?:evaluation\s+metrics?|performance\s+measures?)\b[:\s\-]*", re.IGNORECASE),
    "results": re.compile(r"^(?:\d+[\.\s]+)?(?:results?(?:\s+and\s+discussion)?|findings?|empirical\s+evaluation)\b[:\s\-]*", re.IGNORECASE),
    "limitations": re.compile(r"^(?:\d+[\.\s]+)?(?:limitations?|threats\s+to\s+validity)\b[:\s\-]*", re.IGNORECASE),
    "future_work": re.compile(r"^(?:\d+[\.\s]+)?(?:future\s+work|future\s+directions?)\b[:\s\-]*", re.IGNORECASE),
    "conclusion": re.compile(r"^(?:\d+[\.\s]+)?(?:conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks)\b[:\s\-]*", re.IGNORECASE),
}

# Prompt injection heuristics
INJECTION_PATTERN = re.compile(
    r"(?i)(ignore\s+all\s+(?:previous|above)\s+instructions|system\s*:\s*you\s+are|override\s+system\s+prompt|drop\s+table|delete\s+from|execute\s+script|<\s*script\b|\[INST\]|\[/INST\])",
)


def _sanitize_untrusted_text(text: str, page_num: int) -> Tuple[str, List[str]]:
    """Sanitize untrusted text from PDF documents and detect injection patterns.

    Removes null bytes and control characters while preserving unicode scripts (e.g. Ethiopic).
    """
    warnings: List[str] = []

    # 1. Strip null bytes and dangerous terminal escape codes
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", cleaned)

    # 2. Check for adversarial prompt injection strings
    match = INJECTION_PATTERN.search(cleaned)
    if match:
        warning_msg = (
            f"Adversarial prompt injection pattern detected on page {page_num}: "
            f"'{match.group(0)}'. Extracted text treated strictly as untrusted data."
        )
        logger.warning(warning_msg)
        warnings.append(warning_msg)

    return cleaned, warnings


def _extract_with_pymupdf(pdf_bytes: bytes) -> Tuple[Dict[int, str], Dict[str, Any], int, bool, List[str]]:
    """Extract text and metadata using PyMuPDF.

    Returns (raw_text_by_page, metadata, page_count, is_scanned, warnings).
    """
    raw_text_by_page: Dict[int, str] = {}
    warnings: List[str] = []

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise InvalidPDFError(f"PyMuPDF failed to parse document: {e}") from e

    page_count = len(doc)
    if page_count == 0:
        return {}, {}, 0, False, ["Document contains 0 pages"]

    if page_count > settings.MAX_PDF_PAGES:
        raise PDFSizeExceededError(
            f"PDF page count ({page_count}) exceeds maximum allowed limit of {settings.MAX_PDF_PAGES} pages"
        )

    metadata = {
        "title": doc.metadata.get("title") or None,
        "author": doc.metadata.get("author") or None,
        "subject": doc.metadata.get("subject") or None,
        "creator": doc.metadata.get("creator") or None,
        "creation_date": doc.metadata.get("creationDate") or None,
    }

    total_chars = 0
    total_images = 0

    for i in range(page_count):
        page = doc[i]
        page_num = i + 1
        page_text = page.get_text("text") or ""
        cleaned_text, page_warnings = _sanitize_untrusted_text(page_text, page_num)
        warnings.extend(page_warnings)

        raw_text_by_page[page_num] = cleaned_text
        total_chars += len(cleaned_text.strip())
        if total_chars > settings.MAX_EXTRACTED_CHARS:
            raise PDFSizeExceededError(
                f"Extracted characters ({total_chars}) exceeded maximum allowable limit of {settings.MAX_EXTRACTED_CHARS}"
            )
        total_images += len(page.get_images())

    # Detect scanned image-only PDF
    avg_chars_per_page = total_chars / page_count
    is_scanned = (avg_chars_per_page < 35 and total_images > 0) or (total_chars == 0 and page_count > 0)

    return raw_text_by_page, metadata, page_count, is_scanned, warnings


def _extract_with_pdfplumber(pdf_bytes: bytes) -> Tuple[Dict[int, str], Dict[str, Any], int, bool, List[str]]:
    """Extract text and metadata using pdfplumber as secondary engine.

    Returns (raw_text_by_page, metadata, page_count, is_scanned, warnings).
    """
    raw_text_by_page: Dict[int, str] = {}
    warnings: List[str] = []

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            page_count = len(pdf.pages)
            if page_count == 0:
                return {}, {}, 0, False, ["Document contains 0 pages"]

            if page_count > settings.MAX_PDF_PAGES:
                raise PDFSizeExceededError(
                    f"PDF page count ({page_count}) exceeds maximum allowed limit of {settings.MAX_PDF_PAGES} pages"
                )

            metadata = {
                "title": (pdf.metadata or {}).get("Title"),
                "author": (pdf.metadata or {}).get("Author"),
                "creation_date": (pdf.metadata or {}).get("CreationDate"),
            }

            total_chars = 0
            total_images = 0

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                page_text = page.extract_text() or ""
                cleaned_text, page_warnings = _sanitize_untrusted_text(page_text, page_num)
                warnings.extend(page_warnings)

                raw_text_by_page[page_num] = cleaned_text
                total_chars += len(cleaned_text.strip())
                if total_chars > settings.MAX_EXTRACTED_CHARS:
                    raise PDFSizeExceededError(
                        f"Extracted characters ({total_chars}) exceeded maximum allowable limit of {settings.MAX_EXTRACTED_CHARS}"
                    )
                total_images += len(page.images or [])

            avg_chars = total_chars / page_count
            is_scanned = (avg_chars < 35 and total_images > 0) or (total_chars == 0 and page_count > 0)

            return raw_text_by_page, metadata, page_count, is_scanned, warnings
    except (PDFSizeExceededError, InvalidPDFError):
        raise
    except Exception as e:
        raise InvalidPDFError(f"pdfplumber failed to parse document: {e}") from e


TARGET_SECTION_KEYS = (
    "abstract", "research_objectives", "research_questions",
    "methodology", "dataset", "models_and_algorithms",
    "experimental_setup", "evaluation_metrics", "results",
    "limitations", "future_work", "conclusion",
)


# Running headers/footers printed above the title on page 1 (venue, pages, copyright, DOI).
_RUNNING_HEADER = re.compile(r"^\s*proceedings\s+of\b|\bpages\b|©|\bdoi\s*:", re.IGNORECASE)
# "Workshop" alone may appear in a real title; only a venue line pairs it with these.
_WORKSHOP = re.compile(r"\bworkshop\b", re.IGNORECASE)
_WORKSHOP_VENUE = re.compile(r"\bproceedings\b|\binternational\b", re.IGNORECASE)


def _is_running_header_line(line: str) -> bool:
    if _RUNNING_HEADER.search(line):
        return True
    return bool(_WORKSHOP.search(line) and _WORKSHOP_VENUE.search(line))


def _title_and_authors(
    raw_text_by_page: Dict[int, str],
    metadata_title: Optional[str],
    metadata_author: Optional[str],
    layout_title: Optional[str] = None,
) -> Tuple[Optional[str], List[str]]:
    """Title and authors for a paper.

    Title priority: the layout title (first ``# `` heading pymupdf4llm found on page 1),
    then PDF metadata, then the first substantive first-page line that is not a running
    header (venue/pages/copyright/DOI line). Authors come from metadata.
    """
    first_page_text = raw_text_by_page.get(1, "")
    first_page_lines = [
        line.strip() for line in first_page_text.splitlines()
        if line.strip() and not _is_running_header_line(line)
    ]

    extracted_title = layout_title or metadata_title
    extracted_authors: List[str] = []
    if metadata_author:
        extracted_authors.append(metadata_author)

    if first_page_lines and not extracted_title:
        # Academic conventions: Title is typically the first substantive line before 'abstract'
        for line in first_page_lines[:4]:
            if not SECTION_REGEXES["abstract"].match(line) and len(line) > 10:
                extracted_title = line
                break
    return extracted_title, extracted_authors


def _parse_academic_sections(
    raw_text_by_page: Dict[int, str],
    paper_id: str,
    metadata_title: Optional[str] = None,
    metadata_author: Optional[str] = None,
    layout_title: Optional[str] = None,
) -> ExtractedPaperDataSchema:
    """Detect academic sections and extract evidence with verified page numbers.

    Never invents missing sections.
    """
    sections: Dict[str, Tuple[str, int, str]] = {}  # key -> (text, page_number, heading_name)
    current_key: Optional[str] = None
    current_heading: Optional[str] = None
    current_start_page: int = 1
    current_lines: List[str] = []

    evidence_items: List[EvidenceItemSchema] = []
    extracted_title, extracted_authors = _title_and_authors(
        raw_text_by_page, metadata_title, metadata_author, layout_title,
    )

    # Scan pages sequentially to preserve page references
    for page_num in sorted(raw_text_by_page.keys()):
        page_text = raw_text_by_page[page_num]
        lines = page_text.splitlines()

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Check if line matches a new academic section heading
            matched_key = None
            matched_heading = None
            for key, pattern in SECTION_REGEXES.items():
                if pattern.match(line_str) and len(line_str) < 90:
                    matched_key = key
                    matched_heading = line_str
                    break

            if matched_key:
                # Flush previous section
                if current_key and current_lines:
                    content_str = "\n".join(current_lines).strip()
                    if content_str and current_key not in sections:
                        sections[current_key] = (content_str, current_start_page, current_heading or current_key)
                        # Extract exact verbatim passage from raw page text to preserve evidence integrity
                        start_page_text = raw_text_by_page.get(current_start_page, "")
                        verbatim_passage = ""
                        if current_heading and current_heading in start_page_text:
                            idx = start_page_text.find(current_heading)
                            verbatim_passage = start_page_text[idx:min(idx + 400, len(start_page_text))]
                        elif current_lines and current_lines[0] in start_page_text:
                            idx = start_page_text.find(current_lines[0])
                            verbatim_passage = start_page_text[idx:min(idx + 400, len(start_page_text))]
                        else:
                            verbatim_passage = content_str[:400].strip()

                        evidence_items.append(
                            EvidenceItemSchema(
                                paper_id=paper_id,
                                page_number=current_start_page,
                                section_heading=current_heading,
                                passage=verbatim_passage,
                                evidence_type=EvidenceType.DIRECT_EXTRACT,
                            )
                        )

                current_key = matched_key
                current_heading = matched_heading
                current_start_page = page_num
                current_lines = []
            else:
                if current_key:
                    current_lines.append(line_str)

    # Flush final section
    if current_key and current_lines:
        content_str = "\n".join(current_lines).strip()
        if content_str and current_key not in sections:
            sections[current_key] = (content_str, current_start_page, current_heading or current_key)
            start_page_text = raw_text_by_page.get(current_start_page, "")
            verbatim_passage = ""
            if current_heading and current_heading in start_page_text:
                idx = start_page_text.find(current_heading)
                verbatim_passage = start_page_text[idx:min(idx + 400, len(start_page_text))]
            elif current_lines and current_lines[0] in start_page_text:
                idx = start_page_text.find(current_lines[0])
                verbatim_passage = start_page_text[idx:min(idx + 400, len(start_page_text))]
            else:
                verbatim_passage = content_str[:400].strip()

            evidence_items.append(
                EvidenceItemSchema(
                    paper_id=paper_id,
                    page_number=current_start_page,
                    section_heading=current_heading,
                    passage=verbatim_passage,
                    evidence_type=EvidenceType.DIRECT_EXTRACT,
                )
            )

    # Helper to get section text
    def _get(key: str) -> Optional[str]:
        return sections[key][0] if key in sections else None

    # Track missing sections explicitly
    all_target_keys = [
        "abstract", "research_objectives", "research_questions",
        "methodology", "dataset", "models_and_algorithms",
        "experimental_setup", "evaluation_metrics", "results",
        "limitations", "future_work", "conclusion"
    ]
    missing_sections = [k for k in all_target_keys if k not in sections]

    # Specific check for research questions inside introduction or methodology
    rq_text = _get("research_questions")
    rq_list: Optional[List[str]] = None
    if rq_text:
        rq_list = [line.strip() for line in rq_text.splitlines() if line.strip()]

    return ExtractedPaperDataSchema(
        paper_id=paper_id,
        title=extracted_title,
        authors=extracted_authors,
        abstract=_get("abstract"),
        research_objectives=_get("research_objectives"),
        research_questions=rq_list,
        methodology=_get("methodology"),
        dataset=_get("dataset"),
        models_and_algorithms=_get("models_and_algorithms"),
        experimental_setup=_get("experimental_setup"),
        evaluation_metrics=_get("evaluation_metrics"),
        results=_get("results"),
        limitations=_get("limitations"),
        future_work=_get("future_work"),
        conclusion=_get("conclusion"),
        evidence=evidence_items,
        missing_sections=missing_sections,
    )


# ==============================================================================
# Layout-based section detection (primary strategy)
# ==============================================================================

# pymupdf4llm runs on one dedicated thread so the event loop stays responsive and a
# time budget can be enforced. A timed-out conversion cannot be killed: it finishes
# in the background, and later requests queue behind it (falling back to the regex
# parser if their own budget runs out first).
_LAYOUT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aira-layout")
# Seconds of PDF_PROCESSING_TIMEOUT kept in reserve for the regex fallback.
_LAYOUT_RESERVE_SECONDS = 3.0

# Layout box classes whose text never belongs to a section body.
_SKIPPED_BOX_CLASSES = frozenset({"page-header", "page-footer", "footnote", "picture"})

_NUMBER_PREFIX = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+(?P<title>\S.*)$")
_APPENDIX_PREFIX = re.compile(r"^(?P<num>[A-H](?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z].*)$")
_BARE_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?$")
_URL_OR_FOOTNOTE = re.compile(
    r"(?i)https?://|www\.|\.(?:com|org|io|net|edu)\b|@|^\d+\s*https?|^\d+[a-z]"
)
_RUN_IN_ABSTRACT = re.compile(r"^(?P<head>abstract)\s*[—–:\-.]\s*(?P<rest>\S.*)$", re.IGNORECASE)

# Whole-name vocabulary for unnumbered headings ("Results and Discussion" is split on
# "and" / "&" / "," and every part must be a known name).
_KNOWN_SECTION_PART = re.compile(
    r"""
      abstract | introduction | background | preliminar(?:y|ies) | related\s+works? | literature\s+review
    | (?:proposed\s+|our\s+)?(?:methods?|methodology|approach(?:es)?|framework|system|models?)
    | system\s+(?:description|overview|architecture) | model\s+(?:overview|architecture|description)
    | task\s+(?:description|definition|overview|setup) | problem\s+(?:definition|formulation|statement)
    | data | datasets? | corpus | corpora | benchmarks?
    | data\s+(?:collection|preparation|description|sources?|statistics) | (?:data\s+)?pre-?processing
    | annotation | experiments? | experimental\s+(?:setup|settings?|design|details|results|evaluation)
    | (?:evaluation\s+)?(?:metrics?|measures) | evaluation | implementation(?:\s+details)?
    | training(?:\s+details)? | hyper-?parameters? | baselines? | algorithms?
    | results? | (?:error\s+)?analysis | ablations?(?:\s+stud(?:y|ies))? | discussion
    | limitations? | threats\s+to\s+validity | future\s+(?:work|directions?|research)
    | conclusions? | concluding\s+remarks | summary
    | research\s+(?:questions?|objectives?) | hypothes[ie]s | objectives?
    | references | bibliography | acknowledge?ments? | appendix(?:\s+.*)? | supplementary\s+materials?
    | ethics\s+statement | ethical\s+considerations | broader\s+impacts?
    """,
    re.VERBOSE,
)
# Unnumbered headings that are top-level even in a numbered paper.
_TOP_LEVEL_UNNUMBERED = re.compile(
    r"abstract|references|bibliography|acknowledge?ments?|limitations?(?:\s.*)?|appendix.*"
    r"|ethics\s+statement|ethical\s+considerations|broader\s+impacts?|supplementary\s+materials?"
)
_BACK_MATTER = re.compile(r"references|bibliography")

# Heading title -> schema field, first rule that matches wins. Order matters:
# "Experimental Results" is results, "Evaluation Metrics" is evaluation_metrics.
_NON_TARGET_HEADING = re.compile(
    r"\b(?:introduction|background|related\s+works?|literature|references|bibliography|"
    r"acknowledge?ments?|appendix|ethic\w*|broader\s+impacts?|task\s+(?:description|definition|overview))\b"
)
_HEADING_KEY_RULES: Tuple[Tuple[str, "re.Pattern[str]"], ...] = tuple(
    (key, re.compile(pattern)) for key, pattern in (
        ("abstract", r"^abstract$"),
        ("limitations", r"\blimitations?\b|\bthreats\s+to\s+validity\b"),
        ("conclusion", r"\bconclu(?:sions?|ding\s+remarks)\b"),
        ("future_work", r"\bfuture\s+(?:work|directions?|research)\b"),
        ("evaluation_metrics", r"\bmetrics?\b|\b(?:evaluation|performance)\s+measures?\b"),
        ("results", r"\bresults?\b|\bfindings\b|\bperformance\b"),
        ("experimental_setup", r"\bexperiment(?:s|al)?\b|\bsetup\b|\bsettings?\b|\bhyper-?parameters?\b"
                               r"|\bimplementation\b|\btraining\s+details\b"),
        ("dataset", r"\bdata(?:sets?)?\b|\bcorp(?:us|ora)\b|\bbenchmarks?\b|\bpre-?processing\b|\bannotation\b"),
        ("models_and_algorithms", r"\bmodels?\b|\balgorithms?\b|\barchitectures?\b|\bbaselines?\b"),
        ("methodology", r"\bmethod(?:s|ology)?\b|\bapproach(?:es)?\b|\bsystem\b|\bframework\b|\bproposed\b"),
        ("research_questions", r"\bresearch\s+questions?\b|\bhypothes[ie]s\b"),
        ("research_objectives", r"\bobjectives?\b|\bgoals?\b|\baims?\b"),
    )
)


@dataclass
class _LayoutBox:
    page: int
    cls: str
    bbox: Tuple[float, float, float, float]
    page_width: float
    markdown: str  # pymupdf4llm markdown; only used to read heading text
    raw: str  # PyMuPDF plain text inside the box (same engine as raw_text_by_page)


@dataclass
class _Heading:
    page: int
    display: str  # as printed, e.g. "4.1 Datasets"
    title: str  # without numbering, e.g. "Datasets"
    number: Optional[Tuple[int, ...]]
    back_matter: bool  # after References (appendices)
    depth: int = 1
    key: Optional[str] = None


@dataclass
class SectionSpan:
    """Where a detected section starts and ends (diagnostics and tests)."""
    key: str
    heading: str
    start_page: int
    end_page: int
    text: str


def _clean_control_chars(text: str) -> str:
    cleaned = text.replace("\x00", "")
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", cleaned)


def _extract_layout_boxes(pdf_bytes: bytes) -> List[_LayoutBox]:
    """Run pymupdf4llm layout analysis (no OCR) and return its content boxes.

    Runs on the layout worker thread. Box text is re-read with PyMuPDF's plain-text
    extraction clipped to the box, so it matches ``raw_text_by_page`` verbatim.
    """
    import pymupdf4llm  # lazy: importing it activates PyMuPDF's layout engine

    boxes: List[_LayoutBox] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    try:
        chunks = pymupdf4llm.to_markdown(
            doc, page_chunks=True, use_ocr=False, header=False, footer=False, show_progress=False,
        )
        for chunk in chunks:
            page_num = int(chunk["metadata"]["page_number"])
            page = doc[page_num - 1]
            text = chunk.get("text") or ""
            for box in chunk.get("page_boxes") or []:
                cls = str(box.get("class") or "text")
                if cls in _SKIPPED_BOX_CLASSES:
                    continue
                x0, y0, x1, y1 = (float(v) for v in box["bbox"])
                start, end = box.get("pos") or (0, 0)
                raw = page.get_text("text", clip=pymupdf.Rect(x0, y0, x1, y1)) or ""
                boxes.append(_LayoutBox(
                    page=page_num, cls=cls, bbox=(x0, y0, x1, y1), page_width=float(page.rect.width),
                    markdown=text[start:end], raw=_clean_control_chars(raw),
                ))
    finally:
        doc.close()
    return boxes


def _reading_order(boxes: List[_LayoutBox]) -> List[_LayoutBox]:
    """Order boxes page by page in two-column reading order from their coordinates.

    Full-width boxes (title, wide tables) split a page into bands; inside a band the
    left column is read top to bottom, then the right column. Pages without both a
    left and a right column are read top to bottom.
    """
    from bisect import bisect_left, bisect_right

    ordered: List[_LayoutBox] = []
    for page in sorted({b.page for b in boxes}):
        page_boxes = [b for b in boxes if b.page == page]
        width = page_boxes[0].page_width or max(b.bbox[2] for b in page_boxes)
        mid, tol = width / 2, width * 0.03

        def column(b: _LayoutBox) -> int:
            if b.bbox[2] <= mid + tol:
                return 0
            if b.bbox[0] >= mid - tol:
                return 1
            return -1  # spans both columns

        columns = [column(b) for b in page_boxes]
        if not (0 in columns and 1 in columns):
            ordered.extend(sorted(page_boxes, key=lambda b: (b.bbox[1], b.bbox[0])))
            continue
        full_tops = sorted(b.bbox[1] for b, c in zip(page_boxes, columns) if c == -1)

        def sort_key(item: Tuple[_LayoutBox, int]) -> Tuple[int, int, float]:
            b, col = item
            top = b.bbox[1]
            if col == -1:
                return (2 * bisect_left(full_tops, top) + 1, 0, top)
            return (2 * bisect_right(full_tops, top), col, top)

        ordered.extend(b for b, _ in sorted(zip(page_boxes, columns), key=sort_key))
    return ordered


def _clean_heading_markdown(markdown: str) -> str:
    text = re.sub(r"<sup>.*?</sup>", "", markdown, flags=re.DOTALL)  # footnote markers
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s*#+\s*", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = " ".join(text.split())
    return text.strip(" :")


def _layout_title(boxes: List[_LayoutBox]) -> Optional[str]:
    """The paper title: the first level-1 (``# ``) heading pymupdf4llm marked on page 1."""
    for box in boxes:
        if box.page != 1:
            continue
        if re.match(r"^\s*#(?!#)\s", box.markdown):
            title = _clean_heading_markdown(box.markdown)
            if len(title) > 3 and not _is_running_header_line(title):
                return title
    return None


def _is_known_section_name(title: str) -> bool:
    normalized = " ".join(title.lower().split()).strip(" :.")
    parts = [p for p in re.split(r"\s*(?:,|&|\band\b)\s*", normalized) if p]
    return bool(parts) and all(_KNOWN_SECTION_PART.fullmatch(p) for p in parts)


def _heading_key(title: str) -> Optional[str]:
    """Map a heading title (numbering removed) to an ExtractedPaperDataSchema field."""
    normalized = " ".join(title.lower().split()).strip(" :.")
    if _NON_TARGET_HEADING.search(normalized):
        return None
    for key, pattern in _HEADING_KEY_RULES:
        if pattern.search(normalized):
            return key
    return None


def _parse_heading(text: str, page: int, back_matter: bool) -> Optional[_Heading]:
    """Accept a heading only if it is numbered or a known section name; reject URLs/footnotes."""
    text = text.strip()
    if not text or len(text) > 120 or _URL_OR_FOOTNOTE.search(text):
        return None
    number: Optional[Tuple[int, ...]] = None
    title = text
    numbered = _NUMBER_PREFIX.match(text)
    appendix = _APPENDIX_PREFIX.match(text) if back_matter else None
    if numbered and int(numbered.group("num").split(".")[0]) <= 20:
        number = tuple(int(x) for x in numbered.group("num").split("."))
        title = numbered.group("title").strip()
    elif appendix:
        letter, *rest = appendix.group("num").split(".")
        number = (100 + ord(letter) - ord("A"), *(int(x) for x in rest))
        title = appendix.group("title").strip()
    words = title.split()
    if not re.search(r"[A-Za-z]", title) or len(words) > 12:
        return None
    if title.endswith(".") and len(words) > 4:  # a sentence, not a heading
        return None
    if number is None and not _is_known_section_name(title):
        return None
    return _Heading(page=page, display=text, title=title, number=number, back_matter=back_matter)


# Single generic words that only count as an in-text heading when numbered; as a bare
# line they are too often a table cell or a wrapped word.
_GENERIC_SECTION_WORDS = frozenset({
    "model", "models", "data", "system", "framework", "evaluation", "summary", "analysis",
    "annotation", "training", "baseline", "baselines", "approach", "approaches", "algorithm",
    "algorithms", "corpus", "corpora", "benchmark", "benchmarks", "measures", "metric",
    "metrics", "objective", "objectives", "implementation", "experiment",
})


def _standalone_heading_line(lines: List[str], i: int) -> Optional[Tuple[str, int]]:
    """If lines[i] (plus lines[i+1] for a split "4.1" / "Datasets") is a heading set in
    the body font, return (heading_text, number_of_lines_used)."""
    line, used = lines[i], 1
    if _BARE_NUMBER.match(line) and i + 1 < len(lines):  # number and title on separate lines
        line, used = f"{line.rstrip('.')} {lines[i + 1]}", 2
    numbered = _NUMBER_PREFIX.match(line)
    title = (numbered.group("title") if numbered else line).strip()
    if not title or len(title) > 60 or not (title[0].isupper() or numbered):
        return None
    if title[-1] in ".,;":
        return None
    if not numbered and title.lower().strip(" :") in _GENERIC_SECTION_WORDS:
        return None
    if not _is_known_section_name(title):
        return None
    if i > 0 and not re.search(r"[.!?:)\]]$", lines[i - 1]):
        return None  # previous line does not end a sentence: this is wrapped body text
    return line, used


def _split_body_headings(raw: str) -> List[Tuple[str, str]]:
    """Split a text box at headings set in the body font, which layout analysis reports
    as plain text. Returns [("text", ...) | ("heading", ...)] segments in order."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    segments: List[Tuple[str, str]] = []
    buffer: List[str] = []
    i = 0
    if lines:
        run_in = _RUN_IN_ABSTRACT.match(lines[0])
        if run_in:  # "Abstract—We present ..."
            segments.append(("heading", run_in.group("head")))
            lines = [run_in.group("rest"), *lines[1:]]
            i = 1
            buffer.append(lines[0])
    while i < len(lines):
        found = _standalone_heading_line(lines, i)
        if found:
            if buffer:
                segments.append(("text", "\n".join(buffer)))
                buffer = []
            segments.append(("heading", found[0]))
            i += found[1]
        else:
            buffer.append(lines[i])
            i += 1
    if buffer:
        segments.append(("text", "\n".join(buffer)))
    return segments


def _verbatim_passage(raw_text_by_page: Dict[int, str], heading: _Heading, body: str) -> str:
    """Up to 400 characters copied from the raw page text, starting at the heading."""
    page_text = raw_text_by_page.get(heading.page, "")
    for needle in (heading.display, heading.title):
        match = re.search(rf"(?m)^[ \t]*{re.escape(needle)}[ \t]*$", page_text)
        if match:
            idx = match.start() + (len(match.group(0)) - len(match.group(0).lstrip()))
            return page_text[idx:idx + 400]
    for needle in (heading.display, heading.title, body.splitlines()[0] if body else ""):
        if needle and needle in page_text:
            idx = page_text.find(needle)
            return page_text[idx:idx + 400]
    return body[:400].strip()


def _parse_sections_with_layout(
    boxes: List[_LayoutBox],
    raw_text_by_page: Dict[int, str],
    paper_id: str,
    metadata_title: Optional[str] = None,
    metadata_author: Optional[str] = None,
) -> Tuple[Optional[ExtractedPaperDataSchema], Dict[str, SectionSpan], Optional[str]]:
    """Build sections from layout headings.

    Returns (data, spans, problem). ``data`` is None when no usable heading was found
    (caller falls back silently) or when ``problem`` explains why the headings cannot
    be trusted (caller falls back and reports it).
    """
    # 1. Reading order, then a flat stream of headings and body text.
    items: List[Union[_Heading, Tuple[int, str]]] = []
    back_matter = False
    for box in _reading_order(boxes):
        if box.cls == "section-header":
            segments = [("heading", _clean_heading_markdown(box.markdown))]
        elif box.cls == "text":
            segments = _split_body_headings(box.raw)
        else:
            segments = [("text", box.raw)]
        for kind, value in segments:
            heading = _parse_heading(value, box.page, back_matter) if kind == "heading" else None
            if heading:
                items.append(heading)
                if _BACK_MATTER.fullmatch(heading.title.lower()):
                    back_matter = True
            elif box.cls == "section-header":
                if box.raw.strip():
                    items.append((box.page, box.raw))  # rejected heading: keep its text as body
            elif value.strip():
                items.append((box.page, value))

    headings = [it for it in items if isinstance(it, _Heading)]
    if not headings:
        return None, {}, None

    # 2. Numbering must be strictly increasing (4 < 4.1 < 4.2 < 5) in reading order.
    numbered = [h for h in headings if h.number and not h.back_matter]
    for prev, cur in zip(numbered, numbered[1:]):
        if cur.number <= prev.number:
            return None, {}, (
                f"Heading numbering is out of order after layout reordering "
                f"('{prev.display}' on page {prev.page} before '{cur.display}' on page {cur.page}); "
                f"layout section boundaries are unreliable for this document"
            )

    # 3. Depth: numbered -> number of levels; unnumbered -> top level, or a child of the
    #    current numbered section in a numbered paper.
    has_numbers = bool(numbered)
    current_depth = 0
    for h in headings:
        if h.number:
            h.depth = len(h.number)
            current_depth = h.depth
        elif not has_numbers or _TOP_LEVEL_UNNUMBERED.fullmatch(h.title.lower()):
            h.depth = 1
            current_depth = 0
        else:
            h.depth = current_depth + 1
        h.key = _heading_key(h.title)

    # 4. A section runs to the next heading at the same or a higher level, so a parent
    #    section includes its subsections. The first heading for a field wins.
    spans: Dict[str, SectionSpan] = {}
    span_headings: Dict[str, Tuple[int, _Heading]] = {}  # key -> (stream position, heading)
    for i, item in enumerate(items):
        if not isinstance(item, _Heading) or not item.key or item.key in spans:
            continue
        lines: List[str] = []
        end_page = item.page
        for nxt in items[i + 1:]:
            if isinstance(nxt, _Heading):
                if nxt.depth <= item.depth:
                    break
                lines.append(nxt.display)
                end_page = nxt.page
            else:
                lines.extend(line.strip() for line in nxt[1].splitlines() if line.strip())
                end_page = nxt[0]
        text = "\n".join(lines).strip()
        if text:
            spans[item.key] = SectionSpan(item.key, item.display, item.page, end_page, text)
            span_headings[item.key] = (i, item)

    if not spans:
        return None, {}, None

    evidence_items = [
        EvidenceItemSchema(
            paper_id=paper_id,
            page_number=spans[key].start_page,
            section_heading=spans[key].heading,
            passage=_verbatim_passage(raw_text_by_page, heading, spans[key].text),
            evidence_type=EvidenceType.DIRECT_EXTRACT,
        )
        for key, (_, heading) in sorted(span_headings.items(), key=lambda kv: kv[1][0])
    ]

    title, authors = _title_and_authors(raw_text_by_page, metadata_title, metadata_author, _layout_title(boxes))
    texts = {key: span.text for key, span in spans.items()}
    rq_text = texts.get("research_questions")
    rq_list = [line.strip() for line in rq_text.splitlines() if line.strip()] if rq_text else None

    data = ExtractedPaperDataSchema(
        paper_id=paper_id,
        title=title,
        authors=authors,
        abstract=texts.get("abstract"),
        research_objectives=texts.get("research_objectives"),
        research_questions=rq_list,
        methodology=texts.get("methodology"),
        dataset=texts.get("dataset"),
        models_and_algorithms=texts.get("models_and_algorithms"),
        experimental_setup=texts.get("experimental_setup"),
        evaluation_metrics=texts.get("evaluation_metrics"),
        results=texts.get("results"),
        limitations=texts.get("limitations"),
        future_work=texts.get("future_work"),
        conclusion=texts.get("conclusion"),
        evidence=evidence_items,
        missing_sections=[k for k in TARGET_SECTION_KEYS if k not in spans],
    )
    return data, spans, None


async def _detect_sections_with_layout(
    pdf_bytes: bytes,
    raw_text_by_page: Dict[int, str],
    paper_id: str,
    metadata: Dict[str, Any],
    started_at: float,
) -> Tuple[Optional[ExtractedPaperDataSchema], Optional[str], Optional[str]]:
    """Run layout heading detection inside the PDF time budget.

    Returns (data, warning, layout_title). ``data`` is None whenever the regex parser
    should be used; ``layout_title`` is still returned then so the fallback can use it.
    """
    fallback = "used text-pattern section detection instead"
    timeout = float(settings.PDF_PROCESSING_TIMEOUT)
    budget = timeout - (time.perf_counter() - started_at) - min(_LAYOUT_RESERVE_SECONDS, 0.25 * timeout)
    if budget <= 0.5:
        return None, f"Layout heading detection skipped: PDF time budget exhausted; {fallback}.", None

    loop = asyncio.get_running_loop()
    try:
        boxes = await asyncio.wait_for(
            loop.run_in_executor(_LAYOUT_EXECUTOR, _extract_layout_boxes, pdf_bytes), timeout=budget,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Layout heading detection for '{paper_id}' exceeded {budget:.1f}s")
        return None, f"Layout heading detection exceeded its {budget:.1f}s budget; {fallback}.", None
    except Exception as e:  # any layout/library failure must not fail the extraction
        logger.warning(f"Layout heading detection failed for '{paper_id}': {e}")
        return None, f"Layout heading detection failed ({type(e).__name__}: {e}); {fallback}.", None

    title = _layout_title(boxes)
    data, spans, problem = _parse_sections_with_layout(
        boxes, raw_text_by_page, paper_id, metadata.get("title"), metadata.get("author"),
    )
    if problem:
        logger.warning(f"'{paper_id}': {problem}")
        return None, f"{problem}; {fallback}.", title
    if data is None:
        logger.info(f"No layout headings found for '{paper_id}'; using text-pattern section detection.")
        return None, None, title
    logger.info(f"Layout headings used for '{paper_id}': {sorted(spans)}")
    return data, None, title


async def read_research_paper(
    paper_url_or_file: Union[str, Path, bytes],
    extraction_method: str = "auto",
    paper_id: Optional[str] = None,
) -> PaperExtractionResultSchema:
    """Main Tool: Read and extract academic paper content from URL, path, or bytes.

    Supports:
    - Downloading academic PDFs using URLs (with SSRF protection).
    - Reading locally uploaded PDF files and binary buffers.
    - Extracting text using PyMuPDF with fallback to pdfplumber.
    - Preserving 1-indexed page numbers and document metadata.
    - Identifying incomplete, empty, or scanned documents.
    - Enforcing strict processing timeouts against PDF bombs.
    """
    effective_id = paper_id or (
        str(paper_url_or_file).split("/")[-1].split("?")[0]
        if isinstance(paper_url_or_file, (str, Path)) and str(paper_url_or_file).startswith("http")
        else (Path(paper_url_or_file).stem if isinstance(paper_url_or_file, (str, Path)) else "uploaded_paper")
    )

    async def _execute() -> PaperExtractionResultSchema:
        start_time = time.perf_counter()
        warnings: List[str] = []

        # 1. Determine input source and load PDF bytes
        eff_id = paper_id or "uploaded_paper"
        pdf_bytes: bytes

        if isinstance(paper_url_or_file, bytes):
            pdf_bytes = paper_url_or_file
            eff_id = paper_id or "memory_buffer"
        elif isinstance(paper_url_or_file, (str, Path)):
            str_input = str(paper_url_or_file).strip()
            if str_input.startswith("http://") or str_input.startswith("https://"):
                eff_id = paper_id or str_input.split("/")[-1].split("?")[0] or "web_paper"
                logger.info(f"Downloading PDF from URL: {str_input} | Target ID: {eff_id}")
                pdf_bytes = await download_pdf_safely(str_input)
            else:
                file_path = Path(str_input)
                if not file_path.exists():
                    raise FileNotFoundError(f"Local PDF file not found: {file_path}")
                if file_path.stat().st_size > settings.MAX_PDF_SIZE_BYTES:
                    raise PDFSizeExceededError(
                        f"Local file size ({file_path.stat().st_size} bytes) exceeds limit of {settings.MAX_PDF_SIZE_BYTES} bytes"
                    )
                eff_id = paper_id or file_path.stem
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()
        else:
            raise ValueError(f"Unsupported input type: {type(paper_url_or_file)}")

        # 2. Verify magic header
        if len(pdf_bytes) < 5 or b"%PDF-" not in pdf_bytes[:1024]:
            duration = round(time.perf_counter() - start_time, 3)
            return PaperExtractionResultSchema(
                paper_id=eff_id,
                status=ExtractionStatus.FAILED,
                extraction_method="none",
                page_count=0,
                document_metadata={},
                extracted_data=None,
                is_scanned=False,
                ocr_required=False,
                warnings=["File does not start with valid '%PDF-' header"],
                error_message="Invalid PDF structure or corrupted header",
                raw_text_by_page={},
            )

        # 3. Dual-Engine Extraction
        chosen_method = extraction_method.lower().strip()
        if chosen_method not in ("auto", "pymupdf", "pdfplumber"):
            chosen_method = "auto"

        raw_text: Dict[int, str] = {}
        metadata: Dict[str, Any] = {}
        page_count = 0
        is_scanned = False
        used_method = "none"

        if chosen_method in ("auto", "pymupdf"):
            try:
                raw_text, metadata, page_count, is_scanned, mu_warnings = _extract_with_pymupdf(pdf_bytes)
                warnings.extend(mu_warnings)
                used_method = "pymupdf"
            except (PDFSizeExceededError, InvalidPDFError):
                raise
            except Exception as e:
                logger.warning(f"PyMuPDF failed ({e}). Falling back to pdfplumber...")
                warnings.append(f"PyMuPDF engine failed: {e}")
                if chosen_method == "pymupdf":
                    duration = round(time.perf_counter() - start_time, 3)
                    return PaperExtractionResultSchema(
                        paper_id=eff_id,
                        status=ExtractionStatus.FAILED,
                        extraction_method="pymupdf",
                        page_count=0,
                        warnings=warnings,
                        error_message=f"PyMuPDF extraction failed: {e}",
                    )

        # Fallback to pdfplumber if auto mode failed with PyMuPDF or returned 0 text
        total_extracted_chars = sum(len(txt.strip()) for txt in raw_text.values())
        if (chosen_method == "pdfplumber") or (chosen_method == "auto" and (not raw_text or total_extracted_chars == 0 and not is_scanned)):
            try:
                raw_text, metadata, page_count, is_scanned, plumb_warnings = _extract_with_pdfplumber(pdf_bytes)
                warnings.extend(plumb_warnings)
                used_method = "pdfplumber"
            except (PDFSizeExceededError, InvalidPDFError):
                raise
            except Exception as e:
                logger.error(f"pdfplumber extraction failed: {e}")
                warnings.append(f"pdfplumber engine failed: {e}")

        # Check for empty document
        if page_count == 0:
            return PaperExtractionResultSchema(
                paper_id=eff_id,
                status=ExtractionStatus.FAILED,
                extraction_method=used_method,
                page_count=0,
                warnings=warnings,
                error_message="PDF document contains zero pages",
            )

        # 4. Handle Scanned Documents (Require OCR)
        if is_scanned:
            logger.warning(f"Document '{eff_id}' appears to be a scanned image-only PDF. OCR required.")
            return PaperExtractionResultSchema(
                paper_id=eff_id,
                status=ExtractionStatus.SCANNED_NEEDS_OCR,
                extraction_method=used_method,
                page_count=page_count,
                document_metadata=metadata,
                extracted_data=None,
                is_scanned=True,
                ocr_required=True,
                warnings=warnings + ["Document contains scanned bitmap images with negligible extractable text. OCR is required."],
                error_message="Document is scanned; OCR engine required but not bundled.",
                raw_text_by_page=raw_text,
            )

        # 5. Parse Academic Sections & Evidence References.
        #    Primary: layout headings (PyMuPDF documents only). Fallback: regex on text lines.
        extracted_data: Optional[ExtractedPaperDataSchema] = None
        layout_title: Optional[str] = None
        if used_method == "pymupdf":
            extracted_data, layout_warning, layout_title = await _detect_sections_with_layout(
                pdf_bytes, raw_text, eff_id, metadata, start_time,
            )
            if layout_warning:
                warnings.append(layout_warning)
        if extracted_data is None:
            extracted_data = _parse_academic_sections(
                raw_text_by_page=raw_text,
                paper_id=eff_id,
                metadata_title=metadata.get("title"),
                metadata_author=metadata.get("author"),
                layout_title=layout_title,
            )

        # Determine status
        identified_sections = [
            k for k in (
                "abstract", "research_objectives", "research_questions",
                "methodology", "dataset", "models_and_algorithms",
                "experimental_setup", "evaluation_metrics", "results",
                "limitations", "future_work", "conclusion"
            )
            if getattr(extracted_data, k, None) is not None
        ]

        if (len(identified_sections) >= 2 or total_extracted_chars >= 200) and total_extracted_chars >= 30:
            status = ExtractionStatus.SUCCESS
        elif total_extracted_chars > 0:
            status = ExtractionStatus.PARTIAL
            warnings.append("Extraction yielded sparse text; key academic sections could not be identified.")
        else:
            status = ExtractionStatus.FAILED

        duration = round(time.perf_counter() - start_time, 3)
        logger.info(
            f"Extraction completed for '{eff_id}' in {duration}s | Status: {status} | "
            f"Pages: {page_count} | Method: {used_method} | Evidence count: {len(extracted_data.evidence)}"
        )

        return PaperExtractionResultSchema(
            paper_id=eff_id,
            status=status,
            extraction_method=used_method,
            page_count=page_count,
            document_metadata=metadata,
            extracted_data=extracted_data,
            is_scanned=False,
            ocr_required=False,
            warnings=warnings,
            error_message=None,
            raw_text_by_page=raw_text,
        )

    try:
        return await asyncio.wait_for(_execute(), timeout=settings.PDF_PROCESSING_TIMEOUT)
    except asyncio.TimeoutError as e:
        logger.error(f"Paper processing timed out after {settings.PDF_PROCESSING_TIMEOUT}s for '{effective_id}'")
        raise TimeoutError(f"PDF extraction timed out after {settings.PDF_PROCESSING_TIMEOUT} seconds") from e
