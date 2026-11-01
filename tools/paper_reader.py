"""Tool 2: Research Paper Reader (Category: System & Environment).

Extracts and organizes academic paper dimensions from PDFs using PyMuPDF (primary)
and pdfplumber (fallback). Preserves 1-indexed page numbers, exact section headings,
and verbatim evidence passages. Treats all PDF content as untrusted input.
"""
import asyncio
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


def _parse_academic_sections(
    raw_text_by_page: Dict[int, str],
    paper_id: str,
    metadata_title: Optional[str] = None,
    metadata_author: Optional[str] = None,
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

    # First page inspection for title and authors
    first_page_text = raw_text_by_page.get(1, "")
    first_page_lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]

    extracted_title = metadata_title
    extracted_authors: List[str] = []
    if metadata_author:
        extracted_authors.append(metadata_author)

    if first_page_lines and not extracted_title:
        # Academic conventions: Title is typically the first substantive line before 'abstract'
        for line in first_page_lines[:4]:
            if not SECTION_REGEXES["abstract"].match(line) and len(line) > 10:
                extracted_title = line
                break

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

        # 5. Parse Academic Sections & Evidence References
        extracted_data = _parse_academic_sections(
            raw_text_by_page=raw_text,
            paper_id=eff_id,
            metadata_title=metadata.get("title"),
            metadata_author=metadata.get("author"),
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
