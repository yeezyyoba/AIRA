"""Document Chunking Service for Research Context Retrieval (RAG).

Splits extracted academic paper text into section-aware chunks while preserving
physical page boundaries, document provenance, and verbatim fidelity.
"""
from typing import Any, Dict, List, Optional
import re

from app.core.config import settings
from logging_config.logger import get_logger
from validation.schemas import (
    DocumentChunkSchema,
    ExtractedPaperDataSchema,
    PaperExtractionResultSchema,
)

logger = get_logger("document_chunker")


class DocumentChunker:
    """Configurable, section-aware academic document chunker."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.RAG_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.RAG_CHUNK_OVERLAP

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})"
            )

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks of at most `chunk_size` characters with `chunk_overlap`.

        Preserves sentence and paragraph boundaries without slicing words in half.
        """
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        if len(clean_text) <= self.chunk_size:
            return [clean_text]

        chunks: List[str] = []
        start = 0
        text_len = len(clean_text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            if end < text_len:
                # Find appropriate natural boundary
                split_at = -1
                for sep in ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]:
                    idx = clean_text.rfind(sep, start + self.chunk_overlap, end)
                    if idx != -1:
                        split_at = idx + len(sep)
                        break
                if split_at != -1:
                    end = split_at

            chunk = clean_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end >= text_len:
                break

            # Advance window with overlap
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def _locate_pages(
        self,
        chunk_text: str,
        raw_text_by_page: Dict[int, str],
        fallback_page: int = 1,
    ) -> tuple[int, int]:
        """Locate the starting and ending physical page numbers for a text chunk."""
        if not raw_text_by_page:
            return fallback_page, fallback_page

        chunk_clean = chunk_text.strip()
        needle_len = min(35, len(chunk_clean))
        prefix = chunk_clean[:needle_len]
        suffix = chunk_clean[-needle_len:]

        start_p: Optional[int] = None
        end_p: Optional[int] = None

        for p in sorted(raw_text_by_page.keys()):
            page_text = raw_text_by_page[p]
            if prefix in page_text and start_p is None:
                start_p = p
            if suffix in page_text:
                end_p = p

        p_start = start_p or fallback_page
        p_end = end_p or p_start
        if p_end < p_start:
            p_end = p_start

        return p_start, p_end

    def chunk_paper(
        self,
        paper_id: str,
        extraction_result: PaperExtractionResultSchema,
        document_title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        publication_year: Optional[int] = None,
    ) -> List[DocumentChunkSchema]:
        """Chunk an extracted research paper into structured DocumentChunkSchema items.

        Prioritizes structured academic sections (abstract, methodology, results, etc.).
        Falls back to page-by-page chunking when sections are not delineated.
        """
        if not paper_id:
            raise ValueError("paper_id must be a non-empty string")

        raw_pages = extraction_result.raw_text_by_page or {}
        extracted_data: Optional[ExtractedPaperDataSchema] = extraction_result.extracted_data

        # Determine metadata
        title = document_title or (extracted_data.title if extracted_data else None) or extraction_result.document_metadata.get("title")
        paper_authors = authors or (extracted_data.authors if extracted_data else None) or []
        if not paper_authors and extraction_result.document_metadata.get("author"):
            paper_authors = [extraction_result.document_metadata["author"]]

        chunks: List[DocumentChunkSchema] = []
        chunk_counter = 1

        # Check if we have usable structured academic sections
        has_sections = False
        if extracted_data:
            section_fields = [
                ("Abstract", extracted_data.abstract),
                ("Research Objectives", extracted_data.research_objectives),
                ("Methodology", extracted_data.methodology),
                ("Dataset", extracted_data.dataset),
                ("Models and Algorithms", extracted_data.models_and_algorithms),
                ("Experimental Setup", extracted_data.experimental_setup),
                ("Evaluation Metrics", extracted_data.evaluation_metrics),
                ("Results", extracted_data.results),
                ("Limitations", extracted_data.limitations),
                ("Future Work", extracted_data.future_work),
                ("Conclusion", extracted_data.conclusion),
            ]

            # Also check research questions
            if extracted_data.research_questions:
                rq_text = "\n".join(f"- {rq}" for rq in extracted_data.research_questions)
                section_fields.append(("Research Questions", rq_text))

            # Build evidence map for section starting pages
            evidence_pages: Dict[str, int] = {}
            for ev in extracted_data.evidence:
                if ev.section_heading:
                    evidence_pages[ev.section_heading.lower()] = ev.page_number

            for heading, content in section_fields:
                if not content or not content.strip():
                    continue

                has_sections = True
                fallback_p = evidence_pages.get(heading.lower(), 1)
                sub_chunks = self._split_text(content)

                for sub_chunk in sub_chunks:
                    p_start, p_end = self._locate_pages(sub_chunk, raw_pages, fallback_page=fallback_p)
                    chunk_id = f"{paper_id}_c{chunk_counter:04d}"

                    chunk_schema = DocumentChunkSchema(
                        paper_id=paper_id,
                        chunk_id=chunk_id,
                        page_start=p_start,
                        page_end=p_end,
                        section_heading=heading,
                        text=sub_chunk,
                        source_type="academic_paper",
                        document_title=title,
                        authors=paper_authors,
                        publication_year=publication_year,
                    )
                    chunks.append(chunk_schema)
                    chunk_counter += 1

        # Fallback: if no structured sections were identified, chunk raw pages sequentially
        if not has_sections:
            if not raw_pages:
                logger.warning(f"Paper '{paper_id}' contains no extracted text or pages to chunk.")
                return []

            for p_num in sorted(raw_pages.keys()):
                page_text = raw_pages[p_num]
                if not page_text or not page_text.strip():
                    continue

                page_sub_chunks = self._split_text(page_text)
                for sub_chunk in page_sub_chunks:
                    chunk_id = f"{paper_id}_c{chunk_counter:04d}"
                    chunk_schema = DocumentChunkSchema(
                        paper_id=paper_id,
                        chunk_id=chunk_id,
                        page_start=p_num,
                        page_end=p_num,
                        section_heading=None,
                        text=sub_chunk,
                        source_type="academic_paper",
                        document_title=title,
                        authors=paper_authors,
                        publication_year=publication_year,
                    )
                    chunks.append(chunk_schema)
                    chunk_counter += 1

        logger.info(
            f"Chunked paper '{paper_id}' into {len(chunks)} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks
