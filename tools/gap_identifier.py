"""Tool 7: Research Gap Identifier.

Identifies evidence-supported research gaps from indexed papers:
- Underexplored datasets & languages
- Missing model comparisons & reproducibility gaps
- Methodological limitations & unanswered questions
- Author-proposed future work

Strictly distinguishes AUTHOR_ACKNOWLEDGED_GAP, CROSS_PAPER_GAP, and POTENTIAL_RESEARCH_DIRECTION.
Never claims global novelty based only on a localized corpus.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from validation.schemas import (
    ExtractedPaperDataSchema,
    EvidenceItemSchema,
    EvidenceType,
    GapType,
    PaperExtractionResultSchema,
    ResearchGapSchema,
    ResearchGapsResultSchema,
)

logger = get_logger("gap_identifier")

# Author-acknowledged gaps may only come from these sections. The heading must be the
# whole section name (optionally numbered, optionally "X and Y"), so a body line such
# as "limitation of computation resource languages, we" or "3 Methodology" never counts.
_ACKNOWLEDGEMENT_SECTION = (
    r"(?:limitations?|discussions?|conclusions?|concluding\s+remarks"
    r"|future\s+(?:work|directions?|research)|threats\s+to\s+validity)"
)
_ACKNOWLEDGEMENT_HEADING = re.compile(
    r"^(?:(?:\d{1,2}(?:\.\d{1,2})*|[A-H](?:\.\d{1,2})*)\.?\s+)?"
    r"(?:(?:results|summary|conclusions?)\s+(?:and|&)\s+)?"
    + _ACKNOWLEDGEMENT_SECTION
    + r"(?:\s+(?:and|&)\s+[a-z][a-z\s\-]{0,40})?$",
    re.IGNORECASE,
)


def is_acknowledgement_heading(heading: Optional[str]) -> bool:
    """True if ``heading`` names a Limitations, Discussion, Conclusion or Future Work section."""
    if not heading:
        return False
    cleaned = " ".join(heading.replace("*", "").split()).strip(" :.")
    return bool(_ACKNOWLEDGEMENT_HEADING.match(cleaned))


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _section_provenance(
    data: ExtractedPaperDataSchema,
    section_text: str,
    heading_words: Tuple[str, ...],
) -> Tuple[str, List[EvidenceItemSchema]]:
    """Find the evidence passage(s) the parser recorded for a section and judge their heading.

    Evidence is linked to the section if its heading mentions one of ``heading_words``
    or its passage contains the start of the section text.

    Returns one of:
      ("verified", items)  linked evidence comes from an acknowledgement section heading
      ("rejected", items)  linked evidence comes from any other heading (e.g. Methods)
      ("unknown", [])      no linked evidence (e.g. data built without the PDF parser)
    """
    opening = _normalize(section_text)[:40]
    linked = [
        ev for ev in (data.evidence or [])
        if (ev.section_heading and any(w in ev.section_heading.lower() for w in heading_words))
        or (opening and opening in _normalize(ev.passage))
    ]
    if not linked:
        return "unknown", []
    verified = [ev for ev in linked if is_acknowledgement_heading(ev.section_heading)]
    if verified:
        return "verified", verified
    return "rejected", linked


async def identify_research_gaps(
    paper_ids: List[str],
    research_area: str,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
    router: Optional[ModelRouter] = None,
) -> ResearchGapsResultSchema:
    """Identify evidence-backed research gaps across indexed papers.
    
    Categorizes:
    - AUTHOR_ACKNOWLEDGED_GAP: Directly stated by paper authors in limitations/future work.
    - CROSS_PAPER_GAP: Methodological or evaluation discrepancies between papers (e.g. missing baseline cross-evaluations).
    - POTENTIAL_RESEARCH_DIRECTION: Promising architectural or linguistic extensions supported by findings.
    
    Enforces corpus boundaries and handles insufficient evidence explicitly.
    """
    if not paper_ids:
        raise ValueError("paper_ids must be a non-empty list")
    if not research_area or not research_area.strip():
        raise ValueError("research_area must be a non-empty string")

    repo = repository or default_paper_repo
    store = vector_db or default_vector_store
    llm = router or default_model_router

    loaded_papers: Dict[str, PaperExtractionResultSchema] = {}
    for pid in paper_ids:
        clean_id = pid.strip()
        p = repo.get_paper(clean_id)
        if p:
            loaded_papers[clean_id] = p

    if not loaded_papers:
        logger.warning(f"None of the specified paper IDs ({paper_ids}) were found in the repository.")
        return ResearchGapsResultSchema(
            research_area=research_area,
            paper_ids_analyzed=paper_ids,
            gaps=[],
            insufficient_evidence=True,
            is_degraded=False,
        )

    gaps: List[ResearchGapSchema] = []
    gap_counter = 1

    # 1. Author Acknowledged Gaps (from Limitations and Future Work).
    #    Only text whose recorded heading is a Limitations / Discussion / Conclusion /
    #    Future Work section counts; text the parser took from any other section
    #    (e.g. a Methods sentence mentioning "limitation") is never reported.
    usable_limitations: Dict[str, List[EvidenceItemSchema]] = {}  # pid -> verified evidence ([] = unknown)
    for pid, paper in loaded_papers.items():
        data = paper.extracted_data
        if not data:
            continue

        if data.limitations and len(data.limitations.strip()) > 10:
            provenance, ev_list = _section_provenance(data, data.limitations, ("limitation", "threat"))
            if provenance == "rejected":
                logger.info(
                    f"Ignoring limitations text for '{pid}': it comes from section(s) "
                    f"{[ev.section_heading for ev in ev_list]}, not a Limitations, Discussion, "
                    "Conclusion or Future Work section."
                )
        else:
            provenance, ev_list = "absent", []

        if provenance in ("verified", "unknown"):
            usable_limitations[pid] = ev_list
            # If no dedicated evidence passage exists, preserve text without fabricating page 1
            if not ev_list:
                ev_list = [
                    EvidenceItemSchema(
                        paper_id=pid,
                        page_number=None,
                        section_heading="Limitations",
                        passage=data.limitations.strip()[:300],
                        evidence_type=EvidenceType.INFERENCE,
                    )
                ]

            gaps.append(
                ResearchGapSchema(
                    gap_id=f"gap_{gap_counter:03d}",
                    title=f"Author-Reported Limitation in {pid}",
                    description=data.limitations.strip()[:400],
                    gap_type=GapType.AUTHOR_ACKNOWLEDGED_GAP,
                    supporting_papers=[pid],
                    evidence=ev_list,
                    suggested_direction="Address the specific experimental and modeling constraints identified by the authors.",
                )
            )
            gap_counter += 1

        if data.future_work and len(data.future_work.strip()) > 10:
            provenance, ev_list = _section_provenance(data, data.future_work, ("future",))
            if provenance == "rejected":
                logger.info(
                    f"Ignoring future-work text for '{pid}': it comes from section(s) "
                    f"{[ev.section_heading for ev in ev_list]}, not an acknowledgement section."
                )
        else:
            provenance, ev_list = "absent", []

        if provenance in ("verified", "unknown"):
            if not ev_list:
                ev_list = [
                    EvidenceItemSchema(
                        paper_id=pid,
                        page_number=None,
                        section_heading="Future Work",
                        passage=data.future_work.strip()[:300],
                        evidence_type=EvidenceType.INFERENCE,
                    )
                ]

            gaps.append(
                ResearchGapSchema(
                    gap_id=f"gap_{gap_counter:03d}",
                    title=f"Proposed Future Work in {pid}",
                    description=data.future_work.strip()[:400],
                    gap_type=GapType.AUTHOR_ACKNOWLEDGED_GAP,
                    supporting_papers=[pid],
                    evidence=ev_list,
                    suggested_direction="Implement the specific architectural or data collection extensions suggested by the authors.",
                )
            )
            gap_counter += 1

    # 2. Cross-Paper Gaps (Synthesized from Corpus Discrepancies)
    # Check if multiple papers use isolated datasets without standardized cross-benchmarking
    datasets_seen = set()
    dataset_evidence: List[EvidenceItemSchema] = []
    for pid, paper in loaded_papers.items():
        if paper.extracted_data and paper.extracted_data.dataset and paper.extracted_data.dataset.strip():
            datasets_seen.add(paper.extracted_data.dataset[:50])
            ev_match = None
            for ev in (paper.extracted_data.evidence or []):
                if ev.section_heading and any(w in ev.section_heading.lower() for w in ("dataset", "data", "method", "setup")):
                    ev_match = ev
                    break
            if ev_match:
                dataset_evidence.append(ev_match)
            else:
                dataset_evidence.append(
                    EvidenceItemSchema(
                        paper_id=pid,
                        page_number=None,
                        section_heading="Dataset",
                        passage=f"Evaluates on dataset: {paper.extracted_data.dataset.strip()[:200]}",
                        evidence_type=EvidenceType.INFERENCE,
                    )
                )

    distinct_evidence_papers = set(e.paper_id for e in dataset_evidence)
    if len(datasets_seen) > 1 and len(loaded_papers) >= 2 and len(distinct_evidence_papers) >= 2:
        gaps.append(
            ResearchGapSchema(
                gap_id=f"gap_{gap_counter:03d}",
                title="Lack of Standardized Cross-Dataset Benchmark Evaluation",
                description=(
                    f"The analyzed papers ({', '.join(loaded_papers.keys())}) evaluate models on disparate, non-overlapping datasets. "
                    "The absence of a common shared benchmark prevents direct empirical comparison of generalization capabilities."
                ),
                gap_type=GapType.CROSS_PAPER_GAP,
                supporting_papers=list(distinct_evidence_papers),
                evidence=dataset_evidence,
                suggested_direction="Evaluate models across a unified multi-domain benchmark split.",
            )
        )
        gap_counter += 1

    # 3. Potential Research Directions (from Common Empirical Limitations)
    common_limitations = []
    matched_themes: Dict[str, List[str]] = {}
    direction_evidence: List[EvidenceItemSchema] = []
    theme_keywords = {
        "noise": "noise robustness",
        "compute": "compute cost",
        "scale": "scalability",
        "latency": "latency",
        "domain": "domain transfer",
        "robust": "robustness",
        "generaliz": "generalization",
        "low-resource": "low-resource data scarcity",
    }
    for pid, paper in loaded_papers.items():
        data = paper.extracted_data
        if data and data.limitations and pid in usable_limitations:
            themes = [label for kw, label in theme_keywords.items() if kw in data.limitations.lower()]
            if themes:
                matched_themes[pid] = themes
                common_limitations.append(pid)
                ev_lim = next(iter(usable_limitations[pid]), None)
                if ev_lim:
                    direction_evidence.append(ev_lim)
                else:
                    direction_evidence.append(
                        EvidenceItemSchema(
                            paper_id=pid,
                            page_number=None,
                            section_heading="Limitations",
                            passage=data.limitations.strip()[:250],
                            evidence_type=EvidenceType.INFERENCE,
                        )
                    )

    if len(common_limitations) >= 2 and len(direction_evidence) >= 2:
        gaps.append(
            ResearchGapSchema(
                gap_id=f"gap_{gap_counter:03d}",
                title=f"Shared Reported Limitations Across Studies in {research_area}",
                description=(
                    "Multiple studies report related limitations in their own words: "
                    + "; ".join(f"{pid} ({', '.join(themes)})" for pid, themes in matched_themes.items())
                    + ". Addressing these shared constraints is a potential research direction supported by the cited passages."
                ),
                gap_type=GapType.POTENTIAL_RESEARCH_DIRECTION,
                supporting_papers=list(set(e.paper_id for e in direction_evidence)),
                evidence=direction_evidence,
                suggested_direction=(
                    "Investigate methods targeting the shared limitation themes: "
                    + ", ".join(sorted({t for themes in matched_themes.values() for t in themes})) + "."
                ),
            )
        )
        gap_counter += 1

    # Check for insufficient evidence
    insufficient = (len(gaps) == 0)
    is_degraded = llm.is_degraded_mode() if llm else False

    return ResearchGapsResultSchema(
        research_area=research_area,
        paper_ids_analyzed=list(loaded_papers.keys()),
        gaps=gaps,
        insufficient_evidence=insufficient,
        corpus_scope_disclaimer=(
            f"Identified {len(gaps)} research gaps strictly within the scope of the {len(loaded_papers)} analyzed papers. "
            "This analysis reflects corpus coverage and does not claim global novelty across unindexed literature."
        ),
        is_degraded=is_degraded,
    )
