"""Tool 7: Research Gap Identifier.

Identifies evidence-supported research gaps from indexed papers:
- Underexplored datasets & languages
- Missing model comparisons & reproducibility gaps
- Methodological limitations & unanswered questions
- Author-proposed future work

Strictly distinguishes AUTHOR_ACKNOWLEDGED_GAP, CROSS_PAPER_GAP, and POTENTIAL_RESEARCH_DIRECTION.
Never claims global novelty based only on a localized corpus.
"""
from typing import Any, Dict, List, Optional

from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from validation.schemas import (
    EvidenceItemSchema,
    EvidenceType,
    GapType,
    PaperExtractionResultSchema,
    ResearchGapSchema,
    ResearchGapsResultSchema,
)

logger = get_logger("gap_identifier")


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

    # 1. Author Acknowledged Gaps (from Limitations and Future Work)
    for pid, paper in loaded_papers.items():
        data = paper.extracted_data
        if not data:
            continue

        if data.limitations and len(data.limitations.strip()) > 10:
            ev_list: List[EvidenceItemSchema] = []
            for ev in data.evidence:
                if ev.section_heading and "limitation" in ev.section_heading.lower():
                    ev_list.append(ev)

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
            ev_list: List[EvidenceItemSchema] = []
            for ev in data.evidence:
                if ev.section_heading and "future" in ev.section_heading.lower():
                    ev_list.append(ev)

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
        if data and data.limitations:
            themes = [label for kw, label in theme_keywords.items() if kw in data.limitations.lower()]
            if themes:
                matched_themes[pid] = themes
                common_limitations.append(pid)
                ev_lim = next((e for e in (data.evidence or []) if e.section_heading and "limitation" in e.section_heading.lower()), None)
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
