"""AIRA Research Comparison Report Generation Service.

Generates comprehensive, cited academic research comparison reports combining:
- Objectives and methodology comparisons
- Dataset and model comparisons
- Experimental metric tables
- Strengths and limitations
- Evidence-supported research gaps
- Traceable citations and limitations of analysis

Supports exporting reports to Markdown and JSON formats.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.core.config import settings
from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from tools.gap_identifier import identify_research_gaps
from tools.methodology_analyzer import analyze_research_methodology
from tools.paper_comparator import compare_research_papers
from tools.result_analyzer import analyze_experimental_results
from validation.schemas import (
    MetricComparisonResultSchema,
    ResearchComparisonReportSchema,
    ResearchGapSchema,
)

logger = get_logger("report_service")


async def generate_research_comparison_report(
    paper_ids: List[str],
    research_question: str,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
    router: Optional[ModelRouter] = None,
) -> ResearchComparisonReportSchema:
    """Generate a structured 11-section research comparison report with traceable citations."""
    if not paper_ids or len(paper_ids) < 2:
        raise ValueError("Report generation requires at least two valid paper_ids")
    if not research_question or not research_question.strip():
        raise ValueError("research_question must be a non-empty string")

    clean_question = research_question.strip()
    repo = repository or default_paper_repo
    store = vector_db or default_vector_store
    llm = router or default_model_router

    report_id = f"report_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow().isoformat() + "Z"

    # 1. Run tools to gather structured factual comparisons
    comparison = await compare_research_papers(paper_ids, repository=repo, vector_db=store, router=llm)
    metric_tables = analyze_experimental_results(paper_ids, repository=repo, vector_db=store)
    gaps_result = await identify_research_gaps(paper_ids, research_area=clean_question, repository=repo, vector_db=store, router=llm)

    # 2. Gather individual methodology analyses
    methodologies = {}
    references = []
    for pid in paper_ids:
        meth = await analyze_research_methodology(pid, repository=repo, vector_db=store, router=llm)
        methodologies[pid] = meth
        p_obj = repo.get_paper(pid)
        ref_title = p_obj.extracted_data.title if p_obj and p_obj.extracted_data and p_obj.extracted_data.title else pid
        ref_authors = p_obj.extracted_data.authors if p_obj and p_obj.extracted_data and p_obj.extracted_data.authors else ["Unknown"]
        references.append({
            "paper_id": pid,
            "title": ref_title,
            "authors": ref_authors,
            "page_count": p_obj.page_count if p_obj else 0,
        })

    is_degraded = llm.is_degraded_mode() if llm else False

    # 3. Construct 11 sections
    obj_meth_text = f"Comparative evaluation of {len(paper_ids)} studies addressing: {clean_question}.\n\n"
    for dim in comparison.matrix:
        if dim.dimension_name in ("Research Objectives", "Methodology"):
            obj_meth_text += f"### {dim.dimension_name}\n"
            for pid, val in dim.values_by_paper.items():
                obj_meth_text += f"- **{pid}**: {val or 'Not explicitly documented in paper text'}\n"
            if dim.differences_noted:
                diffs = "; ".join(dim.differences_noted)
                obj_meth_text += f"*Differences*: {diffs}\n\n"

    dataset_text = "### Dataset Characteristics & Scale\n"
    for dim in comparison.matrix:
        if dim.dimension_name in ("Datasets", "Dataset Sizes", "Languages Covered"):
            dataset_text += f"#### {dim.dimension_name}\n"
            for pid, val in dim.values_by_paper.items():
                dataset_text += f"- **{pid}**: {val or 'Not reported'}\n"
            dataset_text += "\n"

    arch_text = "### Models & Algorithmic Architectures\n"
    for dim in comparison.matrix:
        if dim.dimension_name in ("Models and Algorithms", "Experimental Setup"):
            arch_text += f"#### {dim.dimension_name}\n"
            for pid, val in dim.values_by_paper.items():
                arch_text += f"- **{pid}**: {val or 'Not reported'}\n"
            arch_text += "\n"

    exp_text = "### Quantitative Results & Metric Comparisons\n"
    if metric_tables:
        for mt in metric_tables:
            exp_text += f"#### Metric: {mt.metric_name.upper()} (Goal: {mt.goal.value})\n"
            exp_text += f"Status: **{mt.compatibility.value}**\n"
            if mt.incompatibility_reason:
                exp_text += f"> **Warning**: {mt.incompatibility_reason}\n\n"
            if mt.trade_off_notes:
                exp_text += f"> **Trade-off Analysis**: {mt.trade_off_notes}\n\n"

            if mt.ranked_models:
                exp_text += "| Rank | Paper ID | Model | Score | Delta | Evidence Page |\n"
                exp_text += "|---|---|---|---|---|---|\n"
                for r in mt.ranked_models:
                    page_str = f"Page {r['page_number']}" if r.get('page_number') is not None else "Page: Unspecified"
                    exp_text += f"| {r['rank']} | {r['paper_id']} | {r['model_name']} | {r['score']} | {r['delta_from_best']} | {page_str} |\n"
                exp_text += "\n"

            if mt.missing_data_warnings:
                exp_text += "*Statistical Notes*:\n"
                for w in mt.missing_data_warnings:
                    exp_text += f"- {w}\n"
                exp_text += "\n"
    else:
        exp_text += "No compatible numerical metrics extracted across the analyzed papers.\n\n"

    strengths_text = "### Strengths, Limitations & Artifact Availability Across Corpus\n"
    for pid, meth in methodologies.items():
        strengths_text += f"#### Paper: {pid}\n"
        if meth.artifact_availability:
            art = meth.artifact_availability
            strengths_text += f"- **Artifact Availability**: Code: {art.get('code_status', 'UNSPECIFIED')}"
            if art.get("repository_url"):
                strengths_text += f" ({art.get('repository_url')})"
            strengths_text += f" | Dataset: {art.get('dataset_status', 'UNSPECIFIED')}"
            strengths_text += f" | Hyperparameters: {art.get('hyperparameters_status', 'UNSPECIFIED')}"
            strengths_text += f" | Random Seeds: {art.get('random_seeds_status', 'UNSPECIFIED')}\n"
        elif meth.reproducibility_score:
            strengths_text += f"- **Reproducibility Indicator**: {meth.reproducibility_score}\n"

        if meth.reported_limitations:
            for lim in meth.reported_limitations:
                page_str = f"Page {lim.page_number}" if lim.page_number is not None else "Page: Unspecified"
                strengths_text += f"- **Reported Limitation ({page_str})**: {lim.passage}\n"
        else:
            strengths_text += "- No dedicated limitations section delineated in paper text.\n"
        strengths_text += "\n"

    conclusions_text = (
        f"Based on the empirical evidence extracted from {len(paper_ids)} papers, "
        "models demonstrate domain-specific efficacy on the analyzed benchmarks. "
        "However, direct cross-paper superiority cannot be claimed where metric variants or evaluation splits diverge."
    )

    limitations_analysis_text = (
        f"This analysis is strictly bounded to the {len(paper_ids)} papers currently indexed in the knowledge base. "
        "Unreported hyperparameters, unreleased datasets, and missing variance measures (std dev, p-values) "
        "limit absolute statistical comparability."
    )

    return ResearchComparisonReportSchema(
        report_id=report_id,
        research_question=clean_question,
        papers_analyzed=paper_ids,
        objectives_and_methodology_comparison=obj_meth_text.strip(),
        dataset_comparison=dataset_text.strip(),
        model_architecture_comparison=arch_text.strip(),
        experimental_results_comparison=exp_text.strip(),
        strengths_and_limitations=strengths_text.strip(),
        research_gaps=gaps_result.gaps,
        evidence_backed_conclusions=conclusions_text.strip(),
        limitations_of_analysis=limitations_analysis_text.strip(),
        references=references,
        metric_tables=metric_tables,
        is_degraded=is_degraded,
        created_at=created_at,
    )


def export_report_markdown(report: ResearchComparisonReportSchema) -> str:
    """Export comparison report to a structured, publication-ready Markdown document."""
    md = []
    md.append(f"# Research Comparison Report: {report.research_question}\n")
    md.append(f"*Report ID*: {report.report_id} | *Generated*: {report.created_at}\n")
    if report.is_degraded:
        md.append("> **Notice**: Generated in Degraded Mode (Deterministic extraction active; LLM reasoning unavailable).\n")

    md.append("## 1. Research Question")
    md.append(f"{report.research_question}\n")

    md.append("## 2. Papers Analyzed")
    for ref in report.references:
        authors = ", ".join(ref.get("authors", [])) if isinstance(ref.get("authors"), list) else str(ref.get("authors"))
        md.append(f"- **{ref['paper_id']}**: *{ref['title']}* ({authors}) [{ref.get('page_count', 0)} pages]")
    md.append("")

    md.append("## 3. Comparison of Objectives and Methodologies")
    md.append(report.objectives_and_methodology_comparison)
    md.append("")

    md.append("## 4. Dataset Comparison")
    md.append(report.dataset_comparison)
    md.append("")

    md.append("## 5. Model & Architecture Comparison")
    md.append(report.model_architecture_comparison)
    md.append("")

    md.append("## 6. Experimental Results Comparison")
    md.append(report.experimental_results_comparison)
    md.append("")

    md.append("## 7. Strengths and Limitations")
    md.append(report.strengths_and_limitations)
    md.append("")

    md.append("## 8. Identified Research Gaps")
    if report.research_gaps:
        for g in report.research_gaps:
            md.append(f"### [{g.gap_type.value}] {g.title}")
            md.append(f"{g.description}\n")
            sup_papers = ", ".join(g.supporting_papers)
            md.append(f"- **Supporting Papers**: {sup_papers}")
            if g.suggested_direction:
                md.append(f"- **Suggested Direction**: {g.suggested_direction}")
            if g.evidence:
                md.append("- **Evidence Citations**:")
                for ev in g.evidence:
                    page_str = f"Page {ev.page_number}" if ev.page_number is not None else "Page: Unspecified"
                    md.append(f"  - *{ev.paper_id} ({page_str})*: \"{ev.passage}\"")
            md.append("")
    else:
        md.append("No prominent research gaps identified across the specified corpus.\n")

    md.append("## 9. Evidence-Backed Conclusions")
    md.append(report.evidence_backed_conclusions)
    md.append("")

    md.append("## 10. Missing Information & Limitations of the Analysis")
    md.append(report.limitations_of_analysis)
    md.append("")

    md.append("## 11. References to Original Papers")
    for idx, ref in enumerate(report.references, start=1):
        md.append(f"[{idx}] **{ref['paper_id']}**: {ref['title']}. Pages: {ref.get('page_count', 0)}.")
    md.append("")

    return "\n".join(md)


def export_report_json(report: ResearchComparisonReportSchema) -> str:
    """Export comparison report to a standardized JSON string."""
    return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)
