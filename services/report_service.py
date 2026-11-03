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
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional
import uuid

from llm.model_router import ModelRouter, model_router as default_model_router
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from tools.gap_identifier import identify_research_gaps
from tools.methodology_analyzer import analyze_research_methodology
from tools.paper_comparator import compare_research_papers
from tools.result_analyzer import analyze_experimental_results
from validation.schemas import (
    MethodologyAnalysisSchema,
    MetricComparisonResultSchema,
    PaperComparisonSchema,
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
    precomputed: Optional[Dict[str, Any]] = None,
) -> ResearchComparisonReportSchema:
    """Generate a structured 11-section research comparison report with traceable citations.

    ``precomputed`` may carry analyses the agent already executed (comparison,
    methodologies, metric_tables, gaps, citations). They are reused instead of
    re-running the tools, so the report reflects exactly the evidence the agent
    gathered and no LLM call is repeated. Missing entries are computed here.
    """
    if not paper_ids or len(paper_ids) < 2:
        raise ValueError("Report generation requires at least two valid paper_ids")
    if not research_question or not research_question.strip():
        raise ValueError("research_question must be a non-empty string")

    clean_question = research_question.strip()
    repo = repository or default_paper_repo
    store = vector_db or default_vector_store
    llm = router or default_model_router

    report_id = f"report_{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()
    pre = precomputed or {}

    # 1. Reuse or run the tools that gather structured factual comparisons
    comparison = _coerce(pre.get("comparison"), PaperComparisonSchema)
    if comparison is None or set(comparison.paper_ids) != set(paper_ids):
        comparison = await compare_research_papers(paper_ids, repository=repo, vector_db=store, router=llm)

    if pre.get("metric_tables") is not None:
        metric_tables = [
            m for m in (_coerce(item, MetricComparisonResultSchema) for item in pre["metric_tables"]) if m
        ]
    else:
        metric_tables = analyze_experimental_results(paper_ids, repository=repo, vector_db=store)

    if pre.get("gaps") is not None:
        research_gaps = [g for g in (_coerce(item, ResearchGapSchema) for item in pre["gaps"]) if g]
    else:
        gaps_result = await identify_research_gaps(
            paper_ids, research_area=clean_question, repository=repo, vector_db=store, router=llm,
        )
        research_gaps = gaps_result.gaps

    # 2. Gather individual methodology analyses
    reused_methodologies = {
        m.paper_id: m
        for m in (_coerce(item, MethodologyAnalysisSchema) for item in pre.get("methodologies") or [])
        if m
    }
    citations = pre.get("citations") or {}
    methodologies = {}
    references = []
    for pid in paper_ids:
        meth = reused_methodologies.get(pid)
        if meth is None:
            meth = await analyze_research_methodology(pid, repository=repo, vector_db=store, router=llm)
        methodologies[pid] = meth
        p_obj = repo.get_paper(pid)
        verified = (pre.get("paper_metadata") or {}).get(pid) or {}
        extracted_title = p_obj.extracted_data.title if p_obj and p_obj.extracted_data else None
        extracted_authors = p_obj.extracted_data.authors if p_obj and p_obj.extracted_data else None
        ref_title = verified.get("title") or extracted_title or pid
        ref_authors = verified.get("authors") or extracted_authors or []
        citation = citations.get(pid) if isinstance(citations.get(pid), dict) else {}
        ref_year = verified.get("year")
        references.append({
            "paper_id": pid,
            "label": paper_label(pid, ref_title, ref_authors, ref_year),
            "title": ref_title,
            "authors": ref_authors,
            "year": ref_year,
            "doi": verified.get("doi") or citation.get("doi"),
            "page_count": p_obj.page_count if p_obj else 0,
            "citation_text": citation.get("citation_text"),
        })
    labels = {ref["paper_id"]: ref["label"] for ref in references}

    is_degraded = llm.is_degraded_mode() if llm else False

    # 3. Construct 11 sections
    obj_meth_text = f"Comparative evaluation of {len(paper_ids)} studies addressing: {clean_question}.\n\n"
    for dim in comparison.matrix:
        if dim.dimension_name in ("Research Objectives", "Methodology"):
            obj_meth_text += f"### {dim.dimension_name}\n"
            for pid, val in dim.values_by_paper.items():
                obj_meth_text += f"- **{labels.get(pid, pid)}**: {_excerpt(val) if val else 'Not explicitly documented in paper text'}\n"
            if dim.differences_noted:
                diffs = "; ".join(dim.differences_noted)
                obj_meth_text += f"*Differences*: {diffs}\n\n"

    dataset_text = "### Dataset Characteristics & Scale\n"
    for dim in comparison.matrix:
        if dim.dimension_name in ("Datasets", "Dataset Sizes", "Languages Covered"):
            dataset_text += f"#### {dim.dimension_name}\n"
            for pid, val in dim.values_by_paper.items():
                dataset_text += f"- **{labels.get(pid, pid)}**: {_excerpt(val) if val else 'Not reported'}\n"
            dataset_text += "\n"

    arch_text = "### Models & Algorithmic Architectures\n"
    for dim in comparison.matrix:
        if dim.dimension_name in ("Models and Algorithms", "Experimental Setup"):
            arch_text += f"#### {dim.dimension_name}\n"
            for pid, val in dim.values_by_paper.items():
                arch_text += f"- **{labels.get(pid, pid)}**: {_excerpt(val) if val else 'Not reported'}\n"
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
                exp_text += "| Rank | Paper | Model | Score | Delta | Evidence Page |\n"
                exp_text += "|---|---|---|---|---|---|\n"
                for r in mt.ranked_models:
                    page_str = f"Page {r['page_number']}" if r.get('page_number') is not None else "Page: Unspecified"
                    exp_text += f"| {r['rank']} | {labels.get(r['paper_id'], r['paper_id'])} | {r['model_name']} | {r['score']} | {r['delta_from_best']} | {page_str} |\n"
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
        strengths_text += f"#### {labels.get(pid, pid)}\n"
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
                strengths_text += f"- **Reported Limitation ({page_str})**: {_excerpt(lim.passage)}\n"
        else:
            strengths_text += "- No dedicated limitations section delineated in paper text.\n"
        strengths_text += "\n"

    conclusions_text = _evidence_backed_conclusions(paper_ids, metric_tables, research_gaps)

    limitations_analysis_text = (
        f"This analysis is strictly bounded to the {len(paper_ids)} papers currently indexed in the knowledge base. "
        "Unreported hyperparameters, unreleased datasets, and missing variance measures (std dev, p-values) "
        "limit absolute statistical comparability."
    )

    return ResearchComparisonReportSchema(
        report_id=report_id,
        title=f"Research Comparison Report: {clean_question}",
        report_status="COMPLETE",
        research_question=clean_question,
        papers_analyzed=paper_ids,
        objectives_and_methodology_comparison=obj_meth_text.strip(),
        dataset_comparison=dataset_text.strip(),
        model_architecture_comparison=arch_text.strip(),
        experimental_results_comparison=exp_text.strip(),
        strengths_and_limitations=strengths_text.strip(),
        research_gaps=research_gaps,
        evidence_backed_conclusions=conclusions_text.strip(),
        limitations_of_analysis=limitations_analysis_text.strip(),
        references=references,
        metric_tables=metric_tables,
        is_degraded=is_degraded,
        created_at=created_at,
    )


def _last_name(author: str) -> str:
    author = (author or "").strip()
    if "," in author:
        return author.split(",")[0].strip()
    parts = author.split()
    return parts[-1] if parts else ""


def paper_label(pid: str, title: Optional[str] = None, authors: Optional[List[str]] = None,
                year: Optional[int] = None) -> str:
    """Human-readable short name for a paper: "Muhammad et al., 2022".

    Paper IDs are usually DOIs, which readers cannot tell apart. Falls back to
    a shortened title, then to the ID itself; nothing is invented.
    """
    names = [a for a in (authors or []) if a and a.strip()]
    if len(names) == 1:
        who = _last_name(names[0])
    elif len(names) == 2:
        who = f"{_last_name(names[0])} & {_last_name(names[1])}"
    elif len(names) > 2:
        who = f"{_last_name(names[0])} et al."
    else:
        who = ""
    if who:
        return f"{who}, {year}" if year else who
    if title and title != pid:
        return title if len(title) <= 48 else title[:45].rstrip() + "…"
    return pid


EXCERPT_CHARS = 450


def _excerpt(text: Optional[str], limit: int = EXCERPT_CHARS) -> str:
    """A readable, single-paragraph excerpt of extracted paper text.

    The report is for reading; the complete extracted sections remain in the
    paper repository and in the comparison matrix of the JSON export.
    """
    if not text:
        return ""
    flat = " ".join(str(text).replace("-\n", "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return f"{cut} … *(excerpt; {len(flat):,} characters extracted)*"


def _coerce(value: Any, schema: Any) -> Any:
    """Return ``value`` as ``schema`` if possible, else None (never raises)."""
    if value is None:
        return None
    if isinstance(value, schema):
        return value
    if isinstance(value, dict):
        try:
            return schema(**value)
        except Exception as exc:
            logger.warning("Ignoring precomputed %s that failed validation: %s", schema.__name__, exc)
    return None


def _evidence_backed_conclusions(
    paper_ids: List[str],
    metric_tables: List[MetricComparisonResultSchema],
    research_gaps: List[ResearchGapSchema],
) -> str:
    """Build conclusions strictly from computed comparisons; never generic claims."""
    lines: List[str] = []
    for table in metric_tables:
        metric = table.metric_name.upper()
        if table.compatibility.value == "COMPATIBLE" and table.ranked_models:
            best = table.ranked_models[0]
            page = best.get("page_number")
            page_note = f", page {page}" if page is not None else ""
            lines.append(
                f"- On {metric} ({table.goal.value.replace('_', ' ').lower()}), the best reported comparable result is "
                f"{best.get('model_name')} with {best.get('score')} ({best.get('paper_id')}{page_note})."
            )
        else:
            reason = table.incompatibility_reason or "evaluation conditions are not comparable or not reported"
            reason = " ".join(reason.split())
            if len(reason) > 300:
                reason = reason[:297].rstrip() + "..."
            lines.append(f"- {metric}: no cross-paper ranking is made because {reason}.")
    if research_gaps:
        lines.append(f"- {len(research_gaps)} research gap(s) were identified from the papers' own reported limitations and differences.")
    if not lines:
        return (
            f"The {len(paper_ids)} analyzed papers did not report metrics that can be compared on equal terms, "
            "so no empirical ranking or superiority claim is made. See the dimension-by-dimension comparison above."
        )
    return "Conclusions are limited to what the extracted evidence supports:\n" + "\n".join(lines)


def export_report_markdown(report: ResearchComparisonReportSchema) -> str:
    """Export comparison report to a structured, publication-ready Markdown document."""
    md = []
    labels = {
        ref.get("paper_id"): ref.get("label") or paper_label(ref.get("paper_id"), ref.get("title"), ref.get("authors"), ref.get("year"))
        for ref in report.references if ref.get("paper_id")
    }

    def relabel(text: str) -> str:
        for pid, label in labels.items():
            if pid and pid in text:
                text = text.replace(pid, label)
        return text

    md.append(f"# {report.title or ('Research Comparison Report: ' + report.research_question)}\n")
    if report.report_status and report.report_status != "COMPLETE":
        md.append(f"> **Status**: {report.report_status} — sections marked 'Not established' lack sufficient evidence.\n")
    md.append(f"*Report ID*: {report.report_id} | *Generated*: {report.created_at}\n")
    if report.is_degraded:
        md.append("> **Notice**: Generated in Degraded Mode (Deterministic extraction active; LLM reasoning unavailable).\n")

    md.append("## 1. Research Question")
    md.append(f"{report.research_question}\n")

    md.append("## 2. Papers Analyzed")
    for ref in report.references:
        authors = ", ".join(ref.get("authors", [])) if isinstance(ref.get("authors"), list) else str(ref.get("authors"))
        label = ref.get("label") or paper_label(ref.get("paper_id"), ref.get("title"), ref.get("authors"), ref.get("year"))
        doi = f" — https://doi.org/{ref['doi']}" if ref.get("doi") else ""
        md.append(f"- **{label}**: *{ref.get('title')}* ({authors or 'authors not reported'}) [{ref.get('page_count', 0)} pages]{doi}")
    md.append("")

    md.append("## 3. Comparison of Objectives and Methodologies")
    md.append(relabel(report.objectives_and_methodology_comparison))
    md.append("")

    md.append("## 4. Dataset Comparison")
    md.append(relabel(report.dataset_comparison))
    md.append("")

    md.append("## 5. Model & Architecture Comparison")
    md.append(relabel(report.model_architecture_comparison))
    md.append("")

    md.append("## 6. Experimental Results Comparison")
    md.append(relabel(report.experimental_results_comparison))
    md.append("")

    md.append("## 7. Strengths and Limitations")
    md.append(relabel(report.strengths_and_limitations))
    md.append("")

    md.append("## 8. Identified Research Gaps")
    if report.research_gaps:
        for g in report.research_gaps:
            md.append(f"### [{g.gap_type.value}] {relabel(g.title)}")
            md.append(f"{g.description}\n")
            sup_papers = ", ".join(labels.get(p, p) for p in g.supporting_papers)
            md.append(f"- **Supporting Papers**: {sup_papers}")
            if g.suggested_direction:
                md.append(f"- **Suggested Direction**: {g.suggested_direction}")
            if g.evidence:
                md.append("- **Evidence Citations**:")
                for ev in g.evidence:
                    page_str = f"Page {ev.page_number}" if ev.page_number is not None else "Page: Unspecified"
                    md.append(f"  - *{labels.get(ev.paper_id, ev.paper_id)} ({page_str})*: \"{ev.passage}\"")
            md.append("")
    else:
        md.append("No prominent research gaps identified across the specified corpus.\n")

    md.append("## 9. Evidence-Backed Conclusions")
    md.append(relabel(report.evidence_backed_conclusions))
    md.append("")

    md.append("## 10. Missing Information & Limitations of the Analysis")
    md.append(relabel(report.limitations_of_analysis))
    md.append("")

    md.append("## 11. References to Original Papers")
    for idx, ref in enumerate(report.references, start=1):
        if ref.get("citation_text"):
            md.append(f"[{idx}] {ref['citation_text']}")
        else:
            md.append(f"[{idx}] **{labels.get(ref.get('paper_id'), ref.get('paper_id'))}**: {ref.get('title')}. Pages: {ref.get('page_count', 0)}.")
    md.append("")

    md.append("---")
    md.append("*Generated by AIRA (Amharic Intelligent Research Assistant). Content is bounded to the extracted evidence listed above.*")
    return "\n".join(md)


def export_report_json(report: ResearchComparisonReportSchema) -> str:
    """Export comparison report to a standardized JSON string."""
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False)
