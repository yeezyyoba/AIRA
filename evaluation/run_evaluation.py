"""AIRA Phase 9 evaluation runner and evidence-based report generator.

All evaluator outputs are preserved. Failures are isolated per evaluator,
and the report distinguishes measured results from proxies and limitations.
"""
import asyncio
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evaluation.evaluators.agentic_evaluator import AgenticEvaluator
from evaluation.evaluators.analysis_evaluator import AnalysisEvaluator
from evaluation.evaluators.extraction_evaluator import PDFExtractionEvaluator
from evaluation.evaluators.rag_evaluator import RAGEvaluator
from evaluation.evaluators.reliability_evaluator import ReliabilityEvaluator
from evaluation.evaluators.search_evaluator import AcademicSearchEvaluator


def _section(results: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = results.get(key, {})
    return value if isinstance(value, dict) else {}


def _metrics(results: Dict[str, Any], section: str, key: str) -> Dict[str, Any]:
    value = _section(results, section).get(key, {})
    return value if isinstance(value, dict) else {}


def _num(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = data.get(key, default)
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _status(results: Dict[str, Any], key: str) -> str:
    statuses = results.get("evaluation_status", {})
    return statuses.get(key, "not_recorded") if isinstance(statuses, dict) else "not_recorded"


def _result_line(results: Dict[str, Any], key: str) -> str:
    status = _status(results, key)
    if status == "completed":
        return "Completed"
    if status == "failed":
        return "Failed"
    if status == "not_run":
        return "Not run"
    return status.replace("_", " ").capitalize()


def generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate a report that does not overstate what the evaluators measure."""
    search = _section(results, "academic_search")
    search_m = _metrics(results, "academic_search", "aggregate_metrics")
    pdf = _section(results, "pdf_extraction")
    pdf_m = _metrics(results, "pdf_extraction", "aggregate_metrics")
    rag = _section(results, "rag_retrieval")
    rag_m = _metrics(results, "rag_retrieval", "overall_metrics")
    rag_en = _metrics(results, "rag_retrieval", "english_metrics")
    rag_am = _metrics(results, "rag_retrieval", "amharic_metrics")
    analysis = _section(results, "analysis_comparison")
    agent = _section(results, "agentic_workflow")
    reliability = _section(results, "reliability")

    p5 = _num(search_m, "mean_precision_at_5")
    p10 = _num(search_m, "mean_precision_at_10")
    r5 = _num(search_m, "mean_recall_at_5")
    r10 = _num(search_m, "mean_recall_at_10")
    mrr = _num(search_m, "mrr")
    ndcg5 = _num(search_m, "mean_ndcg_at_5")
    ndcg10 = _num(search_m, "mean_ndcg_at_10")

    # Separate digital document field extraction from scanned-document handling.
    archetypes = pdf.get("per_archetype_results", [])
    digital_scores = [
        score
        for item in archetypes if isinstance(item, dict)
        and item.get("paper_id") != "eval_pdf_scanned_04"
        for score in (item.get("field_scores", {}).values(),)
    ]
    digital_values = [v for group in digital_scores for v in group if isinstance(v, (int, float))]
    digital_field_accuracy = (
        sum(digital_values) / len(digital_values) if digital_values else None
    )
    digital_field_count = len(digital_values)

    scanned = [
        item for item in archetypes
        if isinstance(item, dict) and item.get("paper_id") == "eval_pdf_scanned_04"
    ]
    scanned_handling = None
    if scanned:
        scores = scanned[0].get("field_scores", {})
        vals = [v for v in scores.values() if isinstance(v, (int, float))]
        if vals:
            scanned_handling = sum(vals) / len(vals)

    rag_total = rag.get("total_queries", 0)
    rag_evaluated = rag.get("evaluated_queries", rag_m.get("count", 0))
    rag_excluded = rag.get("excluded_queries", max(0, rag_total - rag_evaluated))
    embedding = rag.get("embedding_provider") or rag.get("embedding_provider_evaluated", "Not specified")
    correct_paper = _num(rag_m, "correct_paper_rate")
    correct_page = _num(rag_m, "correct_page_rate")
    passage_support = _num(rag_m, "mean_ground_truth_passage_support",
                           _num(rag_m, "mean_faithfulness"))
    unsupported_passage = _num(rag_m, "mean_unsupported_passage_rate",
                               _num(rag_m, "mean_unsupported_rate"))

    analysis_passed = analysis.get("passed_cases", 0)
    analysis_total = analysis.get("total_cases", 0)
    agent_passed = agent.get("passed_scenarios", 0)
    agent_total = agent.get("total_scenarios", 0)
    stop_accuracy = _num(agent, "stopping_condition_accuracy")
    reliability_passed = reliability.get("passed_fault_injections", 0)
    reliability_total = reliability.get("total_fault_injections", 0)
    reliability_score = _num(reliability, "reliability_score")

    eval_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    failed = [
        key for key, value in results.get("evaluation_status", {}).items()
        if value == "failed"
    ]

    lines = [
        "# AIRA Phase 9: Comprehensive System Evaluation and Reliability Report",
        "",
        f"**Report generated:** {eval_time}",
        "**Evaluation mode:** Offline deterministic and mocked evaluation. These results are not a substitute for live-provider or real-world validation.",
        "**Language scope:** English and Amharic benchmark cases.",
        f"**RAG embedding provider:** {embedding}",
        "",
        "## 1. Executive Summary",
        "",
        "This report summarizes six evaluation components: academic search, PDF extraction, RAG retrieval, research analysis, agentic workflow checks, and reliability checks.",
        "",
        "The benchmark suite primarily uses synthetic documents, mocked academic-provider responses, deterministic fixtures, and mock embeddings. It verifies selected software behaviors under controlled conditions. It does not establish general real-world research quality, live academic-search quality, or generated-answer factuality.",
        "",
        "### Evaluation execution status",
        "",
        "| Component | Execution status |",
        "|---|---|",
    ]

    for key, label in [
        ("academic_search", "Academic search"),
        ("pdf_extraction", "PDF extraction"),
        ("rag_retrieval", "RAG retrieval"),
        ("analysis_comparison", "Research analysis"),
        ("agentic_workflow", "Agentic workflow"),
        ("reliability", "Reliability"),
    ]:
        lines.append(f"| {label} | {_result_line(results, key)} |")

    lines += [
        "",
        f"**Failed evaluators:** {', '.join(failed) if failed else 'None recorded'}",
        "",
        "### Key measured results",
        "",
        "| Area | Result | Interpretation |",
        "|---|---:|---|",
        f"| Search P@5 | {p5:.4f} | Mock-candidate pipeline metric |",
        f"| Search R@5 | {r5:.4f} | Mock-candidate pipeline metric |",
        f"| Search MRR | {mrr:.4f} | Mock-candidate pipeline metric |",
        f"| Search nDCG@5 | {ndcg5:.4f} | Mock-candidate pipeline metric |",
        f"| Digital PDF field accuracy | {f'{digital_field_accuracy:.4f}' if digital_field_accuracy is not None else 'Not available'} | Synthetic digital PDFs only; {digital_field_count} field checks |",
        f"| Scanned PDF handling score | {f'{scanned_handling:.4f}' if scanned_handling is not None else 'Not evaluated'} | OCR-needed detection and graceful handling, not OCR text accuracy |",
        f"| RAG correct-paper rate | {_pct(correct_paper)} | Mock-embedding retrieval plumbing |",
        f"| RAG correct-page rate | {_pct(correct_page)} | Page-level benchmark coverage |",
        f"| Analysis challenge cases | {analysis_passed}/{analysis_total} | Component-level challenge checks |",
        f"| Agentic scenarios | {agent_passed}/{agent_total} | Node/routing scenario checks, not end-to-end task success |",
        f"| Reliability checks | {reliability_passed}/{reliability_total} | Four specific fault-handling checks |",
        "",
        "---",
        "",
        "## 2. Evaluation Architecture",
        "",
        "```mermaid",
        "flowchart TD",
        "    B[Benchmark fixtures] --> S[Academic search evaluator]",
        "    B --> P[PDF extraction evaluator]",
        "    B --> R[RAG evaluator]",
        "    B --> A[Analysis evaluator]",
        "    B --> G[Agentic evaluator]",
        "    B --> L[Reliability evaluator]",
        "    S --> J[JSON results]",
        "    P --> J",
        "    R --> J",
        "    A --> J",
        "    G --> J",
        "    L --> J",
        "    J --> M[Markdown report]",
        "```",
        "",
        "The search evaluator injects mocked provider clients. The PDF evaluator creates synthetic PDFs. The RAG evaluator uses mock embeddings. The analysis and agentic evaluators exercise selected components and routes. The reliability evaluator injects selected provider and input failures.",
        "",
        "## 3. Mathematical Metric Definitions and Formulas",
        "",
        "For a cutoff K, Precision@K uses the fixed denominator K:",
        "",
        r"$$P@K = \frac{|Retrieved_{1:K} \cap Relevant|}{K}$$",
        "",
        "Recall@K is the fraction of known relevant items retrieved:",
        "",
        r"$$R@K = \frac{|Retrieved_{1:K} \cap Relevant|}{|Relevant|}$$",
        "",
        "MRR averages reciprocal rank of the first relevant result. nDCG uses graded relevance judgments and discounts lower-ranked results.",
        "",
        "**Important:** A fixed-denominator P@K can be lower when fewer than K items are returned. This is intentional in the current metric implementation and must be interpreted alongside the actual returned-result count.",
        "",
        "The RAG passage-support score is a lexical evidence-overlap proxy. It is not a measure of LLM-generated answer faithfulness. Page recall measures coverage of expected pages, not recall of all relevant chunks.",
        "",
        "## 4. Benchmark Datasets Catalog",
        "",
        "| Dataset | Size | Evaluation scope |",
        "|---|---:|---|",
        f"| `academic_search_benchmark.json` | {search.get('total_queries', 'Not run')} queries | Graded relevance judgments; mocked candidate records |",
        f"| `pdf_extraction_benchmark.json` | {pdf.get('total_archetypes', 'Not run')} archetypes | Programmatically generated PDF fixtures |",
        f"| `rag_qa_benchmark.json` | {rag_total} questions | Bilingual retrieval; scanned cases may be excluded |",
        f"| `analysis_comparison_benchmark.json` | {analysis_total} cases | Analysis and validation challenge fixtures |",
        f"| `agentic_scenarios_benchmark.json` | {agent_total} scenarios | Selected LangGraph node and routing checks |",
        "| `end_to_end_tasks.json` | 3 tasks | Benchmark definitions; not evidence of execution unless separately run |",
        "",
        "## 5. Detailed Evaluation Results",
        "",
        "### 5.1 Academic Search",
        "",
        f"- Queries: {search.get('total_queries', 'Not run')}",
        f"- P@5: {p5:.4f}; R@5: {r5:.4f}",
        f"- P@10: {p10:.4f}; R@10: {r10:.4f}",
        f"- MRR: {mrr:.4f}",
        f"- nDCG@5: {ndcg5:.4f}; nDCG@10: {ndcg10:.4f}",
        "",
        "The evaluator constructs candidate paper records from the benchmark relevance judgments and supplies them through mocked OpenAlex, Semantic Scholar, and Crossref clients. These metrics assess the controlled search/merge/ranking pipeline; they do not independently measure discovery quality on live academic indexes.",
        "",
        "### 5.2 PDF Extraction",
        "",
        f"- Archetypes: {pdf.get('total_archetypes', 'Not run')}",
        f"- Digital-document field checks: {digital_field_count}",
        f"- Digital-document field accuracy: {f'{digital_field_accuracy:.4f}' if digital_field_accuracy is not None else 'Not available'}",
        f"- Page-count accuracy across archetypes: {_pct(_num(pdf_m, 'page_count_accuracy'))}",
        f"- OCR-required flag accuracy across archetypes: {_pct(_num(pdf_m, 'ocr_detection_accuracy'))}",
        f"- Scanned-document handling score: {f'{scanned_handling:.4f}' if scanned_handling is not None else 'Not evaluated'}",
        "",
        "All PDF fixtures are generated programmatically. The evaluator calls the automatic extraction path once per fixture; it does not perform an independent dual-engine comparison. The scanned fixture measures OCR-required detection and graceful handling only. OCR transcription accuracy is not evaluated, and the aggregate `mean_extraction_score` mixes different criteria across archetypes, so it is not presented as digital field accuracy.",
        "",
        "### 5.3 RAG Retrieval",
        "",
        f"- Benchmark questions: {rag_total}",
        f"- Evaluated questions: {rag_evaluated}",
        f"- Excluded questions: {rag_excluded}",
        f"- Embedding provider: {embedding}",
        f"- Correct-paper rate: {_pct(correct_paper)}",
        f"- Correct-page rate: {_pct(correct_page)}",
        f"- Page precision@5: {_num(rag_m, 'mean_page_precision_at_5'):.4f}",
        f"- Page recall: {_num(rag_m, 'mean_page_recall'):.4f}",
        f"- Keyword recall proxy: {_num(rag_m, 'mean_keyword_recall'):.4f}",
        f"- Ground-truth passage support proxy: {passage_support:.4f}",
        f"- Unsupported passage proxy: {unsupported_passage:.4f}",
        "",
        "The current RAG benchmark uses MockEmbeddingProvider. It validates indexing/retrieval plumbing with synthetic embeddings; semantic retrieval quality, especially cross-lingual Amharic retrieval, is not validated. Passage support is a lexical overlap proxy, not generated-answer faithfulness.",
        "",
        "| Language | Evaluated questions | Correct-paper rate | Correct-page rate |",
        "|---|---:|---:|---:|",
        f"| English | {rag_en.get('count', 0)} | {_pct(_num(rag_en, 'correct_paper_rate'))} | {_pct(_num(rag_en, 'correct_page_rate'))} |",
        f"| Amharic | {rag_am.get('count', 0)} | {_pct(_num(rag_am, 'correct_paper_rate'))} | {_pct(_num(rag_am, 'correct_page_rate'))} |",
        "",
        "Excluded questions:",
    ]

    # The evaluator stores excluded cases in per_query_results,
    # using evaluation_status and exclusion_reason.
    excluded = [
        item for item in rag.get("per_query_results", [])
        if isinstance(item, dict)
        and item.get("evaluation_status") == "NOT_EVALUATED"
    ]

    if excluded:
        for item in excluded:
            lines.append(
                f"- `{item.get('qa_id', 'unknown')}` "
                f"({item.get('language', 'unknown')}): "
                f"{item.get('exclusion_reason', 'No reason recorded')}"
            )
    else:
        lines.append("- No excluded questions recorded.")

    lines += [
        "",
        "### 5.4 Research Analysis",
        "",
        f"- Challenge cases passed: {analysis_passed}/{analysis_total}",
        f"- Challenge-case pass rate: {_pct(_num(analysis, 'robustness_score'))}",
        "",
        "These are component-level challenge tests, not a broad evaluation of research-analysis quality. In particular, the insufficient-evidence case checks the benchmark input count rather than executing the complete agent response, and the contradictory-findings case has a permissive pass condition. The fabricated-citation case tests mocked registry responses. The unsupported-claim case uses a lexical grounding heuristic.",
        "",
        "### 5.5 Agentic Workflow",
        "",
        f"- Scenarios passed: {agent_passed}/{agent_total}",
        f"- Scenario pass rate: {_pct(_num(agent, 'task_success_rate'))}",
        f"- Stopping-condition scenario metric: {_pct(stop_accuracy)}",
        "",
        "The six scenarios test selected nodes and routing conditions. They do not constitute six complete end-to-end research tasks. The evaluator's retry rate and unnecessary-tool-call rate are not instrumented measurements and are intentionally omitted from the headline results. The degraded-mode scenario checks benchmark-declared deterministic tools rather than executing each listed tool.",
        "",
        "### 5.6 Reliability",
        "",
        f"- Specific checks passed: {reliability_passed}/{reliability_total}",
        f"- Check pass rate: {_pct(reliability_score)}",
        "",
        "The checks cover a deterministic search/citation pipeline, partial academic-provider failure with Crossref fallback, corrupted-PDF handling, and blank-query validation. They do not establish general service availability or production reliability.",
        "",
        "---",
        "",
        "## 6. Error and Failure-Mode Analysis",
        "",
        "1. **Live search quality is unmeasured:** the search benchmark supplies candidates derived from the relevance judgments.",
        "2. **Scanned-PDF OCR is not implemented:** the evaluation verifies detection and graceful handling, not transcription.",
        "3. **RAG semantic quality is unmeasured:** mock embeddings verify retrieval plumbing, not semantic or cross-lingual relevance.",
        "4. **Grounding is approximate:** lexical overlap cannot reliably detect every unsupported scientific claim or semantic contradiction.",
        "5. **Agentic evaluation is component-level:** node and routing checks do not demonstrate full-task completion, nor do they instrument all tool-use efficiency metrics.",
        "6. **Small synthetic benchmarks limit generalization:** results should not be interpreted as statistically representative of real research papers or user requests.",
        "",
        "## 7. System Limitations and Recommendations",
        "",
        "1. Add a held-out search benchmark with independently collected live-provider results and human relevance judgments.",
        "2. Evaluate extraction on a representative collection of real digital PDFs, and add OCR implementation plus character/word error rate evaluation for scanned documents.",
        "3. Run RAG evaluation with the production embedding provider and report English and Amharic results separately.",
        "4. Evaluate generated answers using claim-level evidence attribution and human review; retain lexical overlap only as a lightweight proxy.",
        "5. Run complete end-to-end agent tasks and instrument actual tool calls, retries, plan changes, and stop reasons.",
        "6. Expand fault injection to cover timeouts, provider outages, malformed tool outputs, vector-store failures, and recovery behavior.",
        "",
        "The RAG implementation uses `gemini-embedding-2` in its production configuration, while this evaluation uses mock embeddings. The production embedding model has therefore not been benchmarked here.",
        "",
        "## 8. Regression Testing Verification",
        "",
        "The Phase 9 runner does not execute the full pytest suite or frontend build. Those checks must be run separately and their actual outputs recorded.",
        "",
        "Previously reported project test result: **197 passed, 7 warnings**. This is a prior test run, not a result generated by this evaluation runner. Re-run the suite after applying changes.",
        "",
        "No live LLM quota is required by the offline evaluator fixtures as currently designed. This statement does not imply that every project test or every application workflow is offline.",
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


async def _run_one(name: str, evaluator: Any) -> Dict[str, Any]:
    try:
        result = await evaluator.run()
        if not isinstance(result, dict):
            raise TypeError(f"{name} evaluator returned {type(result).__name__}, expected dict")
        if "error" in result and len(result) == 1:
            raise RuntimeError(str(result["error"]))
        return result
    except Exception as exc:
        print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "evaluator_status": "failed",
        }


async def run_all_evaluations(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Run all evaluators independently and save complete or partial results."""
    print("=" * 70)
    print("AIRA Phase 9: System Evaluation and Reliability Testing")
    print("=" * 70)

    evaluators = [
        ("academic_search", "Academic Search", AcademicSearchEvaluator),
        ("pdf_extraction", "PDF Extraction", PDFExtractionEvaluator),
        ("rag_retrieval", "RAG Retrieval", RAGEvaluator),
        ("analysis_comparison", "Research Analysis", AnalysisEvaluator),
        ("agentic_workflow", "Agentic Workflow", AgenticEvaluator),
        ("reliability", "Reliability", ReliabilityEvaluator),
    ]

    results: Dict[str, Any] = {
        "timestamp": time.time(),
        "evaluation_status": {},
    }

    for index, (key, label, evaluator_class) in enumerate(evaluators, start=1):
        print(f"\n[{index}/6] Running {label} Evaluator...")
        result = await _run_one(key, evaluator_class())
        results[key] = result
        failed = "error" in result
        results["evaluation_status"][key] = "failed" if failed else "completed"
        if failed:
            print(f"      -> FAILED: {result['error']}")
        else:
            print("      -> Completed")

        # Persist after each evaluator so an interruption does not lose all progress.
        base_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
        base_dir.mkdir(parents=True, exist_ok=True)
        json_path = base_dir / "evaluation_results.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    base_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
    json_path = base_dir / "evaluation_results.json"
    md_path = base_dir / "evaluation_report.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    generate_markdown_report(results, md_path)

    completed = sum(1 for v in results["evaluation_status"].values() if v == "completed")
    print(f"\nEvaluators completed: {completed}/6")
    print(f"JSON: {json_path}")
    print(f"Report: {md_path}")
    print("Evaluation run finished; inspect evaluation_status for failures.")
    return results


if __name__ == "__main__":
    asyncio.run(run_all_evaluations())
