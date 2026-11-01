"""Automated Pytest Suite for AIRA Phase 9 System Evaluation and Reliability Testing.

Tests:
1. Mathematical metric algorithms (P@K, R@K, MRR, nDCG@K, overlap, grounding).
2. Benchmark dataset integrity and schema validation across all 6 benchmark files.
3. Domain evaluators (Search, PDF Extraction, RAG, Analysis, Agentic, Reliability).
4. Machine-readable JSON and Markdown report generation.
Zero live API quota consumed.
"""
import asyncio
import json
from pathlib import Path
import tempfile
import pytest

from evaluation.evaluators.agentic_evaluator import AgenticEvaluator
from evaluation.evaluators.analysis_evaluator import AnalysisEvaluator
from evaluation.evaluators.extraction_evaluator import PDFExtractionEvaluator
from evaluation.evaluators.rag_evaluator import RAGEvaluator
from evaluation.evaluators.reliability_evaluator import ReliabilityEvaluator
from evaluation.evaluators.search_evaluator import AcademicSearchEvaluator
from evaluation.metrics import (
    dcg_at_k,
    evidence_grounding_eval,
    field_extraction_accuracy,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    text_token_overlap,
)
from evaluation.run_evaluation import generate_markdown_report, run_all_evaluations


# ---------------------------------------------------------------------------
# 1. Mathematical Metrics Tests
# ---------------------------------------------------------------------------

def test_precision_at_k():
    """Verify precision@k calculation under normal, boundary, and empty conditions."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    relevant = {"doc_a", "doc_c", "doc_x"}

    # Top 5: doc_a and doc_c are relevant -> 2 / 5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == 0.4
    # Top 2: doc_a is relevant -> 1 / 2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    # k <= 0 -> 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0
    # empty retrieved -> 0.0
    assert precision_at_k([], relevant, k=5) == 0.0


def test_recall_at_k():
    """Verify recall@k calculation under normal, boundary, and empty conditions."""
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    relevant = {"doc_a", "doc_c"}

    # Both relevant docs retrieved in top 5 -> 2 / 2 = 1.0
    assert recall_at_k(retrieved, relevant, k=5) == 1.0
    # Top 1: only doc_a retrieved -> 1 / 2 = 0.5
    assert recall_at_k(retrieved, relevant, k=1) == 0.5
    # Empty relevant set defaults to 1.0
    assert recall_at_k(retrieved, set(), k=5) == 1.0


def test_reciprocal_rank_and_mrr():
    """Verify reciprocal rank and mean reciprocal rank."""
    # First relevant at rank 1 -> 1.0
    assert reciprocal_rank(["d1", "d2"], {"d1"}) == 1.0
    # First relevant at rank 3 -> 1 / 3 = 0.3333...
    assert pytest.approx(reciprocal_rank(["d1", "d2", "d3"], {"d3"}), 0.001) == 0.3333
    # No relevant items -> 0.0
    assert reciprocal_rank(["d1", "d2"], {"d99"}) == 0.0

    # MRR across 2 queries: (1.0 + 0.5) / 2 = 0.75
    q_ret = [["d1", "d2"], ["d2", "d1"]]
    q_rel = [{"d1"}, {"d1"}]
    assert mean_reciprocal_rank(q_ret, q_rel) == 0.75
    assert mean_reciprocal_rank([], []) == 0.0


def test_dcg_and_ndcg_at_k():
    """Verify DCG and normalized DCG calculations."""
    relevance_map = {"d1": 3.0, "d2": 2.0, "d3": 1.0, "d4": 0.0}

    # Perfect ranking produces nDCG of 1.0
    perfect_ranking = ["d1", "d2", "d3", "d4"]
    assert pytest.approx(ndcg_at_k(perfect_ranking, relevance_map, k=4), 0.001) == 1.0

    # Suboptimal ranking produces nDCG < 1.0
    suboptimal = ["d4", "d3", "d2", "d1"]
    score = ndcg_at_k(suboptimal, relevance_map, k=4)
    assert 0.0 < score < 1.0

    # Empty retrieved or non-positive k returns 0.0
    assert ndcg_at_k([], relevance_map, k=5) == 0.0
    assert ndcg_at_k(perfect_ranking, relevance_map, k=0) == 0.0


def test_text_token_overlap():
    """Verify Jaccard text overlap."""
    t1 = "Amharic language modeling"
    t2 = "amharic neural language modeling"
    overlap = text_token_overlap(t1, t2)
    assert 0.5 <= overlap <= 1.0
    assert text_token_overlap("", "text") == 0.0
    assert text_token_overlap("hello world", "goodbye moon") == 0.0


def test_field_extraction_accuracy():
    """Verify field extraction comparisons across numbers, lists, and text."""
    # Numerical tolerance
    assert field_extraction_accuracy(31.2, 31.2) is True
    assert field_extraction_accuracy("31.2", 31.2) is True
    assert field_extraction_accuracy(50, 80) is False

    # List token overlap
    assert field_extraction_accuracy(["Abebe Bikila", "Almaz Worku"], ["Abebe Bikila", "Almaz Worku"]) is True
    assert field_extraction_accuracy(["Abebe"], ["Abebe", "Almaz"]) is False

    # Substring / text match
    assert field_extraction_accuracy("Neural Machine Translation for Amharic", "Neural Machine Translation") is True
    assert field_extraction_accuracy(None, "expected") is False
    assert field_extraction_accuracy(None, None) is True


def test_evidence_grounding_eval():
    """Verify evidence grounding and unsupported claim classification."""
    evidence = [
        "The model achieves 31.2 BLEU score on 50,000 sentence pairs.",
        "We evaluate on the Amharic-English parallel corpus.",
    ]
    supported_claim = "The model obtained 31.2 BLEU on parallel corpus sentence pairs."
    unsupported_claim = "The model sets a new world record on ImageNet with 99% accuracy."

    faith, unsupp, audit = evidence_grounding_eval(
        claims=[supported_claim, unsupported_claim],
        evidence_passages=evidence,
        overlap_threshold=0.25,
    )

    assert audit[0]["supported"] is True
    assert audit[1]["supported"] is False
    assert faith == 0.5
    assert unsupp == 0.5


# ---------------------------------------------------------------------------
# 2. Benchmark Datasets Integrity Tests
# ---------------------------------------------------------------------------

def test_benchmark_datasets_exist_and_valid():
    """Verify all 6 required benchmark JSON files exist, load without error, and have expected schemas."""
    benchmarks_dir = Path(__file__).resolve().parent.parent / "evaluation" / "benchmarks"
    assert benchmarks_dir.exists(), f"Benchmark directory missing: {benchmarks_dir}"

    files_to_check = [
        "academic_search_benchmark.json",
        "pdf_extraction_benchmark.json",
        "rag_qa_benchmark.json",
        "analysis_comparison_benchmark.json",
        "agentic_scenarios_benchmark.json",
        "end_to_end_tasks.json",
    ]

    for fname in files_to_check:
        fpath = benchmarks_dir / fname
        assert fpath.exists(), f"Benchmark file {fname} not found"
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list), f"Benchmark {fname} root must be a list"
        assert len(data) > 0, f"Benchmark {fname} must not be empty"


# ---------------------------------------------------------------------------
# 3. Domain Evaluators Offline Execution Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_academic_search_evaluator():
    """Verify AcademicSearchEvaluator executes and returns expected IR metrics."""
    evaluator = AcademicSearchEvaluator()
    res = await evaluator.run()

    assert res["evaluator"] == "AcademicSearchEvaluator"
    assert res["total_queries"] == 10
    agg = res["aggregate_metrics"]
    assert agg["mean_precision_at_5"] >= 0.5
    assert agg["mean_recall_at_5"] >= 0.8
    assert agg["mrr"] >= 0.8
    assert agg["mean_ndcg_at_5"] >= 0.8


@pytest.mark.anyio
async def test_pdf_extraction_evaluator():
    """Verify PDFExtractionEvaluator parses 4 archetypes and detects OCR scanned document."""
    evaluator = PDFExtractionEvaluator()
    res = await evaluator.run()

    assert res["evaluator"] == "PDFExtractionEvaluator"
    assert res["total_archetypes"] == 4
    agg = res["aggregate_metrics"]
    assert agg["ocr_detection_accuracy"] == 1.0  # 100% OCR identification
    assert agg["page_count_accuracy"] == 1.0     # 100% page count match


@pytest.mark.anyio
async def test_rag_evaluator():
    """Verify RAGEvaluator indexes corpus, runs bilingual QA queries, and reports provider."""
    evaluator = RAGEvaluator()
    res = await evaluator.run()

    assert res["evaluator"] == "RAGEvaluator"
    assert res["total_queries"] == 8
    assert "MockEmbeddingProvider" in res["embedding_provider_evaluated"]
    overall = res["overall_metrics"]
    assert overall["correct_paper_rate"] >= 0.5
    assert overall["mean_faithfulness"] >= 0.5


@pytest.mark.anyio
async def test_analysis_evaluator():
    """Verify AnalysisEvaluator tests all 5 challenge cases."""
    evaluator = AnalysisEvaluator()
    res = await evaluator.run()

    assert res["evaluator"] == "AnalysisEvaluator"
    assert res["total_cases"] == 5
    assert res["passed_cases"] == 5
    assert res["robustness_score"] == 1.0


@pytest.mark.anyio
async def test_agentic_evaluator():
    """Verify AgenticEvaluator validates all 6 stateful LangGraph workflow scenarios."""
    evaluator = AgenticEvaluator()
    res = await evaluator.run()

    assert res["evaluator"] == "AgenticEvaluator"
    assert res["total_scenarios"] == 6
    assert res["passed_scenarios"] == 6
    assert res["task_success_rate"] == 1.0
    assert res["stopping_condition_accuracy"] == 1.0


@pytest.mark.anyio
async def test_reliability_evaluator():
    """Verify ReliabilityEvaluator validates provider failover, corrupted PDF, and degraded mode."""
    evaluator = ReliabilityEvaluator()
    res = await evaluator.run()

    assert res["evaluator"] == "ReliabilityEvaluator"
    assert res["total_fault_injections"] == 4
    assert res["passed_fault_injections"] == 4
    assert res["reliability_score"] == 1.0
    assert res["degraded_mode_operational"] is True


# ---------------------------------------------------------------------------
# 4. Report Generation Tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_full_evaluation_runner_and_report_generation():
    """Verify complete evaluation runner produces valid JSON and Markdown reports."""
    results = await run_all_evaluations()

    assert "academic_search" in results
    assert "pdf_extraction" in results
    assert "rag_retrieval" in results
    assert "analysis_comparison" in results
    assert "agentic_workflow" in results
    assert "reliability" in results

    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "test_report.md"
        generate_markdown_report(results, report_path)

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# AIRA Phase 9: Comprehensive System Evaluation and Reliability Report" in content
        assert "## 1. Executive Summary" in content
        assert "## 2. Evaluation Architecture" in content
        assert "## 3. Mathematical Metric Definitions and Formulas" in content
        assert "## 4. Benchmark Datasets Catalog" in content
        assert "## 5. Detailed Evaluation Results" in content
        assert "## 6. Error and Failure-Mode Analysis" in content
        assert "## 7. System Limitations and Recommendations" in content
        assert "## 8. Regression Testing Verification" in content
