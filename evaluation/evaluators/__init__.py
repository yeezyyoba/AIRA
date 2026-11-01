"""AIRA Phase 9 Domain Evaluators Package.

Contains specialized evaluators for:
- Academic Search (IR metrics: P@K, R@K, MRR, nDCG@K)
- PDF Extraction (dual-engine field accuracy and scanned OCR detection)
- RAG Retrieval & QA (English & Amharic bilingual retrieval quality)
- Research Analysis & Comparison (metric compatibility, contradictions, unsupported claims)
- Agentic Workflow (graph traversal, reflection, retries, stopping conditions)
- Reliability & Degraded Mode (failovers, network errors, graceful degradation)
"""
from evaluation.evaluators.search_evaluator import AcademicSearchEvaluator
from evaluation.evaluators.extraction_evaluator import PDFExtractionEvaluator
from evaluation.evaluators.rag_evaluator import RAGEvaluator
from evaluation.evaluators.analysis_evaluator import AnalysisEvaluator
from evaluation.evaluators.agentic_evaluator import AgenticEvaluator
from evaluation.evaluators.reliability_evaluator import ReliabilityEvaluator

__all__ = [
    "AcademicSearchEvaluator",
    "PDFExtractionEvaluator",
    "RAGEvaluator",
    "AnalysisEvaluator",
    "AgenticEvaluator",
    "ReliabilityEvaluator",
]
