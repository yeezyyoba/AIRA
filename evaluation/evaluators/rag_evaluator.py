"""Evaluator for Research Context Retrieval (RAG Quality and Bilingual QA).

Measures:
- Precision@5, Recall@5
- Correct-Paper Retrieval Rate
- Correct-Page Retrieval Rate
- Faithfulness and Unsupported Claim Rate
- Bilingual breakdown (English vs Amharic)
Explicitly documents embedding provider type (MockEmbeddingProvider vs GeminiEmbeddingProvider).
Zero live API quota consumed.
"""
import asyncio
import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from evaluation.metrics import (
    evidence_grounding_eval,
    precision_at_k,
    recall_at_k,
    text_token_overlap,
)
from memory.vector_store import ChromaVectorStore
from services.embedding_provider import MockEmbeddingProvider
from tools.paper_indexer import index_research_paper
from tools.rag_retriever import retrieve_research_context
from validation.schemas import (
    EvidenceItemSchema,
    ExtractedPaperDataSchema,
    ExtractionStatus,
    PaperExtractionResultSchema,
    RetrievalResultSchema,
)


class RAGEvaluator:
    """Evaluates RAG context retrieval and bilingual QA grounding."""

    def __init__(self, benchmark_path: Optional[Path] = None):
        if benchmark_path is None:
            benchmark_path = Path(__file__).resolve().parent.parent / "benchmarks" / "rag_qa_benchmark.json"
        self.benchmark_path = benchmark_path
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_data = json.load(f)

    def _create_corpus(self) -> List[PaperExtractionResultSchema]:
        """Create benchmark corpus with target multilingual paper and distractor papers."""
        target_pages = {
            1: (
                "Neural Machine Translation for Low-Resource Amharic: An Empirical Study. "
                "Abebe Bikila, Almaz Worku. Department of Computer Science, Addis Ababa University. "
                "Abstract: We present an empirical evaluation of cross-lingual transfer for low-resource "
                "Amharic neural translation, addressing morphological complexity and vocabulary sparsity. "
                "የአማርኛ ቋንቋ የነርቭ ማሽን ትርጉም ጥናት ማጠቃለያ።"
            ),
            2: (
                "Methodology and Experimental Setup. "
                "Our neural architecture employs a 12-layer cross-lingual Transformer with 8 attention heads and 512 embedding dimensions. "
                "We evaluate sentence representations across 50,000 parallel Amharic-English sentence pairs extracted from aligned public domain publications. "
                "Subword segmentation uses SentencePiece with a unigram vocabulary of 32,000 tokens. "
                "የቀረበው ባለ 12-ደረጃ ትራንስፎርመር ሞዴል እና 50,000 ትይዩ ዓረፍተ ነገሮች ስብስብ።"
            ),
            3: (
                "Results and Quantitative Discussion. "
                "The proposed 12-layer cross-lingual Transformer achieved a BLEU score of 31.2 on the held-out test split, "
                "outperforming previous baselines by 4.8 points. "
                "የቀረበው ባለ 12-ደረጃ ትራንስፎርመር ሞዴል በፈተና መረጃው ላይ 31.2 የBLEU ውጤት አስመዝግቧል።"
            ),
            4: (
                "Limitations and Conclusion. "
                "Morphological inflections in low-resource Ethiopic verbs remain a primary challenge. "
                "ማጠቃለያ እና የወደፊት የጥናት አቅጣጫዎች።"
            ),
        }

        target_extracted = ExtractedPaperDataSchema(
            paper_id="amharic_nmt_2024",
            title="Neural Machine Translation for Low-Resource Amharic: An Empirical Study",
            authors=["Abebe Bikila", "Almaz Worku"],
            abstract=target_pages[1],
            methodology=target_pages[2],
            results=target_pages[3],
            conclusion=target_pages[4],
            evidence=[
                EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=1, section_heading="Abstract", passage="empirical evaluation"),
                EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=2, section_heading="Methodology", passage="12-layer cross-lingual Transformer"),
                EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=3, section_heading="Results", passage="BLEU score of 31.2"),
                EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=4, section_heading="Conclusion", passage="Morphological inflections"),
            ],
        )

        paper_target = PaperExtractionResultSchema(
            paper_id="amharic_nmt_2024",
            status=ExtractionStatus.SUCCESS,
            extraction_method="pymupdf",
            page_count=4,
            raw_text_by_page=target_pages,
            extracted_data=target_extracted,
        )

        distractor_pages_1 = {
            1: "Computer Vision and Deep Learning for Aerial Drone Navigation. Abstract: Real-time visual odometry.",
            2: "Methodology: Convolutional neural networks trained on 100,000 synthetic landscape frames.",
            3: "Results: Mean tracking accuracy reached 94.2% in GPS-denied environments.",
        }
        paper_distractor_1 = PaperExtractionResultSchema(
            paper_id="distractor_cv_2023",
            status=ExtractionStatus.SUCCESS,
            extraction_method="pymupdf",
            page_count=3,
            raw_text_by_page=distractor_pages_1,
            extracted_data=ExtractedPaperDataSchema(
                paper_id="distractor_cv_2023",
                title="Computer Vision and Deep Learning for Aerial Drone Navigation",
                authors=["John Doe"],
                abstract=distractor_pages_1[1],
                methodology=distractor_pages_1[2],
                results=distractor_pages_1[3],
            ),
        )

        distractor_pages_2 = {
            1: "Speech Recognition for Medical Dictation Systems. Abstract: Acoustic modeling of clinical terms.",
            2: "Methodology: Conformer-CTC acoustic models evaluated on 500 hours of clinical audio recordings.",
            3: "Results: Word error rate achieved 5.4% across 12 medical subspecialties.",
        }
        paper_distractor_2 = PaperExtractionResultSchema(
            paper_id="distractor_speech_2022",
            status=ExtractionStatus.SUCCESS,
            extraction_method="pymupdf",
            page_count=3,
            raw_text_by_page=distractor_pages_2,
            extracted_data=ExtractedPaperDataSchema(
                paper_id="distractor_speech_2022",
                title="Speech Recognition for Medical Dictation Systems",
                authors=["Jane Smith"],
                abstract=distractor_pages_2[1],
                methodology=distractor_pages_2[2],
                results=distractor_pages_2[3],
            ),
        )

        return [paper_target, paper_distractor_1, paper_distractor_2]

    async def _setup_test_vector_store(self, temp_dir: str, mock_embedder: MockEmbeddingProvider) -> ChromaVectorStore:
        """Initialize and populate isolated Chroma vector collection with chunked papers."""
        store = ChromaVectorStore(
            collection_name="test_phase9_rag_eval",
            persist_directory=temp_dir,
        )
        papers = self._create_corpus()

        for p in papers:
            await index_research_paper(
                paper_id=p.paper_id,
                extraction_result=p,
                vector_db=store,
                embedding_provider=mock_embedder,
            )

        return store

    async def evaluate_query(
        self,
        qa_item: Dict[str, Any],
        vector_db: ChromaVectorStore,
        mock_embedder: MockEmbeddingProvider,
    ) -> Dict[str, Any]:
        """Evaluate a single QA retrieval test case."""
        qa_id = qa_item["qa_id"]
        lang = qa_item["language"]
        question = qa_item["question"]
        target_paper_id = qa_item["target_paper_id"]
        expected_pages = set(qa_item["expected_pages"])
        expected_keywords = qa_item.get("expected_answer_keywords", [])
        gt_passage = qa_item.get("ground_truth_passage", "")

        retrieval_res: RetrievalResultSchema = await retrieve_research_context(
            question=question,
            top_k=5,
            vector_db=vector_db,
            embedding_provider=mock_embedder,
            isolate_untrusted=False,
        )

        retrieved_items = retrieval_res.items
        retrieved_paper_ids = [item.paper_id for item in retrieved_items]
        retrieved_passages = [item.text for item in retrieved_items]

        # 1. Correct-Paper Rate (is target paper present in top 5?)
        correct_paper = target_paper_id in retrieved_paper_ids

        # 2. Correct-Page Rate (is an expected page present in top 5 for target paper?)
        matched_pages = [
            p
            for item in retrieved_items
            if item.paper_id == target_paper_id
            for p in range(item.page_start, item.page_end + 1)
            if p in expected_pages
        ]
        correct_page = len(matched_pages) > 0

        # 3. Precision@5 and Recall@5
        relevant_chunk_ids = {f"{target_paper_id}_p{p}" for p in expected_pages}
        actual_chunk_ids = [f"{item.paper_id}_p{item.page_start}" for item in retrieved_items]

        p5 = precision_at_k(actual_chunk_ids, relevant_chunk_ids, k=5)
        r5 = recall_at_k(actual_chunk_ids, relevant_chunk_ids, k=5)

        # 4. Keyword presence in retrieved context
        combined_text = " ".join(retrieved_passages).lower()
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in combined_text)
        keyword_ratio = keyword_hits / max(len(expected_keywords), 1)

        # 5. Faithfulness against ground-truth passage
        faithfulness, unsupported_rate, _ = evidence_grounding_eval(
            claims=[gt_passage] if gt_passage else [],
            evidence_passages=retrieved_passages,
            overlap_threshold=0.20,
        )

        return {
            "qa_id": qa_id,
            "language": lang,
            "question": question,
            "target_paper_id": target_paper_id,
            "expected_pages": list(expected_pages),
            "retrieved_count": len(retrieved_items),
            "correct_paper": correct_paper,
            "correct_page": correct_page,
            "precision_at_5": round(p5, 4),
            "recall_at_5": round(r5, 4),
            "keyword_recall": round(keyword_ratio, 4),
            "faithfulness": round(faithfulness, 4),
            "unsupported_rate": round(unsupported_rate, 4),
        }

    async def run(self) -> Dict[str, Any]:
        """Run complete RAG QA benchmark evaluation."""
        mock_embedder = MockEmbeddingProvider()

        with tempfile.TemporaryDirectory() as tmp_dir:
            vector_db = await self._setup_test_vector_store(tmp_dir, mock_embedder)

            results = []
            for item in self.benchmark_data:
                res = await self.evaluate_query(item, vector_db, mock_embedder)
                results.append(res)

        total_queries = len(results)
        if total_queries == 0:
            return {"error": "Empty RAG benchmark dataset"}

        en_queries = [r for r in results if r["language"] == "en"]
        am_queries = [r for r in results if r["language"] == "am"]

        def _calc_stats(subset: List[Dict[str, Any]]) -> Dict[str, float]:
            if not subset:
                return {}
            n = len(subset)
            return {
                "count": n,
                "correct_paper_rate": round(sum(1 for r in subset if r["correct_paper"]) / n, 4),
                "correct_page_rate": round(sum(1 for r in subset if r["correct_page"]) / n, 4),
                "mean_precision_at_5": round(sum(r["precision_at_5"] for r in subset) / n, 4),
                "mean_recall_at_5": round(sum(r["recall_at_5"] for r in subset) / n, 4),
                "mean_keyword_recall": round(sum(r["keyword_recall"] for r in subset) / n, 4),
                "mean_faithfulness": round(sum(r["faithfulness"] for r in subset) / n, 4),
                "mean_unsupported_rate": round(sum(r["unsupported_rate"] for r in subset) / n, 4),
            }

        gemini_live_configured = bool(getattr(settings, "GEMINI_API_KEY", None))

        return {
            "evaluator": "RAGEvaluator",
            "benchmark_dataset": str(self.benchmark_path.name),
            "embedding_provider_evaluated": "MockEmbeddingProvider (Offline Zero-Quota Verification)",
            "live_gemini_embedding_configured": gemini_live_configured,
            "total_queries": total_queries,
            "overall_metrics": _calc_stats(results),
            "english_metrics": _calc_stats(en_queries),
            "amharic_metrics": _calc_stats(am_queries),
            "per_query_results": results,
        }
