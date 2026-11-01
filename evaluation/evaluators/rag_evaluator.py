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

        twocolumn_pages = {
            1: "Named Entity Recognition in Ethiopian Social Media Texts. Abebe Kebede, Solomon Desta. Abstract: Processing informal social media texts introduces significant orthographic variation and nested entities.",
            2: "Section 2: Background and Data. Processing AmhSocialNER-v2 dataset.",
            3: "Section 3: Model Architecture. We deploy AfroXLMR sequence models.",
            4: "Section 4: Methodology. We utilize AfroXLMR backbone with a CRF sequence layer and morphological feature injection.",
            5: "Section 5: Results and Discussion. The AfroXLMR-CRF sequence model reached a Macro-F1 of 84.6% on the AmhSocialNER test benchmark with an annotated corpus of 12,500 posts.",
            6: "Section 6: Conclusion and Future Work. Future work will explore dialectal variants.",
        }
        paper_twocolumn = PaperExtractionResultSchema(
            paper_id="eval_pdf_twocolumn_02",
            status=ExtractionStatus.SUCCESS,
            extraction_method="pymupdf",
            page_count=6,
            raw_text_by_page=twocolumn_pages,
            extracted_data=ExtractedPaperDataSchema(
                paper_id="eval_pdf_twocolumn_02",
                title="Named Entity Recognition in Ethiopian Social Media Texts",
                authors=["Abebe Kebede", "Solomon Desta"],
                abstract=twocolumn_pages[1],
                methodology=twocolumn_pages[4],
                results=twocolumn_pages[5],
                evidence=[
                    EvidenceItemSchema(paper_id="eval_pdf_twocolumn_02", page_number=5, section_heading="Results", passage="The AfroXLMR-CRF sequence model reached a Macro-F1 of 84.6% on the AmhSocialNER test benchmark."),
                ],
            ),
        )

        return [paper_target, paper_twocolumn, paper_distractor_1, paper_distractor_2]


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
        """Evaluate retrieval using actual chunk IDs and explicit page-level labels.

        Scanned-paper questions are excluded because the production indexer
        correctly rejects documents requiring OCR when OCR is unavailable.
        """

        qa_id = qa_item["qa_id"]
        lang = qa_item["language"]
        question = qa_item["question"]
        target_paper_id = qa_item["target_paper_id"]
        expected_pages = set(qa_item["expected_pages"])
        expected_keywords = qa_item.get("expected_answer_keywords", [])
        gt_passage = qa_item.get("ground_truth_passage", "")

        # These benchmark questions refer to a scanned-only fixture.
        # Its manually authored text is not valid evidence of OCR or RAG ability.
        if target_paper_id == "eval_pdf_scanned_04":
            return {
                "qa_id": qa_id,
                "language": lang,
                "question": question,
                "target_paper_id": target_paper_id,
                "expected_pages": sorted(expected_pages),
                "evaluation_status": "NOT_EVALUATED",
                "exclusion_reason": (
                    "Scanned-document RAG is not evaluated: OCR is not implemented "
                    "and the production indexer rejects SCANNED_NEEDS_OCR documents."
                ),
                "retrieved_count": 0,
                "correct_paper": None,
                "correct_page": None,
                "precision_at_5": None,
                "recall_at_5": None,
                "page_precision_at_5": None,
                "page_recall": None,
                "keyword_recall": None,
                "ground_truth_passage_support": None,
                "unsupported_passage_rate": None,
                "faithfulness": None,
                "unsupported_rate": None,
            }

        retrieval_res: RetrievalResultSchema = await retrieve_research_context(
            question=question,
            top_k=5,
            vector_db=vector_db,
            embedding_provider=mock_embedder,
            isolate_untrusted=False,
        )

        retrieved_items = retrieval_res.items

        # Deduplicate by the actual chunk identifier returned by retrieval.
        unique_items = []
        seen_chunk_ids = set()

        for item in retrieved_items:
            if item.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(item.chunk_id)
            unique_items.append(item)

        retrieved_paper_ids = [item.paper_id for item in unique_items]
        retrieved_passages = [item.text for item in unique_items]

        correct_paper = target_paper_id in retrieved_paper_ids

        # A retrieved chunk covers an expected page if its actual page range
        # overlaps that page. No synthetic paper_id_page_number IDs are created.
        matched_pages = {
            page
            for item in unique_items
            if item.paper_id == target_paper_id
            for page in expected_pages
            if item.page_start <= page <= item.page_end
        }

        correct_page = bool(matched_pages)

        relevant_chunks = [
            item
            for item in unique_items
            if item.paper_id == target_paper_id
            and any(
                item.page_start <= page <= item.page_end
                for page in expected_pages
            )
        ]

        # This is chunk-level precision among the returned, deduplicated chunks.
        # Recall is page coverage, NOT chunk recall: the benchmark labels pages,
        # not independently annotated relevant chunk IDs.
        page_precision_at_5 = (
            len(relevant_chunks) / len(unique_items)
            if unique_items else 0.0
        )
        page_recall = (
            len(matched_pages) / len(expected_pages)
            if expected_pages else None
        )

        combined_text = " ".join(retrieved_passages).casefold()
        keyword_hits = sum(
            1 for keyword in expected_keywords
            if keyword.casefold() in combined_text
        )
        keyword_recall = (
            keyword_hits / len(expected_keywords)
            if expected_keywords else None
        )

        # This checks reference-passage support in retrieved context.
        # It does NOT evaluate a generated answer's factual faithfulness.
        if gt_passage.strip() and retrieved_passages:
            passage_support, unsupported_passage_rate, audit = evidence_grounding_eval(
                claims=[gt_passage],
                evidence_passages=retrieved_passages,
                overlap_threshold=0.20,
            )
        else:
            passage_support = None
            unsupported_passage_rate = None
            audit = []

        return {
            "qa_id": qa_id,
            "language": lang,
            "question": question,
            "target_paper_id": target_paper_id,
            "expected_pages": sorted(expected_pages),
            "evaluation_status": "EVALUATED",
            "retrieved_count": len(unique_items),
            "retrieved_chunk_ids": [item.chunk_id for item in unique_items],
            "correct_paper": correct_paper,
            "correct_page": correct_page,
            "matched_expected_pages": sorted(matched_pages),
            "page_precision_at_5": round(page_precision_at_5, 4),
            "page_recall": round(page_recall, 4) if page_recall is not None else None,
            # Compatibility aliases. These are explicitly page/chunk metrics,
            # not conventional IR precision/recall over annotated relevant chunks.
            "precision_at_5": round(page_precision_at_5, 4),
            "recall_at_5": round(page_recall, 4) if page_recall is not None else None,
            "keyword_recall": (
                round(keyword_recall, 4) if keyword_recall is not None else None
            ),
            "ground_truth_passage_support": (
                round(passage_support, 4) if passage_support is not None else None
            ),
            "unsupported_passage_rate": (
                round(unsupported_passage_rate, 4)
                if unsupported_passage_rate is not None else None
            ),
            # Legacy field names retained for existing consumers.
            "faithfulness": (
                round(passage_support, 4) if passage_support is not None else None
            ),
            "unsupported_rate": (
                round(unsupported_passage_rate, 4)
                if unsupported_passage_rate is not None else None
            ),
            "passage_support_audit": audit,
        }

    async def run(self) -> Dict[str, Any]:
        """Run the offline RAG benchmark with explicit exclusions and metric definitions."""

        mock_embedder = MockEmbeddingProvider()

        with tempfile.TemporaryDirectory() as tmp_dir:
            vector_db = await self._setup_test_vector_store(tmp_dir, mock_embedder)

            results = []
            for item in self.benchmark_data:
                result = await self.evaluate_query(item, vector_db, mock_embedder)
                results.append(result)

        evaluated = [
            result for result in results
            if result["evaluation_status"] == "EVALUATED"
        ]
        excluded = [
            result for result in results
            if result["evaluation_status"] != "EVALUATED"
        ]

        def mean_of(subset, key):
            values = [
                result[key] for result in subset
                if result.get(key) is not None
            ]
            if not values:
                return None
            return round(sum(values) / len(values), 4)

        def rate_of(subset, key):
            if not subset:
                return None
            values = [result[key] for result in subset if result.get(key) is not None]
            if not values:
                return None
            return round(
                sum(1 for value in values if value) / len(values), 4
            )

        def calc_stats(subset):
            if not subset:
                return {"count": 0}

            return {
                "count": len(subset),
                "correct_paper_rate": rate_of(subset, "correct_paper"),
                "correct_page_rate": rate_of(subset, "correct_page"),
                "mean_page_precision_at_5": mean_of(subset, "page_precision_at_5"),
                "mean_page_recall": mean_of(subset, "page_recall"),
                "mean_keyword_recall": mean_of(subset, "keyword_recall"),
                "mean_ground_truth_passage_support": mean_of(
                    subset, "ground_truth_passage_support"
                ),
                "mean_unsupported_passage_rate": mean_of(
                    subset, "unsupported_passage_rate"
                ),
                # Backward-compatible aliases. These are not answer faithfulness.
                "mean_precision_at_5": mean_of(subset, "precision_at_5"),
                "mean_recall_at_5": mean_of(subset, "recall_at_5"),
                "mean_faithfulness": mean_of(subset, "faithfulness"),
                "mean_unsupported_rate": mean_of(subset, "unsupported_rate"),
            }

        english = [r for r in evaluated if r["language"] == "en"]
        amharic = [r for r in evaluated if r["language"] == "am"]

        return {
            "evaluator": "RAGEvaluator",
            "benchmark_dataset": str(self.benchmark_path.name),
            "evaluation_mode": "offline_synthetic_corpus_mock_embeddings",
            "embedding_provider_evaluated": (
                "MockEmbeddingProvider (synthetic embeddings; "
                "retrieval plumbing verification only)"
            ),
            "live_embedding_evaluation_performed": False,
            "semantic_retrieval_quality_validated": False,
            "total_queries": len(results),
            "evaluated_queries": len(evaluated),
            "excluded_queries": len(excluded),
            "exclusion_policy": (
                "Scanned-paper questions are excluded because OCR is not implemented "
                "and production indexing rejects SCANNED_NEEDS_OCR documents."
            ),
            "metric_definitions": {
                "page_precision_at_5": (
                    "Relevant returned chunks divided by unique returned chunks; "
                    "a chunk is relevant when its paper ID matches and its page range "
                    "overlaps an expected page."
                ),
                "page_recall": (
                    "Distinct expected pages covered by retrieved chunks divided by "
                    "the number of expected pages. This is page coverage, not "
                    "conventional chunk-level recall."
                ),
                "ground_truth_passage_support": (
                    "Lexical evidence-grounding overlap between the benchmark's "
                    "reference passage and retrieved context. No answer is generated; "
                    "this is not answer faithfulness."
                ),
            },
            "overall_metrics": calc_stats(evaluated),
            "english_metrics": calc_stats(english),
            "amharic_metrics": calc_stats(amharic),
            "per_query_results": results,
        }
