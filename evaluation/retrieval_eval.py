"""Retrieval Quality Evaluation Benchmark for AIRA Research Context Retrieval (RAG).

Provides a reproducible evaluation dataset and scoring tool to measure retrieval quality
across academic research queries in English and Amharic.
Evaluates:
- Recall@5
- Precision@5
- Correct-paper rate
- Correct-page rate
Explicitly documents whether results are generated from mock embeddings or live Gemini embeddings.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from validation.schemas import (
    EvidenceItemSchema,
    ExtractedPaperDataSchema,
    ExtractionStatus,
    PaperExtractionResultSchema,
)
from services.document_chunker import DocumentChunker
from services.embedding_provider import BaseEmbeddingProvider, MockEmbeddingProvider
from memory.vector_store import ChromaVectorStore
from tools.rag_retriever import retrieve_research_context


@dataclass
class EvalQuery:
    query_id: str
    language: str  # 'en' or 'am'
    question: str
    target_paper_id: str
    relevant_pages: List[int]
    expected_topics: List[str]


EVALUATION_DATASET: List[EvalQuery] = [
    # English Queries
    EvalQuery(
        query_id="en_01",
        language="en",
        question="What BLEU score did the neural machine translation model achieve for Amharic?",
        target_paper_id="amharic_nmt_2024",
        relevant_pages=[3],
        expected_topics=["bleu", "score", "translation"],
    ),
    EvalQuery(
        query_id="en_02",
        language="en",
        question="What dataset size and parallel sentence pairs were used in the evaluation?",
        target_paper_id="amharic_nmt_2024",
        relevant_pages=[2],
        expected_topics=["50,000", "sentence pairs", "dataset"],
    ),
    EvalQuery(
        query_id="en_03",
        language="en",
        question="What transformer architecture and layers were employed?",
        target_paper_id="amharic_nmt_2024",
        relevant_pages=[2],
        expected_topics=["transformer", "12-layer", "cross-lingual"],
    ),
    # Amharic Queries (Ethiopic Script)
    EvalQuery(
        query_id="am_01",
        language="am",
        question="የአማርኛ የነርቭ ማሽን ትርጉም ሞዴል ያስመዘገበው የBLEU ውጤት ስንት ነው?",
        target_paper_id="amharic_nmt_2024",
        relevant_pages=[3],
        expected_topics=["bleu", "ውጤት", "31.2"],
    ),
    EvalQuery(
        query_id="am_02",
        language="am",
        question="ጥናቱ የተጠቀመው ትይዩ የመረጃ ስብስብ (parallel dataset) ስንት ዓረፍተ ነገሮችን ይዟል?",
        target_paper_id="amharic_nmt_2024",
        relevant_pages=[2],
        expected_topics=["50,000", "ትይዩ", "ዓረፍተ ነገሮች"],
    ),
    EvalQuery(
        query_id="am_03",
        language="am",
        question="በዚህ ጥናት ውስጥ የተገነባው የሞዴል መዋቅር (architecture) ምን ይመስላል?",
        target_paper_id="amharic_nmt_2024",
        relevant_pages=[2],
        expected_topics=["12-layer", "Transformer", "መዋቅር"],
    ),
]


def create_eval_benchmark_papers() -> List[PaperExtractionResultSchema]:
    """Create benchmark papers including target paper and distractor papers."""
    target_pages = {
        1: (
            "Neural Machine Translation for Low-Resource Amharic: An Empirical Study. "
            "Abebe Bikila, Almaz Worku. Department of Computer Science, Addis Ababa University. "
            "Abstract: We present an empirical evaluation of cross-lingual transfer for low-resource "
            "Amharic neural translation, addressing morphological complexity and vocabulary sparsity."
        ),
        2: (
            "Methodology and Experimental Setup. "
            "Our neural architecture employs a 12-layer Transformer with cross-lingual pretraining. "
            "We construct and evaluate sentence representations across 50,000 parallel Amharic-English sentence pairs "
            "collected from news, parliamentary proceedings, and religious texts. "
            "Subword segmentation uses SentencePiece with a unigram vocabulary of 32,000 tokens."
        ),
        3: (
            "Results and Quantitative Discussion. "
            "The proposed Amharic NMT system achieves a BLEU score of 31.2, outperforming previous baselines by 4.8 points. "
            "Acoustic and semantic evaluations confirm statistically significant improvements across all test subsets. "
            "Human evaluators noted high fluency and grammatical accuracy in Amharic output sentences."
        ),
        4: (
            "Limitations and Conclusion. "
            "Although BLEU reached 31.2, out-of-vocabulary morphological variants still cause hallucinations. "
            "In conclusion, cross-lingual pretraining significantly improves Amharic translation accuracy."
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
            EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=2, section_heading="Methodology", passage="12-layer Transformer"),
            EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=3, section_heading="Results", passage="BLEU score of 31.2"),
            EvidenceItemSchema(paper_id="amharic_nmt_2024", page_number=4, section_heading="Conclusion", passage="cross-lingual pretraining"),
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
        1: "YOLOv8 Object Detection on Satellite Imagery. Abstract: Aerial object tracking in urban environments.",
        2: "Methodology: We use convolutional feature pyramids and bounding box regression on DroneNet dataset.",
        3: "Results: Mean Average Precision (mAP) reached 68.4% on 10,000 aerial tiles.",
    }
    distractor_extracted_1 = ExtractedPaperDataSchema(
        paper_id="distractor_cv_2023",
        title="YOLOv8 Object Detection on Satellite Imagery",
        authors=["John Doe"],
        abstract=distractor_pages_1[1],
        methodology=distractor_pages_1[2],
        results=distractor_pages_1[3],
    )
    paper_distractor_1 = PaperExtractionResultSchema(
        paper_id="distractor_cv_2023",
        status=ExtractionStatus.SUCCESS,
        extraction_method="pymupdf",
        page_count=3,
        raw_text_by_page=distractor_pages_1,
        extracted_data=distractor_extracted_1,
    )

    distractor_pages_2 = {
        1: "Clinical Sentiment Classification in Patient Portals. Abstract: Patient survey text analysis.",
        2: "Methodology: BiLSTM and clinical BioBERT fine-tuning on 15,000 hospital discharge summaries.",
        3: "Results: Macro F1-score of 84.1% for detecting negative sentiment in medical reviews.",
    }
    distractor_extracted_2 = ExtractedPaperDataSchema(
        paper_id="distractor_health_2023",
        title="Clinical Sentiment Classification in Patient Portals",
        authors=["Jane Smith"],
        abstract=distractor_pages_2[1],
        methodology=distractor_pages_2[2],
        results=distractor_pages_2[3],
    )
    paper_distractor_2 = PaperExtractionResultSchema(
        paper_id="distractor_health_2023",
        status=ExtractionStatus.SUCCESS,
        extraction_method="pymupdf",
        page_count=3,
        raw_text_by_page=distractor_pages_2,
        extracted_data=distractor_extracted_2,
    )

    return [paper_target, paper_distractor_1, paper_distractor_2]


@dataclass
class EvalSummary:
    total_queries: int
    recall_at_5: float
    precision_at_5: float
    correct_paper_rate: float
    correct_page_rate: float
    is_synthetic: bool
    embedding_model: str
    details: List[Dict[str, Any]] = field(default_factory=list)


async def run_retrieval_benchmark(
    vector_db: ChromaVectorStore,
    embedder: BaseEmbeddingProvider,
    top_k: int = 5,
) -> EvalSummary:
    """Run reproducible retrieval benchmark across English and Amharic queries."""
    total = len(EVALUATION_DATASET)
    if total == 0:
        return EvalSummary(0, 0.0, 0.0, 0.0, 0.0, embedder.is_synthetic, embedder.model_name)

    correct_paper_hits = 0
    correct_page_hits = 0
    total_relevant_retrieved = 0
    total_retrieved = 0
    details = []

    for eq in EVALUATION_DATASET:
        res = await retrieve_research_context(
            question=eq.question,
            top_k=top_k,
            vector_db=vector_db,
            embedding_provider=embedder,
            isolate_untrusted=False,
        )

        retrieved_items = res.items
        total_retrieved += len(retrieved_items)

        paper_matched = any(item.paper_id == eq.target_paper_id for item in retrieved_items)
        if paper_matched:
            correct_paper_hits += 1

        page_matched = any(
            item.paper_id == eq.target_paper_id and any(
                p in eq.relevant_pages for p in range(item.page_start, item.page_end + 1)
            )
            for item in retrieved_items
        )
        if page_matched:
            correct_page_hits += 1

        relevant_in_top_k = sum(
            1 for item in retrieved_items
            if item.paper_id == eq.target_paper_id and any(
                p in eq.relevant_pages for p in range(item.page_start, item.page_end + 1)
            )
        )
        total_relevant_retrieved += relevant_in_top_k

        details.append({
            "query_id": eq.query_id,
            "language": eq.language,
            "question": eq.question,
            "paper_matched": paper_matched,
            "page_matched": page_matched,
            "relevant_in_top_k": relevant_in_top_k,
            "retrieved_count": len(retrieved_items),
        })

    recall_at_5 = correct_page_hits / total if total > 0 else 0.0
    precision_at_5 = total_relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
    correct_paper_rate = correct_paper_hits / total if total > 0 else 0.0
    correct_page_rate = correct_page_hits / total if total > 0 else 0.0

    return EvalSummary(
        total_queries=total,
        recall_at_5=round(recall_at_5, 4),
        precision_at_5=round(precision_at_5, 4),
        correct_paper_rate=round(correct_paper_rate, 4),
        correct_page_rate=round(correct_page_rate, 4),
        is_synthetic=embedder.is_synthetic,
        embedding_model=embedder.model_name,
        details=details,
    )
