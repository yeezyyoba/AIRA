"""Evaluator for Academic Paper Search (Information Retrieval Metrics).

Measures:
- Precision@5, Precision@10
- Recall@5, Recall@10
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG@5, nDCG@10)
Uses academic_search_benchmark.json with realistic queries across Amharic NLP,
sentiment analysis, NER, MT, handwritten OCR, and low-resource/multilingual models.
Zero live API quota consumed.
"""
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import AsyncMock

from evaluation.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from services.academic_clients import CrossrefClient, OpenAlexClient, SemanticScholarClient
from tools.academic_search import search_academic_papers
from validation.schemas import PaperMetadataSchema


class AcademicSearchEvaluator:
    """Evaluates multi-provider academic search against graded relevance benchmarks."""

    def __init__(self, benchmark_path: Optional[Path] = None):
        if benchmark_path is None:
            benchmark_path = Path(__file__).resolve().parent.parent / "benchmarks" / "academic_search_benchmark.json"
        self.benchmark_path = benchmark_path
        with open(self.benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark_data = json.load(f)

    def _build_mock_clients(self, query_item: Dict[str, Any]) -> Tuple[OpenAlexClient, SemanticScholarClient, CrossrefClient]:
        """Construct mock academic clients populated with candidate papers for the given query."""
        relevance_map = query_item.get("relevance_judgments", {})
        query_id = query_item.get("query_id", "")

        # Generate realistic paper records based on relevance judgments
        openalex_papers = []
        s2_papers = []
        crossref_papers = []

        # Natural benchmark candidate order (independent of ground-truth score)
        candidates = list(relevance_map.items())

        for i, (doc_id, score) in enumerate(candidates):
            title = doc_id.replace("_", " ").title()
            pub_year = 2020 + (i % 5)
            # Realistic deterministic citation count independent of relevance score
            doc_hash = int.from_bytes(hashlib.md5(doc_id.encode("utf-8")).digest()[:4], "big")
            citations = 5 + (doc_hash % 40) + (2025 - pub_year) * 3

            paper_meta = PaperMetadataSchema(
                title=title,
                authors=["Abebe Kebede", "Mulugeta Bekele"] if i % 2 == 0 else ["Solomon Desta"],
                year=pub_year,
                doi=f"10.1145/{query_id.lower()}.{doc_id}",
                abstract=f"Academic study regarding {' '.join(query_item.get('expected_topics', []))}. Evaluated on standard corpora.",
                venue="ACL / EMNLP Anthologies",
                citation_count=citations,
                source="openalex" if i % 2 == 0 else "semantic_scholar",
                sources=["openalex"] if i % 2 == 0 else ["semantic_scholar"],
            )
            if i % 2 == 0:
                openalex_papers.append(paper_meta)
            else:
                s2_papers.append(paper_meta)

            # Realistic multi-source coverage: simulate papers indexed across providers
            if i % 3 == 0 or "nmt" in doc_id or "ner" in doc_id:
                crossref_paper = PaperMetadataSchema(
                    title=title,
                    authors=paper_meta.authors,
                    year=paper_meta.year,
                    doi=paper_meta.doi,
                    abstract=paper_meta.abstract,
                    venue=paper_meta.venue,
                    citation_count=paper_meta.citation_count,
                    source="crossref",
                    sources=["crossref"],
                )
                crossref_papers.append(crossref_paper)


        mock_oa = OpenAlexClient()
        mock_oa.search = AsyncMock(return_value=openalex_papers)

        mock_s2 = SemanticScholarClient()
        mock_s2.search = AsyncMock(return_value=s2_papers)

        mock_cr = CrossrefClient()
        mock_cr.search = AsyncMock(return_value=crossref_papers)

        return mock_oa, mock_s2, mock_cr

    async def evaluate_query(self, query_item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate academic search for a single benchmark query."""
        query_text = query_item["query"]
        relevance_map = query_item["relevance_judgments"]
        # Relevant set defined as relevance score >= 2 (Highly relevant or relevant)
        relevant_docs = {doc_id for doc_id, score in relevance_map.items() if score >= 2}

        mock_oa, mock_s2, mock_cr = self._build_mock_clients(query_item)

        search_result = await search_academic_papers(
            query=query_text,
            max_results=10,
            openalex_client=mock_oa,
            semantic_scholar_client=mock_s2,
            crossref_client=mock_cr,
        )

        retrieved_papers = search_result.get("papers", [])
        # Extract paper identifiers (matching normalized title/key or doi)
        retrieved_ids = []
        for p in retrieved_papers:
            # Map back to benchmark doc_id
            matched_id = None
            p_title = p.get("title", "") if isinstance(p, dict) else getattr(p, "title", "")
            p_doi = p.get("doi", "") or "" if isinstance(p, dict) else getattr(p, "doi", "") or ""
            p_title_norm = p_title.lower().replace(" ", "_").strip()
            for doc_id in relevance_map:
                if doc_id in p_title_norm or doc_id in p_doi:
                    matched_id = doc_id
                    break
            if matched_id:
                retrieved_ids.append(matched_id)
            else:
                retrieved_ids.append(p_title)

        p_5 = precision_at_k(retrieved_ids, relevant_docs, k=5)
        r_5 = recall_at_k(retrieved_ids, relevant_docs, k=5)
        p_10 = precision_at_k(retrieved_ids, relevant_docs, k=10)
        r_10 = recall_at_k(retrieved_ids, relevant_docs, k=10)
        rr = reciprocal_rank(retrieved_ids, relevant_docs)
        ndcg_5 = ndcg_at_k(retrieved_ids, relevance_map, k=5)
        ndcg_10 = ndcg_at_k(retrieved_ids, relevance_map, k=10)

        return {
            "query_id": query_item["query_id"],
            "domain": query_item.get("domain", ""),
            "query": query_text,
            "retrieved_count": len(retrieved_papers),
            "retrieved_ids": retrieved_ids,
            "relevant_count": len(relevant_docs),
            "precision_at_5": round(p_5, 4),
            "recall_at_5": round(r_5, 4) if r_5 is not None else 0.0,
            "precision_at_10": round(p_10, 4),
            "recall_at_10": round(r_10, 4) if r_10 is not None else 0.0,
            "reciprocal_rank": round(rr, 4),
            "ndcg_at_5": round(ndcg_5, 4),
            "ndcg_at_10": round(ndcg_10, 4),
        }

    async def run(self) -> Dict[str, Any]:
        """Run complete academic search evaluation across all benchmark queries."""
        query_evals = []
        for item in self.benchmark_data:
            eval_res = await self.evaluate_query(item)
            query_evals.append(eval_res)

        total_queries = len(query_evals)
        if total_queries == 0:
            return {"error": "Empty search benchmark dataset"}

        mean_p5 = sum(q["precision_at_5"] for q in query_evals) / total_queries
        mean_r5 = sum(q["recall_at_5"] for q in query_evals) / total_queries
        mean_p10 = sum(q["precision_at_10"] for q in query_evals) / total_queries
        mean_r10 = sum(q["recall_at_10"] for q in query_evals) / total_queries
        mrr = sum(q["reciprocal_rank"] for q in query_evals) / total_queries
        mean_ndcg5 = sum(q["ndcg_at_5"] for q in query_evals) / total_queries
        mean_ndcg10 = sum(q["ndcg_at_10"] for q in query_evals) / total_queries

        return {
            "evaluator": "AcademicSearchEvaluator",
            "benchmark_dataset": str(self.benchmark_path.name),
            "total_queries": total_queries,
            "aggregate_metrics": {
                "mean_precision_at_5": round(mean_p5, 4),
                "mean_recall_at_5": round(mean_r5, 4),
                "mean_precision_at_10": round(mean_p10, 4),
                "mean_recall_at_10": round(mean_r10, 4),
                "mrr": round(mrr, 4),
                "mean_ndcg_at_5": round(mean_ndcg5, 4),
                "mean_ndcg_at_10": round(mean_ndcg10, 4),
            },
            "per_query_results": query_evals,
        }
