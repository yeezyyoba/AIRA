"""Tool 3: Research Context Retriever (Category: RAG / Knowledge Base).

Retrieves relevant academic paper chunks from ChromaDB based on semantic similarity
to a user's research question. Preserves exact page citations, section provenance,
and relevance scores.
"""
from typing import List, Optional

from app.core.config import settings
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.embedding_provider import (
    BaseEmbeddingProvider,
    EmbeddingProviderUnavailableError,
    get_embedding_provider,
)
from tools.paper_reader import format_untrusted_content_block
from validation.schemas import RetrievalResultSchema, RetrievedContextItemSchema

logger = get_logger("rag_retriever")


class RetrievalError(RuntimeError):
    """Raised when semantic research retrieval fails."""
    pass


async def retrieve_research_context(
    question: str,
    paper_ids: Optional[List[str]] = None,
    top_k: int = 5,
    vector_db: Optional[ChromaVectorStore] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    isolate_untrusted: bool = True,
) -> RetrievalResultSchema:
    """Tool 3: Retrieve relevant research context passages from ChromaDB.

    Args:
        question: Academic research query or topic.
        paper_ids: Optional list of specific paper IDs to filter against.
        top_k: Maximum number of relevant chunks to retrieve (1 to 50).
        vector_db: ChromaVectorStore instance (defaults to global instance).
        embedding_provider: BaseEmbeddingProvider instance (defaults to configured provider).
        isolate_untrusted: Whether to wrap retrieved academic text in structural isolation tags.

    Returns:
        RetrievalResultSchema with ranked passages and citation provenance.
    """
    if not question or not isinstance(question, str) or not question.strip():
        raise ValueError("Research question must be a non-empty string")

    clean_question = question.strip()
    clamped_top_k = max(1, min(top_k, settings.RAG_MAX_TOP_K))

    store = vector_db or default_vector_store
    embedder = embedding_provider or get_embedding_provider()

    filter_info = {}
    if paper_ids:
        clean_pids = [p.strip() for p in paper_ids if p and p.strip()]
        if clean_pids:
            filter_info["paper_ids"] = clean_pids

    # 1. Check if embedding provider is configured and available
    if not embedder.is_available():
        logger.warning(
            f"Retrieval requested for query '{clean_question[:40]}', but embedding provider "
            f"'{embedder.model_name}' is unconfigured or unavailable."
        )
        return RetrievalResultSchema(
            question=clean_question,
            results_count=0,
            top_k=clamped_top_k,
            embedding_model=embedder.model_name,
            items=[],
            filters_applied=filter_info,
            is_degraded=True,
            warning="Semantic retrieval is unavailable because no embedding provider is configured or available.",
        )

    # 2. Check if collection is empty
    if store.count() == 0:
        logger.info("ChromaDB vector store contains 0 documents. Returning empty context.")
        return RetrievalResultSchema(
            question=clean_question,
            results_count=0,
            top_k=clamped_top_k,
            embedding_model=embedder.model_name,
            items=[],
            filters_applied=filter_info,
            is_degraded=False,
            warning="No research papers have been indexed in the vector store yet.",
        )

    # 3. Generate query embedding
    try:
        query_vec = await embedder.embed_query(clean_question)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        return RetrievalResultSchema(
            question=clean_question,
            results_count=0,
            top_k=clamped_top_k,
            embedding_model=embedder.model_name,
            items=[],
            filters_applied=filter_info,
            is_degraded=True,
            warning=f"Query embedding generation failed: {e}",
        )

    # 4. Perform vector search
    try:
        raw_items = store.query(
            query_embedding=query_vec,
            top_k=clamped_top_k,
            paper_ids=filter_info.get("paper_ids"),
        )
    except Exception as e:
        logger.error(f"Vector search failed in ChromaDB: {e}")
        raise RetrievalError(f"Vector store search failed: {e}") from e

    # 5. Format and sanitize retrieved items (wrap untrusted text to prevent prompt injection)
    is_synth = getattr(embedder, "is_synthetic", False)
    formatted_items: List[RetrievedContextItemSchema] = []
    for item in raw_items:
        text_val = format_untrusted_content_block(item.text) if isolate_untrusted else item.text
        formatted_items.append(
            RetrievedContextItemSchema(
                chunk_id=item.chunk_id,
                paper_id=item.paper_id,
                text=text_val,
                document_title=item.document_title,
                section_heading=item.section_heading,
                page_start=item.page_start,
                page_end=item.page_end,
                similarity_score=item.similarity_score,
                distance=item.distance,
                authors=item.authors,
                publication_year=item.publication_year,
                evidence_type=item.evidence_type,
                is_synthetic=is_synth,
            )
        )

    logger.info(
        f"Retrieved {len(formatted_items)} context passages for question '{clean_question[:40]}' "
        f"(top_k={clamped_top_k}, filters={filter_info}, synthetic={is_synth})"
    )

    return RetrievalResultSchema(
        question=clean_question,
        results_count=len(formatted_items),
        top_k=clamped_top_k,
        embedding_model=embedder.model_name,
        items=formatted_items,
        filters_applied=filter_info,
        is_degraded=False,
        is_synthetic=is_synth,
        warning=None if formatted_items else "No matching research passages found for this question and filter criteria.",
    )
