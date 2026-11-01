"""Academic Paper Indexer Tool (Category: RAG / Knowledge Base).

Indexes previously extracted academic papers into ChromaDB for semantic research retrieval.
Enforces text usability verification, section-aware chunking, batch embeddings,
and atomic replacement with failure rollback.
"""
from typing import Optional

from app.core.config import settings
from logging_config.logger import get_logger
from memory.vector_store import ChromaVectorStore, vector_store as default_vector_store
from services.document_chunker import DocumentChunker
from services.embedding_provider import (
    BaseEmbeddingProvider,
    EmbeddingProviderUnavailableError,
    get_embedding_provider,
)
from services.paper_repository import PaperRepository, paper_repository as default_paper_repo
from validation.schemas import (
    ExtractionStatus,
    IndexingStatus,
    IndexingSummarySchema,
    PaperExtractionResultSchema,
)

logger = get_logger("paper_indexer")


class PaperNotFoundError(ValueError):
    """Raised when the requested paper ID cannot be found in the repository."""
    pass


class UnusablePaperTextError(ValueError):
    """Raised when an extracted paper has no extractable text or is a scanned image without OCR."""
    pass


class IndexingError(RuntimeError):
    """Raised when chunking, embedding, or vector persistence fails."""
    pass


async def index_research_paper(
    paper_id: str,
    extraction_result: Optional[PaperExtractionResultSchema] = None,
    repository: Optional[PaperRepository] = None,
    vector_db: Optional[ChromaVectorStore] = None,
    embedding_provider: Optional[BaseEmbeddingProvider] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> IndexingSummarySchema:
    """Index a previously extracted academic research paper into ChromaDB.

    Steps:
    1. Retrieve extracted paper from repository or input argument.
    2. Verify that the document text is usable.
    3. Split paper into section-aware chunks preserving page boundaries.
    4. Generate embeddings in batches via the configured embedding provider.
    5. Atomically replace previous chunks for this paper in ChromaDB.
    6. Return detailed indexing summary.
    """
    if not paper_id or not isinstance(paper_id, str):
        raise ValueError("paper_id must be a non-empty string")

    clean_id = paper_id.strip()
    repo = repository or default_paper_repo
    store = vector_db or default_vector_store
    embedder = embedding_provider or get_embedding_provider()

    # Check embedding provider availability
    if not embedder.is_available():
        logger.warning(f"Indexing requested for '{clean_id}', but embedding provider '{embedder.model_name}' is unavailable.")
        return IndexingSummarySchema(
            paper_id=clean_id,
            chunk_count=0,
            embedding_model=embedder.model_name,
            dimension=embedder.dimension,
            status=IndexingStatus.UNAVAILABLE,
            message="Semantic indexing is unavailable because no embedding provider is configured or available.",
        )

    # 1. Obtain extracted paper content
    paper_data = extraction_result or repo.get_paper(clean_id)
    if not paper_data:
        logger.error(f"Paper '{clean_id}' was not found in the extracted papers repository.")
        raise PaperNotFoundError(f"Paper '{clean_id}' not found. Paper must be extracted before indexing.")

    # 2. Validate text usability
    if paper_data.status == ExtractionStatus.SCANNED_NEEDS_OCR or paper_data.is_scanned:
        raise UnusablePaperTextError(
            f"Paper '{clean_id}' is a scanned image-only PDF and contains no extractable text for indexing."
        )

    total_chars = sum(len(txt.strip()) for txt in (paper_data.raw_text_by_page or {}).values())
    if total_chars < 10 and not paper_data.extracted_data:
        raise UnusablePaperTextError(
            f"Paper '{clean_id}' has insufficient text content ({total_chars} chars) for indexing."
        )

    # 3. Chunk the document
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_paper(
        paper_id=clean_id,
        extraction_result=paper_data,
    )

    if not chunks:
        logger.warning(f"Paper '{clean_id}' yielded 0 chunks during document chunking.")
        return IndexingSummarySchema(
            paper_id=clean_id,
            chunk_count=0,
            embedding_model=embedder.model_name,
            dimension=embedder.dimension,
            status=IndexingStatus.FAILED,
            message="No usable chunks could be produced from paper text.",
        )

    # 4. Generate embeddings
    chunk_texts = [c.text for c in chunks]
    try:
        embeddings = await embedder.embed_documents(chunk_texts)
    except Exception as e:
        logger.error(f"Failed to generate embeddings for paper '{clean_id}': {e}")
        return IndexingSummarySchema(
            paper_id=clean_id,
            chunk_count=0,
            embedding_model=embedder.model_name,
            dimension=embedder.dimension,
            status=IndexingStatus.FAILED,
            message=f"Embedding generation failed: {e}",
        )

    # 5. Persist chunks in ChromaDB with atomic replacement
    try:
        indexed_count = store.add_chunks(chunks=chunks, embeddings=embeddings)
    except Exception as e:
        logger.error(f"Failed to persist chunks in vector store for '{clean_id}': {e}")
        return IndexingSummarySchema(
            paper_id=clean_id,
            chunk_count=0,
            embedding_model=embedder.model_name,
            dimension=embedder.dimension,
            status=IndexingStatus.FAILED,
            message=f"Vector database persistence failed: {e}",
        )

    logger.info(
        f"Indexed paper '{clean_id}' successfully: {indexed_count} chunks stored in ChromaDB "
        f"(model: {embedder.model_name}, dim: {embedder.dimension})"
    )

    return IndexingSummarySchema(
        paper_id=clean_id,
        chunk_count=indexed_count,
        embedding_model=embedder.model_name,
        dimension=embedder.dimension,
        status=IndexingStatus.SUCCESS,
        message=f"Successfully indexed {indexed_count} chunks into ChromaDB.",
    )
