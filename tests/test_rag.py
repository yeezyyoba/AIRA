"""Offline Unit and Integration Tests for Phase 5: Research Context Retrieval (RAG).

Covers:
- Section-aware document chunking and page boundary preservation.
- Chunk overlap and sentence-boundary integrity.
- Empty and short document chunking.
- Stable deterministic chunk IDs.
- Provider-independent embedding interface.
- Batch embedding generation.
- Invalid embedding dimensions, NaN, and Inf validation.
- Missing/Null embedding configuration and degraded mode.
- Provider timeout and rate-limit error handling.
- Paper repository persistence and retrieval.
- Indexing tool with atomic replacement and error rollback.
- Retrieval tool with ranking, paper-specific filtering, and prompt injection isolation.
- FastAPI REST endpoints: index, retrieve, indexing-status, and delete-index.
"""
import io
import math
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from memory.vector_store import ChromaVectorStore
from services.document_chunker import DocumentChunker
from services.embedding_provider import (
    BaseEmbeddingProvider,
    EmbeddingError,
    EmbeddingProviderUnavailableError,
    EmbeddingRateLimitError,
    EmbeddingTimeoutError,
    InvalidEmbeddingError,
    MockEmbeddingProvider,
    NullEmbeddingProvider,
    get_embedding_provider,
)
from services.paper_repository import PaperRepository
from tools.paper_indexer import (
    IndexingError,
    PaperNotFoundError,
    UnusablePaperTextError,
    index_research_paper,
)
from tools.rag_retriever import retrieve_research_context
from validation.schemas import (
    DocumentChunkSchema,
    EvidenceItemSchema,
    EvidenceType,
    ExtractedPaperDataSchema,
    ExtractionStatus,
    IndexingStatus,
    PaperExtractionResultSchema,
)


# ==============================================================================
# Helper Fixtures & Synthetic Extraction Results
# ==============================================================================

def create_sample_extraction_result(paper_id: str = "amharic_nlp_2024") -> PaperExtractionResultSchema:
    """Create a structured PaperExtractionResultSchema for RAG testing."""
    abstract_text = (
        "We present a comprehensive empirical survey of neural machine translation for Amharic. "
        "Low-resource constraints require novel subword tokenization and back-translation strategies."
    )
    methodology_text = (
        "Our neural architecture employs a 12-layer Transformer with cross-lingual pretraining. "
        "We evaluate sentence representations across 50,000 parallel Amharic-English sentence pairs."
    )
    results_text = (
        "The proposed approach achieves a BLEU score of 31.2, outperforming existing baselines by 4.8 points. "
        "Acoustic and semantic evaluations confirm statistically significant improvements."
    )
    conclusion_text = (
        "In conclusion, cross-lingual transfer learning offers a viable path for under-resourced Ethiopic languages."
    )

    extracted_data = ExtractedPaperDataSchema(
        paper_id=paper_id,
        title="Neural Machine Translation for Low-Resource Amharic",
        authors=["Dr. Abebe Bikila", "Prof. Almaz Worku"],
        abstract=abstract_text,
        methodology=methodology_text,
        results=results_text,
        conclusion=conclusion_text,
        evidence=[
            EvidenceItemSchema(
                paper_id=paper_id,
                page_number=1,
                section_heading="Abstract",
                passage=abstract_text[:100],
            ),
            EvidenceItemSchema(
                paper_id=paper_id,
                page_number=2,
                section_heading="Methodology",
                passage=methodology_text[:100],
            ),
            EvidenceItemSchema(
                paper_id=paper_id,
                page_number=3,
                section_heading="Results",
                passage=results_text[:100],
            ),
        ],
    )

    raw_text_by_page = {
        1: f"Title\nAbstract\n{abstract_text}\n",
        2: f"Methodology\n{methodology_text}\n",
        3: f"Results\n{results_text}\nConclusion\n{conclusion_text}\n",
    }

    return PaperExtractionResultSchema(
        paper_id=paper_id,
        status=ExtractionStatus.SUCCESS,
        extraction_method="pymupdf",
        page_count=3,
        document_metadata={"title": "Neural Machine Translation for Low-Resource Amharic"},
        extracted_data=extracted_data,
        raw_text_by_page=raw_text_by_page,
    )


# ==============================================================================
# 1. Document Chunking Tests
# ==============================================================================

def test_document_chunker_section_aware_and_page_preservation():
    """Verify that chunker creates chunks mapped to academic sections with valid page ranges."""
    extraction = create_sample_extraction_result("amharic_nmt")
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

    chunks = chunker.chunk_paper(
        paper_id="amharic_nmt",
        extraction_result=extraction,
        document_title="Neural Machine Translation for Low-Resource Amharic",
        authors=["Dr. Abebe Bikila"],
        publication_year=2024,
    )

    assert len(chunks) >= 4
    sections = [c.section_heading for c in chunks]
    assert "Abstract" in sections
    assert "Methodology" in sections
    assert "Results" in sections
    assert "Conclusion" in sections

    # Check page numbers
    for c in chunks:
        assert c.page_start >= 1
        assert c.page_end >= c.page_start
        assert c.paper_id == "amharic_nmt"
        assert c.authors == ["Dr. Abebe Bikila"]
        assert c.publication_year == 2024


def test_document_chunker_stable_deterministic_ids():
    """Verify that chunk IDs follow the deterministic format {paper_id}_c{index:04d}."""
    extraction = create_sample_extraction_result("stable_test")
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks1 = chunker.chunk_paper("stable_test", extraction)
    chunks2 = chunker.chunk_paper("stable_test", extraction)

    assert len(chunks1) == len(chunks2)
    for i in range(len(chunks1)):
        assert chunks1[i].chunk_id == f"stable_test_c{i+1:04d}"
        assert chunks1[i].chunk_id == chunks2[i].chunk_id
        assert chunks1[i].text == chunks2[i].text


def test_document_chunker_handles_missing_sections_fallback():
    """Verify that documents with missing section headings chunk by page without crashing."""
    raw_pages = {
        1: "First page unformatted text without any section headings at all.",
        2: "Second page continuing discussion of language modeling.",
    }
    extraction = PaperExtractionResultSchema(
        paper_id="unsectioned_doc",
        status=ExtractionStatus.PARTIAL,
        extraction_method="pymupdf",
        page_count=2,
        extracted_data=None,  # No sections identified
        raw_text_by_page=raw_pages,
    )

    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_paper("unsectioned_doc", extraction)

    assert len(chunks) == 2
    assert chunks[0].section_heading is None
    assert chunks[0].page_start == 1
    assert chunks[1].page_start == 2


def test_document_chunker_short_and_empty_documents():
    """Verify handling of very short text and empty documents."""
    chunker = DocumentChunker()

    # Short document -> exactly 1 chunk
    short_extraction = PaperExtractionResultSchema(
        paper_id="short_doc",
        status=ExtractionStatus.SUCCESS,
        extraction_method="pymupdf",
        page_count=1,
        raw_text_by_page={1: "Short text."},
    )
    short_chunks = chunker.chunk_paper("short_doc", short_extraction)
    assert len(short_chunks) == 1
    assert short_chunks[0].text == "Short text."

    # Empty document -> 0 chunks
    empty_extraction = PaperExtractionResultSchema(
        paper_id="empty_doc",
        status=ExtractionStatus.FAILED,
        extraction_method="none",
        page_count=0,
        raw_text_by_page={},
    )
    empty_chunks = chunker.chunk_paper("empty_doc", empty_extraction)
    assert len(empty_chunks) == 0


def test_document_chunker_invalid_overlap_raises():
    """Verify that chunk_overlap >= chunk_size raises ValueError."""
    with pytest.raises(ValueError, match="strictly less"):
        DocumentChunker(chunk_size=100, chunk_overlap=150)


# ==============================================================================
# 2. Embedding Provider Tests
# ==============================================================================

@pytest.mark.anyio
async def test_mock_embedding_provider_determinism_and_dimension():
    """Verify MockEmbeddingProvider generates valid, normalized vectors deterministically."""
    provider = MockEmbeddingProvider(dimension=128)
    assert provider.is_available() is True

    vec1 = await provider.embed_query("Amharic text")
    vec2 = await provider.embed_query("Amharic text")
    vec3 = await provider.embed_query("Different topic")

    assert len(vec1) == 128
    assert vec1 == vec2  # Exactly deterministic
    assert vec1 != vec3

    # Check unit norm (length ~ 1.0)
    norm = math.sqrt(sum(x * x for x in vec1))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


@pytest.mark.anyio
async def test_embedding_batch_generation():
    """Verify that batch embedding handles multiple items and obeys batch sizing."""
    provider = MockEmbeddingProvider(dimension=64, batch_size=2)
    texts = ["Text A", "Text B", "Text C", "Text D", "Text E"]

    embeddings = await provider.embed_documents(texts)
    assert len(embeddings) == 5
    for emb in embeddings:
        assert len(emb) == 64


@pytest.mark.anyio
async def test_embedding_validation_dimension_mismatch():
    """Verify that provider raises InvalidEmbeddingError if returned vector dimension mismatches."""
    class BadDimensionProvider(BaseEmbeddingProvider):
        def is_available(self) -> bool:
            return True

        async def _generate_embeddings(self, texts):
            # Returns 3 dimensions when 4 is expected
            return [[0.1, 0.2, 0.3] for _ in texts]

    bad_provider = BadDimensionProvider(model_name="bad-dim", dimension=4)
    with pytest.raises(InvalidEmbeddingError, match="dimension mismatch"):
        await bad_provider.embed_documents(["test text"])


@pytest.mark.anyio
async def test_embedding_validation_nan_and_inf():
    """Verify that NaN or infinite numerical values in embeddings raise InvalidEmbeddingError."""
    class NanEmbeddingProvider(BaseEmbeddingProvider):
        def is_available(self) -> bool:
            return True

        async def _generate_embeddings(self, texts):
            return [[0.1, float("nan"), 0.3, 0.4] for _ in texts]

    class InfEmbeddingProvider(BaseEmbeddingProvider):
        def is_available(self) -> bool:
            return True

        async def _generate_embeddings(self, texts):
            return [[0.1, float("inf"), 0.3, 0.4] for _ in texts]

    nan_prov = NanEmbeddingProvider(model_name="nan-prov", dimension=4)
    with pytest.raises(InvalidEmbeddingError, match="invalid numerical value"):
        await nan_prov.embed_documents(["test text"])

    inf_prov = InfEmbeddingProvider(model_name="inf-prov", dimension=4)
    with pytest.raises(InvalidEmbeddingError, match="invalid numerical value"):
        await inf_prov.embed_documents(["test text"])


@pytest.mark.anyio
async def test_null_embedding_provider_raises_unavailable():
    """Verify that NullEmbeddingProvider correctly reports unavailable and raises on call."""
    null_prov = NullEmbeddingProvider()
    assert null_prov.is_available() is False

    with pytest.raises(EmbeddingProviderUnavailableError, match="unavailable"):
        await null_prov.embed_documents(["sample text"])


@pytest.mark.anyio
async def test_embedding_timeout_and_rate_limit_handling():
    """Verify that provider timeouts raise EmbeddingTimeoutError."""
    class SlowProvider(BaseEmbeddingProvider):
        def is_available(self) -> bool:
            return True

        async def _generate_embeddings(self, texts):
            import asyncio
            await asyncio.sleep(0.5)
            return [[0.1, 0.2] for _ in texts]

    slow_prov = SlowProvider(model_name="slow", dimension=2, timeout=0.05)
    with pytest.raises(EmbeddingTimeoutError, match="timed out"):
        await slow_prov.embed_documents(["sample"])


# ==============================================================================
# 3. Paper Repository & Indexing Tool Tests
# ==============================================================================

@pytest.mark.anyio
async def test_paper_repository_save_and_retrieve():
    """Verify that extracted papers are persisted and loaded from disk safely."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = PaperRepository(storage_dir=tmpdir)
        paper = create_sample_extraction_result("repo_paper_01")

        # Save
        saved_path = repo.save_paper(paper)
        assert saved_path.exists()
        assert repo.has_paper("repo_paper_01")
        assert "repo_paper_01" in repo.list_papers()

        # Load
        loaded = repo.get_paper("repo_paper_01")
        assert loaded is not None
        assert loaded.paper_id == "repo_paper_01"
        assert loaded.page_count == 3
        assert loaded.extracted_data.abstract is not None


@pytest.mark.anyio
async def test_index_research_paper_success():
    """Verify complete indexing workflow with MockEmbeddingProvider and ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "papers"
        chroma_dir = Path(tmpdir) / "chroma"

        repo = PaperRepository(storage_dir=str(repo_dir))
        store = ChromaVectorStore(persist_directory=str(chroma_dir), dimension=32)
        embedder = MockEmbeddingProvider(dimension=32)

        paper = create_sample_extraction_result("index_success_paper")
        repo.save_paper(paper)

        summary = await index_research_paper(
            paper_id="index_success_paper",
            repository=repo,
            vector_db=store,
            embedding_provider=embedder,
        )

        assert summary.status == IndexingStatus.SUCCESS
        assert summary.paper_id == "index_success_paper"
        assert summary.chunk_count >= 4
        assert summary.dimension == 32
        assert store.count() == summary.chunk_count


@pytest.mark.anyio
async def test_index_research_paper_not_found():
    """Verify that indexing an unextracted paper raises PaperNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = PaperRepository(storage_dir=tmpdir)
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)

        with pytest.raises(PaperNotFoundError, match="not found"):
            await index_research_paper(
                paper_id="non_existent_paper_id",
                repository=repo,
                vector_db=store,
            )


@pytest.mark.anyio
async def test_index_research_paper_unusable_scanned_pdf():
    """Verify that scanned image-only PDFs raise UnusablePaperTextError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = PaperRepository(storage_dir=tmpdir)
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)

        scanned = PaperExtractionResultSchema(
            paper_id="scanned_paper",
            status=ExtractionStatus.SCANNED_NEEDS_OCR,
            extraction_method="pymupdf",
            is_scanned=True,
            ocr_required=True,
            page_count=5,
            raw_text_by_page={},
        )
        repo.save_paper(scanned)

        with pytest.raises(UnusablePaperTextError, match="scanned image-only PDF"):
            await index_research_paper(
                paper_id="scanned_paper",
                repository=repo,
                vector_db=store,
            )


@pytest.mark.anyio
async def test_index_research_paper_degraded_when_provider_unavailable():
    """Verify that indexer reports UNAVAILABLE when embedding provider is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = PaperRepository(storage_dir=tmpdir)
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)
        paper = create_sample_extraction_result("degraded_index_paper")
        repo.save_paper(paper)

        summary = await index_research_paper(
            paper_id="degraded_index_paper",
            repository=repo,
            vector_db=store,
            embedding_provider=NullEmbeddingProvider(),
        )

        assert summary.status == IndexingStatus.UNAVAILABLE
        assert "no embedding provider" in summary.message


# ==============================================================================
# 4. Retrieval Tool Tests
# ==============================================================================

@pytest.mark.anyio
async def test_retrieve_research_context_semantic_search():
    """Verify semantic retrieval returns relevant ranked chunks with provenance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        chroma_dir = Path(tmpdir) / "chroma"
        store = ChromaVectorStore(persist_directory=str(chroma_dir), dimension=32)
        embedder = MockEmbeddingProvider(dimension=32)

        paper = create_sample_extraction_result("retrieval_paper")
        chunker = DocumentChunker()
        chunks = chunker.chunk_paper("retrieval_paper", paper)
        embeds = await embedder.embed_documents([c.text for c in chunks])
        store.add_chunks(chunks, embeds)

        # Retrieve
        result = await retrieve_research_context(
            question="What is the BLEU score and evaluation results?",
            top_k=3,
            vector_db=store,
            embedding_provider=embedder,
            isolate_untrusted=True,
        )

        assert result.results_count > 0
        assert len(result.items) <= 3
        assert result.items[0].paper_id == "retrieval_paper"
        assert result.items[0].similarity_score >= 0.0

        # Verify prompt injection containment wrapper
        assert result.items[0].text.startswith("<untrusted_academic_content>")
        assert result.items[0].text.endswith("</untrusted_academic_content>")


@pytest.mark.anyio
async def test_retrieve_research_context_empty_collection():
    """Verify that retrieval from an empty collection returns 0 results with an explicit warning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)
        embedder = MockEmbeddingProvider(dimension=16)

        result = await retrieve_research_context(
            question="Any research findings?",
            vector_db=store,
            embedding_provider=embedder,
        )

        assert result.results_count == 0
        assert len(result.items) == 0
        assert "No research papers have been indexed" in result.warning


@pytest.mark.anyio
async def test_retrieve_research_context_degraded_provider():
    """Verify retrieval returns degraded status when embedding provider is disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)
        result = await retrieve_research_context(
            question="Test query",
            vector_db=store,
            embedding_provider=NullEmbeddingProvider(),
        )

        assert result.is_degraded is True
        assert result.results_count == 0
        assert "no embedding provider" in result.warning


# ==============================================================================
# 5. FastAPI Endpoint Tests
# ==============================================================================

def test_fastapi_papers_rag_endpoints_lifecycle():
    """Verify full REST lifecycle: extraction -> indexing -> retrieve -> status -> delete."""
    client = TestClient(app)

    # 1. Read / extract a synthetic paper via endpoint
    from tests.test_paper_reader import create_synthetic_pdf
    pdf_bytes = create_synthetic_pdf([
        "Deep Learning for Amharic Text\n\nAbstract\nWe study neural architectures for Amharic.\n\nMethodology\nTransformer models.\n"
    ])

    upload_resp = client.post(
        "/api/v1/papers/upload",
        files={"file": ("amharic_deep.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"paper_id": "amharic_deep_01"},
    )
    assert upload_resp.status_code == 200

    # 2. Index the paper
    index_resp = client.post(
        "/api/v1/papers/amharic_deep_01/index",
        json={"chunk_size": 500, "chunk_overlap": 50},
    )
    assert index_resp.status_code == 200
    idx_data = index_resp.json()
    assert idx_data["paper_id"] == "amharic_deep_01"
    assert idx_data["status"] == "SUCCESS"
    assert idx_data["chunk_count"] > 0

    # 3. Check indexing status
    status_resp = client.get("/api/v1/papers/amharic_deep_01/indexing-status")
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["is_indexed"] is True
    assert st_data["chunk_count"] > 0

    # 4. Retrieve context
    retrieve_resp = client.post(
        "/api/v1/papers/retrieve",
        json={"question": "What neural architecture was used?", "top_k": 3},
    )
    assert retrieve_resp.status_code == 200
    ret_data = retrieve_resp.json()
    assert ret_data["results_count"] > 0
    assert len(ret_data["items"]) > 0

    # 5. Delete paper index
    del_resp = client.delete("/api/v1/papers/amharic_deep_01/index")
    assert del_resp.status_code == 200
    del_data = del_resp.json()
    assert del_data["deleted_chunks"] > 0


def test_fastapi_index_endpoint_not_found():
    """Verify POST /api/v1/papers/{paper_id}/index returns 404 for unextracted paper."""
    client = TestClient(app)
    response = client.post("/api/v1/papers/unknown_ghost_paper/index")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


# ==============================================================================
# 6. Phase 5 Audit & Model Migration Tests
# ==============================================================================

def test_gemini_embedding_model_configuration():
    """Verify gemini-embedding-2 is configured as default embedding model with 768 dimensions."""
    from app.core.config import settings
    assert settings.EMBEDDING_MODEL == "gemini-embedding-2"
    assert settings.EMBEDDING_DIMENSION == 768


def test_production_environment_rejects_mock_embeddings():
    """Verify MockEmbeddingProvider is rejected in production unless explicitly allowed."""
    with patch("services.embedding_provider.settings.ENVIRONMENT", "production"):
        with patch("services.embedding_provider.settings.ALLOW_MOCK_EMBEDDINGS", False):
            with patch("services.embedding_provider.settings.EMBEDDING_PROVIDER", "mock"):
                with pytest.raises(ValueError, match="strictly prohibited in production"):
                    get_embedding_provider()

        # When explicitly allowed in production
        with patch("services.embedding_provider.settings.ALLOW_MOCK_EMBEDDINGS", True):
            with patch("services.embedding_provider.settings.EMBEDDING_PROVIDER", "mock"):
                provider = get_embedding_provider()
                assert isinstance(provider, MockEmbeddingProvider)
                assert provider.is_synthetic is True


@pytest.mark.anyio
async def test_atomic_reindexing_rollback_on_failure():
    """Verify that if chunk addition fails during re-indexing, previous chunks are restored intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)
        embedder = MockEmbeddingProvider(dimension=16)

        # Initial indexing: paper_1 with 2 chunks
        chunk1 = DocumentChunkSchema(
            paper_id="paper_1",
            chunk_id="paper_1_c0001",
            page_start=1,
            page_end=1,
            text="Initial chunk text content 1",
        )
        chunk2 = DocumentChunkSchema(
            paper_id="paper_1",
            chunk_id="paper_1_c0002",
            page_start=1,
            page_end=1,
            text="Initial chunk text content 2",
        )
        vecs = await embedder.embed_documents([chunk1.text, chunk2.text])
        store.add_chunks([chunk1, chunk2], vecs)

        assert store.count() == 2
        initial_records = store.get_paper_records("paper_1")
        assert len(initial_records["ids"]) == 2

        # Prepare new chunks that will fail during insertion (e.g. inject an error into collection.add)
        new_chunk = DocumentChunkSchema(
            paper_id="paper_1",
            chunk_id="paper_1_c9999",
            page_start=2,
            page_end=2,
            text="New replacement chunk text",
        )
        new_vecs = await embedder.embed_documents([new_chunk.text])

        # Patch collection.add so that the first call inside add_chunks fails
        original_add = store.collection.add
        call_count = 0

        def failing_add(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated disk I/O failure during chunk persist")
            return original_add(*args, **kwargs)

        with patch.object(store.collection, "add", side_effect=failing_add):
            with pytest.raises(Exception, match="Simulated disk I/O failure"):
                store.add_chunks([new_chunk], new_vecs)

        # Verify old records were restored and NOT destroyed
        restored_records = store.get_paper_records("paper_1")
        assert len(restored_records["ids"]) == 2
        assert set(restored_records["ids"]) == {"paper_1_c0001", "paper_1_c0002"}
        assert store.count() == 2


@pytest.mark.anyio
async def test_retrieval_synthetic_flag_propagation():
    """Verify is_synthetic flag is True for mock embeddings and present in result schemas."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=16)
        embedder = MockEmbeddingProvider(dimension=16)
        assert embedder.is_synthetic is True

        chunk = DocumentChunkSchema(
            paper_id="test_p",
            chunk_id="test_p_c1",
            page_start=1,
            page_end=1,
            text="Sample research text on Amharic linguistics",
        )
        vec = await embedder.embed_query(chunk.text)
        store.add_chunks([chunk], [vec])

        result = await retrieve_research_context(
            question="Amharic linguistics",
            vector_db=store,
            embedding_provider=embedder,
        )

        assert result.is_synthetic is True
        assert len(result.items) > 0
        assert result.items[0].is_synthetic is True


@pytest.mark.anyio
async def test_retrieval_benchmark_evaluation():
    """Verify reproducible retrieval quality evaluation benchmark executes and computes metrics."""
    from evaluation.retrieval_eval import create_eval_benchmark_papers, run_retrieval_benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ChromaVectorStore(persist_directory=tmpdir, dimension=32)
        embedder = MockEmbeddingProvider(dimension=32)
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)

        # Index benchmark papers
        benchmark_papers = create_eval_benchmark_papers()
        for p in benchmark_papers:
            chunks = chunker.chunk_paper(p.paper_id, p)
            if chunks:
                vecs = await embedder.embed_documents([c.text for c in chunks])
                store.add_chunks(chunks, vecs)

        assert store.count() > 0

        # Run benchmark
        summary = await run_retrieval_benchmark(vector_db=store, embedder=embedder, top_k=5)

        assert summary.total_queries == 6
        assert summary.is_synthetic is True
        assert 0.0 <= summary.recall_at_5 <= 1.0
        assert 0.0 <= summary.precision_at_5 <= 1.0
        assert 0.0 <= summary.correct_paper_rate <= 1.0
        assert 0.0 <= summary.correct_page_rate <= 1.0
        assert len(summary.details) == 6


# ==============================================================================
# 7. Gemini Embedding 2 Batch Semantics & Mocked Provider Tests
# ==============================================================================

@pytest.mark.anyio
async def test_gemini_embedding_provider_batch_16_independent_vectors():
    """Verify GeminiEmbeddingProvider sends 16 separate Content objects and receives 16 independent vectors."""
    from services.embedding_provider import GeminiEmbeddingProvider
    from google.genai import types

    provider = GeminiEmbeddingProvider(api_key="fake-key-for-test", dimension=768, batch_size=16)

    # 16 distinct input chunks
    inputs = [f"Research chunk number {i}: empirical findings on NLP." for i in range(16)]

    # Mock client and response
    mock_resp = types.EmbedContentResponse(
        embeddings=[
            types.ContentEmbedding(values=[float(i)] * 768)
            for i in range(16)
        ]
    )

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        embeddings = await provider.embed_documents(inputs)

        # 1. Assert exactly 16 embedding vectors returned
        assert len(embeddings) == 16

        # 2. Assert each vector matches configured dimension
        for vec in embeddings:
            assert len(vec) == 768

        # 3. Assert ordering and distinct association: vector i matches input i
        for i, vec in enumerate(embeddings):
            assert vec[0] == float(i)

        # 4. Verify request format: embed_content called with contents list of Content objects
        mock_client.models.embed_content.assert_called_once()
        _, kwargs = mock_client.models.embed_content.call_args

        assert kwargs["model"] == "gemini-embedding-2"
        contents_arg = kwargs["contents"]
        assert len(contents_arg) == 16
        for idx, c in enumerate(contents_arg):
            assert isinstance(c, types.Content)
            assert c.parts[0].text == inputs[idx]

        config_arg = kwargs["config"]
        assert config_arg.output_dimensionality == 768
        assert config_arg.task_type == "RETRIEVAL_DOCUMENT"


@pytest.mark.anyio
async def test_gemini_embedding_provider_single_input_and_query_semantics():
    """Verify single-input embedding works and embed_query applies RETRIEVAL_QUERY task_type."""
    from services.embedding_provider import GeminiEmbeddingProvider
    from google.genai import types

    provider = GeminiEmbeddingProvider(api_key="fake-key-for-test", dimension=768)

    mock_resp = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.5] * 768)]
    )

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = mock_resp

    with patch("google.genai.Client", return_value=mock_client):
        query_vec = await provider.embed_query("What is the BLEU score?")

        assert len(query_vec) == 768
        assert query_vec[0] == 0.5

        mock_client.models.embed_content.assert_called_once()
        _, kwargs = mock_client.models.embed_content.call_args

        assert len(kwargs["contents"]) == 1
        assert kwargs["contents"][0].parts[0].text == "What is the BLEU score?"
        assert kwargs["config"].task_type == "RETRIEVAL_QUERY"
        assert kwargs["config"].output_dimensionality == 768


@pytest.mark.anyio
async def test_gemini_embedding_provider_rejects_aggregated_or_mismatched_response():
    """Verify that if Gemini returns an aggregated (single) vector for multi-item input, it is rejected."""
    from services.embedding_provider import GeminiEmbeddingProvider
    from google.genai import types

    provider = GeminiEmbeddingProvider(api_key="fake-key-for-test", dimension=768, batch_size=16)

    # 4 distinct input texts
    inputs = ["Text 1", "Text 2", "Text 3", "Text 4"]

    # Erroneous aggregated response containing only 1 vector for 4 inputs
    aggregated_resp = types.EmbedContentResponse(
        embeddings=[types.ContentEmbedding(values=[0.1] * 768)]
    )

    mock_client = MagicMock()
    mock_client.models.embed_content.return_value = aggregated_resp

    with patch("google.genai.Client", return_value=mock_client):
        with pytest.raises(InvalidEmbeddingError, match="Expected exactly one independent vector per input"):
            await provider.embed_documents(inputs)


@pytest.mark.anyio
async def test_gemini_embedding_provider_surfaces_provider_errors():
    """Verify that provider rate limits and general failures raise expected exception types."""
    from services.embedding_provider import GeminiEmbeddingProvider

    provider = GeminiEmbeddingProvider(api_key="fake-key-for-test", dimension=768)

    # 1. Quota / Rate limit (429)
    mock_client_429 = MagicMock()
    mock_client_429.models.embed_content.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    with patch("google.genai.Client", return_value=mock_client_429):
        with pytest.raises(EmbeddingRateLimitError, match="rate limit exceeded"):
            await provider.embed_documents(["Test chunk"])

    # 2. General provider failure
    mock_client_500 = MagicMock()
    mock_client_500.models.embed_content.side_effect = RuntimeError("503 Service Unavailable: upstream timeout")

    with patch("google.genai.Client", return_value=mock_client_500):
        with pytest.raises(EmbeddingError, match="Gemini embedding failed"):
            await provider.embed_documents(["Test chunk"])


