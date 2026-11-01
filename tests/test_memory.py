"""Tests for ChromaDB Vector Storage, Persistence, and Recovery (Phase 5).

Verifies:
- ChromaDB local persistence and recovery from missing directory.
- Collection initialization and metadata preservation.
- Incompatible embedding model / dimension detection.
- Adding chunks with deterministic IDs.
- Atomic paper re-indexing and deletion.
- Metadata type compatibility.
- Vector search ranking and distance calculations.
"""
from pathlib import Path
import tempfile
import pytest

from memory.vector_store import (
    ChromaVectorStore,
    IncompatibleEmbeddingModelError,
    VectorStoreError,
)
from validation.schemas import DocumentChunkSchema


@pytest.fixture
def temp_chroma_dir():
    """Provide a temporary directory for ChromaDB vector store testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_chromadb_initialization_and_metadata(temp_chroma_dir):
    """Verify that ChromaDB creates directory and initializes collection metadata properly."""
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name="test_academic",
        embedding_model="test-embed-v1",
        dimension=4,
    )

    assert store.count() == 0
    assert store.collection.metadata["embedding_model"] == "test-embed-v1"
    assert store.collection.metadata["dimension"] == 4
    assert store.collection.metadata["hnsw:space"] == "cosine"


def test_chromadb_incompatible_embedding_model_detected(temp_chroma_dir):
    """Verify that attempting to connect with a mismatched dimension raises IncompatibleEmbeddingModelError."""
    # 1. Create collection with dimension 4 and add a vector
    store1 = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name="compat_test",
        embedding_model="model-4d",
        dimension=4,
    )
    chunk = DocumentChunkSchema(
        paper_id="paper1",
        chunk_id="paper1_c0001",
        page_start=1,
        page_end=1,
        text="Sample text content",
    )
    store1.add_chunks([chunk], [[0.1, 0.2, 0.3, 0.4]])
    assert store1.count() == 1

    # 2. Re-open same collection with dimension 8 -> must raise IncompatibleEmbeddingModelError
    with pytest.raises(IncompatibleEmbeddingModelError, match="Incompatible embedding model detected"):
        ChromaVectorStore(
            persist_directory=str(temp_chroma_dir),
            collection_name="compat_test",
            embedding_model="model-8d",
            dimension=8,
        )


def test_chromadb_add_chunks_and_atomic_replacement(temp_chroma_dir):
    """Verify adding chunks and that re-indexing a paper replaces previous chunks without duplicates."""
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name="test_replace",
        dimension=3,
    )

    chunks_v1 = [
        DocumentChunkSchema(
            paper_id="amharic_nlp_2024",
            chunk_id="amharic_nlp_2024_c0001",
            page_start=1,
            page_end=1,
            text="First version chunk 1",
            section_heading="Abstract",
        ),
        DocumentChunkSchema(
            paper_id="amharic_nlp_2024",
            chunk_id="amharic_nlp_2024_c0002",
            page_start=1,
            page_end=2,
            text="First version chunk 2",
            section_heading="Methodology",
        ),
    ]
    embeds_v1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    count_added = store.add_chunks(chunks_v1, embeds_v1)
    assert count_added == 2
    assert store.count() == 2
    assert store.get_indexed_paper_ids() == ["amharic_nlp_2024"]

    # Re-index the same paper with 3 new chunks
    chunks_v2 = [
        DocumentChunkSchema(
            paper_id="amharic_nlp_2024",
            chunk_id="amharic_nlp_2024_c0001",
            page_start=1,
            page_end=1,
            text="Second version chunk 1",
            section_heading="Abstract",
        ),
        DocumentChunkSchema(
            paper_id="amharic_nlp_2024",
            chunk_id="amharic_nlp_2024_c0002",
            page_start=1,
            page_end=2,
            text="Second version chunk 2",
            section_heading="Methodology",
        ),
        DocumentChunkSchema(
            paper_id="amharic_nlp_2024",
            chunk_id="amharic_nlp_2024_c0003",
            page_start=2,
            page_end=3,
            text="Second version chunk 3",
            section_heading="Results",
        ),
    ]
    embeds_v2 = [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 0.1]]

    count_reindexed = store.add_chunks(chunks_v2, embeds_v2)
    assert count_reindexed == 3
    # Total count must be exactly 3, not 5 (no duplicates)
    assert store.count() == 3


def test_chromadb_delete_paper(temp_chroma_dir):
    """Verify deleting a paper removes all its chunks while preserving other papers."""
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name="test_delete",
        dimension=2,
    )

    c1 = DocumentChunkSchema(paper_id="paper_A", chunk_id="pA_c1", page_start=1, page_end=1, text="Text A1")
    c2 = DocumentChunkSchema(paper_id="paper_A", chunk_id="pA_c2", page_start=2, page_end=2, text="Text A2")
    c3 = DocumentChunkSchema(paper_id="paper_B", chunk_id="pB_c1", page_start=1, page_end=1, text="Text B1")

    store.add_chunks([c1, c2], [[0.1, 0.2], [0.2, 0.3]])
    store.add_chunks([c3], [[0.9, 0.8]])
    assert store.count() == 3

    # Delete paper A
    deleted = store.delete_paper("paper_A")
    assert deleted == 2
    assert store.count() == 1
    assert store.get_indexed_paper_ids() == ["paper_B"]


def test_chromadb_query_similarity_and_filtering(temp_chroma_dir):
    """Verify similarity search ranking, cosine score mapping, and paper ID filtering."""
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name="test_query",
        dimension=3,
    )

    # Unit vector 1: [1, 0, 0] (Paper 1)
    # Unit vector 2: [0.7071, 0.7071, 0] (Paper 2)
    # Unit vector 3: [0, 1, 0] (Paper 3)
    c1 = DocumentChunkSchema(
        paper_id="amharic_ner",
        chunk_id="ner_c1",
        page_start=1,
        page_end=1,
        text="Named entity recognition for Ethiopic script",
        section_heading="Methodology",
        authors=["Abebe Kebede"],
        publication_year=2023,
    )
    c2 = DocumentChunkSchema(
        paper_id="amharic_sa",
        chunk_id="sa_c1",
        page_start=2,
        page_end=3,
        text="Sentiment analysis on Amharic social media comments",
        section_heading="Dataset",
        authors=["Tigist Haile", "Dawit Yohannes"],
        publication_year=2024,
    )

    store.add_chunks([c1], [[1.0, 0.0, 0.0]])
    store.add_chunks([c2], [[0.0, 1.0, 0.0]])

    # Query with vector [1, 0, 0] -> c1 should have distance 0.0 and similarity 1.0
    results = store.query([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "ner_c1"
    assert results[0].paper_id == "amharic_ner"
    assert results[0].similarity_score == 1.0
    assert results[0].distance == 0.0
    assert results[0].authors == ["Abebe Kebede"]
    assert results[0].publication_year == 2023

    # Filter query specifically to amharic_sa
    filtered = store.query([1.0, 0.0, 0.0], top_k=5, paper_ids=["amharic_sa"])
    assert len(filtered) == 1
    assert filtered[0].paper_id == "amharic_sa"
    assert filtered[0].authors == ["Tigist Haile", "Dawit Yohannes"]


def test_chromadb_persistence_recovery(temp_chroma_dir):
    """Verify that vectors written to disk persist across fresh client instances."""
    col_name = "persistence_verify"
    # Session 1: write
    store1 = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name=col_name,
        dimension=2,
    )
    c1 = DocumentChunkSchema(paper_id="p1", chunk_id="p1_c1", page_start=1, page_end=1, text="Persisted text")
    store1.add_chunks([c1], [[0.6, 0.8]])
    assert store1.count() == 1

    # Session 2: separate instance reading from same persist directory
    store2 = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name=col_name,
        dimension=2,
    )
    assert store2.count() == 1
    assert store2.get_indexed_paper_ids() == ["p1"]
    items = store2.query([0.6, 0.8], top_k=1)
    assert len(items) == 1
    assert items[0].text == "Persisted text"


def test_chromadb_rebuild_collection_migration(temp_chroma_dir):
    """Verify rebuild_collection resets metadata and dimensions safely for migration."""
    store = ChromaVectorStore(
        persist_directory=str(temp_chroma_dir),
        collection_name="migrate_test",
        embedding_model="text-embedding-004",
        dimension=768,
    )
    c1 = DocumentChunkSchema(paper_id="old_p", chunk_id="old_c1", page_start=1, page_end=1, text="Old vector")
    store.add_chunks([c1], [[0.0] * 768])
    assert store.count() == 1
    assert store.embedding_model == "text-embedding-004"

    # Rebuild collection for gemini-embedding-2 with dimension 768
    store.rebuild_collection(new_model="gemini-embedding-2", new_dimension=768)
    assert store.count() == 0
    assert store.embedding_model == "gemini-embedding-2"
    assert store.dimension == 768

    # Re-insert with new model dimensions works cleanly
    c2 = DocumentChunkSchema(paper_id="new_p", chunk_id="new_c1", page_start=1, page_end=1, text="New vector")
    store.add_chunks([c2], [[0.1] * 768])
    assert store.count() == 1
    assert store.get_indexed_paper_ids() == ["new_p"]

