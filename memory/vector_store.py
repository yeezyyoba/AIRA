"""ChromaDB Vector Store Service for AIRA Research Context Retrieval (RAG).

Manages local persistence, collection initialization, incompatible model detection,
atomic paper indexing/deletion, and semantic similarity search.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import chromadb
from chromadb.api.models.Collection import Collection

from app.core.config import settings
from logging_config.logger import get_logger
from validation.schemas import DocumentChunkSchema, EvidenceType, RetrievedContextItemSchema

logger = get_logger("vector_store")


class VectorStoreError(RuntimeError):
    """Base exception for vector store operations."""
    pass


class IncompatibleEmbeddingModelError(VectorStoreError):
    """Raised when existing collection vectors were generated with an incompatible model/dimension."""
    pass


class ChromaVectorStore:
    """Manages persistent ChromaDB vector storage and semantic retrieval for research papers."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "academic_papers",
        embedding_model: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> None:
        self.persist_directory = Path(persist_directory or settings.CHROMA_PERSIST_DIRECTORY)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.dimension = dimension or settings.EMBEDDING_DIMENSION

        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[Collection] = None
        self._init_collection()

    def _init_collection(self) -> None:
        """Initialize ChromaDB client and get or create the collection with model compatibility checks."""
        try:
            self._client = chromadb.PersistentClient(path=str(self.persist_directory))

            # Check if collection already exists
            existing_collections = [c.name for c in self._client.list_collections()]

            if self.collection_name in existing_collections:
                col = self._client.get_collection(
                    name=self.collection_name,
                    embedding_function=None,
                )
                meta = col.metadata or {}
                stored_dim = meta.get("dimension")
                stored_model = meta.get("embedding_model")

                # Detect incompatible model dimensions if collection already contains data
                if stored_dim is not None and stored_dim != self.dimension and col.count() > 0:
                    err_msg = (
                        f"Incompatible embedding model detected: collection '{self.collection_name}' has "
                        f"{col.count()} vectors with dimension {stored_dim} ({stored_model}), "
                        f"but current configuration requires dimension {self.dimension} ({self.embedding_model})."
                    )
                    logger.error(err_msg)
                    raise IncompatibleEmbeddingModelError(err_msg)

                self._collection = col
            else:
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,
                    metadata={
                        "embedding_model": self.embedding_model,
                        "dimension": self.dimension,
                        "hnsw:space": "cosine",
                    },
                )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' initialized at '{self.persist_directory}' "
                f"(items: {self._collection.count()}, model: {self.embedding_model}, dim: {self.dimension})"
            )
        except IncompatibleEmbeddingModelError:
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            raise VectorStoreError(f"Vector store initialization failed: {e}") from e

    @property
    def collection(self) -> Collection:
        """Return the active collection, reinitializing if necessary."""
        if self._collection is None:
            self._init_collection()
        return self._collection

    def _sanitize_metadata(self, chunk: DocumentChunkSchema) -> Dict[str, Union[str, int, float, bool]]:
        """Convert chunk metadata to primitive types supported by ChromaDB."""
        meta: Dict[str, Union[str, int, float, bool]] = {
            "paper_id": chunk.paper_id,
            "chunk_id": chunk.chunk_id,
            "page_start": int(chunk.page_start),
            "page_end": int(chunk.page_end),
            "section_heading": chunk.section_heading or "unspecified",
            "source_type": chunk.source_type or "academic_paper",
            "document_title": chunk.document_title or "Untitled Document",
            "authors": ", ".join(chunk.authors) if chunk.authors else "Unknown",
        }
        if chunk.publication_year is not None:
            meta["publication_year"] = int(chunk.publication_year)
        return meta

    def get_paper_records(self, paper_id: str) -> Dict[str, Any]:
        """Fetch all raw records (ids, documents, embeddings, metadatas) for a paper.
        
        Used for creating snapshots prior to re-indexing to enable atomic rollbacks.
        """
        if not paper_id:
            return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

        try:
            res = self.collection.get(
                where={"paper_id": paper_id},
                include=["documents", "embeddings", "metadatas"],
            )

            raw_ids = res.get("ids")
            ids = list(raw_ids) if raw_ids is not None else []

            raw_docs = res.get("documents")
            documents = list(raw_docs) if raw_docs is not None else []

            raw_embeddings = res.get("embeddings")
            embeddings = [list(e) for e in raw_embeddings] if raw_embeddings is not None and len(raw_embeddings) > 0 else []

            raw_meta = res.get("metadatas")
            metadatas = list(raw_meta) if raw_meta is not None else []

            return {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            }
        except Exception as e:
            logger.error(f"Failed to fetch paper records for '{paper_id}': {e}")
            return {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

    def add_chunks(
        self,
        chunks: List[DocumentChunkSchema],
        embeddings: List[List[float]],
    ) -> int:
        """Add chunks and embeddings to the collection with genuine atomic snapshot rollback.
        
        If inserting new chunks fails, any previous chunks for the paper are restored.
        """
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})")

        paper_id = chunks[0].paper_id

        # 1. Snapshot existing records for this paper
        old_records = self.get_paper_records(paper_id)
        has_existing = len(old_records["ids"]) > 0

        # 2. Prepare new records
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Union[str, int, float, bool]]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(self._sanitize_metadata(chunk))

        # 3. Perform atomic swap with rollback guarantee
        try:
            if has_existing:
                self.delete_paper(paper_id)

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Successfully indexed {len(chunks)} chunks for paper '{paper_id}'")
            return len(chunks)
        except Exception as e:
            logger.error(f"Failed to add chunks to ChromaDB for paper '{paper_id}': {e}. Triggering rollback.")
            # Rollback: Clean up any partial additions and restore old snapshot
            try:
                self.delete_paper(paper_id)
            except Exception:
                pass

            if has_existing:
                try:
                    restore_kwargs = {
                        "ids": old_records["ids"],
                        "documents": old_records["documents"],
                        "metadatas": old_records["metadatas"],
                    }
                    if old_records["embeddings"] and len(old_records["embeddings"]) == len(old_records["ids"]):
                        restore_kwargs["embeddings"] = old_records["embeddings"]
                    self.collection.add(**restore_kwargs)
                    logger.info(f"Successfully rolled back and restored {len(old_records['ids'])} previous chunks for paper '{paper_id}'")
                except Exception as restore_err:
                    logger.critical(f"Catastrophic failure: Failed to restore backup chunks for paper '{paper_id}': {restore_err}")

            raise VectorStoreError(f"Failed to persist chunks in vector store: {e}") from e

    def delete_paper(self, paper_id: str) -> int:
        """Delete all chunks belonging to a paper from ChromaDB."""
        if not paper_id:
            return 0

        try:
            initial_count = self.collection.count()
            self.collection.delete(where={"paper_id": paper_id})
            deleted = initial_count - self.collection.count()
            if deleted > 0:
                logger.info(f"Deleted {deleted} chunks for paper '{paper_id}' from vector store")
            return deleted
        except Exception as e:
            logger.error(f"Failed to delete paper '{paper_id}' from ChromaDB: {e}")
            raise VectorStoreError(f"Failed to delete paper '{paper_id}' from vector store: {e}") from e

    def rebuild_collection(
        self,
        new_model: str,
        new_dimension: int,
    ) -> None:
        """Recreate the ChromaDB collection for a new embedding model/dimension safely.
        
        Deletes the old incompatible collection and creates an empty one with updated metadata.
        """
        try:
            existing = [c.name for c in self._client.list_collections()]
            if self.collection_name in existing:
                self._client.delete_collection(name=self.collection_name)

            self.embedding_model = new_model
            self.dimension = new_dimension
            self._collection = self._client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={
                    "embedding_model": self.embedding_model,
                    "dimension": self.dimension,
                    "hnsw:space": "cosine",
                },
            )
            logger.info(f"Rebuilt collection '{self.collection_name}' with model '{new_model}' (dim {new_dimension})")
        except Exception as e:
            logger.error(f"Failed to rebuild collection '{self.collection_name}': {e}")
            raise VectorStoreError(f"Collection rebuild failed: {e}") from e

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        paper_ids: Optional[List[str]] = None,
    ) -> List[RetrievedContextItemSchema]:
        """Query vector database for chunks similar to the query embedding."""
        total_docs = self.collection.count()
        if total_docs == 0:
            logger.info("ChromaDB collection is empty. Returning 0 results.")
            return []

        n_results = max(1, min(top_k, total_docs))

        # Build filter clause if paper_ids specified
        where: Optional[Dict[str, Any]] = None
        if paper_ids:
            clean_ids = [pid.strip() for pid in paper_ids if pid and pid.strip()]
            if len(clean_ids) == 1:
                where = {"paper_id": clean_ids[0]}
            elif len(clean_ids) > 1:
                where = {"paper_id": {"$in": clean_ids}}

        try:
            query_kwargs: Dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n_results,
            }
            if where is not None:
                query_kwargs["where"] = where

            res = self.collection.query(**query_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise VectorStoreError(f"Vector search query failed: {e}") from e

        items: List[RetrievedContextItemSchema] = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return []

        ids = res["ids"][0]
        documents = res["documents"][0] if res.get("documents") else []
        metadatas = res["metadatas"][0] if res.get("metadatas") else []
        distances = res["distances"][0] if res.get("distances") else []

        for idx, chunk_id in enumerate(ids):
            doc_text = documents[idx] if idx < len(documents) else ""
            meta = metadatas[idx] if idx < len(metadatas) else {}
            dist = float(distances[idx]) if idx < len(distances) else 1.0

            # Cosine distance to similarity score in [0.0, 1.0]
            similarity = max(0.0, min(1.0, 1.0 - dist))

            authors_str = str(meta.get("authors", ""))
            authors_list = [a.strip() for a in authors_str.split(",") if a.strip() and a != "Unknown"]

            item = RetrievedContextItemSchema(
                chunk_id=chunk_id,
                paper_id=str(meta.get("paper_id", "")),
                text=doc_text,
                document_title=str(meta.get("document_title")) if meta.get("document_title") != "Untitled Document" else None,
                section_heading=str(meta.get("section_heading")) if meta.get("section_heading") != "unspecified" else None,
                page_start=int(meta.get("page_start", 1)),
                page_end=int(meta.get("page_end", 1)),
                similarity_score=round(similarity, 4),
                distance=round(dist, 4),
                authors=authors_list,
                publication_year=int(meta["publication_year"]) if "publication_year" in meta else None,
                evidence_type=EvidenceType.DIRECT_EXTRACT,
            )
            items.append(item)

        return items

    def count(self) -> int:
        """Return the total number of chunks currently stored in the collection."""
        return self.collection.count()

    def get_indexed_paper_ids(self) -> List[str]:
        """Return a sorted list of unique paper IDs currently indexed in ChromaDB."""
        if self.count() == 0:
            return []

        try:
            res = self.collection.get()
            paper_ids: Set[str] = set()
            for meta in (res.get("metadatas") or []):
                if meta and "paper_id" in meta:
                    paper_ids.add(str(meta["paper_id"]))
            return sorted(list(paper_ids))
        except Exception as e:
            logger.error(f"Failed to fetch indexed paper IDs: {e}")
            return []

    def reset(self) -> None:
        """Delete all records in the collection."""
        try:
            self._client.delete_collection(name=self.collection_name)
            self._collection = None
            self._init_collection()
            logger.info(f"Reset ChromaDB collection '{self.collection_name}' successfully.")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB collection: {e}")
            raise VectorStoreError(f"Failed to reset vector store: {e}") from e


# Global default vector store instance
vector_store = ChromaVectorStore()
