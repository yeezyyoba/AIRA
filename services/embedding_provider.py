"""Embedding Provider Abstraction for AIRA Research Context Retrieval (RAG).

Provides a provider-independent embedding interface decoupled from the LLM reasoning router.
Validates dimensions and numerical values, enforces batching and timeouts,
and supports Gemini, deterministic Mock (for offline tests), and Null providers.
"""
from abc import ABC, abstractmethod
import asyncio
import hashlib
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import settings
from logging_config.logger import get_logger

logger = get_logger("embedding_provider")


class EmbeddingError(RuntimeError):
    """Base exception for embedding failures."""
    pass


class EmbeddingProviderUnavailableError(EmbeddingError):
    """Raised when the configured embedding provider is disabled or unavailable."""
    pass


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when the embedding provider rate limits the request."""
    pass


class EmbeddingTimeoutError(EmbeddingError):
    """Raised when embedding generation times out."""
    pass


class InvalidEmbeddingError(EmbeddingError, ValueError):
    """Raised when an embedding contains NaN, Inf, or invalid dimensions."""
    pass


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(
        self,
        model_name: str,
        dimension: int,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
        is_synthetic: bool = False,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self.timeout = timeout or settings.EMBEDDING_TIMEOUT_SECONDS
        self.is_synthetic = is_synthetic

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and available for requests."""
        pass

    @abstractmethod
    async def _generate_embeddings(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
    ) -> List[List[float]]:
        """Provider-specific implementation for generating embeddings for a batch of texts."""
        pass

    def validate_embedding(self, vec: List[float], label: str = "") -> None:
        """Validate embedding dimensionality and numerical values."""
        if not isinstance(vec, (list, np.ndarray)):
            raise InvalidEmbeddingError(f"Embedding must be a list or array, got {type(vec)}")

        if len(vec) != self.dimension:
            raise InvalidEmbeddingError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {len(vec)} for '{label[:30]}'"
            )

        for v in vec:
            if math.isnan(v) or math.isinf(v):
                raise InvalidEmbeddingError(
                    f"Embedding contains invalid numerical value ({v}) for '{label[:30]}'"
                )

    async def embed_documents(
        self,
        texts: List[str],
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """Generate and validate embeddings for a list of document texts in batches.
        
        Guarantees that each input text receives an independent, unaggregated embedding vector
        and preserves input sequence ordering.
        """
        if not self.is_available():
            raise EmbeddingProviderUnavailableError(
                f"Embedding provider '{self.model_name}' is currently unavailable or unconfigured."
            )

        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                try:
                    gen_coro = self._generate_embeddings(batch, task_type=task_type)
                except TypeError:
                    gen_coro = self._generate_embeddings(batch)

                batch_embeddings = await asyncio.wait_for(
                    gen_coro,
                    timeout=self.timeout,
                )
            except asyncio.TimeoutError as e:
                logger.error(f"Embedding generation timed out after {self.timeout}s for batch of {len(batch)} items")
                raise EmbeddingTimeoutError(f"Embedding generation timed out after {self.timeout} seconds") from e
            except EmbeddingError:
                raise
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise EmbeddingError(f"Embedding provider '{self.model_name}' failed: {e}") from e

            if len(batch_embeddings) != len(batch):
                raise InvalidEmbeddingError(
                    f"Embedding provider returned {len(batch_embeddings)} embeddings for {len(batch)} input texts"
                )

            for idx, vec in enumerate(batch_embeddings):
                self.validate_embedding(vec, label=batch[idx])
                all_embeddings.append(vec)

        return all_embeddings

    async def embed_query(self, text: str) -> List[float]:
        """Generate and validate an embedding for a search query string using query-specific semantics."""
        if not text or not text.strip():
            raise ValueError("Query text for embedding must be a non-empty string")

        results = await self.embed_documents([text.strip()], task_type="RETRIEVAL_QUERY")
        return results[0]


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, offline embedding provider for development and testing.

    Derives normalized unit vectors from SHA-256 hashes of text content.
    Never makes network calls or consumes LLM API quota.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        model = model_name or settings.EMBEDDING_MODEL
        dim = dimension or settings.EMBEDDING_DIMENSION
        super().__init__(
            model_name=model,
            dimension=dim,
            batch_size=batch_size,
            timeout=timeout,
            is_synthetic=True,
        )

    def is_available(self) -> bool:
        return True

    async def _generate_embeddings(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for t in texts:
            # Deterministic SHA-256 pseudo-random generation
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dimension).astype(float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider using Google GenAI SDK.
    
    Uses gemini-embedding-2 with explicitly configured output_dimensionality,
    guaranteeing distinct, unaggregated embeddings for individual batched inputs
    via separate Content objects and explicit task types (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or settings.GEMINI_API_KEY
        model = model_name or settings.EMBEDDING_MODEL
        dim = dimension or settings.EMBEDDING_DIMENSION
        super().__init__(
            model_name=model,
            dimension=dim,
            batch_size=batch_size,
            timeout=timeout,
            is_synthetic=False,
        )

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    async def _generate_embeddings(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
    ) -> List[List[float]]:
        if not self.is_available():
            raise EmbeddingProviderUnavailableError("GEMINI_API_KEY is not configured")

        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=self.api_key)

            # Build config with output_dimensionality and task_type
            config_kwargs: Dict[str, Any] = {}
            if self.dimension:
                config_kwargs["output_dimensionality"] = self.dimension
            if task_type:
                config_kwargs["task_type"] = task_type

            try:
                config = types.EmbedContentConfig(**config_kwargs)
            except Exception:
                config = config_kwargs

            # Wrap each text item as an independent Content object with its own Part
            # This guarantees that multiple inputs are never merged or aggregated
            contents = [
                types.Content(parts=[types.Part.from_text(text=t)])
                for t in texts
            ]

            resp = client.models.embed_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )

            embeddings: List[List[float]] = []
            if hasattr(resp, "embeddings") and resp.embeddings:
                for item in resp.embeddings:
                    embeddings.append(list(item.values))
            elif hasattr(resp, "embedding") and resp.embedding:
                embeddings.append(list(resp.embedding.values))
            else:
                raise InvalidEmbeddingError("Gemini embedding response did not contain embedding vectors")

            if len(embeddings) != len(texts):
                raise InvalidEmbeddingError(
                    f"Gemini API returned {len(embeddings)} embeddings for {len(texts)} input texts. "
                    "Expected exactly one independent vector per input."
                )

            return embeddings
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg:
                raise EmbeddingRateLimitError(f"Gemini embedding rate limit exceeded: {e}") from e
            if isinstance(e, InvalidEmbeddingError):
                raise
            raise EmbeddingError(f"Gemini embedding failed: {e}") from e


class NullEmbeddingProvider(BaseEmbeddingProvider):
    """Null embedding provider representing unconfigured or disabled semantic indexing."""

    def __init__(self) -> None:
        super().__init__(model_name="none", dimension=0, is_synthetic=False)

    def is_available(self) -> bool:
        return False

    async def _generate_embeddings(
        self,
        texts: List[str],
        task_type: Optional[str] = None,
    ) -> List[List[float]]:
        raise EmbeddingProviderUnavailableError(
            "Semantic indexing is unavailable because no embedding provider is configured."
        )


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory to initialize the configured embedding provider based on application settings.
    
    Guards production environments against synthetic mock embeddings unless explicitly allowed.
    """
    provider_name = (settings.EMBEDDING_PROVIDER or "mock").lower().strip()
    is_production = getattr(settings, "ENVIRONMENT", "development").lower() == "production"

    if provider_name == "mock":
        if is_production and not getattr(settings, "ALLOW_MOCK_EMBEDDINGS", False):
            raise ValueError(
                "Mock embedding provider is strictly prohibited in production environments. "
                "Configure a valid live embedding provider (e.g. 'gemini') or set ALLOW_MOCK_EMBEDDINGS=true."
            )
        return MockEmbeddingProvider()

    if provider_name == "gemini":
        if not settings.GEMINI_API_KEY:
            logger.warning("EMBEDDING_PROVIDER is 'gemini' but GEMINI_API_KEY is missing. Using NullEmbeddingProvider.")
            return NullEmbeddingProvider()
        return GeminiEmbeddingProvider()
    elif provider_name in ("none", "null", "disabled"):
        return NullEmbeddingProvider()
    else:
        logger.warning(f"Unrecognized EMBEDDING_PROVIDER '{provider_name}'. Defaulting to MockEmbeddingProvider.")
        if is_production and not getattr(settings, "ALLOW_MOCK_EMBEDDINGS", False):
            raise ValueError(
                f"Unrecognized EMBEDDING_PROVIDER '{provider_name}' in production environment."
            )
        return MockEmbeddingProvider()
