"""DenseRetriever — semantic recall via embedding + vector store.

Combines an embedding provider (query vectorization) with a vector store
(similarity search) to produce semantic retrieval results.

Flow:
    query -> embedding.embed([query]) -> vector_store.query(vector, top_k, filters)
          -> normalize to List[RetrievalResult]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Embedding-based semantic retriever."""

    def __init__(
        self,
        settings: "Settings | None" = None,
        embedding_client: "BaseEmbedding | None" = None,
        vector_store: "BaseVectorStore | None" = None,
    ):
        """Initialize DenseRetriever.

        Components may be injected (tests) or built lazily from settings.

        Args:
            settings: Settings used to build components if not injected.
            embedding_client: Optional BaseEmbedding (built from settings if None).
            vector_store: Optional BaseVectorStore (built from settings if None).
        """
        self._embedding = embedding_client
        self._store = vector_store

        if self._embedding is None or self._store is None:
            if settings is None:
                raise ValueError(
                    "DenseRetriever requires either injected components or settings"
                )
            if self._embedding is None:
                from src.libs.embedding.embedding_factory import EmbeddingFactory
                self._embedding = EmbeddingFactory.create(settings)
            if self._store is None:
                from src.libs.vector_store.vector_store_factory import (
                    VectorStoreFactory,
                )
                self._store = VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Retrieve semantically relevant chunks for a query.

        Args:
            query: The (already normalized) query text.
            top_k: Maximum number of results.
            filters: Optional metadata filters passed to the vector store.
            trace: Optional trace context.

        Returns:
            List of RetrievalResult sorted by descending relevance.
        """
        if trace:
            trace.start_stage("dense_retrieval")

        if not query or not query.strip():
            if trace:
                trace.end_stage(details={"count": 0, "reason": "empty_query"})
            return []

        query_vectors = self._embedding.embed([query], trace=trace)
        if not query_vectors:
            logger.warning("Embedding returned no vector for query")
            if trace:
                trace.end_stage(details={"count": 0, "reason": "no_vector"})
            return []

        query_vector = query_vectors[0]
        hits = self._store.query(
            vector=query_vector, top_k=top_k, filters=filters, trace=trace
        )

        results = [
            RetrievalResult(
                chunk_id=h.id,
                score=h.score,
                text=h.text,
                metadata=h.metadata,
            )
            for h in hits
        ]

        if trace:
            trace.end_stage(
                details={
                    "count": len(results),
                    "method": "vector_search",
                    "provider": self._embedding.provider_name,
                }
            )

        return results
