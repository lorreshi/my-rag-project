"""SparseRetriever — BM25 keyword retrieval.

Loads a BM25 index from disk (data/db/bm25/) and queries it with keywords,
then resolves the matched chunk_ids back to text + metadata via the vector
store's get_by_ids().

Flow:
    keywords -> bm25.query(keywords, top_k) -> [(chunk_id, score)]
             -> vector_store.get_by_ids(chunk_ids) -> {id: {text, metadata}}
             -> merge -> List[RetrievalResult]
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.ingestion.storage.bm25_indexer import BM25Indexer
    from src.libs.vector_store.base_vector_store import BaseVectorStore

logger = logging.getLogger(__name__)


class SparseRetriever:
    """BM25-based keyword retriever."""

    def __init__(
        self,
        settings: "Settings | None" = None,
        bm25_indexer: "BM25Indexer | None" = None,
        vector_store: "BaseVectorStore | None" = None,
        index_dir: str = "data/db/bm25",
        auto_load: bool = True,
    ):
        """Initialize SparseRetriever.

        Args:
            settings: Settings used to build the vector store if not injected.
            bm25_indexer: Optional BM25Indexer (built + loaded from disk if None).
            vector_store: Optional BaseVectorStore (built from settings if None).
            index_dir: BM25 index directory (used when building the indexer).
            auto_load: If True and a fresh indexer is built, load it from disk.
        """
        self._bm25 = bm25_indexer
        self._store = vector_store

        if self._bm25 is None:
            from src.ingestion.storage.bm25_indexer import BM25Indexer
            self._bm25 = BM25Indexer(index_dir=index_dir)
            if auto_load:
                try:
                    self._bm25.load()
                except FileNotFoundError:
                    logger.warning(
                        "BM25 index not found in %s; sparse retrieval will be empty",
                        index_dir,
                    )

        if self._store is None:
            if settings is None:
                raise ValueError(
                    "SparseRetriever requires an injected vector_store or settings"
                )
            from src.libs.vector_store.vector_store_factory import VectorStoreFactory
            self._store = VectorStoreFactory.create(settings)

    def retrieve(
        self,
        keywords: list[str],
        top_k: int = 20,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Retrieve chunks by BM25 keyword scoring.

        Args:
            keywords: Tokenized query keywords (from QueryProcessor).
            top_k: Maximum number of results.
            trace: Optional trace context.

        Returns:
            List of RetrievalResult with BM25 scores, sorted by descending score.
        """
        if trace:
            trace.start_stage("sparse_retrieval")

        if not keywords:
            if trace:
                trace.end_stage(details={"count": 0, "reason": "no_keywords"})
            return []

        scored = self._bm25.query(keywords, top_k=top_k)  # [(chunk_id, score)]
        if not scored:
            if trace:
                trace.end_stage(details={"count": 0})
            return []

        chunk_ids = [cid for cid, _ in scored]
        records = self._store.get_by_ids(chunk_ids)
        by_id = {r["id"]: r for r in records}

        results: list[RetrievalResult] = []
        for chunk_id, score in scored:
            rec = by_id.get(chunk_id)
            if rec is None:
                # Index references a chunk no longer in the store; skip safely.
                logger.warning(
                    "BM25 hit %s missing from vector store; skipping", chunk_id
                )
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score,
                    text=rec.get("text", ""),
                    metadata=rec.get("metadata", {}),
                )
            )

        if trace:
            trace.end_stage(details={"count": len(results), "method": "bm25"})

        return results
