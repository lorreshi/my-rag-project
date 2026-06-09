"""HybridSearch — orchestrates Dense + Sparse + Fusion with metadata filtering.

Flow:
    query
      -> QueryProcessor.process()            (keywords + filters + normalized)
      -> DenseRetriever.retrieve(normalized)  (semantic candidates)
         SparseRetriever.retrieve(keywords)   (keyword candidates)
      -> Fusion.fuse([dense, sparse])         (RRF unified ranking)
      -> _apply_metadata_filters()            (post-filter safety net)
      -> Top-K

Resilience: if either retrieval path raises, HybridSearch logs the failure and
degrades to the surviving path (single-route results) rather than failing the
whole query.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import ReciprocalRankFusion
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid (dense + sparse) retrieval orchestrator."""

    def __init__(
        self,
        query_processor: "QueryProcessor",
        dense_retriever: "DenseRetriever",
        sparse_retriever: "SparseRetriever",
        fusion: "ReciprocalRankFusion",
        settings: "Settings | None" = None,
        candidate_multiplier: int = 2,
    ):
        """Initialize HybridSearch.

        Args:
            query_processor: Produces keywords + filters from the raw query.
            dense_retriever: Semantic retriever.
            sparse_retriever: BM25 retriever.
            fusion: RRF fusion.
            settings: Optional settings (for default top_k values).
            candidate_multiplier: Each route fetches top_k * multiplier
                candidates before fusion, improving recall before the cut.
        """
        self._qp = query_processor
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._fusion = fusion
        self._settings = settings
        self._multiplier = max(1, candidate_multiplier)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Run hybrid search and return the Top-K fused results.

        Args:
            query: Raw user query.
            top_k: Number of final results.
            filters: Optional metadata filters.
            trace: Optional trace context.

        Returns:
            Top-K RetrievalResult after fusion + metadata filtering.
        """
        if trace:
            trace.start_stage("hybrid_search")

        processed = self._qp.process(query, filters=filters, trace=trace)
        candidate_k = top_k * self._multiplier

        dense_results = self._run_dense(processed, candidate_k, trace)
        sparse_results = self._run_sparse(processed, candidate_k, trace)

        # Fuse available routes; if one is empty, fusion still works.
        fused = self._fusion.fuse([dense_results, sparse_results], trace=trace)

        # Post-filter safety net (covers cases the stores didn't pre-filter).
        filtered = self._apply_metadata_filters(fused, processed.filters)

        final = filtered[:top_k]

        if trace:
            trace.end_stage(
                details={
                    "dense_count": len(dense_results),
                    "sparse_count": len(sparse_results),
                    "fused_count": len(fused),
                    "final_count": len(final),
                }
            )

        return final

    # ------------------------------------------------------------------
    # Route execution with degradation
    # ------------------------------------------------------------------

    def _run_dense(self, processed, candidate_k, trace) -> list[RetrievalResult]:
        try:
            return self._dense.retrieve(
                processed.normalized_query,
                top_k=candidate_k,
                filters=processed.filters or None,
                trace=trace,
            )
        except Exception as exc:
            logger.warning("Dense retrieval failed, degrading to sparse-only: %s", exc)
            return []

    def _run_sparse(self, processed, candidate_k, trace) -> list[RetrievalResult]:
        try:
            return self._sparse.retrieve(
                processed.keywords, top_k=candidate_k, trace=trace
            )
        except Exception as exc:
            logger.warning("Sparse retrieval failed, degrading to dense-only: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Metadata filtering (post-filter safety net)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_metadata_filters(
        candidates: list[RetrievalResult],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Apply equality metadata filters with missing->include policy.

        For each filter key, a candidate is kept if its metadata value equals
        the filter value. If the candidate's metadata is missing the key, it is
        included (lenient) to avoid wrongly dropping recall on incomplete data.
        """
        if not filters:
            return candidates

        def _keep(item: RetrievalResult) -> bool:
            for key, value in filters.items():
                if key not in item.metadata:
                    continue  # missing -> include
                if item.metadata[key] != value:
                    return False
            return True

        return [c for c in candidates if _keep(c)]

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides: Any) -> "HybridSearch":
        """Build a HybridSearch with real components from settings."""
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.fusion import ReciprocalRankFusion
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.sparse_retriever import SparseRetriever

        qp = overrides.get("query_processor") or QueryProcessor()
        dense = overrides.get("dense_retriever") or DenseRetriever(settings=settings)
        sparse = overrides.get("sparse_retriever") or SparseRetriever(settings=settings)
        fusion = overrides.get("fusion") or ReciprocalRankFusion(
            k=getattr(settings.retrieval, "rrf_k", 60)
            if hasattr(settings, "retrieval") else 60
        )
        return cls(qp, dense, sparse, fusion, settings=settings)
