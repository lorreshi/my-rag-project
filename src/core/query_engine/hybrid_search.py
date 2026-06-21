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
from src.core.query_engine.metadata_filter import (
    STRUCTURED_FILTER_KEYS,
    match_filters,
)

if TYPE_CHECKING:
    from src.core.query_engine.dense_retriever import DenseRetriever
    from src.core.query_engine.fusion import BaseFusion
    from src.core.query_engine.query_processor import QueryProcessor
    from src.core.query_engine.sparse_retriever import SparseRetriever
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)


class HybridSearch:
    """Hybrid (dense + sparse) retrieval orchestrator."""

    # Structured metadata fields produced at split time (e.g. by
    # TableAwareSplitter); STRICT missing policy. Kept as a class alias of the
    # shared constant so existing references continue to work.
    _STRUCTURED_FILTER_KEYS = STRUCTURED_FILTER_KEYS

    def __init__(
        self,
        query_processor: "QueryProcessor",
        dense_retriever: "DenseRetriever",
        sparse_retriever: "SparseRetriever",
        fusion: "BaseFusion",
        settings: "Settings | None" = None,
        candidate_multiplier: int = 2,
        top_k_dense: int = 20,
        top_k_sparse: int = 20,
        sparse_filter_overfetch: int = 4,
        enable_synonym_expansion: bool = False,
    ):
        """Initialize HybridSearch.

        Args:
            query_processor: Produces keywords + filters from the raw query.
            dense_retriever: Semantic retriever.
            sparse_retriever: BM25 retriever.
            fusion: Fusion strategy (RRF / weighted_sum via FusionFactory).
            settings: Optional settings (for default top_k values).
            candidate_multiplier: Each route fetches its configured candidate
                width * multiplier before fusion, improving recall before the cut.
            top_k_dense: Base dense candidate width (from settings.retrieval).
            top_k_sparse: Base sparse candidate width (from settings.retrieval).
        """
        self._qp = query_processor
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._fusion = fusion
        self._settings = settings
        self._multiplier = max(1, candidate_multiplier)
        self._top_k_dense = max(1, top_k_dense)
        self._top_k_sparse = max(1, top_k_sparse)
        self._sparse_overfetch = max(1, sparse_filter_overfetch)
        self._synonym_expansion = enable_synonym_expansion

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
        # Config-driven candidate pool widths (no longer top_k * hardcoded 2).
        dense_k = max(top_k, self._top_k_dense) * self._multiplier
        sparse_k = max(top_k, self._top_k_sparse) * self._multiplier

        dense_results = self._run_dense(processed, dense_k, trace)
        sparse_results = self._run_sparse(processed, sparse_k, trace)

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
        keywords = (
            processed.expanded_keywords
            if self._synonym_expansion
            else processed.keywords
        )
        try:
            return self._sparse.retrieve(
                keywords,
                top_k=candidate_k,
                filters=processed.filters or None,
                overfetch=self._sparse_overfetch,
                trace=trace,
            )
        except Exception as exc:
            logger.warning("Sparse retrieval failed, degrading to dense-only: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Metadata filtering (post-filter safety net)
    # ------------------------------------------------------------------

    @classmethod
    def _apply_metadata_filters(
        cls,
        candidates: list[RetrievalResult],
        filters: dict[str, Any],
    ) -> list[RetrievalResult]:
        """Apply equality metadata filters with a per-key missing policy.

        For each filter key a candidate is kept only if its metadata value
        equals the filter value (present-but-different is always excluded).

        The missing-key policy depends on the key:
        - Structured fields (``_STRUCTURED_FILTER_KEYS``, e.g. ``sheet_name``):
          STRICT — a candidate missing the key is excluded. This makes
          structured filters (filter by ``sheet_name``) return only matching
          chunks (Requirement 7.1).
        - Generic fields (doc_type/collection/...): LENIENT — a candidate
          missing the key is included, avoiding wrongly dropping recall on
          incomplete data.
        """
        if not filters:
            return candidates

        def _keep(item: RetrievalResult) -> bool:
            return match_filters(item.metadata, filters, cls._STRUCTURED_FILTER_KEYS)

        return [c for c in candidates if _keep(c)]

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides: Any) -> "HybridSearch":
        """Build a HybridSearch with real components from settings."""
        from src.core.query_engine.dense_retriever import DenseRetriever
        from src.core.query_engine.fusion_factory import FusionFactory
        from src.core.query_engine.query_processor import QueryProcessor
        from src.core.query_engine.sparse_retriever import SparseRetriever
        from src.libs.tokenizer import TokenizerFactory

        retrieval = getattr(settings, "retrieval", None)
        filter_extractor = None
        if getattr(retrieval, "enable_filter_extraction", False):
            from src.core.query_engine.filter_extractor import RuleBasedFilterExtractor
            filter_extractor = RuleBasedFilterExtractor()
        synonym_map: dict[str, list[str]] = {}
        if getattr(retrieval, "enable_synonym_expansion", False):
            synonym_map = cls._load_synonyms(getattr(retrieval, "synonym_source", ""))
        qp = overrides.get("query_processor") or QueryProcessor(
            tokenizer=TokenizerFactory.create(settings),
            nfkc=getattr(retrieval, "enable_nfkc", True),
            casefold=getattr(retrieval, "normalize_casefold", True),
            to_simplified=getattr(retrieval, "normalize_to_simplified", False),
            filter_extractor=filter_extractor,
            synonym_map=synonym_map,
        )
        dense = overrides.get("dense_retriever") or DenseRetriever(settings=settings)
        sparse = overrides.get("sparse_retriever") or SparseRetriever(settings=settings)
        fusion = overrides.get("fusion") or FusionFactory.create(settings)
        return cls(
            qp,
            dense,
            sparse,
            fusion,
            settings=settings,
            candidate_multiplier=getattr(retrieval, "candidate_multiplier", 2),
            top_k_dense=getattr(retrieval, "top_k_dense", 20),
            top_k_sparse=getattr(retrieval, "top_k_sparse", 20),
            sparse_filter_overfetch=getattr(retrieval, "sparse_filter_overfetch", 4),
            enable_synonym_expansion=getattr(retrieval, "enable_synonym_expansion", False),
        )

    @staticmethod
    def _load_synonyms(path: str) -> dict[str, list[str]]:
        """Load a synonym map (JSON: term -> [aliases]); degrade to {} on error."""
        if not path:
            return {}
        import json
        import os

        if not os.path.exists(path):
            logger.warning("Synonym source not found: %s; expansion disabled", path)
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("Synonym source %s is not a dict; ignoring", path)
                return {}
            return {str(k): list(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning("Failed to load synonym source %s: %s", path, exc)
            return {}
