"""Reranker — Core-layer reranking orchestration with fallback.

Wraps a pluggable ``libs.reranker`` backend. Converts RetrievalResult <->
RerankCandidate, applies the backend on the Top-M candidates, and on any
failure (exception, timeout signal, or backend self-reported failure) falls
back to the incoming fusion ranking — never breaking the query.

Each returned RetrievalResult carries ``metadata["rerank_fallback"]`` so callers
and traces can tell whether reranking actually took effect.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.types import RetrievalResult
from src.libs.reranker.base_reranker import RerankCandidate

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext
    from src.libs.reranker.base_reranker import BaseReranker

logger = logging.getLogger(__name__)


class Reranker:
    """Core reranking stage with graceful fallback to fusion ranking."""

    def __init__(
        self,
        backend: "BaseReranker | None" = None,
        settings: "Settings | None" = None,
        top_m: int = 30,
    ):
        """Initialize the Core reranker.

        Args:
            backend: A libs.reranker BaseReranker. Built from settings if None.
            settings: Settings used to build the backend / read top_m.
            top_m: Max candidates to send to the backend (cost/latency control).
        """
        self._backend = backend
        if self._backend is None:
            if settings is None:
                from src.libs.reranker.base_reranker import NoneReranker
                self._backend = NoneReranker()
            else:
                from src.libs.reranker.reranker_factory import RerankerFactory
                self._backend = RerankerFactory.create(settings)

        if settings is not None and hasattr(settings, "rerank"):
            self._top_m = getattr(settings.rerank, "top_m", top_m)
        else:
            self._top_m = top_m

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates, falling back to fusion order on failure.

        Args:
            query: The user query.
            candidates: Fusion-ranked candidates (descending relevance).
            top_k: Optional limit on returned results.
            trace: Optional trace context.

        Returns:
            Reranked (or fallback) RetrievalResult list. Each item's metadata
            includes ``rerank_fallback`` (bool) and ``rerank_backend``.
        """
        if trace:
            trace.start_stage("rerank")

        backend_name = getattr(self._backend, "backend_name", "unknown")

        if not candidates:
            if trace:
                trace.end_stage(
                    details={"backend": backend_name, "fallback": False, "count": 0}
                )
            return []

        # NoneReranker: identity — not a fallback, just disabled reranking.
        if backend_name == "none":
            result = self._finalize(candidates, top_k, fallback=False, backend=backend_name)
            if trace:
                trace.end_stage(
                    details={"backend": backend_name, "fallback": False,
                             "count": len(result)}
                )
            return result

        head = candidates[: self._top_m]
        tail = candidates[self._top_m :]

        fallback = False
        try:
            rerank_candidates = [
                RerankCandidate(id=c.chunk_id, text=c.text, score=c.score)
                for c in head
            ]
            reranked = self._backend.rerank(query, rerank_candidates, trace=trace)

            # Backend may self-report failure (e.g. CrossEncoder timeout/error).
            if getattr(self._backend, "has_failed", False):
                raise RuntimeError("reranker backend reported failure")

            by_id = {c.chunk_id: c for c in head}
            reordered: list[RetrievalResult] = []
            for rc in reranked:
                original = by_id.get(rc.id)
                if original is None:
                    continue
                reordered.append(
                    RetrievalResult(
                        chunk_id=rc.id,
                        score=rc.score,
                        text=original.text,
                        metadata=dict(original.metadata),
                    )
                )
            # Append any non-reranked tail (kept in fusion order).
            merged = reordered + tail
        except Exception as exc:
            logger.warning(
                "Rerank backend '%s' failed (%s); falling back to fusion order",
                backend_name, exc,
            )
            fallback = True
            merged = list(candidates)

        result = self._finalize(merged, top_k, fallback=fallback, backend=backend_name)

        if trace:
            trace.end_stage(
                details={
                    "backend": backend_name,
                    "method": backend_name,
                    "fallback": fallback,
                    "count": len(result),
                }
            )

        return result

    @staticmethod
    def _finalize(
        results: list[RetrievalResult],
        top_k: int | None,
        fallback: bool,
        backend: str,
    ) -> list[RetrievalResult]:
        """Tag fallback/backend metadata and apply the top_k cut."""
        for r in results:
            r.metadata["rerank_fallback"] = fallback
            r.metadata["rerank_backend"] = backend
        if top_k is not None:
            return results[:top_k]
        return results
