"""Fusion — Reciprocal Rank Fusion (RRF) of multiple ranked result lists.

RRF combines several ranked lists into one without relying on the absolute
scores of each list. For a document d appearing at rank r (1-based) in a list:

    contribution = 1 / (k + r)

The fused score is the sum of contributions across all lists in which the
document appears. Larger fused score = more relevant.

This is deterministic: ties are broken by descending fused score then by
chunk_id (lexicographic), so the same input always yields the same output.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Standard RRF constant (Cormack et al., 2009).
_DEFAULT_K = 60

# Canonical route order used to map a ``fusion_weights`` dict (keyed by route
# name) onto the positional ``result_lists`` passed to ``fuse``.
_DEFAULT_ROUTE_NAMES: tuple[str, ...] = ("dense", "sparse")


class BaseFusion(ABC):
    """Abstract base for fusion strategies (RRF, weighted sum, ...).

    Implementations combine several ranked ``RetrievalResult`` lists into a
    single ranking. They must be deterministic for a given input.
    """

    @abstractmethod
    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Fuse ranked lists into one ranking sorted by descending relevance."""
        ...


class ReciprocalRankFusion(BaseFusion):
    """Fuse ranked RetrievalResult lists via (optionally weighted) RRF.

    For a document ``d`` at rank ``r`` (1-based) in list ``i`` with weight
    ``w_i``::

        contribution = w_i / (k + r)

    The fused score sums contributions across every list in which ``d``
    appears. With no weights configured every ``w_i`` is ``1.0`` and the output
    is identical to plain RRF (backward compatible).
    """

    def __init__(
        self,
        k: int = _DEFAULT_K,
        weights: dict[str, float] | Sequence[float] | None = None,
        route_names: Sequence[str] = _DEFAULT_ROUTE_NAMES,
    ):
        """Initialize RRF.

        Args:
            k: RRF constant; larger k dampens the influence of top ranks.
            weights: Optional per-route weights. Either a positional sequence
                (weight per list index) or a dict keyed by route name (mapped
                to list indices via *route_names*). ``None`` => all weights 1.0.
            route_names: Canonical order used to map a weights dict onto the
                positional ``result_lists`` (default ``("dense", "sparse")``).
        """
        if k <= 0:
            raise ValueError("RRF k must be a positive integer")
        self._k = k
        self._weights = weights
        self._route_names = tuple(route_names)

    def _weight_for(self, list_idx: int) -> float:
        """Return the weight for the list at *list_idx* (default 1.0)."""
        w = self._weights
        if not w:
            return 1.0
        if isinstance(w, dict):
            if list_idx < len(self._route_names):
                return float(w.get(self._route_names[list_idx], 1.0))
            return 1.0
        # positional sequence
        if 0 <= list_idx < len(w):
            return float(w[list_idx])
        return 1.0

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        """Fuse multiple ranked lists into a single ranking.

        Args:
            result_lists: Each inner list is a ranked list (rank 0 = best).
            top_k: Optional limit on the number of fused results.
            trace: Optional trace context.

        Returns:
            Fused RetrievalResult list sorted by descending RRF score (ties
            broken by chunk_id). The ``score`` field holds the RRF score; the
            ``text``/``metadata`` are taken from the first list in which the
            document appears.
        """
        if trace:
            trace.start_stage("fusion")

        rrf_scores: dict[str, float] = {}
        # Keep the best-known text/metadata for each chunk_id (first occurrence).
        payloads: dict[str, RetrievalResult] = {}

        for list_idx, results in enumerate(result_lists):
            weight = self._weight_for(list_idx)
            for rank, item in enumerate(results, start=1):
                cid = item.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + weight / (self._k + rank)
                if cid not in payloads:
                    payloads[cid] = item

        fused = [
            RetrievalResult(
                chunk_id=cid,
                score=score,
                text=payloads[cid].text,
                metadata=payloads[cid].metadata,
            )
            for cid, score in rrf_scores.items()
        ]
        # Deterministic ordering: score desc, then chunk_id asc
        fused.sort(key=lambda r: (-r.score, r.chunk_id))

        if top_k is not None:
            fused = fused[:top_k]

        if trace:
            trace.end_stage(
                details={
                    "num_lists": len(result_lists),
                    "fused_count": len(fused),
                    "k": self._k,
                    "method": "rrf",
                    "algorithm": "rrf",
                    "weighted": bool(self._weights),
                }
            )

        return fused
