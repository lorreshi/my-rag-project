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


def _resolve_weight(
    weights: "dict[str, float] | Sequence[float] | None",
    route_names: "Sequence[str]",
    list_idx: int,
) -> float:
    """Resolve the weight for the list at *list_idx* (default 1.0).

    *weights* may be a positional sequence (weight per index) or a dict keyed
    by route name (mapped to indices via *route_names*). Missing/out-of-range
    entries default to 1.0.
    """
    if not weights:
        return 1.0
    if isinstance(weights, dict):
        if list_idx < len(route_names):
            return float(weights.get(route_names[list_idx], 1.0))
        return 1.0
    if 0 <= list_idx < len(weights):
        return float(weights[list_idx])
    return 1.0


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
        return _resolve_weight(self._weights, self._route_names, list_idx)

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


class WeightedSumFusion(BaseFusion):
    """Fuse ranked lists via a weighted sum of min-max normalized scores.

    Unlike RRF (rank-based), this strategy uses the absolute ``score`` of each
    candidate. Because dense (cosine) and sparse (BM25) scores live on
    different scales, each list is min-max normalized to ``[0, 1]`` before
    weighting::

        fused_score(d) = Σ_i  w_i * norm_i(d)

    A list whose scores are all equal (e.g. a single item) normalizes to 1.0.
    Documents absent from a list contribute 0 from that list. Ties are broken
    by chunk_id (ascending), matching ``ReciprocalRankFusion``.
    """

    def __init__(
        self,
        weights: "dict[str, float] | Sequence[float] | None" = None,
        route_names: "Sequence[str]" = _DEFAULT_ROUTE_NAMES,
    ):
        self._weights = weights
        self._route_names = tuple(route_names)

    def _weight_for(self, list_idx: int) -> float:
        return _resolve_weight(self._weights, self._route_names, list_idx)

    @staticmethod
    def _min_max_normalize(results: list[RetrievalResult]) -> dict[str, float]:
        """Map each chunk_id to its min-max normalized score within the list."""
        if not results:
            return {}
        scores = [r.score for r in results]
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span == 0:
            # All equally relevant (incl. single-item lists) -> full weight.
            return {r.chunk_id: 1.0 for r in results}
        return {r.chunk_id: (r.score - lo) / span for r in results}

    def fuse(
        self,
        result_lists: list[list[RetrievalResult]],
        top_k: int | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[RetrievalResult]:
        if trace:
            trace.start_stage("fusion")

        scores: dict[str, float] = {}
        payloads: dict[str, RetrievalResult] = {}

        for list_idx, results in enumerate(result_lists):
            weight = self._weight_for(list_idx)
            normalized = self._min_max_normalize(results)
            for item in results:
                cid = item.chunk_id
                scores[cid] = scores.get(cid, 0.0) + weight * normalized[cid]
                if cid not in payloads:
                    payloads[cid] = item

        fused = [
            RetrievalResult(
                chunk_id=cid,
                score=score,
                text=payloads[cid].text,
                metadata=payloads[cid].metadata,
            )
            for cid, score in scores.items()
        ]
        fused.sort(key=lambda r: (-r.score, r.chunk_id))

        if top_k is not None:
            fused = fused[:top_k]

        if trace:
            trace.end_stage(
                details={
                    "num_lists": len(result_lists),
                    "fused_count": len(fused),
                    "method": "weighted_sum",
                    "algorithm": "weighted_sum",
                    "weighted": bool(self._weights),
                }
            )

        return fused
