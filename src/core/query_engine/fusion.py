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
from typing import TYPE_CHECKING

from src.core.types import RetrievalResult

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger(__name__)

# Standard RRF constant (Cormack et al., 2009).
_DEFAULT_K = 60


class ReciprocalRankFusion:
    """Fuse ranked RetrievalResult lists via RRF."""

    def __init__(self, k: int = _DEFAULT_K):
        """Initialize RRF.

        Args:
            k: RRF constant; larger k dampens the influence of top ranks.
        """
        if k <= 0:
            raise ValueError("RRF k must be a positive integer")
        self._k = k

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

        for results in result_lists:
            for rank, item in enumerate(results, start=1):
                cid = item.chunk_id
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self._k + rank)
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
                    "algorithm": "rrf",
                }
            )

        return fused
