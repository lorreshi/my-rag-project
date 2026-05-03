"""Reranker abstract base class and NoneReranker fallback.

All reranker implementations must implement this interface.
NoneReranker is the default fallback that preserves original ordering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


@dataclass
class RerankCandidate:
    """A candidate document for reranking."""

    id: str
    text: str
    score: float = 0.0


class BaseReranker(ABC):
    """Abstract base class for reranker backends."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: "TraceContext | None" = None,
    ) -> list[RerankCandidate]:
        """Rerank candidates by relevance to *query*.

        Args:
            query: The user query string.
            candidates: List of candidates to rerank.
            trace: Optional trace context for observability.

        Returns:
            Candidates sorted by descending relevance with updated scores.
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return a human-readable backend identifier (e.g. 'cross_encoder')."""
        ...


class NoneReranker(BaseReranker):
    """Fallback reranker that preserves the original ordering unchanged."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: "TraceContext | None" = None,
    ) -> list[RerankCandidate]:
        # Return candidates as-is, no reranking
        return list(candidates)

    @property
    def backend_name(self) -> str:
        return "none"
