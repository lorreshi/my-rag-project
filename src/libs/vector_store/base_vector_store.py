"""VectorStore abstract base class.

Defines the contract for all vector database backends.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


@dataclass
class VectorRecord:
    """A single record to upsert into the vector store."""

    id: str
    vector: list[float]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """A single result returned from a vector query."""

    id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVectorStore(ABC):
    """Abstract base class for vector store backends."""

    @abstractmethod
    def upsert(
        self,
        records: list[VectorRecord],
        trace: "TraceContext | None" = None,
    ) -> int:
        """Insert or update records. Returns the number of records upserted."""
        ...

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[QueryResult]:
        """Query the store by vector similarity.

        Args:
            vector: The query embedding vector.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.
            trace: Optional trace context for observability.

        Returns:
            List of QueryResult sorted by descending relevance.
        """
        ...

    @abstractmethod
    def delete_by_metadata(
        self,
        filter: dict[str, Any],
        trace: "TraceContext | None" = None,
    ) -> int:
        """Delete records matching the metadata filter. Returns count deleted."""
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return a human-readable backend identifier (e.g. 'chroma')."""
        ...
