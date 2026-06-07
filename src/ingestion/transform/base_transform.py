"""Transform abstract base class.

All transform steps in the ingestion pipeline must implement this interface.
A Transform receives a list of Chunks and returns a (possibly modified) list
of Chunks. Transforms are composable and order-independent where possible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext
    from src.core.types import Chunk


class BaseTransform(ABC):
    """Abstract base class for ingestion transforms."""

    @abstractmethod
    def transform(
        self,
        chunks: list["Chunk"],
        trace: "TraceContext | None" = None,
    ) -> list["Chunk"]:
        """Apply transformation to a list of chunks.

        Args:
            chunks: Input chunks to transform.
            trace: Optional trace context for observability.

        Returns:
            Transformed list of chunks (same length, modified in-place or copied).

        Contract:
            - Must not raise on individual chunk failures; log and skip/preserve.
            - Must preserve chunk.id and chunk.source_ref unchanged.
            - May modify chunk.text and chunk.metadata.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable transform name (e.g. 'chunk_refiner')."""
        ...
