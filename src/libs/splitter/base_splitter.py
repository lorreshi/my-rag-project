"""Splitter abstract base class.

All splitter implementations must implement this interface so the pipeline
can call ``splitter.split_text(text)`` without knowing which strategy is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """Abstract base class for text splitting strategies."""

    @abstractmethod
    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        """Split *text* into a list of chunks.

        Args:
            text: The full document text to split.
            trace: Optional trace context for observability.

        Returns:
            List of text chunks.
        """
        ...

    @property
    @abstractmethod
    def splitter_type(self) -> str:
        """Return a human-readable splitter identifier (e.g. 'recursive')."""
        ...
