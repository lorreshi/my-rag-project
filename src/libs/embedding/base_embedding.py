"""Embedding abstract base class.

All embedding providers must implement this interface so the rest of the
codebase can call ``embedding.embed(texts)`` without knowing which backend
is in use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(
        self,
        texts: list[str],
        trace: "TraceContext | None" = None,
    ) -> list[list[float]]:
        """Compute embedding vectors for a batch of texts.

        Args:
            texts: List of text strings to embed.
            trace: Optional trace context for observability.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return a human-readable provider identifier (e.g. 'openai')."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...
