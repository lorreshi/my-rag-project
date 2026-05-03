"""Embedding Factory — configuration-driven provider instantiation.

Usage::

    from src.core.settings import load_settings
    from src.libs.embedding.embedding_factory import EmbeddingFactory

    settings = load_settings()
    embedding = EmbeddingFactory.create(settings)
    vectors = embedding.embed(["hello world"])
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding

if TYPE_CHECKING:
    from src.core.settings import Settings

# Registry: provider name -> lazy constructor
_REGISTRY: dict[str, Callable[["Settings"], BaseEmbedding]] = {}


def register_provider(
    name: str, factory_fn: Callable[["Settings"], BaseEmbedding]
) -> None:
    """Register an embedding provider constructor under *name*."""
    _REGISTRY[name] = factory_fn


class EmbeddingFactory:
    """Create a BaseEmbedding instance based on ``settings.embedding.provider``."""

    @staticmethod
    def create(settings: "Settings") -> BaseEmbedding:
        """Instantiate the embedding backend specified in *settings*.

        Raises:
            ValueError: if the provider is unknown / not registered.
        """
        provider = settings.embedding.provider.lower()
        factory_fn = _REGISTRY.get(provider)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown embedding provider '{provider}'. "
                f"Available: {available}"
            )
        return factory_fn(settings)
