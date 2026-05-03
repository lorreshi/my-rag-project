"""VectorStore Factory — configuration-driven backend instantiation."""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.vector_store.base_vector_store import BaseVectorStore

if TYPE_CHECKING:
    from src.core.settings import Settings

_REGISTRY: dict[str, Callable[["Settings"], BaseVectorStore]] = {}


def register_backend(
    name: str, factory_fn: Callable[["Settings"], BaseVectorStore]
) -> None:
    """Register a vector store backend constructor under *name*."""
    _REGISTRY[name] = factory_fn


class VectorStoreFactory:
    """Create a BaseVectorStore based on ``settings.vector_store.backend``."""

    @staticmethod
    def create(settings: "Settings") -> BaseVectorStore:
        """Instantiate the vector store backend specified in *settings*.

        Raises:
            ValueError: if the backend is unknown / not registered.
        """
        backend = settings.vector_store.backend.lower()
        factory_fn = _REGISTRY.get(backend)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown vector store backend '{backend}'. "
                f"Available: {available}"
            )
        return factory_fn(settings)
