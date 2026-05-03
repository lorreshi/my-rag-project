"""Reranker Factory — configuration-driven backend instantiation.

The 'none' backend is always available as a built-in fallback.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.reranker.base_reranker import BaseReranker, NoneReranker

if TYPE_CHECKING:
    from src.core.settings import Settings

_REGISTRY: dict[str, Callable[["Settings"], BaseReranker]] = {
    "none": lambda _s: NoneReranker(),
}


def register_backend(
    name: str, factory_fn: Callable[["Settings"], BaseReranker]
) -> None:
    """Register a reranker backend constructor under *name*."""
    _REGISTRY[name] = factory_fn


class RerankerFactory:
    """Create a BaseReranker based on ``settings.rerank.backend``."""

    @staticmethod
    def create(settings: "Settings") -> BaseReranker:
        """Instantiate the reranker backend specified in *settings*.

        Raises:
            ValueError: if the backend is unknown / not registered.
        """
        backend = settings.rerank.backend.lower()
        factory_fn = _REGISTRY.get(backend)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown reranker backend '{backend}'. "
                f"Available: {available}"
            )
        return factory_fn(settings)
