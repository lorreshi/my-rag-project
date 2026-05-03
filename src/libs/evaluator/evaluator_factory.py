"""Evaluator Factory — configuration-driven backend instantiation.

The 'custom' backend is always available as a built-in.
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.libs.evaluator.custom_evaluator import CustomEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings

_REGISTRY: dict[str, Callable[["Settings"], BaseEvaluator]] = {
    "custom": lambda _s: CustomEvaluator(),
}


def register_backend(
    name: str, factory_fn: Callable[["Settings"], BaseEvaluator]
) -> None:
    """Register an evaluator backend constructor under *name*."""
    _REGISTRY[name] = factory_fn


class EvaluatorFactory:
    """Create BaseEvaluator instances based on settings."""

    @staticmethod
    def create(settings: "Settings", backend: str = "custom") -> BaseEvaluator:
        """Instantiate the evaluator specified by *backend*.

        Raises:
            ValueError: if the backend is unknown / not registered.
        """
        key = backend.lower()
        factory_fn = _REGISTRY.get(key)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown evaluator backend '{key}'. "
                f"Available: {available}"
            )
        return factory_fn(settings)
