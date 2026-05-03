"""LLM Factory — configuration-driven provider instantiation.

Usage::

    from src.core.settings import load_settings
    from src.libs.llm.llm_factory import LLMFactory

    settings = load_settings()
    llm = LLMFactory.create(settings)
    response = llm.chat([ChatMessage(role="user", content="Hello")])
"""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.llm.base_llm import BaseLLM

if TYPE_CHECKING:
    from src.core.settings import Settings

# Registry: provider name -> lazy constructor
# Each entry is a callable(settings) -> BaseLLM.
# Concrete providers register themselves here when implemented (Phase B7).
_REGISTRY: dict[str, Callable[["Settings"], BaseLLM]] = {}


def register_provider(name: str, factory_fn: Callable[["Settings"], BaseLLM]) -> None:
    """Register a provider constructor under *name*."""
    _REGISTRY[name] = factory_fn


class LLMFactory:
    """Create a BaseLLM instance based on ``settings.llm.provider``."""

    @staticmethod
    def create(settings: "Settings") -> BaseLLM:
        """Instantiate the LLM backend specified in *settings*.

        Raises:
            ValueError: if the provider is unknown / not registered.
        """
        provider = settings.llm.provider.lower()
        factory_fn = _REGISTRY.get(provider)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown LLM provider '{provider}'. "
                f"Available: {available}"
            )
        return factory_fn(settings)
