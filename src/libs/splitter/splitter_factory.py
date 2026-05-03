"""Splitter Factory — configuration-driven strategy instantiation."""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from src.core.settings import Settings

_REGISTRY: dict[str, Callable[["Settings"], BaseSplitter]] = {}


def register_splitter(
    name: str, factory_fn: Callable[["Settings"], BaseSplitter]
) -> None:
    """Register a splitter constructor under *name*."""
    _REGISTRY[name] = factory_fn


class SplitterFactory:
    """Create a BaseSplitter based on config (future: settings.splitter.type)."""

    @staticmethod
    def create(settings: "Settings", splitter_type: str = "recursive") -> BaseSplitter:
        """Instantiate the splitter specified by *splitter_type*.

        Raises:
            ValueError: if the splitter type is unknown / not registered.
        """
        key = splitter_type.lower()
        factory_fn = _REGISTRY.get(key)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unknown splitter type '{key}'. "
                f"Available: {available}"
            )
        return factory_fn(settings)
