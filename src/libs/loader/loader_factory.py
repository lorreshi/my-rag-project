"""Loader Factory — extension-driven loader routing.

Follows the project's existing factory pattern (matching ``SplitterFactory``'s
``register_*`` style). Loaders self-register an extension → constructor mapping;
``LoaderFactory.create(path)`` routes by file suffix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.libs.loader.base_loader import BaseLoader

# ext (lowercase, with leading dot, e.g. ".pdf") -> zero-arg constructor
_REGISTRY: dict[str, Callable[[], BaseLoader]] = {}


def register_loader(
    extensions: list[str], factory_fn: Callable[[], BaseLoader]
) -> None:
    """Register *factory_fn* for each extension in *extensions*.

    Extensions are normalized to lowercase with a leading dot (e.g. ``.pdf``).
    """
    for ext in extensions:
        key = ext.lower()
        if not key.startswith("."):
            key = "." + key
        _REGISTRY[key] = factory_fn


class LoaderFactory:
    """Create a BaseLoader based on a file path's extension."""

    @staticmethod
    def create(path: str) -> BaseLoader:
        """Instantiate the loader registered for ``Path(path).suffix``.

        Raises:
            ValueError: if the file extension is not registered. The message
                includes the sorted list of available extensions.
        """
        ext = Path(path).suffix.lower()
        factory_fn = _REGISTRY.get(ext)
        if factory_fn is None:
            available = ", ".join(sorted(_REGISTRY)) or "(none)"
            raise ValueError(
                f"Unsupported file extension '{ext}'. "
                f"Available extensions: {available}"
            )
        return factory_fn()
