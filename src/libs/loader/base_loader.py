"""Loader abstract base class.

All document loaders must implement this interface. A Loader is responsible
for parsing a raw file into a unified Document object (text + metadata).
It does NOT perform chunking — that's the Splitter's job.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.core.types import Document


class BaseLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, path: str) -> Document:
        """Parse a file and return a Document.

        Args:
            path: Path to the source file.

        Returns:
            Document with text in canonical Markdown format and metadata
            containing at least 'source_path'.

        Raises:
            FileNotFoundError: if *path* does not exist.
            ValueError: if the file format is unsupported or unparseable.
        """
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of supported file extensions (e.g. ['.pdf'])."""
        ...
