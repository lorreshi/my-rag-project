"""Recursive Character Text Splitter — default splitting strategy.

Implements the same algorithm as LangChain's RecursiveCharacterTextSplitter
without the heavy dependency chain. Splits text by trying separators in order
of priority (Markdown headings → paragraphs → sentences → characters),
recursing into smaller separators when chunks exceed the size limit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

# Markdown-aware separators, from coarsest to finest
_DEFAULT_SEPARATORS = [
    "\n## ",      # H2 heading
    "\n### ",     # H3 heading
    "\n#### ",    # H4 heading
    "\n\n",       # paragraph break
    "\n",         # line break
    ". ",         # sentence boundary
    " ",          # word boundary
    "",           # character-level (last resort)
]


class RecursiveSplitter(BaseSplitter):
    """Split text recursively using a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or _DEFAULT_SEPARATORS

    def split_text(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[str]:
        if not text.strip():
            return []
        return self._split(text, self._separators)

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the first applicable separator."""
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        # Find the best separator for this text
        sep = separators[-1]  # fallback
        for s in separators:
            if s == "":
                sep = s
                break
            if s in text:
                sep = s
                break

        # Split by the chosen separator
        if sep:
            parts = text.split(sep)
        else:
            # Character-level split
            parts = list(text)

        # Merge small parts into chunks respecting chunk_size
        chunks: list[str] = []
        current = ""
        remaining_seps = separators[separators.index(sep) + 1:] if sep in separators else separators[-1:]

        for part in parts:
            piece = part if not sep else (sep + part if current else part)
            candidate = current + piece if current else piece

            if len(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If this single piece is too large, recurse with finer separators
                if len(piece.strip()) > self._chunk_size and remaining_seps:
                    chunks.extend(self._split(piece, remaining_seps))
                    current = ""
                else:
                    current = piece

        if current.strip():
            chunks.append(current.strip())

        # Apply overlap: prepend tail of previous chunk to next chunk
        if self._chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self._chunk_overlap:]
                merged = prev_tail + " " + chunks[i]
                if len(merged) <= self._chunk_size:
                    overlapped.append(merged)
                else:
                    overlapped.append(chunks[i])
            chunks = overlapped

        return chunks

    @property
    def splitter_type(self) -> str:
        return "recursive"


def _create_recursive(settings: "Settings") -> RecursiveSplitter:
    # Future: read chunk_size/overlap from settings
    return RecursiveSplitter()


register_splitter("recursive", _create_recursive)
