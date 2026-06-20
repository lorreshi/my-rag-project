"""Recursive Character Text Splitter — default splitting strategy.

Implements the same algorithm as LangChain's RecursiveCharacterTextSplitter
without the heavy dependency chain. Splits text by trying separators in order
of priority (Markdown headings → paragraphs → lines → CJK/English sentence
boundaries → commas → words → characters), recursing into smaller separators
when chunks exceed the size limit.

The separator table is CJK-aware: long Chinese paragraphs are broken at
Chinese punctuation (。！？；，) *before* ever falling back to the character
level, so prose never degrades into single-character fragments (Property 11).
"""
from __future__ import annotations

from typing import Callable, TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter, SplitPiece
from src.libs.splitter.length import token_length
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

# Markdown-aware separators, from coarsest to finest.
# Order (design §2.1a): structure (headings) → paragraph → line →
# CJK sentence enders → English sentence enders → commas → word → character.
_DEFAULT_SEPARATORS = [
    "\n## ",      # H2 heading (structure)
    "\n### ",     # H3 heading
    "\n#### ",    # H4 heading
    "\n\n",       # paragraph break
    "\n",         # line break
    "。",          # CJK full stop
    "！",          # CJK exclamation
    "？",          # CJK question mark
    "；",          # CJK semicolon
    ". ",         # English sentence boundary
    "! ",         # English exclamation
    "? ",         # English question
    "; ",         # English semicolon
    "，",          # CJK comma (finer)
    ", ",         # English comma
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
        length_fn: Callable[[str], int] | None = None,
        size_unit: str = "char",
        token_encoding: str = "cl100k_base",
    ):
        """Create a recursive splitter.

        Args:
            chunk_size: Maximum chunk size, measured by the active length
                function (characters by default, tokens when ``size_unit`` is
                ``"token"``).
            chunk_overlap: Amount of trailing context (in the same unit) carried
                from the previous chunk into the next.
            separators: Custom separator hierarchy; defaults to the CJK-aware
                :data:`_DEFAULT_SEPARATORS`.
            length_fn: Explicit size-measuring function. When provided it takes
                precedence over ``size_unit``.
            size_unit: ``"char"`` (default, ``len``) or ``"token"`` (tiktoken
                counter via ``token_encoding``). Only consulted when
                ``length_fn`` is not given.
            token_encoding: tiktoken encoding name used when
                ``size_unit == "token"`` (default ``cl100k_base``, aligned with
                the embedding model).
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or _DEFAULT_SEPARATORS

        # Resolve the size-measuring function (design §2.1b). An explicit
        # length_fn always wins; otherwise size_unit selects char vs token.
        if length_fn is not None:
            self._length = length_fn
        elif size_unit == "token":
            self._length = token_length(token_encoding)
        else:
            self._length = len

    def split(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[SplitPiece]:
        """Split *text* into prose pieces with empty (structureless) metadata."""
        if not text.strip():
            return []
        return [SplitPiece(t) for t in self._split(text, self._separators)]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using the first applicable separator."""
        if self._length(text) <= self._chunk_size:
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

            if self._length(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # If this single piece is too large, recurse with finer separators.
                # When no finer separators remain (character level), the piece is
                # kept as-is — a structureless fallback block (the "兜底块" that
                # Property 12 exempts from the size bound), which also guarantees
                # token measurement can never cause an infinite loop.
                if self._length(piece.strip()) > self._chunk_size and remaining_seps:
                    chunks.extend(self._split(piece, remaining_seps))
                    current = ""
                else:
                    current = piece

        if current.strip():
            chunks.append(current.strip())

        # Apply overlap: prepend tail of previous chunk to next chunk.
        # Concatenate directly without inserting a separator — CJK-friendly.
        if self._chunk_overlap > 0 and len(chunks) > 1:
            overlapped: list[str] = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1][-self._chunk_overlap:]
                merged = prev_tail + chunks[i]
                if self._length(merged) <= self._chunk_size:
                    overlapped.append(merged)
                else:
                    overlapped.append(chunks[i])
            chunks = overlapped

        return chunks

    @property
    def splitter_type(self) -> str:
        return "recursive"


def _create_recursive(settings: "Settings") -> RecursiveSplitter:
    """Create a RecursiveSplitter from ``settings.splitter`` (default size).

    Reads chunk_size/chunk_overlap/size_unit/token_encoding from the splitter
    config, falling back to RecursiveSplitter defaults when the splitter config
    is absent (keeps the factory robust for minimal Settings objects).
    """
    return build_recursive_splitter(settings)


def build_recursive_splitter(
    settings: "Settings",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveSplitter:
    """Build a RecursiveSplitter using the effective size parameters.

    The size unit and token encoding always come from ``settings.splitter``
    (with ``getattr`` fallbacks). ``chunk_size``/``chunk_overlap`` may be
    overridden explicitly (e.g. per-collection overrides resolved by the
    chunker); when not given they default to the splitter config values.
    """
    splitter_cfg = getattr(settings, "splitter", None)

    if chunk_size is None:
        chunk_size = getattr(splitter_cfg, "chunk_size", 512)
    if chunk_overlap is None:
        chunk_overlap = getattr(splitter_cfg, "chunk_overlap", 64)
    size_unit = getattr(splitter_cfg, "size_unit", "token")
    token_encoding = getattr(splitter_cfg, "token_encoding", "cl100k_base")

    return RecursiveSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        size_unit=size_unit,
        token_encoding=token_encoding,
    )


register_splitter("recursive", _create_recursive)
