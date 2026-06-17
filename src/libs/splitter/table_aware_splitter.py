"""Table-aware splitter for Markdown tables (design §2.2, Property 13).

Splits Markdown produced from spreadsheet-like sources (xlsx) into
self-contained, structure-aware chunks:

1. **Table detection** — scans for contiguous ``| ... |`` rows shaped as a
   *header row* + *separator row* (``|---|``, alignment variants like
   ``:---:`` allowed) + zero or more *data rows*. Non-table prose is delegated
   to a fallback :class:`RecursiveSplitter`; those fallback pieces carry an
   empty ``metadata`` dict (Requirement 5.5 / Property 14).
2. **Sheet boundaries** — an ``## {sheet_name}`` H2 heading marks a sheet
   boundary. Rows after that heading (and before the next ``## ``) belong to
   that sheet. Rows from different sheets are **never** mixed into one chunk.
   When no sheet heading precedes a table, ``sheet_name`` is ``None``.
3. **Row chunking + header repetition** — data rows are packed into chunks by a
   token budget (``chunk_size``); every chunk repeats the header (header line +
   separator line) so each chunk is self-contained. A small table becomes a
   single chunk.
4. **Structured metadata** — each table :class:`SplitPiece` carries
   ``sheet_name``, ``row_start``/``row_end`` (1-based, inclusive, over the
   table's data rows) and ``is_table=True``.

Validates: Requirements 5.2, 5.3, 5.4, 5.5 (Property 13)
"""
from __future__ import annotations

import re
from typing import Callable, TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter, SplitPiece
from src.libs.splitter.length import token_length
from src.libs.splitter.recursive_splitter import build_recursive_splitter
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

# H2 heading used as a sheet boundary: "## Sheet1" (but not "### ...").
_SHEET_HEADING_RE = re.compile(r"^##(?!#)\s+(.+?)\s*$")

# A separator cell is dashes with optional leading/trailing alignment colons,
# e.g. "---", ":---", "---:", ":---:".
_SEPARATOR_CELL_RE = re.compile(r"^:?-+:?$")

# Sentinel: no enclosing sheet heading seen yet.
_NO_SHEET: str | None = None


def _is_table_row(line: str) -> bool:
    """Return True if *line* looks like a Markdown table row (``| ... |``)."""
    stripped = line.strip()
    return stripped.startswith("|") and len(stripped) > 1


def _split_cells(line: str) -> list[str]:
    """Split a table row into trimmed cell texts (drop the outer borders)."""
    stripped = line.strip()
    inner = stripped.strip("|")
    return [cell.strip() for cell in inner.split("|")]


def _is_separator_row(line: str) -> bool:
    """Return True if *line* is a Markdown table separator row (``|---|``)."""
    if not _is_table_row(line):
        return False
    cells = _split_cells(line)
    if not cells:
        return False
    return all(_SEPARATOR_CELL_RE.match(cell) for cell in cells)


class TableAwareSplitter(BaseSplitter):
    """Split Markdown tables row-wise with repeated headers and sheet metadata."""

    def __init__(
        self,
        chunk_size: int = 512,
        size_unit: str = "token",
        token_encoding: str = "cl100k_base",
        fallback_splitter: BaseSplitter | None = None,
        length_fn: Callable[[str], int] | None = None,
    ):
        """Create a table-aware splitter.

        Args:
            chunk_size: Maximum chunk size (token budget) for packing data rows,
                measured by the active length function.
            size_unit: ``"token"`` (default, tiktoken) or ``"char"`` (``len``).
                Only consulted when ``length_fn`` is not given.
            token_encoding: tiktoken encoding used when ``size_unit == "token"``.
            fallback_splitter: Splitter used for non-table prose. Defaults to a
                :class:`RecursiveSplitter` constructed from the same size params.
            length_fn: Explicit size-measuring function (overrides ``size_unit``).
        """
        self._chunk_size = chunk_size

        if length_fn is not None:
            self._length = length_fn
        elif size_unit == "token":
            self._length = token_length(token_encoding)
        else:
            self._length = len

        if fallback_splitter is not None:
            self._fallback = fallback_splitter
        else:
            from src.libs.splitter.recursive_splitter import RecursiveSplitter

            self._fallback = RecursiveSplitter(
                chunk_size=chunk_size,
                chunk_overlap=0,
                size_unit=size_unit,
                token_encoding=token_encoding,
            )

    @property
    def splitter_type(self) -> str:
        return "table_aware"

    def split(
        self,
        text: str,
        trace: "TraceContext | None" = None,
    ) -> list[SplitPiece]:
        """Split *text* into table chunks (with metadata) and prose fallbacks."""
        if not text.strip():
            return []

        lines = text.split("\n")
        pieces: list[SplitPiece] = []
        prose_buf: list[str] = []
        current_sheet: str | None = _NO_SHEET

        def flush_prose() -> None:
            """Emit buffered non-table lines via the recursive fallback."""
            if not prose_buf:
                return
            joined = "\n".join(prose_buf)
            prose_buf.clear()
            if not joined.strip():
                return
            for piece in self._fallback.split(joined, trace):
                # Non-table fallback pieces carry empty metadata (Property 14).
                pieces.append(SplitPiece(piece.text, {}))

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]

            # Sheet boundary: an H2 heading sets the current sheet name. The
            # heading text itself is treated as prose (flushed via fallback).
            heading = _SHEET_HEADING_RE.match(line.strip())
            if heading:
                current_sheet = heading.group(1)
                prose_buf.append(line)
                i += 1
                continue

            # Table start: a table row immediately followed by a separator row.
            if (
                _is_table_row(line)
                and i + 1 < n
                and _is_separator_row(lines[i + 1])
            ):
                flush_prose()
                header = line
                separator = lines[i + 1]
                j = i + 2
                data_rows: list[str] = []
                while j < n and _is_table_row(lines[j]) and not _is_separator_row(lines[j]):
                    data_rows.append(lines[j])
                    j += 1
                pieces.extend(
                    self._build_table_chunks(
                        header, separator, data_rows, current_sheet
                    )
                )
                i = j
                continue

            # Anything else is prose; buffer it for the fallback splitter.
            prose_buf.append(line)
            i += 1

        flush_prose()
        return pieces

    def _build_table_chunks(
        self,
        header: str,
        separator: str,
        data_rows: list[str],
        sheet_name: str | None,
    ) -> list[SplitPiece]:
        """Pack *data_rows* into chunks, repeating the header in each chunk.

        ``row_start``/``row_end`` are 1-based, inclusive indices over the
        table's own data rows.
        """
        header_block = f"{header}\n{separator}"

        # Header-only table (no data rows): emit the header as a single chunk.
        if not data_rows:
            meta = {
                "sheet_name": sheet_name,
                "row_start": 0,
                "row_end": 0,
                "is_table": True,
            }
            return [SplitPiece(header_block, meta)]

        chunks: list[SplitPiece] = []
        total = len(data_rows)
        idx = 0
        while idx < total:
            start = idx
            current = [data_rows[idx]]
            idx += 1
            # Greedily add rows while the chunk stays within the token budget.
            while idx < total:
                candidate = current + [data_rows[idx]]
                candidate_text = header_block + "\n" + "\n".join(candidate)
                if self._length(candidate_text) <= self._chunk_size:
                    current = candidate
                    idx += 1
                else:
                    break

            text = header_block + "\n" + "\n".join(current)
            meta = {
                "sheet_name": sheet_name,
                "row_start": start + 1,  # 1-based inclusive
                "row_end": idx,          # idx now points one past the last row
                "is_table": True,
            }
            chunks.append(SplitPiece(text, meta))

        return chunks


def _create_table_aware(settings: "Settings") -> TableAwareSplitter:
    """Create a TableAwareSplitter from ``settings.splitter``.

    Reads ``chunk_size``/``size_unit``/``token_encoding`` from the splitter
    config (with ``getattr`` fallbacks) and constructs an internal recursive
    fallback splitter for non-table prose.
    """
    splitter_cfg = getattr(settings, "splitter", None)
    chunk_size = getattr(splitter_cfg, "chunk_size", 512)
    size_unit = getattr(splitter_cfg, "size_unit", "token")
    token_encoding = getattr(splitter_cfg, "token_encoding", "cl100k_base")

    fallback = build_recursive_splitter(settings)

    return TableAwareSplitter(
        chunk_size=chunk_size,
        size_unit=size_unit,
        token_encoding=token_encoding,
        fallback_splitter=fallback,
    )


register_splitter("table_aware", _create_table_aware)
