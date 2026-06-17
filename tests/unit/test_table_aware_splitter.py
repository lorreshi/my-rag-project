"""Tests for TableAwareSplitter (T9, design §2.2, Property 13).

Validates: Requirements 5.2, 5.3, 5.4, 5.5

Conventions fixed by these tests:
* ``sheet_name`` is ``None`` when no ``## {sheet_name}`` heading precedes a table.
* ``row_start``/``row_end`` are **1-based, inclusive** indices over a table's
  own data rows (separator/header rows are not counted).
* Non-table prose pieces carry an empty ``metadata`` dict.
"""
from __future__ import annotations

import pytest

from src.libs.splitter.base_splitter import SplitPiece
from src.libs.splitter.table_aware_splitter import TableAwareSplitter


def _table_pieces(pieces: list[SplitPiece]) -> list[SplitPiece]:
    return [p for p in pieces if p.metadata.get("is_table")]


def _prose_pieces(pieces: list[SplitPiece]) -> list[SplitPiece]:
    return [p for p in pieces if not p.metadata.get("is_table")]


@pytest.mark.unit
class TestMultiSheet:
    """Two ``## sheet`` sections, each with its own table."""

    MD = (
        "## Sheet1\n"
        "\n"
        "| Name | Age |\n"
        "| --- | --- |\n"
        "| Alice | 30 |\n"
        "| Bob | 25 |\n"
        "\n"
        "## Sheet2\n"
        "\n"
        "| City | Pop |\n"
        "| --- | --- |\n"
        "| NYC | 8 |\n"
    )

    def test_does_not_cross_sheets_and_headers_repeat(self):
        # Large budget -> each small table collapses into a single chunk.
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        pieces = splitter.split(self.MD)
        tables = _table_pieces(pieces)

        assert len(tables) == 2
        sheets = [p.metadata["sheet_name"] for p in tables]
        assert sheets == ["Sheet1", "Sheet2"]

        # Each table chunk begins with its own header row.
        assert tables[0].text.startswith("| Name | Age |")
        assert tables[1].text.startswith("| City | Pop |")

        # No sheet bleed: Sheet1 rows never appear in the Sheet2 chunk.
        assert "Alice" not in tables[1].text
        assert "NYC" not in tables[0].text

    def test_metadata_fields_present(self):
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        tables = _table_pieces(splitter.split(self.MD))
        for piece in tables:
            assert piece.metadata["is_table"] is True
            assert "sheet_name" in piece.metadata
            assert "row_start" in piece.metadata
            assert "row_end" in piece.metadata
        # Sheet1 has 2 data rows, Sheet2 has 1.
        assert (tables[0].metadata["row_start"], tables[0].metadata["row_end"]) == (1, 2)
        assert (tables[1].metadata["row_start"], tables[1].metadata["row_end"]) == (1, 1)


@pytest.mark.unit
class TestOversizedTable:
    """A table whose data rows exceed the token/char budget splits into chunks."""

    def _build_md(self, n_rows: int) -> str:
        header = "| n |\n| - |"
        rows = "\n".join(f"| {i} |" for i in range(n_rows))
        return f"{header}\n{rows}\n"

    def test_splits_into_multiple_chunks_with_continuous_ranges(self):
        n_rows = 10
        md = self._build_md(n_rows)
        # chunk_size=30 (char) fits the header block + ~3 data rows per chunk.
        splitter = TableAwareSplitter(chunk_size=30, size_unit="char")
        tables = _table_pieces(splitter.split(md))

        assert len(tables) > 1

        # Every chunk repeats the header (and separator).
        for piece in tables:
            assert piece.text.startswith("| n |\n| - |")

        # Row ranges are continuous, 1-based, and cover every data row exactly once.
        ranges = [(p.metadata["row_start"], p.metadata["row_end"]) for p in tables]
        assert ranges[0][0] == 1
        for (s, e) in ranges:
            assert s <= e
        for prev, nxt in zip(ranges, ranges[1:]):
            assert nxt[0] == prev[1] + 1
        assert ranges[-1][1] == n_rows

    def test_each_chunk_contains_only_its_own_rows(self):
        md = self._build_md(6)
        splitter = TableAwareSplitter(chunk_size=30, size_unit="char")
        tables = _table_pieces(splitter.split(md))
        # The union of data values across chunks equals the full row set.
        seen: list[str] = []
        for piece in tables:
            for line in piece.text.splitlines():
                if line.startswith("| ") and line not in ("| n |", "| - |"):
                    seen.append(line.strip())
        expected = [f"| {i} |" for i in range(6)]
        assert sorted(seen) == sorted(expected)


@pytest.mark.unit
class TestMixedText:
    """Tables interleaved with ordinary prose paragraphs."""

    MD = (
        "Intro paragraph that is not a table at all.\n"
        "\n"
        "| Name | Age |\n"
        "| --- | --- |\n"
        "| Alice | 30 |\n"
        "\n"
        "Closing remarks after the table.\n"
    )

    def test_prose_falls_back_with_empty_metadata(self):
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        pieces = splitter.split(self.MD)

        prose = _prose_pieces(pieces)
        assert prose, "expected at least one non-table fallback piece"
        for piece in prose:
            assert piece.metadata == {}
        joined_prose = "\n".join(p.text for p in prose)
        assert "Intro paragraph" in joined_prose
        assert "Closing remarks" in joined_prose

    def test_table_part_marked_is_table(self):
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        tables = _table_pieces(splitter.split(self.MD))
        assert len(tables) == 1
        assert tables[0].text.startswith("| Name | Age |")
        assert tables[0].metadata["is_table"] is True
        # No sheet heading -> sheet_name is None.
        assert tables[0].metadata["sheet_name"] is None


@pytest.mark.unit
class TestSmallTable:

    def test_small_table_is_single_chunk(self):
        md = (
            "| a | b |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
            "| 3 | 4 |\n"
        )
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        tables = _table_pieces(splitter.split(md))
        assert len(tables) == 1
        assert tables[0].text.startswith("| a | b |")
        assert (tables[0].metadata["row_start"], tables[0].metadata["row_end"]) == (1, 2)

    def test_alignment_separator_variants_recognized(self):
        md = (
            "| a | b |\n"
            "|:---|---:|\n"
            "| 1 | 2 |\n"
        )
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        tables = _table_pieces(splitter.split(md))
        assert len(tables) == 1
        assert tables[0].metadata["is_table"] is True


@pytest.mark.unit
class TestSplitTextDerived:

    def test_split_text_returns_plain_strings(self):
        md = (
            "| a | b |\n"
            "| --- | --- |\n"
            "| 1 | 2 |\n"
        )
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        chunks = splitter.split_text(md)
        assert all(isinstance(c, str) for c in chunks)
        assert any(c.startswith("| a | b |") for c in chunks)

    def test_empty_input_returns_empty(self):
        splitter = TableAwareSplitter(chunk_size=10_000, size_unit="char")
        assert splitter.split("") == []
        assert splitter.split("   \n  ") == []

    def test_splitter_type(self):
        assert TableAwareSplitter().splitter_type == "table_aware"
