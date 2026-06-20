"""Tests for DocumentChunker doc_type routing (T10, Property 14).

Validates: Requirements 5.1, 5.6

The chunker selects a splitter based on ``document.metadata["doc_type"]``:
- a ``doc_type`` configured in ``settings.splitter.by_doc_type`` routes to the
  dedicated splitter (e.g. ``xlsx`` -> ``table_aware``);
- any other (or missing) ``doc_type`` falls back to the default recursive
  splitter, whose prose pieces carry empty structured metadata;
- with no ``by_doc_type`` configured, everything uses the default splitter
  (behavior identical to before this task).
"""
from __future__ import annotations

import pytest

# Importing the module self-registers the "table_aware" splitter with the
# SplitterFactory (recursive is registered via the splitter package import).
import src.libs.splitter.table_aware_splitter  # noqa: F401
import src.libs.splitter  # noqa: F401  (registers "recursive")

from src.core.settings import Settings, SplitterConfig
from src.core.types import Document
from src.ingestion.chunking.document_chunker import DocumentChunker


# A small Markdown table under an H2 sheet heading, plus a trailing prose line.
_TABLE_TEXT = """## Summary
| Name | Score |
| --- | --- |
| Alice | 90 |
| Bob | 85 |
"""

_PROSE_TEXT = "First paragraph of prose.\n\nSecond paragraph of prose."


def _settings(by_doc_type: dict[str, str] | None = None) -> Settings:
    """Build Settings with a char-unit splitter and optional routing map."""
    return Settings(
        splitter=SplitterConfig(
            type="recursive",
            size_unit="char",
            chunk_size=512,
            chunk_overlap=0,
            by_doc_type=by_doc_type or {},
        )
    )


def _doc(doc_type: str | None, text: str) -> Document:
    meta = {"source_path": f"sample.{doc_type or 'bin'}", "doc_hash": "h"}
    if doc_type is not None:
        meta["doc_type"] = doc_type
    return Document(id=f"doc_{doc_type or 'none'}", text=text, metadata=meta)


@pytest.mark.unit
class TestDocTypeRouting:
    """Property 14: route by doc_type; default recursive otherwise."""

    def test_xlsx_routes_to_table_aware(self):
        """doc_type=xlsx selects table_aware -> chunks carry table metadata."""
        chunker = DocumentChunker(_settings({"xlsx": "table_aware"}))
        chunks = chunker.split_document(_doc("xlsx", _TABLE_TEXT))

        # At least one chunk must be a structured table chunk.
        table_chunks = [c for c in chunks if c.metadata.get("is_table")]
        assert table_chunks, "expected table_aware to produce table chunks"
        tc = table_chunks[0]
        assert tc.metadata["is_table"] is True
        assert tc.metadata["sheet_name"] == "Summary"
        assert "row_start" in tc.metadata and "row_end" in tc.metadata

    def test_unconfigured_doc_type_uses_recursive(self):
        """doc_type=docx is NOT in by_doc_type -> default recursive, no table meta."""
        chunker = DocumentChunker(_settings({"xlsx": "table_aware"}))
        chunks = chunker.split_document(_doc("docx", _PROSE_TEXT))

        assert chunks, "expected recursive to produce chunks"
        for c in chunks:
            # Prose (recursive) pieces carry no structured table metadata.
            assert "is_table" not in c.metadata
            assert "sheet_name" not in c.metadata

    def test_pdf_and_markdown_use_recursive(self):
        """Other configured-but-unmatched doc_types fall back to recursive."""
        chunker = DocumentChunker(_settings({"xlsx": "table_aware"}))
        for doc_type in ("pdf", "markdown"):
            chunks = chunker.split_document(_doc(doc_type, _PROSE_TEXT))
            assert chunks
            for c in chunks:
                assert "is_table" not in c.metadata
                assert "sheet_name" not in c.metadata

    def test_missing_doc_type_uses_recursive(self):
        """A Document without a doc_type falls back to the default splitter."""
        chunker = DocumentChunker(_settings({"xlsx": "table_aware"}))
        chunks = chunker.split_document(_doc(None, _PROSE_TEXT))
        assert chunks
        for c in chunks:
            assert "is_table" not in c.metadata

    def test_no_by_doc_type_everything_recursive(self):
        """With no routing configured, even xlsx uses the default recursive."""
        chunker = DocumentChunker(_settings())  # empty by_doc_type
        chunks = chunker.split_document(_doc("xlsx", _TABLE_TEXT))
        assert chunks
        for c in chunks:
            # No dedicated splitter selected -> no table structure metadata.
            assert "is_table" not in c.metadata
            assert "sheet_name" not in c.metadata

    def test_routing_hit_and_miss_in_same_chunker(self):
        """Same chunker routes xlsx->table_aware and pdf->recursive (Property 14)."""
        chunker = DocumentChunker(_settings({"xlsx": "table_aware"}))

        # Hit: xlsx -> table_aware (structured metadata present)
        xlsx_chunks = chunker.split_document(_doc("xlsx", _TABLE_TEXT))
        assert any(c.metadata.get("is_table") for c in xlsx_chunks)

        # Miss: pdf -> recursive (no structured metadata)
        pdf_chunks = chunker.split_document(_doc("pdf", _PROSE_TEXT))
        assert all("is_table" not in c.metadata for c in pdf_chunks)
