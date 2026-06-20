"""Tests for XlsxLoader (T14).

XlsxLoader converts .xlsx to canonical Markdown tables via MarkItDown, emitting
a ``## {sheet_name}`` H2 marker before each sheet's table (stable order) so the
downstream TableAwareSplitter can attribute table blocks to their sheet.

openpyxl/pandas (MarkItDown's xlsx backend) are not assumed to be installed, so
these tests mock MarkItDown to fix the conversion output contract.

Validates: Requirements 1.3, 1.4
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.libs.loader  # noqa: F401  (ensures built-in loaders are registered)
from src.core.types import Document
from src.libs.loader.loader_factory import LoaderFactory
from src.libs.loader.xlsx_loader import XlsxLoader


def _write_xlsx(tmp_path: Path, name: str = "book.xlsx") -> Path:
    # Content is arbitrary — MarkItDown is mocked in these tests.
    p = tmp_path / name
    p.write_bytes(b"PK\x03\x04 fake xlsx bytes")
    return p


class _FakeMarkItDown:
    """Returns a fixed Markdown string from convert()."""

    def __init__(self, markdown: str):
        self._markdown = markdown
        self.calls: list[str] = []

    def convert(self, path: str):
        self.calls.append(path)
        return SimpleNamespace(text_content=self._markdown)


class _RaisingMarkItDown:
    """Raises on convert() to exercise graceful degradation."""

    def convert(self, path: str):
        raise RuntimeError("boom: missing optional dependency")


# A single-sheet conversion as MarkItDown renders it.
_SINGLE_SHEET_MD = (
    "## Sheet1\n"
    "| name | age |\n"
    "| --- | --- |\n"
    "| alice | 30 |\n"
)

# A two-sheet conversion as MarkItDown renders it (stable order: Users, Orders).
_MULTI_SHEET_MD = (
    "## Users\n"
    "| name | age |\n"
    "| --- | --- |\n"
    "| alice | 30 |\n\n"
    "## Orders\n"
    "| id | total |\n"
    "| --- | --- |\n"
    "| 1 | 99 |\n"
)


@pytest.mark.unit
class TestXlsxLoader:

    def test_single_sheet_produces_heading_and_table(self, tmp_path: Path):
        xlsx = _write_xlsx(tmp_path)
        fake = _FakeMarkItDown(_SINGLE_SHEET_MD)
        loader = XlsxLoader(markitdown=fake)

        doc = loader.load(str(xlsx))

        assert isinstance(doc, Document)
        # Sheet marker present + Markdown table follows it.
        assert "## Sheet1" in doc.text
        assert "| name | age |" in doc.text
        assert "| --- | --- |" in doc.text
        # Metadata complete; doc_id prefixed xlsx_.
        assert doc.metadata["source_path"] == str(xlsx)
        assert doc.metadata["doc_type"] == "xlsx"
        assert doc.metadata["doc_hash"]
        assert doc.metadata["images"] == []
        assert doc.images == []
        assert doc.id.startswith("xlsx_")
        assert doc.id == f"xlsx_{doc.metadata['doc_hash'][:12]}"
        assert fake.calls == [str(xlsx)]

    def test_multi_sheet_headings_stable_order_with_tables(self, tmp_path: Path):
        xlsx = _write_xlsx(tmp_path)
        loader = XlsxLoader(markitdown=_FakeMarkItDown(_MULTI_SHEET_MD))

        doc = loader.load(str(xlsx))

        # Both sheet headings present.
        assert "## Users" in doc.text
        assert "## Orders" in doc.text
        # Order is stable (Users before Orders).
        assert doc.text.index("## Users") < doc.text.index("## Orders")
        # Each heading is followed by its corresponding table header.
        users_section = doc.text[doc.text.index("## Users"):doc.text.index("## Orders")]
        orders_section = doc.text[doc.text.index("## Orders"):]
        assert "| name | age |" in users_section
        assert "| id | total |" in orders_section

    def test_non_h2_sheet_heading_normalized_to_h2(self, tmp_path: Path):
        # A heading emitted at a different level is normalized to H2.
        md = "# Sheet1\n| a |\n| --- |\n| 1 |\n"
        xlsx = _write_xlsx(tmp_path)
        loader = XlsxLoader(markitdown=_FakeMarkItDown(md))

        doc = loader.load(str(xlsx))

        assert "## Sheet1" in doc.text
        # No leftover H1 heading line.
        assert "# Sheet1\n" not in doc.text.replace("## Sheet1", "")

    def test_markitdown_failure_degrades_without_raising(self, tmp_path: Path, caplog):
        xlsx = _write_xlsx(tmp_path)
        loader = XlsxLoader(markitdown=_RaisingMarkItDown())

        with caplog.at_level("WARNING"):
            doc = loader.load(str(xlsx))

        # Degraded Document: empty text but complete metadata, no fatal error.
        assert doc.text == ""
        assert doc.metadata["source_path"] == str(xlsx)
        assert doc.metadata["doc_type"] == "xlsx"
        assert doc.metadata["doc_hash"]
        assert doc.metadata["images"] == []
        assert doc.id == f"xlsx_{doc.metadata['doc_hash'][:12]}"
        assert any("failed" in rec.getMessage().lower() for rec in caplog.records)

    def test_none_text_content_degrades_to_empty(self, tmp_path: Path):
        xlsx = _write_xlsx(tmp_path)
        loader = XlsxLoader(markitdown=_FakeMarkItDown(markdown=None))  # type: ignore[arg-type]

        doc = loader.load(str(xlsx))
        assert doc.text == ""

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path):
        txt = tmp_path / "book.txt"
        txt.write_text("hello", encoding="utf-8")
        loader = XlsxLoader(markitdown=_FakeMarkItDown("x"))
        with pytest.raises(ValueError):
            loader.load(str(txt))

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        loader = XlsxLoader(markitdown=_FakeMarkItDown("x"))
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "nope.xlsx"))

    def test_supported_extensions(self):
        loader = XlsxLoader(markitdown=_FakeMarkItDown("x"))
        assert loader.supported_extensions == [".xlsx"]

    def test_factory_routes_xlsx_to_xlsx_loader(self):
        assert isinstance(LoaderFactory.create("x.xlsx"), XlsxLoader)
