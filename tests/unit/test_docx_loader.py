"""Tests for DocxLoader (T13).

Validates: Requirements 1.3, 1.4, 1.6
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import src.libs.loader  # noqa: F401  (ensures built-in loaders are registered)
from src.core.types import Document
from src.libs.loader.docx_loader import DocxLoader
from src.libs.loader.loader_factory import LoaderFactory


def _write_docx(tmp_path: Path, name: str = "doc.docx") -> Path:
    # Content is arbitrary — MarkItDown is mocked in these tests.
    p = tmp_path / name
    p.write_bytes(b"PK\x03\x04 fake docx bytes")
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
        raise RuntimeError("boom: conversion failed")


@pytest.mark.unit
class TestDocxLoader:

    def test_converts_docx_to_document(self, tmp_path: Path):
        docx = _write_docx(tmp_path)
        markdown = "# Heading\n\nSome converted **markdown** body.\n"
        fake = _FakeMarkItDown(markdown)
        loader = DocxLoader(markitdown=fake)

        doc = loader.load(str(docx))

        assert isinstance(doc, Document)
        assert doc.text == markdown
        assert doc.metadata["source_path"] == str(docx)
        assert doc.metadata["doc_type"] == "docx"
        assert doc.metadata["doc_hash"]
        assert doc.metadata["images"] == []
        assert doc.images == []
        assert doc.id.startswith("docx_")
        assert doc.id == f"docx_{doc.metadata['doc_hash'][:12]}"
        assert fake.calls == [str(docx)]

    def test_none_text_content_degrades_to_empty(self, tmp_path: Path):
        docx = _write_docx(tmp_path)
        fake = _FakeMarkItDown(markdown=None)  # type: ignore[arg-type]
        loader = DocxLoader(markitdown=fake)

        doc = loader.load(str(docx))
        assert doc.text == ""

    def test_markitdown_failure_degrades_without_raising(
        self, tmp_path: Path, caplog
    ):
        docx = _write_docx(tmp_path)
        loader = DocxLoader(markitdown=_RaisingMarkItDown())

        with caplog.at_level("WARNING"):
            doc = loader.load(str(docx))

        # Degraded Document: empty text but complete metadata, no fatal error.
        assert doc.text == ""
        assert doc.metadata["source_path"] == str(docx)
        assert doc.metadata["doc_type"] == "docx"
        assert doc.metadata["doc_hash"]
        assert doc.metadata["images"] == []
        assert doc.id == f"docx_{doc.metadata['doc_hash'][:12]}"
        assert any(
            "failed" in rec.getMessage().lower() for rec in caplog.records
        )

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path):
        txt = tmp_path / "doc.txt"
        txt.write_text("hello", encoding="utf-8")
        loader = DocxLoader(markitdown=_FakeMarkItDown("x"))
        with pytest.raises(ValueError):
            loader.load(str(txt))

    def test_missing_file_raises_file_not_found(self, tmp_path: Path):
        loader = DocxLoader(markitdown=_FakeMarkItDown("x"))
        with pytest.raises(FileNotFoundError):
            loader.load(str(tmp_path / "nope.docx"))

    def test_supported_extensions(self):
        loader = DocxLoader(markitdown=_FakeMarkItDown("x"))
        assert loader.supported_extensions == [".docx"]

    def test_factory_routes_docx_to_docx_loader(self):
        assert isinstance(LoaderFactory.create("x.docx"), DocxLoader)
