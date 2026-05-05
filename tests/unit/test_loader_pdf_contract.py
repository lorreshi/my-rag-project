"""Tests for BaseLoader and PdfLoader (C3)."""
from __future__ import annotations

import pytest
from pathlib import Path

from src.libs.loader.base_loader import BaseLoader
from src.libs.loader.pdf_loader import PdfLoader
from src.core.types import Document


@pytest.mark.unit
class TestBaseLoaderInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseLoader()


@pytest.mark.unit
class TestPdfLoader:

    @pytest.fixture
    def loader(self) -> PdfLoader:
        return PdfLoader()

    def test_load_simple_pdf(self, loader):
        doc = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert isinstance(doc, Document)
        assert doc.id.startswith("pdf_")
        assert len(doc.text) > 0
        assert "Hello" in doc.text

    def test_metadata_source_path(self, loader):
        doc = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert doc.source_path == "tests/fixtures/sample_documents/simple.pdf"

    def test_metadata_doc_type(self, loader):
        doc = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert doc.metadata["doc_type"] == "pdf"

    def test_metadata_doc_hash(self, loader):
        doc = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert "doc_hash" in doc.metadata
        assert len(doc.metadata["doc_hash"]) == 64  # SHA256 hex

    def test_metadata_images_field_exists(self, loader):
        doc = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert "images" in doc.metadata
        # Simple PDF has no images
        assert doc.metadata["images"] == []

    def test_deterministic_id(self, loader):
        doc1 = loader.load("tests/fixtures/sample_documents/simple.pdf")
        doc2 = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert doc1.id == doc2.id

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load("/nonexistent/file.pdf")

    def test_wrong_extension(self, loader, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a pdf")
        with pytest.raises(ValueError, match="only supports .pdf"):
            loader.load(str(txt_file))

    def test_supported_extensions(self, loader):
        assert loader.supported_extensions == [".pdf"]

    def test_images_property_empty(self, loader):
        doc = loader.load("tests/fixtures/sample_documents/simple.pdf")
        assert doc.images == []
