"""Tests for DocumentChunker (C4)."""
from __future__ import annotations

import pytest

from src.core.types import Document, Chunk, ImageRef
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.libs.splitter.base_splitter import BaseSplitter
from src.libs.splitter.splitter_factory import register_splitter, _REGISTRY
from src.core.settings import Settings


class FakeSplitter(BaseSplitter):
    """Splits by double newline for predictable testing."""

    def split_text(self, text, trace=None):
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    @property
    def splitter_type(self) -> str:
        return "fake"


@pytest.fixture(autouse=True)
def _register_fake():
    saved = dict(_REGISTRY)
    register_splitter("fake", lambda s: FakeSplitter())
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


def _make_doc(text: str = "Para one.\n\nPara two.\n\nPara three.",
              images: list | None = None) -> Document:
    meta = {"source_path": "test.pdf", "doc_type": "pdf", "doc_hash": "abc123"}
    if images is not None:
        meta["images"] = images
    return Document(id="doc_test", text=text, metadata=meta)


@pytest.mark.unit
class TestDocumentChunker:

    def test_basic_split(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        doc = _make_doc()
        chunks = chunker.split_document(doc)
        assert len(chunks) == 3
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_unique(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        chunks = chunker.split_document(_make_doc())
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_deterministic(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        doc = _make_doc()
        chunks1 = chunker.split_document(doc)
        chunks2 = chunker.split_document(doc)
        assert [c.id for c in chunks1] == [c.id for c in chunks2]

    def test_chunk_id_format(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        chunks = chunker.split_document(_make_doc())
        # Format: {doc_id}_{index:04d}_{hash_8chars}
        assert chunks[0].id.startswith("doc_test_0000_")
        assert chunks[1].id.startswith("doc_test_0001_")
        assert len(chunks[0].id.split("_")[-1]) == 8

    def test_metadata_inherited(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        chunks = chunker.split_document(_make_doc())
        for c in chunks:
            assert c.metadata["source_path"] == "test.pdf"
            assert c.metadata["doc_type"] == "pdf"
            assert c.metadata["doc_hash"] == "abc123"

    def test_chunk_index_added(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        chunks = chunker.split_document(_make_doc())
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    def test_source_ref(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        chunks = chunker.split_document(_make_doc())
        for c in chunks:
            assert c.source_ref == "doc_test"

    def test_no_images_no_image_field(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        doc = _make_doc("No images here.\n\nJust text.")
        chunks = chunker.split_document(doc)
        for c in chunks:
            assert "images" not in c.metadata
            assert "image_refs" not in c.metadata

    def test_image_refs_distributed(self):
        images = [
            {"id": "img1", "path": "/tmp/img1.png", "page": 0,
             "text_offset": 0, "text_length": 14},
        ]
        text = "[IMAGE: img1] description.\n\nNo image here."
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        doc = _make_doc(text, images=images)
        chunks = chunker.split_document(doc)

        # First chunk has the image
        assert "images" in chunks[0].metadata
        assert chunks[0].metadata["image_refs"] == ["img1"]
        assert len(chunks[0].metadata["images"]) == 1

        # Second chunk has no image
        assert "images" not in chunks[1].metadata

    def test_multiple_images_in_one_chunk(self):
        images = [
            {"id": "img1", "path": "/tmp/img1.png", "page": 0,
             "text_offset": 0, "text_length": 14},
            {"id": "img2", "path": "/tmp/img2.png", "page": 1,
             "text_offset": 20, "text_length": 14},
        ]
        text = "[IMAGE: img1] and [IMAGE: img2] together."
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        doc = _make_doc(text, images=images)
        chunks = chunker.split_document(doc)
        assert chunks[0].metadata["image_refs"] == ["img1", "img2"]
        assert len(chunks[0].metadata["images"]) == 2

    def test_empty_document(self):
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        doc = Document(id="empty", text="", metadata={"source_path": "x"})
        chunks = chunker.split_document(doc)
        assert chunks == []

    def test_serializable(self):
        import json
        chunker = DocumentChunker(Settings(), splitter_type="fake")
        chunks = chunker.split_document(_make_doc())
        for c in chunks:
            json.dumps(c.to_dict())  # Should not raise
