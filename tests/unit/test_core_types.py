"""Tests for core data types / contracts (C1)."""
from __future__ import annotations

import json
import pytest

from src.core.types import Document, Chunk, ChunkRecord, ImageRef


@pytest.mark.unit
class TestImageRef:

    def test_to_dict_roundtrip(self):
        ref = ImageRef(id="abc_1_0", path="data/images/col/abc_1_0.png", page=1,
                       text_offset=50, text_length=18, position={"x": 0, "y": 100})
        d = ref.to_dict()
        restored = ImageRef.from_dict(d)
        assert restored.id == "abc_1_0"
        assert restored.path == "data/images/col/abc_1_0.png"
        assert restored.page == 1
        assert restored.text_offset == 50
        assert restored.text_length == 18
        assert restored.position == {"x": 0, "y": 100}

    def test_from_dict_defaults(self):
        ref = ImageRef.from_dict({"id": "img1"})
        assert ref.path == ""
        assert ref.page == 0
        assert ref.text_offset == 0

    def test_json_serializable(self):
        ref = ImageRef(id="x", path="/tmp/x.png")
        s = json.dumps(ref.to_dict())
        assert "x" in s


@pytest.mark.unit
class TestDocument:

    def test_basic_creation(self):
        doc = Document(id="doc1", text="Hello world", metadata={"source_path": "test.pdf"})
        assert doc.id == "doc1"
        assert doc.source_path == "test.pdf"

    def test_source_path_required_in_metadata(self):
        doc = Document(id="d", text="", metadata={})
        assert doc.source_path == ""  # empty but doesn't crash

    def test_to_dict_roundtrip(self):
        doc = Document(id="d1", text="content", metadata={"source_path": "a.pdf", "page": 3})
        d = doc.to_dict()
        restored = Document.from_dict(d)
        assert restored.id == "d1"
        assert restored.text == "content"
        assert restored.metadata["source_path"] == "a.pdf"

    def test_images_property(self):
        img_data = [{"id": "img1", "path": "/tmp/img1.png", "page": 2,
                     "text_offset": 10, "text_length": 15}]
        doc = Document(id="d", text="text [IMAGE: img1] more",
                       metadata={"source_path": "x.pdf", "images": img_data})
        imgs = doc.images
        assert len(imgs) == 1
        assert isinstance(imgs[0], ImageRef)
        assert imgs[0].id == "img1"
        assert imgs[0].text_offset == 10

    def test_images_empty(self):
        doc = Document(id="d", text="no images", metadata={"source_path": "x.pdf"})
        assert doc.images == []

    def test_json_serializable(self):
        doc = Document(id="d", text="hi", metadata={"source_path": "f.pdf"})
        s = json.dumps(doc.to_dict())
        assert "source_path" in s


@pytest.mark.unit
class TestChunk:

    def test_basic_creation(self):
        chunk = Chunk(id="d_0001_abc", text="chunk text",
                      metadata={"source_path": "a.pdf"},
                      start_offset=0, end_offset=100, source_ref="doc1")
        assert chunk.id == "d_0001_abc"
        assert chunk.source_ref == "doc1"

    def test_to_dict_roundtrip(self):
        chunk = Chunk(id="c1", text="hello", metadata={"k": "v"},
                      start_offset=5, end_offset=10, source_ref="d1")
        d = chunk.to_dict()
        restored = Chunk.from_dict(d)
        assert restored.id == "c1"
        assert restored.start_offset == 5
        assert restored.end_offset == 10
        assert restored.source_ref == "d1"

    def test_image_refs_property(self):
        chunk = Chunk(id="c", text="", metadata={"image_refs": ["img1", "img2"]})
        assert chunk.image_refs == ["img1", "img2"]

    def test_image_refs_empty(self):
        chunk = Chunk(id="c", text="", metadata={})
        assert chunk.image_refs == []

    def test_json_serializable(self):
        chunk = Chunk(id="c", text="data", metadata={"source_path": "x"})
        s = json.dumps(chunk.to_dict())
        assert "source_path" in s


@pytest.mark.unit
class TestChunkRecord:

    def test_basic_creation(self):
        rec = ChunkRecord(id="r1", text="text",
                          dense_vector=[0.1, 0.2, 0.3],
                          sparse_vector={"hello": 1.5, "world": 0.8})
        assert rec.dense_vector == [0.1, 0.2, 0.3]
        assert rec.sparse_vector["hello"] == 1.5

    def test_to_dict_roundtrip(self):
        rec = ChunkRecord(id="r1", text="t", metadata={"k": "v"},
                          dense_vector=[1.0], sparse_vector={"a": 0.5})
        d = rec.to_dict()
        restored = ChunkRecord.from_dict(d)
        assert restored.id == "r1"
        assert restored.dense_vector == [1.0]
        assert restored.sparse_vector == {"a": 0.5}

    def test_defaults_empty(self):
        rec = ChunkRecord(id="r", text="")
        assert rec.dense_vector == []
        assert rec.sparse_vector == {}
        assert rec.metadata == {}

    def test_json_serializable(self):
        rec = ChunkRecord(id="r", text="x", dense_vector=[0.1],
                          sparse_vector={"t": 1.0})
        s = json.dumps(rec.to_dict())
        assert "dense_vector" in s
        assert "sparse_vector" in s
