"""Unit tests for VectorUpserter — deterministic IDs and idempotent upsert."""
from __future__ import annotations

import pytest

from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeVectorStore(BaseVectorStore):
    """In-memory fake store keyed by id (mimics upsert semantics)."""

    def __init__(self):
        self.records: dict[str, dict] = {}
        self.upsert_calls = 0

    def upsert(self, records, trace=None):
        self.upsert_calls += 1
        for r in records:
            self.records[r.id] = {"vector": r.vector, "text": r.text, "metadata": r.metadata}
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def delete_by_metadata(self, filter, trace=None):
        return 0

    def get_by_ids(self, ids):
        return [
            {"id": rid, "text": self.records[rid]["text"], "metadata": self.records[rid]["metadata"]}
            for rid in ids if rid in self.records
        ]

    @property
    def backend_name(self):
        return "fake"


def _chunk(text: str, source_path: str = "doc.pdf", index: int = 0, cid: str = "c0") -> Chunk:
    return Chunk(
        id=cid,
        text=text,
        metadata={"source_path": source_path, "chunk_index": index},
        source_ref="doc",
    )


class TestVectorId:
    """The vector store id must equal chunk.id (shared id space with BM25)."""

    def test_vector_id_is_chunk_id(self):
        c = _chunk("hello world", "doc.pdf", 0, cid="pdf_abc_0000_deadbeef")
        assert VectorUpserter.make_vector_id(c) == "pdf_abc_0000_deadbeef"

    def test_distinct_chunk_ids_distinct_vector_ids(self):
        c1 = _chunk("hello world", "doc.pdf", 0, cid="pdf_abc_0000_11111111")
        c2 = _chunk("hello there", "doc.pdf", 1, cid="pdf_abc_0001_22222222")
        assert VectorUpserter.make_vector_id(c1) != VectorUpserter.make_vector_id(c2)

    def test_same_chunk_id_same_vector_id(self):
        c1 = _chunk("hello world", "doc.pdf", 0, cid="pdf_abc_0000_11111111")
        c2 = _chunk("hello world", "doc.pdf", 0, cid="pdf_abc_0000_11111111")
        assert VectorUpserter.make_vector_id(c1) == VectorUpserter.make_vector_id(c2)


class TestUpsert:
    def test_returns_ids_in_order(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        chunks = [_chunk("a", "d.pdf", 0, "c0"), _chunk("b", "d.pdf", 1, "c1")]
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        ids = up.upsert(chunks, vectors)
        assert len(ids) == 2
        assert ids[0] == VectorUpserter.make_vector_id(chunks[0])
        assert ids[1] == VectorUpserter.make_vector_id(chunks[1])

    def test_idempotent_no_duplicates(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        chunks = [_chunk("hello", "d.pdf", 0)]
        vectors = [[1.0, 2.0]]
        up.upsert(chunks, vectors)
        up.upsert(chunks, vectors)  # second time
        # same id -> single record in store
        assert len(store.records) == 1
        assert store.upsert_calls == 2

    def test_changed_content_creates_new_record(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        # Changed content -> DocumentChunker assigns a new chunk.id (content hash)
        up.upsert([_chunk("v1", "d.pdf", 0, cid="pdf_d_0000_v1hash00")], [[1.0]])
        up.upsert([_chunk("v2", "d.pdf", 0, cid="pdf_d_0000_v2hash00")], [[2.0]])
        # different chunk.id -> 2 records
        assert len(store.records) == 2

    def test_record_contains_text_and_metadata(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        chunk = _chunk("content here", "d.pdf", 0)
        chunk.metadata["title"] = "My Title"
        ids = up.upsert([chunk], [[0.5]])
        rec = store.records[ids[0]]
        assert rec["text"] == "content here"
        assert rec["metadata"]["title"] == "My Title"

    def test_batch_order_preserved(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        chunks = [_chunk(f"t{i}", "d.pdf", i, f"c{i}") for i in range(5)]
        vectors = [[float(i)] for i in range(5)]
        ids = up.upsert(chunks, vectors)
        for chunk, vid in zip(chunks, ids):
            assert store.records[vid]["text"] == chunk.text

    def test_empty(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        assert up.upsert([], []) == []

    def test_length_mismatch_raises(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        with pytest.raises(ValueError):
            up.upsert([_chunk("a")], [[1.0], [2.0]])

    def test_requires_store(self):
        with pytest.raises(ValueError):
            VectorUpserter(None)


class TestTrace:
    def test_trace_records_stage(self):
        store = FakeVectorStore()
        up = VectorUpserter(store)
        trace = TraceContext(trace_type="ingestion")
        up.upsert([_chunk("a", "d.pdf", 0)], [[1.0]], trace=trace)
        stages = {s.name: s for s in trace.stages}
        assert "vector_upserter" in stages
        assert stages["vector_upserter"].details["upserted"] == 1
