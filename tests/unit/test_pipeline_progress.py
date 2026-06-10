"""Unit tests for IngestionPipeline on_progress callback (F5)."""
from __future__ import annotations

import pytest

from src.core.settings import Settings
from src.core.types import Document, Chunk
from src.ingestion.pipeline import IngestionPipeline, IngestionResult


# --- Minimal fakes for a fully in-memory, network-free pipeline -------------

class FakeLoader:
    def load(self, path):
        return Document(id="doc1", text="hello world content", metadata={"source_path": path})


class FakeChunker:
    def split_document(self, document):
        return [
            Chunk(id="c0", text="hello", metadata={"source_path": document.source_path, "chunk_index": 0}, source_ref=document.id),
            Chunk(id="c1", text="world", metadata={"source_path": document.source_path, "chunk_index": 1}, source_ref=document.id),
        ]


class FakeTransform:
    def __init__(self, name="t"):
        self._name = name

    @property
    def name(self):
        return self._name

    def transform(self, chunks, trace=None):
        return chunks


class _Encoded:
    def __init__(self, chunk):
        self.chunk = chunk
        self.dense_vector = [0.1, 0.2]
        self.sparse_vector = None


class FakeBatch:
    def process(self, chunks, trace=None):
        return [_Encoded(c) for c in chunks]


class FakeUpserter:
    def upsert(self, chunks, dense_vectors, trace=None):
        return [c.id for c in chunks]


class FakeBM25:
    def add_documents(self, svs):
        pass

    def save(self):
        pass


def _pipeline():
    return IngestionPipeline(
        loader=FakeLoader(),
        chunker=FakeChunker(),
        transforms=[FakeTransform("refiner"), FakeTransform("enricher")],
        batch_processor=FakeBatch(),
        vector_upserter=FakeUpserter(),
        bm25_indexer=FakeBM25(),
        integrity_checker=None,
        image_storage=None,
    )


class TestProgressCallback:
    def test_callback_invoked_for_each_stage(self):
        seen = []
        _pipeline().run("doc.pdf", on_progress=lambda s, c, t: seen.append(s))
        for stage in ("load", "split", "transform", "encode", "store"):
            assert stage in seen

    def test_callback_args_are_ints(self):
        records = []
        _pipeline().run("doc.pdf", on_progress=lambda s, c, t: records.append((s, c, t)))
        for stage, cur, total in records:
            assert isinstance(stage, str)
            assert isinstance(cur, int)
            assert isinstance(total, int)

    def test_stage_completes_current_equals_total(self):
        records = []
        _pipeline().run("doc.pdf", on_progress=lambda s, c, t: records.append((s, c, t)))
        # For load: should see (load, 0, 1) and (load, 1, 1)
        load_events = [(c, t) for s, c, t in records if s == "load"]
        assert (0, 1) in load_events
        assert (1, 1) in load_events

    def test_transform_progress_tracks_count(self):
        records = []
        _pipeline().run("doc.pdf", on_progress=lambda s, c, t: records.append((s, c, t)))
        transform_events = [(c, t) for s, c, t in records if s == "transform"]
        # 2 transforms -> total should be 2, final event (2, 2)
        assert (2, 2) in transform_events

    def test_none_callback_no_effect(self):
        # Should run cleanly with no callback
        result = _pipeline().run("doc.pdf", on_progress=None)
        assert isinstance(result, IngestionResult)
        assert result.total_chunks == 2

    def test_callback_not_called_when_none(self):
        # Ensure passing None doesn't raise and result is valid
        result = _pipeline().run("doc.pdf")
        assert not result.skipped
        assert len(result.vector_ids) == 2


class TestProgressOrdering:
    def test_stage_order(self):
        order = []
        _pipeline().run("doc.pdf", on_progress=lambda s, c, t: order.append(s))
        # First occurrence index of each stage should be increasing
        first = {}
        for i, s in enumerate(order):
            first.setdefault(s, i)
        assert first["load"] < first["split"] < first["transform"]
        assert first["transform"] < first["encode"] < first["store"]
