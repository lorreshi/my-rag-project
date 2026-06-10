"""Unit tests for DocumentManager (G2)."""
from __future__ import annotations

from src.ingestion.document_manager import (
    DocumentManager,
    DocumentInfo,
    DeleteResult,
)
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeStore(BaseVectorStore):
    """In-memory store keyed by id, supporting metadata get/delete."""

    def __init__(self, records=None):
        # records: list of {"id","text","metadata"}
        self._records = records or []

    def upsert(self, records, trace=None):
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        return []

    def get_by_ids(self, ids):
        return [r for r in self._records if r["id"] in ids]

    def get_by_metadata(self, filter):
        if not filter:
            return list(self._records)
        return [
            r for r in self._records
            if all(r["metadata"].get(k) == v for k, v in filter.items())
        ]

    def delete_by_metadata(self, filter, trace=None):
        before = len(self._records)
        self._records = [
            r for r in self._records
            if not all(r["metadata"].get(k) == v for k, v in filter.items())
        ]
        return before - len(self._records)

    @property
    def backend_name(self):
        return "fake"


class FakeBM25:
    def __init__(self):
        self.removed = []

    def remove_document(self, source):
        self.removed.append(source)


class FakeImages:
    def __init__(self):
        self.deleted = []

    def delete_by_doc_hash(self, doc_hash):
        self.deleted.append(doc_hash)
        return 2


class FakeIntegrity:
    def __init__(self):
        self.removed = []

    def compute_sha256(self, path):
        return "hash_" + path

    def remove_record(self, file_hash):
        self.removed.append(file_hash)


def _records():
    return [
        {"id": "a0", "text": "x", "metadata": {"source_path": "a.pdf", "collection": "c1"}},
        {"id": "a1", "text": "y", "metadata": {"source_path": "a.pdf", "collection": "c1", "image_refs": ["i1"]}},
        {"id": "b0", "text": "z", "metadata": {"source_path": "b.pdf", "collection": "c2"}},
    ]


class TestListDocuments:
    def test_groups_by_source(self):
        dm = DocumentManager(FakeStore(_records()))
        docs = dm.list_documents()
        sources = {d.source_path for d in docs}
        assert sources == {"a.pdf", "b.pdf"}

    def test_chunk_counts(self):
        dm = DocumentManager(FakeStore(_records()))
        docs = {d.source_path: d for d in dm.list_documents()}
        assert docs["a.pdf"].chunk_count == 2
        assert docs["b.pdf"].chunk_count == 1

    def test_image_counts(self):
        dm = DocumentManager(FakeStore(_records()))
        docs = {d.source_path: d for d in dm.list_documents()}
        assert docs["a.pdf"].image_count == 1

    def test_collection_filter(self):
        dm = DocumentManager(FakeStore(_records()))
        docs = dm.list_documents(collection="c2")
        assert [d.source_path for d in docs] == ["b.pdf"]

    def test_returns_document_info(self):
        dm = DocumentManager(FakeStore(_records()))
        docs = dm.list_documents()
        assert all(isinstance(d, DocumentInfo) for d in docs)

    def test_empty_store(self):
        dm = DocumentManager(FakeStore([]))
        assert dm.list_documents() == []


class TestDocumentDetail:
    def test_returns_chunks(self):
        dm = DocumentManager(FakeStore(_records()))
        detail = dm.get_document_detail("a.pdf")
        assert detail.chunk_count == 2
        assert detail.source_path == "a.pdf"


class TestDelete:
    def test_deletes_chunks(self):
        store = FakeStore(_records())
        dm = DocumentManager(store)
        result = dm.delete_document("a.pdf")
        assert result.chunks_deleted == 2
        # subsequent list excludes deleted doc
        assert "a.pdf" not in {d.source_path for d in dm.list_documents()}

    def test_coordinates_all_stores(self, tmp_path):
        # create a real file so integrity hashing works
        f = tmp_path / "a.pdf"
        f.write_bytes(b"%PDF fake")
        store = FakeStore([
            {"id": "a0", "text": "x", "metadata": {"source_path": str(f), "collection": "c1"}},
        ])
        bm25, images, integrity = FakeBM25(), FakeImages(), FakeIntegrity()
        dm = DocumentManager(store, bm25, images, integrity)
        result = dm.delete_document(str(f))
        assert result.chunks_deleted == 1
        assert result.bm25_removed is True
        assert result.images_deleted == 2
        assert result.integrity_removed is True
        assert bm25.removed == [str(f)]

    def test_returns_delete_result(self):
        dm = DocumentManager(FakeStore(_records()))
        assert isinstance(dm.delete_document("a.pdf"), DeleteResult)

    def test_delete_missing_file_no_crash(self):
        store = FakeStore(_records())
        dm = DocumentManager(store, FakeBM25(), FakeImages(), FakeIntegrity())
        # b.pdf doesn't exist on disk -> integrity hashing returns "", images skip
        result = dm.delete_document("b.pdf")
        assert result.chunks_deleted == 1


class TestStats:
    def test_collection_stats(self):
        dm = DocumentManager(FakeStore(_records()))
        stats = dm.get_collection_stats()
        assert stats.document_count == 2
        assert stats.chunk_count == 3

    def test_collection_stats_filtered(self):
        dm = DocumentManager(FakeStore(_records()))
        stats = dm.get_collection_stats(collection="c1")
        assert stats.document_count == 1
        assert stats.chunk_count == 2
