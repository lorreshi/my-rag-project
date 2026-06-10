"""Unit tests for DataService (G3)."""
from __future__ import annotations

from src.ingestion.document_manager import DocumentManager
from src.observability.dashboard.services.data_service import DataService
from src.libs.vector_store.base_vector_store import BaseVectorStore


class FakeStore(BaseVectorStore):
    def __init__(self, records):
        self._records = records

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
        return 0

    @property
    def backend_name(self):
        return "fake"


class FakeImages:
    def __init__(self, mapping):
        self._mapping = mapping

    def get_path(self, image_id):
        return self._mapping.get(image_id)


def _records():
    return [
        {"id": "a0", "text": "alpha", "metadata": {"source_path": "a.pdf", "collection": "c1", "chunk_index": 0}},
        {"id": "a1", "text": "beta", "metadata": {"source_path": "a.pdf", "collection": "c1", "chunk_index": 1, "image_refs": ["img1"]}},
        {"id": "b0", "text": "gamma", "metadata": {"source_path": "b.pdf", "collection": "c2", "chunk_index": 0}},
    ]


def _service(images=None):
    store = FakeStore(_records())
    dm = DocumentManager(store, image_storage=images)
    return DataService(dm, image_storage=images)


class TestListing:
    def test_list_documents(self):
        docs = _service().list_documents()
        assert len(docs) == 2
        assert all("source_path" in d for d in docs)

    def test_list_documents_collection_filter(self):
        docs = _service().list_documents(collection="c2")
        assert [d["source_path"] for d in docs] == ["b.pdf"]

    def test_list_collections(self):
        assert _service().list_collections() == ["c1", "c2"]


class TestChunks:
    def test_get_chunks(self):
        chunks = _service().get_chunks("a.pdf")
        assert len(chunks) == 2

    def test_get_chunks_missing(self):
        assert _service().get_chunks("missing.pdf") == []


class TestImages:
    def test_get_image_path(self):
        svc = _service(images=FakeImages({"img1": "/data/img1.png"}))
        assert svc.get_image_path("img1") == "/data/img1.png"

    def test_get_image_path_no_storage(self):
        assert _service().get_image_path("img1") is None

    def test_chunk_images_resolves(self):
        svc = _service(images=FakeImages({"img1": "/data/img1.png"}))
        chunk = {"metadata": {"image_refs": ["img1"]}}
        entries = svc.chunk_images(chunk)
        assert entries == [{"id": "img1", "path": "/data/img1.png"}]

    def test_chunk_images_no_refs(self):
        svc = _service(images=FakeImages({}))
        assert svc.chunk_images({"metadata": {}}) == []


class TestPageImport:
    def test_data_browser_render_callable(self):
        from src.observability.dashboard.pages import data_browser
        assert callable(data_browser.render)
