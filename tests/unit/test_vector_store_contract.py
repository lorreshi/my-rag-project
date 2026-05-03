"""Contract tests for VectorStore abstract interface and factory (B4)."""

import pytest

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
)
from src.libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
    register_backend,
    _REGISTRY,
)
from src.core.settings import Settings, VectorStoreConfig


class InMemoryVectorStore(BaseVectorStore):
    """Minimal in-memory implementation for contract testing."""

    def __init__(self):
        self._store: dict[str, VectorRecord] = {}

    def upsert(self, records, trace=None):
        for r in records:
            self._store[r.id] = r
        return len(records)

    def query(self, vector, top_k=10, filters=None, trace=None):
        # Return all records sorted by id (deterministic), limited to top_k
        results = [
            QueryResult(id=r.id, score=1.0, text=r.text, metadata=r.metadata)
            for r in sorted(self._store.values(), key=lambda r: r.id)
        ]
        return results[:top_k]

    def delete_by_metadata(self, filter, trace=None):
        to_delete = [
            rid for rid, r in self._store.items()
            if all(r.metadata.get(k) == v for k, v in filter.items())
        ]
        for rid in to_delete:
            del self._store[rid]
        return len(to_delete)

    @property
    def backend_name(self) -> str:
        return "in_memory"


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(saved)


@pytest.mark.unit
class TestBaseVectorStoreInterface:

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseVectorStore()

    def test_upsert_returns_count(self):
        store = InMemoryVectorStore()
        count = store.upsert([
            VectorRecord(id="1", vector=[0.1, 0.2], text="hello"),
            VectorRecord(id="2", vector=[0.3, 0.4], text="world"),
        ])
        assert count == 2

    def test_query_returns_query_results(self):
        store = InMemoryVectorStore()
        store.upsert([VectorRecord(id="a", vector=[1.0], text="doc a")])
        results = store.query(vector=[1.0], top_k=5)
        assert len(results) == 1
        assert isinstance(results[0], QueryResult)
        assert results[0].id == "a"
        assert results[0].text == "doc a"

    def test_query_top_k_limits(self):
        store = InMemoryVectorStore()
        store.upsert([VectorRecord(id=str(i), vector=[0.0]) for i in range(10)])
        results = store.query(vector=[0.0], top_k=3)
        assert len(results) == 3

    def test_upsert_idempotent(self):
        store = InMemoryVectorStore()
        store.upsert([VectorRecord(id="x", vector=[1.0], text="v1")])
        store.upsert([VectorRecord(id="x", vector=[1.0], text="v2")])
        results = store.query(vector=[0.0], top_k=10)
        assert len(results) == 1
        assert results[0].text == "v2"

    def test_delete_by_metadata(self):
        store = InMemoryVectorStore()
        store.upsert([
            VectorRecord(id="1", vector=[0.0], metadata={"source": "a.pdf"}),
            VectorRecord(id="2", vector=[0.0], metadata={"source": "b.pdf"}),
        ])
        deleted = store.delete_by_metadata({"source": "a.pdf"})
        assert deleted == 1
        assert len(store.query(vector=[0.0], top_k=10)) == 1

    def test_metadata_preserved(self):
        store = InMemoryVectorStore()
        store.upsert([VectorRecord(id="m", vector=[0.0], metadata={"k": "v"})])
        results = store.query(vector=[0.0])
        assert results[0].metadata == {"k": "v"}


@pytest.mark.unit
class TestVectorStoreFactory:

    def test_create_registered(self):
        register_backend("in_memory", lambda s: InMemoryVectorStore())
        store = VectorStoreFactory.create(
            Settings(vector_store=VectorStoreConfig(backend="in_memory"))
        )
        assert isinstance(store, InMemoryVectorStore)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store backend 'nope'"):
            VectorStoreFactory.create(
                Settings(vector_store=VectorStoreConfig(backend="nope"))
            )
