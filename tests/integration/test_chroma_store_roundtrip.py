"""Integration tests for ChromaStore — real upsert→query roundtrip (B7.6).

Uses a temporary directory for Chroma persistence, cleaned up after each test.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from src.libs.vector_store.base_vector_store import VectorRecord, QueryResult
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.core.settings import Settings, VectorStoreConfig

# Trigger registration
import src.libs.vector_store.chroma_store as mod  # noqa: F401
from src.libs.vector_store.chroma_store import ChromaStore


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    """Create a ChromaStore backed by a temp directory."""
    return ChromaStore(
        persist_path=str(tmp_path / "chroma_test"),
        collection_name="test_collection",
    )


@pytest.mark.integration
class TestChromaStoreRoundtrip:

    def test_upsert_and_query(self, store: ChromaStore):
        records = [
            VectorRecord(id="doc1", vector=[1.0, 0.0, 0.0], text="hello world"),
            VectorRecord(id="doc2", vector=[0.0, 1.0, 0.0], text="foo bar"),
            VectorRecord(id="doc3", vector=[0.0, 0.0, 1.0], text="baz qux"),
        ]
        count = store.upsert(records)
        assert count == 3

        # Query with vector close to doc1
        results = store.query(vector=[0.9, 0.1, 0.0], top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], QueryResult)
        # doc1 should be the closest match
        assert results[0].id == "doc1"
        assert results[0].score > 0.5

    def test_top_k_limits_results(self, store: ChromaStore):
        records = [
            VectorRecord(id=f"d{i}", vector=[float(i == j) for j in range(5)], text=f"doc {i}")
            for i in range(5)
        ]
        store.upsert(records)
        results = store.query(vector=[1.0, 0.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_upsert_idempotent(self, store: ChromaStore):
        r = VectorRecord(id="x", vector=[1.0, 0.0], text="version1")
        store.upsert([r])
        r2 = VectorRecord(id="x", vector=[1.0, 0.0], text="version2")
        store.upsert([r2])
        results = store.query(vector=[1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].text == "version2"

    def test_metadata_preserved(self, store: ChromaStore):
        r = VectorRecord(
            id="m1", vector=[1.0, 0.0], text="meta test",
            metadata={"source": "test.pdf", "page": 3},
        )
        store.upsert([r])
        results = store.query(vector=[1.0, 0.0], top_k=1)
        assert results[0].metadata["source"] == "test.pdf"
        assert results[0].metadata["page"] == 3

    def test_delete_by_metadata(self, store: ChromaStore):
        store.upsert([
            VectorRecord(id="a", vector=[1.0, 0.0], metadata={"source": "a.pdf"}),
            VectorRecord(id="b", vector=[0.0, 1.0], metadata={"source": "b.pdf"}),
        ])
        deleted = store.delete_by_metadata({"source": "a.pdf"})
        assert deleted == 1
        results = store.query(vector=[1.0, 0.0], top_k=10)
        assert len(results) == 1
        assert results[0].id == "b"

    def test_empty_upsert(self, store: ChromaStore):
        assert store.upsert([]) == 0

    def test_query_empty_store(self, store: ChromaStore):
        results = store.query(vector=[1.0, 0.0, 0.0], top_k=5)
        assert results == []

    def test_backend_name(self, store: ChromaStore):
        assert store.backend_name == "chroma"


@pytest.mark.integration
class TestChromaStoreFactory:

    def test_factory_creates_chroma(self, tmp_path: Path):
        settings = Settings(
            vector_store=VectorStoreConfig(
                backend="chroma",
                persist_path=str(tmp_path / "chroma_factory"),
            )
        )
        store = VectorStoreFactory.create(settings)
        assert isinstance(store, ChromaStore)
        assert store.backend_name == "chroma"
