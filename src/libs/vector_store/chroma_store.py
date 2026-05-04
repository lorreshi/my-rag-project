"""ChromaDB VectorStore implementation.

Uses chromadb as an embedded local vector database.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import chromadb

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
)
from src.libs.vector_store.vector_store_factory import register_backend

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


class ChromaStore(BaseVectorStore):
    """Chroma-backed vector store."""

    def __init__(
        self,
        persist_path: str = "./data/db/chroma",
        collection_name: str = "default",
    ):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        records: list[VectorRecord],
        trace: "TraceContext | None" = None,
    ) -> int:
        if not records:
            return 0

        ids = [r.id for r in records]
        embeddings = [r.vector for r in records]
        documents = [r.text for r in records]
        # Chroma rejects empty dicts — pass None instead
        metadatas = [r.metadata if r.metadata else None for r in records]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(records)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        trace: "TraceContext | None" = None,
    ) -> list[QueryResult]:
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": top_k,
        }
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)

        query_results: list[QueryResult] = []
        if results and results["ids"]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
            documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

            for i, rid in enumerate(ids):
                # Chroma returns distances; convert to similarity score
                score = 1.0 - distances[i] if distances[i] is not None else 0.0
                query_results.append(QueryResult(
                    id=rid,
                    score=score,
                    text=documents[i] or "",
                    metadata=metadatas[i] or {},
                ))

        return query_results

    def delete_by_metadata(
        self,
        filter: dict[str, Any],
        trace: "TraceContext | None" = None,
    ) -> int:
        # Get matching IDs first, then delete
        results = self._collection.get(where=filter)
        if not results or not results["ids"]:
            return 0
        ids = results["ids"]
        self._collection.delete(ids=ids)
        return len(ids)

    @property
    def backend_name(self) -> str:
        return "chroma"


def _create_chroma(settings: Any) -> ChromaStore:
    cfg = settings.vector_store
    return ChromaStore(
        persist_path=cfg.persist_path or "./data/db/chroma",
    )


register_backend("chroma", _create_chroma)
