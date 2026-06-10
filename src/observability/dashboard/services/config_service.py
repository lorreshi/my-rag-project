"""ConfigService — read and format configuration for the dashboard.

Wraps a Settings object and exposes component cards (LLM / Embedding /
Vector Store / Retrieval / Rerank) plus data statistics for the Overview page.
Pure data layer — no Streamlit imports, fully unit-testable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.settings import Settings


class ConfigService:
    """Format Settings into dashboard-friendly structures."""

    def __init__(self, settings: "Settings"):
        self._settings = settings

    def component_cards(self) -> list[dict[str, Any]]:
        """Return a list of component config cards for display."""
        s = self._settings
        return [
            {
                "name": "LLM",
                "provider": s.llm.provider,
                "model": s.llm.model,
                "details": {"temperature": s.llm.temperature, "max_tokens": s.llm.max_tokens},
            },
            {
                "name": "Embedding",
                "provider": s.embedding.provider,
                "model": s.embedding.model,
                "details": {},
            },
            {
                "name": "Vision LLM",
                "provider": s.vision_llm.provider,
                "model": s.vision_llm.model,
                "details": {},
            },
            {
                "name": "Vector Store",
                "provider": s.vector_store.backend,
                "model": "",
                "details": {"persist_path": s.vector_store.persist_path},
            },
            {
                "name": "Retrieval",
                "provider": s.retrieval.sparse_backend,
                "model": s.retrieval.fusion_algorithm,
                "details": {
                    "top_k_dense": s.retrieval.top_k_dense,
                    "top_k_sparse": s.retrieval.top_k_sparse,
                    "top_k_final": s.retrieval.top_k_final,
                },
            },
            {
                "name": "Rerank",
                "provider": s.rerank.backend,
                "model": s.rerank.model,
                "details": {"top_m": s.rerank.top_m},
            },
        ]

    def data_stats(self, vector_store: Any | None = None) -> dict[str, Any]:
        """Return data statistics (chunk count etc.) if a store is available."""
        stats: dict[str, Any] = {"chunk_count": 0, "backend": self._settings.vector_store.backend}
        if vector_store is not None:
            getter = getattr(vector_store, "get_collection_stats", None)
            if callable(getter):
                try:
                    stats.update(getter())
                except Exception:
                    pass
        return stats

    def summary(self) -> dict[str, Any]:
        """Return a compact summary dict for the overview header."""
        return {
            "llm": f"{self._settings.llm.provider}/{self._settings.llm.model}",
            "embedding": f"{self._settings.embedding.provider}/{self._settings.embedding.model}",
            "vector_store": self._settings.vector_store.backend,
            "rerank": self._settings.rerank.backend,
        }
