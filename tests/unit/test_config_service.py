"""Unit tests for ConfigService and dashboard app import (G1)."""
from __future__ import annotations

from src.core.settings import (
    Settings, LLMConfig, EmbeddingConfig, VectorStoreConfig,
    RetrievalConfig, RerankConfig, VisionLLMConfig,
)
from src.observability.dashboard.services.config_service import ConfigService


def _settings():
    return Settings(
        llm=LLMConfig(provider="openai", model="gpt-4o", temperature=0.0, max_tokens=2048),
        embedding=EmbeddingConfig(provider="openai", model="text-embedding-3-small"),
        vision_llm=VisionLLMConfig(provider="azure", model="gpt-4o"),
        vector_store=VectorStoreConfig(backend="chroma", persist_path="./data/db/chroma"),
        retrieval=RetrievalConfig(sparse_backend="bm25", fusion_algorithm="rrf"),
        rerank=RerankConfig(backend="cross_encoder", model="ms-marco", top_m=30),
    )


class TestComponentCards:
    def test_returns_all_components(self):
        cards = ConfigService(_settings()).component_cards()
        names = {c["name"] for c in cards}
        assert {"LLM", "Embedding", "Vision LLM", "Vector Store", "Retrieval", "Rerank"} <= names

    def test_llm_card(self):
        cards = ConfigService(_settings()).component_cards()
        llm = next(c for c in cards if c["name"] == "LLM")
        assert llm["provider"] == "openai"
        assert llm["model"] == "gpt-4o"
        assert llm["details"]["max_tokens"] == 2048

    def test_retrieval_card_details(self):
        cards = ConfigService(_settings()).component_cards()
        ret = next(c for c in cards if c["name"] == "Retrieval")
        assert ret["details"]["top_k_final"] == 10

    def test_rerank_card(self):
        cards = ConfigService(_settings()).component_cards()
        rr = next(c for c in cards if c["name"] == "Rerank")
        assert rr["provider"] == "cross_encoder"
        assert rr["details"]["top_m"] == 30


class TestDataStats:
    def test_no_store_returns_zero(self):
        stats = ConfigService(_settings()).data_stats(None)
        assert stats["chunk_count"] == 0
        assert stats["backend"] == "chroma"

    def test_store_stats_merged(self):
        class FakeStore:
            def get_collection_stats(self):
                return {"chunk_count": 42, "collection": "default"}

        stats = ConfigService(_settings()).data_stats(FakeStore())
        assert stats["chunk_count"] == 42
        assert stats["collection"] == "default"

    def test_store_error_swallowed(self):
        class BadStore:
            def get_collection_stats(self):
                raise RuntimeError("db down")

        stats = ConfigService(_settings()).data_stats(BadStore())
        assert stats["chunk_count"] == 0


class TestSummary:
    def test_summary_fields(self):
        summary = ConfigService(_settings()).summary()
        assert summary["llm"] == "openai/gpt-4o"
        assert summary["vector_store"] == "chroma"
        assert summary["rerank"] == "cross_encoder"


class TestDashboardImport:
    def test_app_imports(self):
        # Importing the app module must not crash (no Streamlit runtime needed).
        from src.observability.dashboard import app
        assert hasattr(app, "main")

    def test_overview_has_render(self):
        from src.observability.dashboard.pages import overview
        assert callable(overview.render)

    def test_all_pages_have_render(self):
        from src.observability.dashboard.pages import (
            data_browser, ingestion_manager, ingestion_traces, query_traces,
        )
        for mod in (data_browser, ingestion_manager, ingestion_traces, query_traces):
            assert callable(mod.render)
