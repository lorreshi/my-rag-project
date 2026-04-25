"""Smoke test: verify all top-level packages and key sub-packages are importable."""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """Ensure the project skeleton is importable after A1 scaffolding."""

    def test_top_level_packages(self):
        import src.mcp_server
        import src.core
        import src.ingestion
        import src.libs
        import src.observability

    def test_core_subpackages(self):
        import src.core.query_engine
        import src.core.response
        import src.core.trace

    def test_ingestion_subpackages(self):
        import src.ingestion.chunking
        import src.ingestion.transform
        import src.ingestion.embedding
        import src.ingestion.storage

    def test_libs_subpackages(self):
        import src.libs.loader
        import src.libs.llm
        import src.libs.embedding
        import src.libs.splitter
        import src.libs.vector_store
        import src.libs.reranker
        import src.libs.evaluator

    def test_observability_subpackages(self):
        import src.observability.dashboard
        import src.observability.evaluation

    def test_mcp_server_tools(self):
        import src.mcp_server.tools
