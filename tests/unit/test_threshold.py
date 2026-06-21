"""Tests for T14: relevance threshold / abstain gate.

Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5
"""
from __future__ import annotations

import pytest

from src.core.query_engine.threshold import apply_threshold
from src.core.types import RetrievalResult


def _r(cid, score):
    return RetrievalResult(chunk_id=cid, score=score, text=cid, metadata={})


@pytest.mark.unit
class TestApplyThreshold:
    def test_top_above_threshold_returns_all(self):
        results = [_r("a", 0.9), _r("b", 0.4)]
        assert apply_threshold(results, 0.5) == results

    def test_top_at_threshold_returns_all(self):
        results = [_r("a", 0.5)]
        assert apply_threshold(results, 0.5) == results

    def test_top_below_threshold_abstains(self):
        results = [_r("a", 0.3), _r("b", 0.2)]
        assert apply_threshold(results, 0.5) == []

    def test_threshold_zero_disabled(self):
        results = [_r("a", -1.0)]
        assert apply_threshold(results, 0.0) == results

    def test_negative_threshold_disabled(self):
        results = [_r("a", -5.0)]
        assert apply_threshold(results, -1.0) == results

    def test_empty_results(self):
        assert apply_threshold([], 0.5) == []


@pytest.mark.unit
class TestMcpToolAbstain:
    def test_tool_returns_empty_build_on_abstain(self):
        from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

        class FakeHybrid:
            def search(self, query, top_k=10, filters=None, trace=None):
                return [_r("a", 0.1)]

        captured = {}

        class FakeBuilder:
            def build(self, results, query):
                captured["n"] = len(results)
                return {"content": []}

        tool = QueryKnowledgeHubTool(
            hybrid_search=FakeHybrid(),
            reranker=None,
            response_builder=FakeBuilder(),
            min_score_threshold=0.5,
        )
        tool({"query": "q"})
        assert captured["n"] == 0  # abstained -> empty result set to builder
