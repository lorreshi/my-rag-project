"""Tests for T11: MultiQueryTransform (rewrite + concurrency cap + cache + degrade).

Validates: Requirements 8.2, 8.3, 8.4, 8.5
"""
from __future__ import annotations

import pytest

from src.core.query_engine.query_transform import MultiQueryTransform
from src.libs.llm.base_llm import ChatResponse


class FakeLLM:
    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return ChatResponse(content=self.content)

    @property
    def provider_name(self):
        return "fake"


class BoomLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        raise RuntimeError("llm down")

    @property
    def provider_name(self):
        return "boom"


@pytest.mark.unit
class TestMultiQuery:
    def test_original_kept_first_and_variants_added(self):
        llm = FakeLLM("变体一\n变体二\n变体三")
        tq = MultiQueryTransform(llm, n=3, max_concurrency=4).transform("原始问题")
        assert tq.dense_queries[0] == "原始问题"
        assert tq.dense_queries == ["原始问题", "变体一", "变体二", "变体三"]
        assert tq.used_llm is True
        assert tq.degraded is False

    def test_parses_numbered_and_bulleted_lines(self):
        llm = FakeLLM("1. alpha\n2) beta\n- gamma\n* delta")
        tq = MultiQueryTransform(llm, n=4, max_concurrency=10).transform("q")
        assert tq.dense_queries == ["q", "alpha", "beta", "gamma", "delta"]

    def test_dedup_against_original(self):
        llm = FakeLLM("q\nother")
        tq = MultiQueryTransform(llm, max_concurrency=10).transform("q")
        assert tq.dense_queries == ["q", "other"]

    def test_max_concurrency_caps_dense_queries(self):
        llm = FakeLLM("a\nb\nc\nd\ne")
        tq = MultiQueryTransform(llm, n=5, max_concurrency=2).transform("q")
        assert len(tq.dense_queries) == 2

    def test_cache_avoids_second_llm_call(self):
        llm = FakeLLM("v1\nv2")
        mq = MultiQueryTransform(llm, cache_enabled=True)
        mq.transform("same")
        mq.transform("same")
        assert llm.calls == 1

    def test_cache_disabled_calls_each_time(self):
        llm = FakeLLM("v1")
        mq = MultiQueryTransform(llm, cache_enabled=False)
        mq.transform("q")
        mq.transform("q")
        assert llm.calls == 2

    def test_llm_failure_degrades_to_single_query(self):
        llm = BoomLLM()
        tq = MultiQueryTransform(llm).transform("q")
        assert tq.dense_queries == ["q"]
        assert tq.degraded is True
        assert tq.used_llm is False
