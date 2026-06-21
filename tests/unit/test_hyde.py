"""Tests for T12: HyDETransform (hypothetical doc + augment + doc_type gating + degrade).

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5
"""
from __future__ import annotations

import pytest

from src.core.query_engine.query_transform import HyDETransform
from src.libs.llm.base_llm import ChatResponse


class FakeLLM:
    def __init__(self, content="假设性答案文档。"):
        self.content = content
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return ChatResponse(content=self.content)

    @property
    def provider_name(self):
        return "fake"


class BoomLLM:
    def chat(self, messages):
        raise RuntimeError("llm down")

    @property
    def provider_name(self):
        return "boom"


@pytest.mark.unit
class TestHyDE:
    def test_augment_includes_query_and_hypo(self):
        llm = FakeLLM("hypo doc")
        tq = HyDETransform(llm, augment=True).transform("q")
        assert tq.dense_queries == ["q", "hypo doc"]
        assert tq.used_llm is True

    def test_replace_uses_only_hypo(self):
        llm = FakeLLM("hypo doc")
        tq = HyDETransform(llm, augment=False).transform("q")
        assert tq.dense_queries == ["hypo doc"]

    def test_doc_type_skip_bypasses_llm(self):
        llm = FakeLLM("hypo")
        tq = HyDETransform(llm, skip_doc_types=["xlsx"]).transform("q", doc_type="xlsx")
        assert tq.dense_queries == ["q"]
        assert tq.used_llm is False
        assert tq.degraded is False
        assert llm.calls == 0

    def test_doc_type_not_skipped_runs_hyde(self):
        llm = FakeLLM("hypo")
        tq = HyDETransform(llm, skip_doc_types=["xlsx"]).transform("q", doc_type="pdf")
        assert tq.dense_queries == ["q", "hypo"]

    def test_llm_failure_degrades(self):
        tq = HyDETransform(BoomLLM()).transform("q")
        assert tq.dense_queries == ["q"]
        assert tq.degraded is True

    def test_empty_hypo_degrades(self):
        tq = HyDETransform(FakeLLM(content="   ")).transform("q")
        assert tq.dense_queries == ["q"]
        assert tq.degraded is True
