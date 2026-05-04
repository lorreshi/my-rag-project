"""Tests for LLM Reranker (B7.7). LLM calls are mocked."""
from __future__ import annotations

import json
import pytest

from src.libs.reranker.base_reranker import RerankCandidate
from src.libs.reranker.llm_reranker import LLMReranker
from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse


class FakeLLM(BaseLLM):
    """Fake LLM that returns a predetermined response."""

    def __init__(self, response_content: str = "[]"):
        self._response = response_content

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        return ChatResponse(content=self._response)

    @property
    def provider_name(self) -> str:
        return "fake"


def _candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", text="first doc", score=0.9),
        RerankCandidate(id="b", text="second doc", score=0.8),
        RerankCandidate(id="c", text="third doc", score=0.7),
    ]


@pytest.mark.unit
class TestLLMReranker:

    def test_rerank_success(self):
        llm = FakeLLM(json.dumps(["c", "a", "b"]))
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        result = rr.rerank("test query", _candidates())
        assert [c.id for c in result] == ["c", "a", "b"]

    def test_scores_assigned_by_rank(self):
        llm = FakeLLM(json.dumps(["b", "a", "c"]))
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        result = rr.rerank("query", _candidates())
        # First should have highest score
        assert result[0].score > result[1].score
        assert result[1].score > result[2].score

    def test_empty_candidates(self):
        llm = FakeLLM("[]")
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        assert rr.rerank("query", []) == []

    def test_fallback_on_invalid_json(self):
        llm = FakeLLM("not valid json")
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        result = rr.rerank("query", _candidates())
        # Should return original order on failure
        assert [c.id for c in result] == ["a", "b", "c"]

    def test_fallback_on_llm_exception(self):
        class FailingLLM(BaseLLM):
            def chat(self, messages):
                raise RuntimeError("LLM down")
            @property
            def provider_name(self):
                return "failing"

        rr = LLMReranker(llm=FailingLLM(), prompt_template="rank them")
        result = rr.rerank("query", _candidates())
        assert [c.id for c in result] == ["a", "b", "c"]

    def test_missing_ids_appended(self):
        # LLM only returns 2 of 3 IDs
        llm = FakeLLM(json.dumps(["c", "a"]))
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        result = rr.rerank("query", _candidates())
        ids = [c.id for c in result]
        assert ids[:2] == ["c", "a"]
        assert "b" in ids  # missing one appended

    def test_handles_markdown_code_fence(self):
        response = '```json\n["b", "c", "a"]\n```'
        llm = FakeLLM(response)
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        result = rr.rerank("query", _candidates())
        assert [c.id for c in result] == ["b", "c", "a"]

    def test_backend_name(self):
        llm = FakeLLM("[]")
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        assert rr.backend_name == "llm"

    def test_parse_response_not_array(self):
        llm = FakeLLM('{"not": "array"}')
        rr = LLMReranker(llm=llm, prompt_template="rank them")
        # Should fallback since parse raises ValueError
        result = rr.rerank("query", _candidates())
        assert [c.id for c in result] == ["a", "b", "c"]

    def test_reads_prompt_from_file(self, tmp_path):
        prompt_file = tmp_path / "rerank.txt"
        prompt_file.write_text("Custom prompt template")
        llm = FakeLLM(json.dumps(["a", "b", "c"]))
        rr = LLMReranker(llm=llm, prompt_path=str(prompt_file))
        result = rr.rerank("query", _candidates())
        assert [c.id for c in result] == ["a", "b", "c"]
