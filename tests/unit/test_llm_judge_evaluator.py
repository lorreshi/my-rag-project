"""Unit tests for LLMJudgeEvaluator.

A scripted stub LLM returns canned JSON so the metric logic is exercised
without any network / real model. A stub embedding returns controllable
vectors for the answer_relevancy cosine path.
"""
from __future__ import annotations

import json

import pytest

from src.libs.evaluator.llm_judge_evaluator import (
    LLMJudgeEvaluator,
    _cosine,
    _parse_json,
)
from src.libs.llm.base_llm import ChatResponse


class ScriptedLLM:
    """Returns the next canned reply on each chat() call."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return ChatResponse(content=self._replies.pop(0))

    @property
    def provider_name(self):
        return "scripted"


class StubEmbedding:
    """Maps preset texts to preset vectors; unknown texts -> zeros."""

    def __init__(self, mapping, dim=3):
        self._mapping = mapping
        self._dim = dim

    def embed(self, texts, trace=None):
        return [self._mapping.get(t, [0.0] * self._dim) for t in texts]

    @property
    def provider_name(self):
        return "stub"

    @property
    def dimension(self):
        return self._dim


class TestFaithfulness:
    def test_partial_support(self):
        reply = json.dumps({"claims": [
            {"claim": "a", "supported": True},
            {"claim": "b", "supported": False},
        ]})
        ev = LLMJudgeEvaluator(ScriptedLLM([reply]), metrics=("faithfulness",))
        r = ev.evaluate("q", [], [], generated_answer="ans", contexts=["ctx"])
        assert r["faithfulness"] == 0.5

    def test_empty_answer_is_zero(self):
        ev = LLMJudgeEvaluator(ScriptedLLM([]), metrics=("faithfulness",))
        r = ev.evaluate("q", [], [], generated_answer="", contexts=["ctx"])
        assert r["faithfulness"] == 0.0

    def test_refusal_is_vacuously_faithful(self):
        # No LLM call expected for a refusal.
        llm = ScriptedLLM([])
        ev = LLMJudgeEvaluator(llm, metrics=("faithfulness",))
        r = ev.evaluate("q", [], [], generated_answer="根据已知资料无法回答", contexts=["ctx"])
        assert r["faithfulness"] == 1.0
        assert llm.calls == []

    def test_no_contexts_is_zero(self):
        ev = LLMJudgeEvaluator(ScriptedLLM([]), metrics=("faithfulness",))
        r = ev.evaluate("q", [], [], generated_answer="some claim", contexts=[])
        assert r["faithfulness"] == 0.0


class TestAnswerRelevancy:
    def test_cosine_with_embedding(self):
        # reverse-question identical vector to query -> cosine 1.0
        reply = json.dumps({"questions": ["q1", "q2"]})
        emb = StubEmbedding({
            "原始问题": [1.0, 0.0, 0.0],
            "q1": [1.0, 0.0, 0.0],
            "q2": [0.0, 1.0, 0.0],  # orthogonal -> cosine 0
        })
        ev = LLMJudgeEvaluator(ScriptedLLM([reply]), embedding=emb,
                               metrics=("answer_relevancy",))
        r = ev.evaluate("原始问题", [], [], generated_answer="ans")
        # mean(cos(q,q1)=1.0, cos(q,q2)=0.0) = 0.5
        assert r["answer_relevancy"] == 0.5

    def test_refusal_is_zero(self):
        llm = ScriptedLLM([])
        ev = LLMJudgeEvaluator(llm, metrics=("answer_relevancy",))
        r = ev.evaluate("q", [], [], generated_answer="文档中未提供该信息")
        assert r["answer_relevancy"] == 0.0
        assert llm.calls == []

    def test_llm_fallback_without_embedding(self):
        # No embedding -> falls back to direct LLM 0-1 rating.
        replies = [json.dumps({"questions": ["q1"]}), json.dumps({"score": 0.8})]
        ev = LLMJudgeEvaluator(ScriptedLLM(replies), embedding=None,
                               metrics=("answer_relevancy",))
        r = ev.evaluate("q", [], [], generated_answer="ans")
        assert r["answer_relevancy"] == 0.8


class TestContextPrecision:
    def test_average_precision_ranking(self):
        # verdicts [T, F, T] -> AP = (1/1 + 2/3) / 2 = 0.8333
        reply = json.dumps({"verdicts": [True, False, True]})
        ev = LLMJudgeEvaluator(ScriptedLLM([reply]), metrics=("context_precision",))
        r = ev.evaluate("q", [], [], ground_truth="gt", contexts=["x", "y", "z"])
        assert abs(r["context_precision"] - 0.8333) < 1e-3

    def test_no_useful_context_is_zero(self):
        reply = json.dumps({"verdicts": [False, False]})
        ev = LLMJudgeEvaluator(ScriptedLLM([reply]), metrics=("context_precision",))
        r = ev.evaluate("q", [], [], ground_truth="gt", contexts=["x", "y"])
        assert r["context_precision"] == 0.0

    def test_no_contexts_is_zero(self):
        ev = LLMJudgeEvaluator(ScriptedLLM([]), metrics=("context_precision",))
        r = ev.evaluate("q", [], [], ground_truth="gt", contexts=[])
        assert r["context_precision"] == 0.0


class TestRobustness:
    def test_malformed_json_degrades_to_zero(self):
        ev = LLMJudgeEvaluator(ScriptedLLM(["not json at all"]),
                               metrics=("faithfulness",))
        r = ev.evaluate("q", [], [], generated_answer="claim", contexts=["ctx"])
        assert r["faithfulness"] == 0.0

    def test_llm_exception_isolated(self):
        class BoomLLM:
            def chat(self, messages):
                raise RuntimeError("boom")
            provider_name = "boom"

        ev = LLMJudgeEvaluator(BoomLLM(), metrics=("faithfulness",))
        r = ev.evaluate("q", [], [], generated_answer="claim", contexts=["ctx"])
        assert r["faithfulness"] == 0.0

    def test_evaluator_name(self):
        assert LLMJudgeEvaluator(ScriptedLLM([])).evaluator_name == "llm_judge"


class TestHelpers:
    def test_parse_json_plain(self):
        assert _parse_json('{"a": 1}') == {"a": 1}

    def test_parse_json_with_code_fence(self):
        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_parse_json_embedded(self):
        assert _parse_json('here you go: {"a": 1} done') == {"a": 1}

    def test_parse_json_garbage(self):
        assert _parse_json("no json") == {}

    def test_cosine_identical(self):
        assert _cosine([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_cosine_orthogonal(self):
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_cosine_zero_vector(self):
        assert _cosine([0, 0], [1, 1]) is None
