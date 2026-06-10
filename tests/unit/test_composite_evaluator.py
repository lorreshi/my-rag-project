"""Unit tests for CompositeEvaluator (H2)."""
from __future__ import annotations

import pytest

from src.libs.evaluator.base_evaluator import BaseEvaluator
from src.observability.evaluation.composite_evaluator import CompositeEvaluator
from src.core.settings import Settings, EvaluationConfig


class StubEvaluator(BaseEvaluator):
    def __init__(self, name, metrics):
        self._name = name
        self._metrics = metrics

    def evaluate(self, query, retrieved_ids, golden_ids, generated_answer="", ground_truth=""):
        return dict(self._metrics)

    @property
    def evaluator_name(self):
        return self._name


class FailingEvaluator(BaseEvaluator):
    def evaluate(self, *a, **k):
        raise RuntimeError("boom")

    @property
    def evaluator_name(self):
        return "failing"


class TestComposite:
    def test_merges_distinct_metrics(self):
        comp = CompositeEvaluator([
            StubEvaluator("custom", {"hit_rate": 1.0, "mrr": 0.5}),
            StubEvaluator("ragas", {"faithfulness": 0.8}),
        ])
        result = comp.evaluate("q", ["c1"], ["c1"])
        assert result["hit_rate"] == 1.0
        assert result["mrr"] == 0.5
        assert result["faithfulness"] == 0.8

    def test_namespaces_on_collision(self):
        comp = CompositeEvaluator([
            StubEvaluator("a", {"score": 0.1}),
            StubEvaluator("b", {"score": 0.9}),
        ])
        result = comp.evaluate("q", [], [])
        assert result["a.score"] == 0.1
        assert result["b.score"] == 0.9
        assert "score" not in result

    def test_failure_isolated(self):
        comp = CompositeEvaluator([
            StubEvaluator("custom", {"hit_rate": 1.0}),
            FailingEvaluator(),
        ])
        result = comp.evaluate("q", ["c1"], ["c1"])
        # custom still contributes despite the other failing
        assert result["hit_rate"] == 1.0

    def test_requires_at_least_one(self):
        with pytest.raises(ValueError):
            CompositeEvaluator([])

    def test_name_lists_children(self):
        comp = CompositeEvaluator([
            StubEvaluator("custom", {}),
            StubEvaluator("ragas", {}),
        ])
        assert "custom" in comp.evaluator_name
        assert "ragas" in comp.evaluator_name


class TestFromSettings:
    def test_builds_from_backends(self):
        settings = Settings(evaluation=EvaluationConfig(backends=["custom"]))
        comp = CompositeEvaluator.from_settings(settings)
        result = comp.evaluate("q", ["c1"], ["c1"])
        assert "hit_rate" in result

    def test_default_to_custom(self):
        settings = Settings(evaluation=EvaluationConfig(backends=[]))
        comp = CompositeEvaluator.from_settings(settings)
        assert "custom" in comp.evaluator_name

    def test_composite_with_ragas_and_custom(self):
        settings = Settings(evaluation=EvaluationConfig(backends=["custom", "ragas"]))
        comp = CompositeEvaluator.from_settings(settings)
        # ragas builds lazily without importing ragas; name reflects both
        assert "custom" in comp.evaluator_name
        assert "ragas" in comp.evaluator_name
