"""Unit tests for RagasEvaluator (H1).

Uses an injectable score_fn so Ragas itself is never required. Also verifies
the graceful ImportError when Ragas is unavailable.
"""
from __future__ import annotations

import builtins

import pytest

from src.observability.evaluation.ragas_evaluator import RagasEvaluator
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.core.settings import Settings


def _fake_score(**kwargs):
    return {
        "faithfulness": 0.8,
        "answer_relevancy": 0.7,
        "context_precision": 0.9,
    }


class TestEvaluateWithScoreFn:
    def test_returns_metrics(self):
        ev = RagasEvaluator(score_fn=_fake_score)
        result = ev.evaluate(
            query="q", retrieved_ids=["c1"], golden_ids=["c1"],
            generated_answer="ans", ground_truth="truth",
        )
        assert result["faithfulness"] == 0.8
        assert result["answer_relevancy"] == 0.7
        assert result["context_precision"] == 0.9

    def test_metrics_are_floats(self):
        ev = RagasEvaluator(score_fn=lambda **k: {"faithfulness": 1, "answer_relevancy": 0, "context_precision": 1})
        result = ev.evaluate("q", ["c1"], ["c1"])
        assert all(isinstance(v, float) for v in result.values())

    def test_missing_metric_defaults_zero(self):
        ev = RagasEvaluator(score_fn=lambda **k: {"faithfulness": 0.5})
        result = ev.evaluate("q", ["c1"], ["c1"])
        assert result["faithfulness"] == 0.5
        assert result["answer_relevancy"] == 0.0
        assert result["context_precision"] == 0.0

    def test_custom_metrics_subset(self):
        ev = RagasEvaluator(metrics=("faithfulness",), score_fn=_fake_score)
        result = ev.evaluate("q", ["c1"], ["c1"])
        assert set(result.keys()) == {"faithfulness"}

    def test_evaluator_name(self):
        assert RagasEvaluator(score_fn=_fake_score).evaluator_name == "ragas"

    def test_score_fn_receives_inputs(self):
        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _fake_score()

        ev = RagasEvaluator(score_fn=_capture)
        ev.evaluate("my query", ["c1"], ["c2"], generated_answer="a", ground_truth="g", contexts=["ctx"])
        assert captured["query"] == "my query"
        assert captured["contexts"] == ["ctx"]


class TestGracefulDegradation:
    def test_import_error_when_ragas_missing(self, monkeypatch):
        # Force `import ragas` to fail.
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "ragas" or name.startswith("ragas."):
                raise ImportError("no ragas")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        ev = RagasEvaluator()  # no score_fn -> will try real ragas
        with pytest.raises(ImportError) as exc_info:
            ev.evaluate("q", ["c1"], ["c1"], generated_answer="a", contexts=["ctx"])
        assert "ragas" in str(exc_info.value).lower()


class TestFactoryRegistration:
    def test_ragas_registered(self):
        # Should build without importing ragas (lazy LLM/factory wiring).
        ev = EvaluatorFactory.create(Settings(), backend="ragas")
        assert ev.evaluator_name == "ragas"

    def test_custom_still_available(self):
        ev = EvaluatorFactory.create(Settings(), backend="custom")
        assert ev.evaluator_name == "custom"
