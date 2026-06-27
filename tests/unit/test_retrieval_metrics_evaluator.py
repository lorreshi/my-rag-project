"""Unit tests for RetrievalMetricsEvaluator (recall@k / ndcg@k / map@k)."""
from __future__ import annotations

import math

import pytest

from src.libs.evaluator.retrieval_metrics_evaluator import RetrievalMetricsEvaluator


class TestRecall:
    def test_all_relevant_found(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "x", "b"], ["a", "b"])
        assert r["recall"] == 1.0

    def test_partial_recall(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "x", "y"], ["a", "b"])
        assert r["recall"] == 0.5

    def test_no_relevant_found(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["x", "y"], ["a", "b"])
        assert r["recall"] == 0.0

    def test_k_truncates_candidates(self):
        # Relevant 'b' sits at rank 3 but k=2, so it is not counted.
        ev = RetrievalMetricsEvaluator(k=2)
        r = ev.evaluate("q", ["a", "x", "b"], ["a", "b"])
        assert r["recall"] == 0.5


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "b", "x"], ["a", "b"])
        assert r["ndcg"] == 1.0

    def test_known_value(self):
        # retrieved [a, x, b, y], relevant {a, b}
        # DCG  = 1/log2(2) + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "x", "b", "y"], ["a", "b"])
        expected = 1.5 / (1 / math.log2(2) + 1 / math.log2(3))
        assert abs(r["ndcg"] - round(expected, 4)) < 1e-4

    def test_relevant_lower_scores_less(self):
        ev = RetrievalMetricsEvaluator(k=10)
        top = ev.evaluate("q", ["a", "x"], ["a"])["ndcg"]
        deep = ev.evaluate("q", ["x", "y", "z", "a"], ["a"])["ndcg"]
        assert top > deep


class TestMAP:
    def test_known_value(self):
        # retrieved [a, x, b, y], relevant {a, b}
        # AP = (1/1 + 2/3) / 2 = 0.8333
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "x", "b", "y"], ["a", "b"])
        assert abs(r["map"] - 0.8333) < 1e-3

    def test_perfect_is_one(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "b"], ["a", "b"])
        assert r["map"] == 1.0


class TestEdgeCases:
    def test_empty_golden_yields_zeros(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a", "b"], [])
        assert r == {"recall": 0.0, "ndcg": 0.0, "map": 0.0}

    def test_empty_retrieved(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", [], ["a"])
        assert r["recall"] == 0.0
        assert r["ndcg"] == 0.0
        assert r["map"] == 0.0

    def test_metric_names_and_types(self):
        ev = RetrievalMetricsEvaluator(k=10)
        r = ev.evaluate("q", ["a"], ["a"])
        assert set(r) == {"recall", "ndcg", "map"}
        assert all(isinstance(v, float) for v in r.values())

    def test_evaluator_name(self):
        assert RetrievalMetricsEvaluator().evaluator_name == "retrieval_metrics"
