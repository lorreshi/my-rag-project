"""Unit tests for EvalRunner (H3)."""
from __future__ import annotations

import json

import pytest

from src.core.types import RetrievalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import EvalRunner, EvalReport


class FakeHybrid:
    """Returns preconfigured results keyed by query."""

    def __init__(self, mapping):
        # mapping: {query: [chunk_id, ...]}
        self._mapping = mapping

    def search(self, query, top_k=10, filters=None, trace=None):
        ids = self._mapping.get(query, [])
        return [RetrievalResult(chunk_id=cid, score=1.0) for cid in ids[:top_k]]


def _write_set(tmp_path, cases):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"test_cases": cases}), encoding="utf-8")
    return str(path)


class TestRun:
    def test_perfect_recall(self, tmp_path):
        path = _write_set(tmp_path, [
            {"query": "q1", "expected_chunk_ids": ["a", "b"]},
        ])
        hybrid = FakeHybrid({"q1": ["a", "b", "c"]})
        runner = EvalRunner(hybrid, CustomEvaluator())
        report = runner.run(path)
        assert report.aggregate_metrics["hit_rate"] == 1.0
        assert report.aggregate_metrics["mrr"] == 1.0  # 'a' at rank 1

    def test_miss(self, tmp_path):
        path = _write_set(tmp_path, [
            {"query": "q1", "expected_chunk_ids": ["x"]},
        ])
        hybrid = FakeHybrid({"q1": ["a", "b"]})
        runner = EvalRunner(hybrid, CustomEvaluator())
        report = runner.run(path)
        assert report.aggregate_metrics["hit_rate"] == 0.0

    def test_mrr_rank_two(self, tmp_path):
        path = _write_set(tmp_path, [
            {"query": "q1", "expected_chunk_ids": ["b"]},
        ])
        hybrid = FakeHybrid({"q1": ["a", "b"]})
        runner = EvalRunner(hybrid, CustomEvaluator())
        report = runner.run(path)
        assert report.aggregate_metrics["mrr"] == 0.5

    def test_aggregates_multiple_cases(self, tmp_path):
        path = _write_set(tmp_path, [
            {"query": "q1", "expected_chunk_ids": ["a"]},
            {"query": "q2", "expected_chunk_ids": ["z"]},
        ])
        hybrid = FakeHybrid({"q1": ["a"], "q2": ["y"]})
        runner = EvalRunner(hybrid, CustomEvaluator())
        report = runner.run(path)
        # one hit, one miss -> hit_rate 0.5
        assert report.aggregate_metrics["hit_rate"] == 0.5

    def test_per_query_detail(self, tmp_path):
        path = _write_set(tmp_path, [{"query": "q1", "expected_chunk_ids": ["a"]}])
        hybrid = FakeHybrid({"q1": ["a", "b"]})
        runner = EvalRunner(hybrid, CustomEvaluator())
        report = runner.run(path)
        assert len(report.per_query) == 1
        assert report.per_query[0].retrieved_ids == ["a", "b"]
        assert report.per_query[0].expected_ids == ["a"]

    def test_top_k_respected(self, tmp_path):
        path = _write_set(tmp_path, [{"query": "q1", "expected_chunk_ids": ["c"]}])
        hybrid = FakeHybrid({"q1": ["a", "b", "c"]})
        runner = EvalRunner(hybrid, CustomEvaluator(), top_k=2)
        report = runner.run(path)
        # top_k=2 -> only a,b retrieved -> miss
        assert report.aggregate_metrics["hit_rate"] == 0.0

    def test_report_to_dict(self, tmp_path):
        path = _write_set(tmp_path, [{"query": "q1", "expected_chunk_ids": ["a"]}])
        runner = EvalRunner(FakeHybrid({"q1": ["a"]}), CustomEvaluator())
        d = runner.run(path).to_dict()
        assert set(d.keys()) == {"num_cases", "aggregate_metrics", "per_query"}

    def test_missing_file_raises(self, tmp_path):
        runner = EvalRunner(FakeHybrid({}), CustomEvaluator())
        with pytest.raises(FileNotFoundError):
            runner.run(str(tmp_path / "nope.json"))

    def test_empty_test_set(self, tmp_path):
        path = _write_set(tmp_path, [])
        report = EvalRunner(FakeHybrid({}), CustomEvaluator()).run(path)
        assert report.num_cases == 0
        assert report.aggregate_metrics == {}


class TestGoldenSetFixture:
    def test_fixture_loads(self):
        runner = EvalRunner(FakeHybrid({}), CustomEvaluator())
        report = runner.run("tests/fixtures/golden_test_set.json")
        assert report.num_cases >= 5
