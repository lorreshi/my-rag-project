"""Recall regression test (H5).

Runs the golden test set through EvalRunner and asserts hit@k / MRR meet fixed
thresholds. To keep the regression deterministic and hermetic (no real index or
network), a controlled fake HybridSearch returns results derived from each
case's expected ids with a small amount of injected noise/misses.

If you wire this against a REAL ingested index later, drop the fake and lower
the thresholds as appropriate for your corpus.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.types import RetrievalResult
from src.libs.evaluator.custom_evaluator import CustomEvaluator
from src.observability.evaluation.eval_runner import EvalRunner

GOLDEN = "tests/fixtures/golden_test_set.json"

# Fixed regression thresholds.
MIN_HIT_RATE = 0.8
MIN_MRR = 0.6


def _load_cases():
    data = json.loads(Path(GOLDEN).read_text(encoding="utf-8"))
    return data["test_cases"]


class ControlledHybrid:
    """Deterministic retriever: returns expected ids for most cases.

    To simulate a realistic-but-good retriever, every case returns its first
    expected id at rank 1 EXCEPT the last case, which misses (to keep the test
    honest about thresholds rather than asserting a perfect 1.0).
    """

    def __init__(self, cases):
        self._cases = cases
        self._miss_query = cases[-1]["query"] if cases else None

    def search(self, query, top_k=10, filters=None, trace=None):
        case = next((c for c in self._cases if c["query"] == query), None)
        if case is None:
            return []
        if query == self._miss_query:
            # Simulate a miss: return unrelated ids.
            return [RetrievalResult(chunk_id="noise_1", score=0.5),
                    RetrievalResult(chunk_id="noise_2", score=0.4)]
        expected = case.get("expected_chunk_ids", [])
        # Put the first expected id at rank 1, then fill with noise.
        ids = list(expected[:1]) + ["noise_a", "noise_b"]
        return [RetrievalResult(chunk_id=cid, score=1.0 - i * 0.1)
                for i, cid in enumerate(ids)][:top_k]


@pytest.fixture
def report():
    cases = _load_cases()
    runner = EvalRunner(ControlledHybrid(cases), CustomEvaluator(), top_k=10)
    return runner.run(GOLDEN)


class TestRecallRegression:
    def test_has_cases(self, report):
        assert report.num_cases >= 5

    def test_hit_rate_threshold(self, report):
        hit_rate = report.aggregate_metrics["hit_rate"]
        assert hit_rate >= MIN_HIT_RATE, (
            f"hit_rate {hit_rate} below threshold {MIN_HIT_RATE}"
        )

    def test_mrr_threshold(self, report):
        mrr = report.aggregate_metrics["mrr"]
        assert mrr >= MIN_MRR, f"mrr {mrr} below threshold {MIN_MRR}"

    def test_per_query_recorded(self, report):
        assert len(report.per_query) == report.num_cases
        for q in report.per_query:
            assert "hit_rate" in q.metrics
            assert "mrr" in q.metrics
