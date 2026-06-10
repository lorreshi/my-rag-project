"""EvalRunner — run retrieval over a golden test set and compute metrics.

Loads ``golden_test_set.json``, runs each query through HybridSearch, compares
the retrieved chunk ids / sources against the expected ones using an evaluator,
and aggregates into an EvalReport (hit_rate, mrr, per-query detail).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Per-query evaluation detail."""

    query: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "expected_ids": self.expected_ids,
            "metrics": self.metrics,
        }


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    num_cases: int = 0
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    per_query: list[QueryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_cases": self.num_cases,
            "aggregate_metrics": self.aggregate_metrics,
            "per_query": [q.to_dict() for q in self.per_query],
        }


class EvalRunner:
    """Run a golden-set evaluation against HybridSearch."""

    def __init__(
        self,
        hybrid_search: Any,
        evaluator: "BaseEvaluator",
        top_k: int = 10,
    ):
        self._hybrid = hybrid_search
        self._evaluator = evaluator
        self._top_k = top_k

    def run(self, test_set_path: str) -> EvalReport:
        """Run evaluation over all test cases in *test_set_path*."""
        cases = self._load_cases(test_set_path)
        report = EvalReport(num_cases=len(cases))

        metric_sums: dict[str, float] = {}
        for case in cases:
            query = case.get("query", "")
            expected_ids = case.get("expected_chunk_ids", [])

            results = self._hybrid.search(query, top_k=self._top_k)
            retrieved_ids = [r.chunk_id for r in results]

            metrics = self._evaluator.evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=expected_ids,
            )
            for k, v in metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v

            report.per_query.append(QueryResult(
                query=query,
                retrieved_ids=retrieved_ids,
                expected_ids=expected_ids,
                metrics=metrics,
            ))

        n = len(cases)
        report.aggregate_metrics = {
            k: round(v / n, 4) for k, v in metric_sums.items()
        } if n else {}
        return report

    @staticmethod
    def _load_cases(test_set_path: str) -> list[dict[str, Any]]:
        path = Path(test_set_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden test set not found: {test_set_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("test_cases", [])

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> "EvalRunner":
        """Build with real HybridSearch + composite evaluator from settings."""
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator

        hybrid = overrides.get("hybrid_search") or HybridSearch.from_settings(settings)
        evaluator = overrides.get("evaluator") or CompositeEvaluator.from_settings(settings)
        top_k = overrides.get("top_k", getattr(settings.retrieval, "top_k_final", 10))
        return cls(hybrid, evaluator, top_k=top_k)
