"""CompositeEvaluator — combine multiple evaluators and merge their metrics.

Runs each wrapped BaseEvaluator and merges the resulting metric dicts. A
failure in one evaluator is isolated (logged, skipped) so the others still
produce results. Metric names are namespaced by evaluator when collisions
occur, otherwise kept flat for convenience.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.libs.evaluator.base_evaluator import BaseEvaluator

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class CompositeEvaluator(BaseEvaluator):
    """Aggregate several evaluators into one."""

    def __init__(self, evaluators: list[BaseEvaluator]):
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")
        self._evaluators = evaluators

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        generated_answer: str = "",
        ground_truth: str = "",
    ) -> dict[str, float]:
        """Run all evaluators and merge their metrics.

        On metric-name collisions across evaluators, both are kept under
        ``{evaluator_name}.{metric}`` keys; non-colliding metrics stay flat.
        """
        per_eval: dict[str, dict[str, float]] = {}
        for ev in self._evaluators:
            name = ev.evaluator_name
            try:
                per_eval[name] = ev.evaluate(
                    query, retrieved_ids, golden_ids, generated_answer, ground_truth
                )
            except Exception as exc:
                logger.warning("Evaluator '%s' failed: %s", name, exc)
                per_eval[name] = {}

        return self._merge(per_eval)

    @staticmethod
    def _merge(per_eval: dict[str, dict[str, float]]) -> dict[str, float]:
        """Merge per-evaluator metric dicts, namespacing only on collisions."""
        # Count metric occurrences across evaluators.
        counts: dict[str, int] = {}
        for metrics in per_eval.values():
            for m in metrics:
                counts[m] = counts.get(m, 0) + 1

        merged: dict[str, float] = {}
        for ev_name, metrics in per_eval.items():
            for m, v in metrics.items():
                if counts[m] > 1:
                    merged[f"{ev_name}.{m}"] = v
                else:
                    merged[m] = v
        return merged

    @property
    def evaluator_name(self) -> str:
        return "composite(" + "+".join(e.evaluator_name for e in self._evaluators) + ")"

    @classmethod
    def from_settings(cls, settings: "Settings") -> "CompositeEvaluator":
        """Build from settings.evaluation.backends (e.g. [ragas, custom])."""
        from src.libs.evaluator.evaluator_factory import EvaluatorFactory

        backends = getattr(settings.evaluation, "backends", None) or ["custom"]
        evaluators = [EvaluatorFactory.create(settings, backend=b) for b in backends]
        return cls(evaluators)
