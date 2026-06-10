"""RagasEvaluator — wrap the Ragas framework behind BaseEvaluator.

Implements faithfulness / answer relevancy / context precision via Ragas.
Ragas (and an LLM) are heavy optional dependencies, so:
- Import is lazy: nothing is imported until evaluate() actually needs Ragas.
- If Ragas is unavailable, a clear ImportError is raised with install guidance.
- An injectable ``score_fn`` allows unit tests to exercise the logic without
  installing Ragas or making network/LLM calls.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from src.libs.evaluator.base_evaluator import BaseEvaluator

logger = logging.getLogger(__name__)

# Metrics this evaluator reports.
_METRICS = ("faithfulness", "answer_relevancy", "context_precision")

_INSTALL_HINT = (
    "RagasEvaluator requires the 'ragas' package. Install it with "
    "`pip install ragas` (note: this pulls in datasets/langchain and an LLM "
    "client). Alternatively use the built-in 'custom' evaluator."
)


class RagasEvaluator(BaseEvaluator):
    """Ragas-backed evaluator for generation quality metrics."""

    def __init__(
        self,
        llm: Any | None = None,
        metrics: tuple[str, ...] = _METRICS,
        score_fn: Callable[..., dict[str, float]] | None = None,
    ):
        """Initialize.

        Args:
            llm: An LLM client Ragas should use (provider-specific).
            metrics: Which metrics to compute.
            score_fn: Optional injectable scorer for testing. When provided,
                Ragas itself is never imported.
        """
        self._llm = llm
        self._metrics = metrics
        self._score_fn = score_fn

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        generated_answer: str = "",
        ground_truth: str = "",
        contexts: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute Ragas metrics for a single sample.

        Returns a dict containing the configured metric names. Missing inputs
        (e.g. no generated answer) yield 0.0 for the affected metrics rather
        than failing.
        """
        if self._score_fn is not None:
            return self._normalize(self._score_fn(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
                contexts=contexts or [],
            ))

        return self._normalize(self._ragas_score(
            query, generated_answer, ground_truth, contexts or []
        ))

    def _ragas_score(
        self,
        query: str,
        answer: str,
        ground_truth: str,
        contexts: list[str],
    ) -> dict[str, float]:
        """Run real Ragas scoring. Raises ImportError if Ragas is missing."""
        try:
            import ragas  # noqa: F401
            from datasets import Dataset  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        # Lazy, defensive import of metrics + evaluate entrypoint.
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
            from datasets import Dataset
        except ImportError as exc:  # pragma: no cover - version differences
            raise ImportError(_INSTALL_HINT) from exc

        metric_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
        }
        selected = [metric_map[m] for m in self._metrics if m in metric_map]

        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth],
        })

        kwargs: dict[str, Any] = {"metrics": selected}
        if self._llm is not None:
            kwargs["llm"] = self._llm
        result = ragas_evaluate(dataset, **kwargs)

        scores: dict[str, float] = {}
        for m in self._metrics:
            try:
                scores[m] = float(result[m])
            except Exception:
                scores[m] = 0.0
        return scores

    def _normalize(self, raw: dict[str, Any]) -> dict[str, float]:
        """Ensure all configured metrics are present as floats."""
        out: dict[str, float] = {}
        for m in self._metrics:
            try:
                out[m] = float(raw.get(m, 0.0))
            except (TypeError, ValueError):
                out[m] = 0.0
        return out

    @property
    def evaluator_name(self) -> str:
        return "ragas"


def _create_ragas(settings: Any) -> RagasEvaluator:
    """Factory: build a RagasEvaluator, wiring an LLM from settings if possible."""
    llm = None
    try:
        from src.libs.llm.llm_factory import LLMFactory
        llm = LLMFactory.create(settings)
    except Exception:  # pragma: no cover - config dependent
        llm = None
    return RagasEvaluator(llm=llm)
