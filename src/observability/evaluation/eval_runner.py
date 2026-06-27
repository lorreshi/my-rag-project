"""EvalRunner — run retrieval (+ optional generation) over a golden test set.

Loads a golden test set, runs each query through HybridSearch, optionally
generates an answer with an LLM (RAG generation step), then scores the result
with an evaluator and aggregates into an EvalReport.

Two layers of metrics are supported:
- Retrieval metrics (hit_rate, mrr): need only retrieved vs expected chunk ids.
- Generation metrics (faithfulness, answer_relevancy, context_precision): need
  the generated answer + retrieved context texts, so they require an LLM. These
  are produced only when an LLM is wired in (``generate=True``).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.libs.evaluator.base_evaluator import BaseEvaluator
    from src.libs.llm.base_llm import BaseLLM

logger = logging.getLogger(__name__)

# Metrics that are only meaningful for answerable cases (a question with no
# answer in the corpus has no "expected" chunk to hit, no topical answer to be
# relevant to, and no positive reference for context precision). For such cases
# only faithfulness matters: it verifies the system refused without inventing
# facts.
_UNANSWERABLE_METRICS = ("faithfulness",)

_DEFAULT_GEN_PROMPT = (
    "你是一个严谨的问答助手。请【只依据下面提供的资料】回答问题，"
    "不要使用资料之外的知识。如果资料中没有相关信息，请明确回答"
    "“根据已知资料无法回答”。\n\n【资料】\n{contexts}\n\n【问题】{query}\n\n【回答】"
)


@dataclass
class QueryResult:
    """Per-query evaluation detail."""

    query: str
    retrieved_ids: list[str]
    expected_ids: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    generated_answer: str = ""
    answerable: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "expected_ids": self.expected_ids,
            "metrics": self.metrics,
            "generated_answer": self.generated_answer,
            "answerable": self.answerable,
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
    """Run a golden-set evaluation against HybridSearch (+ optional generation)."""

    def __init__(
        self,
        hybrid_search: Any,
        evaluator: "BaseEvaluator",
        top_k: int = 10,
        llm: "BaseLLM | None" = None,
        gen_prompt: str = _DEFAULT_GEN_PROMPT,
    ):
        self._hybrid = hybrid_search
        self._evaluator = evaluator
        self._top_k = top_k
        self._llm = llm
        self._gen_prompt = gen_prompt

    def run(self, test_set_path: str) -> EvalReport:
        """Run evaluation over all test cases in *test_set_path*."""
        cases = self._load_cases(test_set_path)
        report = EvalReport(num_cases=len(cases))

        # Per-metric (sum, count) so each metric is averaged only over the cases
        # where it is meaningful.
        for case in cases:
            query = case.get("query", "")
            expected_ids = case.get("expected_chunk_ids", [])
            ground_truth = case.get("ground_truth", "")
            answerable = case.get("answerable", True)

            results = self._hybrid.search(query, top_k=self._top_k)
            retrieved_ids = [r.chunk_id for r in results]
            contexts = [getattr(r, "text", "") or "" for r in results]

            generated_answer = ""
            if self._llm is not None:
                generated_answer = self._generate(query, contexts)

            metrics = self._evaluator.evaluate(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=expected_ids,
                generated_answer=generated_answer,
                ground_truth=ground_truth,
                contexts=contexts,
            )

            report.per_query.append(QueryResult(
                query=query,
                retrieved_ids=retrieved_ids,
                expected_ids=expected_ids,
                metrics=metrics,
                generated_answer=generated_answer,
                answerable=answerable,
            ))

        report.aggregate_metrics = self._aggregate(report.per_query)
        return report

    @staticmethod
    def _aggregate(per_query: list[QueryResult]) -> dict[str, float]:
        """Average each metric only over the cases where it is meaningful.

        Unanswerable cases contribute only the metrics in
        ``_UNANSWERABLE_METRICS`` (faithfulness); their hit_rate / mrr /
        answer_relevancy / context_precision are excluded so a correct refusal
        does not drag those averages down.
        """
        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        for q in per_query:
            for k, v in q.metrics.items():
                if not q.answerable and k not in _UNANSWERABLE_METRICS:
                    continue
                metric_sums[k] = metric_sums.get(k, 0.0) + v
                metric_counts[k] = metric_counts.get(k, 0) + 1
        return {
            k: round(metric_sums[k] / metric_counts[k], 4)
            for k in metric_sums if metric_counts.get(k)
        }

    def _generate(self, query: str, contexts: list[str]) -> str:
        """RAG generation: answer *query* grounded in the retrieved *contexts*."""
        from src.libs.llm.base_llm import ChatMessage

        context_block = "\n\n".join(
            f"[{i+1}] {c}" for i, c in enumerate(contexts) if c
        )
        prompt = self._gen_prompt.format(contexts=context_block, query=query)
        try:
            resp = self._llm.chat([ChatMessage(role="user", content=prompt)])
            return (resp.content or "").strip()
        except Exception as exc:
            logger.warning("Generation failed for query %r: %s", query, exc)
            return ""

    @staticmethod
    def _load_cases(test_set_path: str) -> list[dict[str, Any]]:
        path = Path(test_set_path)
        if not path.exists():
            raise FileNotFoundError(f"Golden test set not found: {test_set_path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("test_cases", [])

    @classmethod
    def from_settings(cls, settings: Any, **overrides: Any) -> "EvalRunner":
        """Build with real HybridSearch + composite evaluator from settings.

        Pass ``generate=True`` (or inject ``llm=...``) to enable the generation
        step required by generation-quality metrics.
        """
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.observability.evaluation.composite_evaluator import CompositeEvaluator

        hybrid = overrides.get("hybrid_search") or HybridSearch.from_settings(settings)
        evaluator = overrides.get("evaluator") or CompositeEvaluator.from_settings(settings)
        top_k = overrides.get("top_k", getattr(settings.retrieval, "top_k_final", 10))

        llm = overrides.get("llm")
        if llm is None and overrides.get("generate"):
            from src.libs.llm.llm_factory import LLMFactory
            llm = LLMFactory.create(settings)

        return cls(hybrid, evaluator, top_k=top_k, llm=llm)
