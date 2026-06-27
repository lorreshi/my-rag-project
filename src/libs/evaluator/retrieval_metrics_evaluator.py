"""RetrievalMetricsEvaluator — rank-aware retrieval quality metrics.

Complements ``CustomEvaluator`` (hit_rate / mrr, "first-hit" view) with metrics
that account for completeness and ranking quality. All metrics are computed from
retrieved vs golden chunk ids only — no LLM, deterministic, cheap.

Metrics
-------
recall@k    : |relevant ∩ retrieved@k| / |relevant|
              召回了多少“已知相关块”。hit_rate 只看有没有命中，recall 看找全没找全。
ndcg@k      : Normalized Discounted Cumulative Gain（位置折扣的排序质量）。
              相关块越靠前得分越高；支持将来扩展为分级相关。二值相关下，
              DCG = Σ rel_i / log2(i+1)，再除以理想排序的 IDCG 归一到 0~1。
map@k       : Mean Average Precision（此处为单条样本的 AP）。奖励把所有相关块
              都尽量靠前排列。AP = Σ_i (Precision@i · rel_i) / |relevant|。

注意：这些指标默认 ``golden_ids`` 列出了“全部”相关块。若 golden 只标注了
含答案的少数块（非穷举），应将结果理解为“对已知相关块的下界”。
"""
from __future__ import annotations

import math

from src.libs.evaluator.base_evaluator import BaseEvaluator


class RetrievalMetricsEvaluator(BaseEvaluator):
    """Compute recall@k, ndcg@k and map@k from retrieved vs golden ids."""

    def __init__(self, k: int = 10):
        self._k = k

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        generated_answer: str = "",
        ground_truth: str = "",
        contexts: list[str] | None = None,
    ) -> dict[str, float]:
        golden = set(golden_ids)
        if not golden:
            # No known-relevant chunks (e.g. unanswerable case): these metrics
            # are not meaningful. EvalRunner excludes them from aggregation.
            return {"recall": 0.0, "ndcg": 0.0, "map": 0.0}

        top = retrieved_ids[: self._k]
        rel = [1 if cid in golden else 0 for cid in top]

        return {
            "recall": round(self._recall(rel, len(golden)), 4),
            "ndcg": round(self._ndcg(rel, len(golden)), 4),
            "map": round(self._average_precision(rel, len(golden)), 4),
        }

    @staticmethod
    def _recall(rel: list[int], num_relevant: int) -> float:
        return sum(rel) / num_relevant if num_relevant else 0.0

    @staticmethod
    def _ndcg(rel: list[int], num_relevant: int) -> float:
        # DCG with binary gains: Σ rel_i / log2(rank+1).
        dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rel))
        # Ideal DCG: all relevant packed at the top (capped by k = len(rel)).
        ideal_hits = min(num_relevant, len(rel))
        idcg = sum(1 / math.log2(i + 2) for i in range(ideal_hits))
        return dcg / idcg if idcg else 0.0

    @staticmethod
    def _average_precision(rel: list[int], num_relevant: int) -> float:
        cumulative_hits = 0
        ap = 0.0
        for i, r in enumerate(rel, start=1):
            if r:
                cumulative_hits += 1
                ap += cumulative_hits / i
        return ap / num_relevant if num_relevant else 0.0

    @property
    def evaluator_name(self) -> str:
        return "retrieval_metrics"


def _create_retrieval_metrics(settings) -> "RetrievalMetricsEvaluator":
    """Factory: k defaults to retrieval.top_k_final when available."""
    k = getattr(getattr(settings, "retrieval", None), "top_k_final", 10)
    return RetrievalMetricsEvaluator(k=k)
