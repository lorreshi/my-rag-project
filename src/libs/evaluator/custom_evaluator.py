"""Custom lightweight evaluator — hit_rate and MRR metrics.

This is the minimal built-in evaluator that requires no external dependencies.
"""
from __future__ import annotations

from src.libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Compute hit_rate and MRR from retrieved vs golden document IDs."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        generated_answer: str = "",
        ground_truth: str = "",
    ) -> dict[str, float]:
        golden_set = set(golden_ids)

        # Hit Rate: 1.0 if any golden doc appears in retrieved, else 0.0
        hit = 1.0 if golden_set & set(retrieved_ids) else 0.0

        # MRR: 1/rank of the first golden doc found in retrieved list
        mrr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in golden_set:
                mrr = 1.0 / rank
                break

        return {"hit_rate": hit, "mrr": mrr}

    @property
    def evaluator_name(self) -> str:
        return "custom"
