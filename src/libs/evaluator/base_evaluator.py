"""Evaluator abstract base class.

All evaluation backends must implement this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    """Abstract base class for RAG evaluation backends."""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        generated_answer: str = "",
        ground_truth: str = "",
        contexts: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate retrieval/generation quality.

        Args:
            query: The user query.
            retrieved_ids: IDs of retrieved documents.
            golden_ids: IDs of ground-truth relevant documents.
            generated_answer: The generated answer (if applicable).
            ground_truth: The reference answer (if applicable).
            contexts: Retrieved chunk texts, in rank order (for generation
                metrics that judge an answer against its supporting context).

        Returns:
            Dictionary of metric_name -> score.
        """
        ...

    @property
    @abstractmethod
    def evaluator_name(self) -> str:
        """Return a human-readable evaluator identifier."""
        ...
