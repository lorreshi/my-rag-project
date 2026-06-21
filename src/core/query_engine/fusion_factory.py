"""Fusion Factory — configuration-driven fusion strategy instantiation.

Reads ``settings.retrieval.fusion_algorithm`` (``rrf`` by default, or
``weighted_sum``) and returns a ``BaseFusion``. Mirrors the project's other
factories (LLM / Embedding / Reranker / Tokenizer): config decides the
implementation, unknown values raise ``ValueError`` at build time.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.query_engine.fusion import (
    BaseFusion,
    ReciprocalRankFusion,
    WeightedSumFusion,
)

if TYPE_CHECKING:
    from src.core.settings import Settings

_DEFAULT_RRF_K = 60


class FusionFactory:
    """Create a BaseFusion based on ``settings.retrieval.fusion_algorithm``."""

    @staticmethod
    def create(settings: "Settings") -> BaseFusion:
        """Instantiate the configured fusion strategy.

        Defaults to ``rrf`` when the ``retrieval.fusion_algorithm`` field is
        absent so missing config never raises. Per-route weights are read from
        ``retrieval.fusion_weights`` (empty => unweighted).

        Raises:
            ValueError: if an unknown fusion algorithm is configured.
        """
        retrieval = getattr(settings, "retrieval", None)
        name = (getattr(retrieval, "fusion_algorithm", None) or "rrf").lower()
        weights = getattr(retrieval, "fusion_weights", None) or None

        if name == "rrf":
            return ReciprocalRankFusion(
                k=getattr(retrieval, "rrf_k", _DEFAULT_RRF_K),
                weights=weights,
            )
        if name == "weighted_sum":
            return WeightedSumFusion(weights=weights)

        raise ValueError(
            f"Unknown fusion_algorithm '{name}'. Available: rrf, weighted_sum"
        )
