"""QueryTransform — pluggable dense-side query transformation (#8/#9).

A strategy component that turns one raw query into one or more *dense query
texts*, each of which is embedded and retrieved separately; the resulting
ranked lists are fused (alongside the single sparse list) by the existing
``Fusion`` — so the fusion layer needs no change.

Modes (selected by ``retrieval.query_transform``):
- ``none``        -> NoOpTransform: a single dense query (== baseline).
- ``multi_query`` -> LLM rewrites into N variants (T11).
- ``hyde``        -> LLM generates a hypothetical document (T12).

All strategies MUST degrade to the single original query on failure (never
break the query); ``TransformedQuery.degraded`` records when that happened.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.trace.trace_context import TraceContext


@dataclass
class TransformedQuery:
    """Output of a QueryTransform.

    Attributes:
        dense_queries: One or more query texts to embed + retrieve separately.
            Always non-empty and always contains the original query (first).
        used_llm: Whether an LLM call was made.
        degraded: True if the strategy fell back to the single original query
            after a failure.
    """

    dense_queries: list[str] = field(default_factory=list)
    used_llm: bool = False
    degraded: bool = False


class BaseQueryTransform(ABC):
    """Transform a raw query into one or more dense query texts."""

    @abstractmethod
    def transform(
        self, query: str, trace: "TraceContext | None" = None
    ) -> TransformedQuery:
        ...


class NoOpTransform(BaseQueryTransform):
    """Identity transform: a single dense query equal to the original."""

    def transform(
        self, query: str, trace: "TraceContext | None" = None
    ) -> TransformedQuery:
        return TransformedQuery(dense_queries=[query], used_llm=False, degraded=False)
