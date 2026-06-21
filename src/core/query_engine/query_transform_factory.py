"""QueryTransform Factory — config-driven dense query-transform strategy.

Reads ``settings.retrieval.query_transform`` (``none`` | ``multi_query`` |
``hyde``). Defaults to ``none`` (NoOpTransform) so missing config never raises.
LLM-backed strategies build their LLM lazily via ``LLMFactory``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.query_engine.query_transform import BaseQueryTransform, NoOpTransform

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = logging.getLogger(__name__)


class QueryTransformFactory:
    """Create a BaseQueryTransform from ``settings.retrieval.query_transform``."""

    @staticmethod
    def create(settings: "Settings") -> BaseQueryTransform:
        retrieval = getattr(settings, "retrieval", None)
        mode = (getattr(retrieval, "query_transform", None) or "none").lower()

        if mode == "none":
            return NoOpTransform()

        if mode == "multi_query":
            from src.core.query_engine.query_transform import MultiQueryTransform
            from src.libs.llm.llm_factory import LLMFactory
            return MultiQueryTransform(
                llm=LLMFactory.create(settings),
                n=getattr(retrieval, "multi_query_count", 3),
                max_concurrency=getattr(retrieval, "query_transform_concurrency", 4),
                cache_enabled=getattr(retrieval, "query_transform_cache", True),
            )

        if mode == "hyde":
            from src.core.query_engine.query_transform import HyDETransform
            from src.libs.llm.llm_factory import LLMFactory
            return HyDETransform(
                llm=LLMFactory.create(settings),
                augment=getattr(retrieval, "hyde_augment", True),
                skip_doc_types=getattr(retrieval, "hyde_skip_doc_types", None),
            )

        raise ValueError(
            f"Unknown query_transform '{mode}'. Available: none, multi_query, hyde"
        )
