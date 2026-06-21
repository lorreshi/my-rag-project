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


import logging

logger = logging.getLogger(__name__)

_MULTI_QUERY_PROMPT = (
    "你是一个检索查询改写助手。请把下面的用户问题改写成 {n} 个语义等价但措辞"
    "不同的检索查询，覆盖同义表达与不同问法。每行一个，不要编号，不要解释。\n\n"
    "用户问题：{query}\n"
)


class MultiQueryTransform(BaseQueryTransform):
    """LLM multi-query expansion: rewrite into N variants, each retrieved alone.

    The original query is always kept (first). Rewrites are cached per query.
    On any LLM failure the transform degrades to the single original query.
    ``max_concurrency`` bounds how many dense queries (and thus concurrent
    embedding calls) are produced.
    """

    def __init__(
        self,
        llm,
        n: int = 3,
        max_concurrency: int = 4,
        cache_enabled: bool = True,
        cache: "dict[str, list[str]] | None" = None,
        prompt_template: str = _MULTI_QUERY_PROMPT,
    ):
        self._llm = llm
        self._n = max(1, n)
        self._max_concurrency = max(1, max_concurrency)
        self._cache_enabled = cache_enabled
        self._cache = cache if cache is not None else {}
        self._prompt_template = prompt_template

    def transform(self, query, trace=None) -> TransformedQuery:
        try:
            variants: list[str] | None = None
            if self._cache_enabled and query in self._cache:
                variants = self._cache[query]
            if variants is None:
                variants = self._rewrite(query)
                if self._cache_enabled:
                    self._cache[query] = variants

            dense: list[str] = [query]
            seen = {query}
            for v in variants:
                v = v.strip()
                if v and v not in seen:
                    seen.add(v)
                    dense.append(v)
            # Bound concurrent embedding work.
            dense = dense[: self._max_concurrency]
            return TransformedQuery(dense_queries=dense, used_llm=True, degraded=False)
        except Exception as exc:
            logger.warning("multi_query failed; degrading to single query: %s", exc)
            return TransformedQuery(dense_queries=[query], used_llm=False, degraded=True)

    def _rewrite(self, query: str) -> list[str]:
        from src.libs.llm.base_llm import ChatMessage

        prompt = self._prompt_template.format(n=self._n, query=query)
        resp = self._llm.chat([ChatMessage(role="user", content=prompt)])
        return self._parse(getattr(resp, "content", "") or "")

    @staticmethod
    def _parse(text: str) -> list[str]:
        """Split LLM output into variant lines, stripping bullets/numbering."""
        import re

        out: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:\d+[.)、]|[-*•])\s*", "", line).strip()
            if line:
                out.append(line)
        return out


class HyDETransform(BaseQueryTransform):
    """HyDE: embed a hypothetical answer document instead of / alongside query.

    ``augment=True`` keeps the original query (so dense_queries = [query, hypo]);
    ``augment=False`` uses only the hypothetical document. When the target
    ``doc_type`` is in ``skip_doc_types`` (structured data prone to
    hallucinated specifics), HyDE is skipped (single original query). LLM
    failure degrades to the single original query.
    """

    def __init__(
        self,
        llm,
        augment: bool = True,
        skip_doc_types: "list[str] | None" = None,
        prompt_template: str = (
            "请基于你的知识，为下面的问题写一段简洁、像文档摘录的假设性答案"
            "（2-4 句，只输出答案正文）：\n\n问题：{query}\n"
        ),
    ):
        self._llm = llm
        self._augment = augment
        self._skip = set(skip_doc_types or [])
        self._prompt_template = prompt_template

    def transform(self, query, trace=None, doc_type: str | None = None) -> TransformedQuery:
        # doc_type gating: structured types skip HyDE.
        if doc_type is not None and doc_type in self._skip:
            return TransformedQuery(dense_queries=[query], used_llm=False, degraded=False)
        try:
            from src.libs.llm.base_llm import ChatMessage

            prompt = self._prompt_template.format(query=query)
            resp = self._llm.chat([ChatMessage(role="user", content=prompt)])
            hypo = (getattr(resp, "content", "") or "").strip()
            if not hypo:
                raise ValueError("empty hypothetical document")
            dense = [query, hypo] if self._augment else [hypo]
            return TransformedQuery(dense_queries=dense, used_llm=True, degraded=False)
        except Exception as exc:
            logger.warning("hyde failed; degrading to single query: %s", exc)
            return TransformedQuery(dense_queries=[query], used_llm=False, degraded=True)
