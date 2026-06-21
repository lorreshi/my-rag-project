"""MCP tool: query_knowledge_hub.

Primary retrieval entry point. Runs HybridSearch + Reranker and builds an
MCP tool result (Markdown answer + structured citations) via ResponseBuilder.

The tool is exposed to the protocol layer via ``register`` which registers its
schema and handler with a ProtocolHandler.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from src.core.response.response_builder import ResponseBuilder
from src.mcp_server.protocol_handler import InvalidParams

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.trace.trace_context import TraceContext

logger = logging.getLogger("mcp-server")

TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = (
    "Search the private knowledge base using hybrid retrieval (dense + sparse) "
    "with reranking. Returns the most relevant passages with citations."
)
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The natural-language query."},
        "top_k": {"type": "integer", "description": "Number of results (default 10)."},
        "collection": {"type": "string", "description": "Optional collection filter."},
    },
    "required": ["query"],
}


class QueryKnowledgeHubTool:
    """Callable tool that performs hybrid search + rerank + response build."""

    def __init__(
        self,
        hybrid_search: Any,
        reranker: Any | None = None,
        response_builder: ResponseBuilder | None = None,
        default_top_k: int = 10,
        min_score_threshold: float = 0.0,
    ):
        self._hybrid = hybrid_search
        self._reranker = reranker
        self._builder = response_builder or ResponseBuilder()
        self._default_top_k = default_top_k
        self._min_score_threshold = min_score_threshold

    def __call__(
        self,
        arguments: dict[str, Any],
        trace: "TraceContext | None" = None,
    ) -> dict[str, Any]:
        """Execute the tool. Returns an MCP tool result dict."""
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise InvalidParams("'query' is required and must be a non-empty string")

        top_k = arguments.get("top_k", self._default_top_k)
        if not isinstance(top_k, int) or top_k <= 0:
            raise InvalidParams("'top_k' must be a positive integer")

        collection = arguments.get("collection")
        filters = {"collection": collection} if collection else None

        candidates = self._hybrid.search(
            query, top_k=top_k, filters=filters, trace=trace
        )

        if self._reranker is not None and candidates:
            results = self._reranker.rerank(query, candidates, top_k=top_k, trace=trace)
        else:
            results = candidates[:top_k]

        # Optional abstain gate (default off): drop low-relevance result sets.
        if self._min_score_threshold and self._min_score_threshold > 0 and results:
            from src.core.query_engine.threshold import apply_threshold
            results = apply_threshold(results, self._min_score_threshold)

        return self._builder.build(results, query)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, handler) -> None:
        """Register this tool with a ProtocolHandler."""
        handler.register_tool(
            name=TOOL_NAME,
            description=TOOL_DESCRIPTION,
            input_schema=INPUT_SCHEMA,
            handler=self.__call__,
        )

    @classmethod
    def from_settings(cls, settings: "Settings", **overrides: Any) -> "QueryKnowledgeHubTool":
        """Build the tool with real HybridSearch + Reranker from settings."""
        from src.core.query_engine.hybrid_search import HybridSearch
        from src.core.query_engine.reranker import Reranker

        hybrid = overrides.get("hybrid_search") or HybridSearch.from_settings(settings)
        reranker = overrides.get("reranker")
        if reranker is None:
            reranker = Reranker(settings=settings)
        return cls(
            hybrid_search=hybrid,
            reranker=reranker,
            default_top_k=getattr(settings.retrieval, "top_k_final", 10),
            min_score_threshold=getattr(settings.retrieval, "min_score_threshold", 0.0),
        )
