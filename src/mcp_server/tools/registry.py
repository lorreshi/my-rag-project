"""Tool registry — wire the built-in MCP tools onto a ProtocolHandler.

E1 builds the transport loop and E2 the protocol handler, but neither wires the
concrete tools (E3-E5) onto the handler. This module closes that gap: it builds
the default tool set from a Settings object and registers each one with the
handler so a live server (``python -m src.mcp_server.server``) actually exposes
``query_knowledge_hub``, ``list_collections`` and ``get_document_summary``.

Registration is resilient: a tool whose construction fails (e.g. missing config,
no embedding backend) is logged to stderr and skipped, so the server still
starts and serves whatever tools could be built.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.mcp_server.protocol_handler import ProtocolHandler

logger = logging.getLogger("mcp-server")

DEFAULT_DOCUMENTS_DIR = "data/documents"


def register_default_tools(
    handler: "ProtocolHandler",
    settings: "Settings | None" = None,
    documents_base_dir: str = DEFAULT_DOCUMENTS_DIR,
) -> list[str]:
    """Register the built-in tools onto *handler*.

    Args:
        handler: The ProtocolHandler to register tools with.
        settings: Loaded Settings. If None, ``load_settings()`` is used.
        documents_base_dir: Base directory scanned by ``list_collections``.

    Returns:
        The names of the tools that were successfully registered.
    """
    if settings is None:
        from src.core.settings import load_settings
        settings = load_settings()

    registered: list[str] = []

    # E3: query_knowledge_hub (hybrid search + rerank).
    # Registered lazily: building HybridSearch + Reranker eagerly loads heavy
    # models (and can do blocking network checks), which under stdio would
    # delay the read loop and time out the client's initialize. The retrieval
    # stack is built on the first tools/call instead.
    def _build_query(h):
        from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
        QueryKnowledgeHubTool.register_lazy(h, settings)

    # E4: list_collections (scan documents dir).
    def _build_list(h):
        from src.mcp_server.tools.list_collections import ListCollectionsTool
        ListCollectionsTool(documents_base_dir=documents_base_dir).register(h)

    # E5: get_document_summary (vector store metadata lookup).
    def _build_summary(h):
        import src.libs.vector_store.chroma_store  # noqa: F401  (register backend)
        from src.libs.vector_store.vector_store_factory import VectorStoreFactory
        from src.mcp_server.tools.get_document_summary import (
            DocumentLookup,
            GetDocumentSummaryTool,
        )
        store = VectorStoreFactory.create(settings)
        GetDocumentSummaryTool(DocumentLookup(store)).register(h)

    builders: list[tuple[str, Any]] = [
        ("query_knowledge_hub", _build_query),
        ("list_collections", _build_list),
        ("get_document_summary", _build_summary),
    ]

    for name, build in builders:
        try:
            build(handler)
            registered.append(name)
            logger.info("Registered tool: %s", name)
        except Exception:  # don't let one tool failure stop the server
            logger.exception("Failed to register tool '%s'; skipping", name)

    if not registered:
        logger.warning("No tools registered; server will expose an empty tool list")
    return registered
